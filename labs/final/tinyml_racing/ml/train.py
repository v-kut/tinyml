"""Training entry point: clone the pure-pursuit expert, optimize that policy with
PPO against the simulator's reward, then distill it into the smaller student that
ships. All three stages run by default and only the PPO one is required;
everything lands under `<runs_root>/<run_name>/`. See
`tinyml_racing/ml/README.md` for stage contracts.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np

from tinyml_racing import progress
from tinyml_racing.ml.config import PolicyKwargs, RacingEnvConfig, TrainConfig
from tinyml_racing.ml.config_cli import add_config_arguments, configs_from_args
from tinyml_racing.ml.config_io import config_to_dict
from tinyml_racing.ml.regression.dataset import (
    Dataset,
    collect,
    pure_pursuit_teacher,
    snapshot_teacher,
)
from tinyml_racing.ml.regression.fit import FitResult, fit_policy
from tinyml_racing.ml.rl.ppo import WarmStart, reward_clip, train_ppo
from tinyml_racing.ml.rollout import closed_loop, eval_seeds
from tinyml_racing.ml.snapshot import load_snapshot, publish_snapshot
from tinyml_racing.sim.car import CarParams
from tinyml_racing.utils import Run, save_json, setup_logging

logger = logging.getLogger(__name__)

# `VecNormalize`'s own reward-scaling epsilon. Matched so the value targets the
# critic is fit to are in the units PPO will present rewards in.
REWARD_EPSILON = 1e-8


def link_latest(runs_root: str | Path, run_dir: Path) -> Path | None:
    """Point `<runs_root>/latest` at `run_dir`, or return None if it is not a link.

    Swapped through a temporary name and `os.replace`, so a reader resolving it
    mid-swap sees the old run or the new one. The target is relative, so the
    runs tree survives being moved.
    """
    root = Path(runs_root)
    link = root / "latest"
    if link.exists() and not link.is_symlink():
        return None

    tmp = root / ".latest.tmp"
    tmp.unlink(missing_ok=True)
    target = os.path.relpath(run_dir.resolve(), root.resolve())
    tmp.symlink_to(target, target_is_directory=True)
    Path.replace(tmp, link)
    return link


def publish(
    run: Run,
    stage_path: Path,
    result: FitResult,
    env_cfg: RacingEnvConfig,
    policy_kwargs: PolicyKwargs,
    steps: int,
) -> None:
    """Hand a fitted policy to the run, see `snapshot.publish_snapshot`."""
    publish_snapshot(
        stage_path,
        run.snapshot,
        result.policy,
        env_cfg,
        policy_kwargs,
        steps,
        obs_norm=result.obs_norm,
    )


def score(path: Path, env_cfg: RacingEnvConfig, n_tracks: int, label: str) -> None:
    """Lap held-out tracks with the policy at `path` and log the result.

    Read back off disk, because that file is what the next stage and the deploy
    tools use. Both detectors are scored whenever the sensor carries noise.
    """
    snapshot = load_snapshot(path)
    lidar = env_cfg.lidar
    # Skipped rather than logged twice when the run already trained clean:
    # `evaluation_variant(clean_sensor=False)` is then the same config, and a
    # duplicated line costs a second full pass to say nothing.
    sensors = [True] if lidar.noise_std == 0.0 and lidar.dropout_prob == 0.0 else [True, False]
    for clean_sensor in sensors:
        eval_cfg = env_cfg.evaluation_variant(clean_sensor=clean_sensor)
        stats = closed_loop(
            snapshot.act,
            eval_cfg,
            eval_seeds(eval_cfg, n_tracks),
            eval_cfg.episode_steps(CarParams().dt),
        )
        logger.info(
            "%s over %d held-out tracks (%s sensor): "
            "reward %.1f +/- %.1f, %.2f laps, %.0f%% crashed",
            label,
            n_tracks,
            "clean" if clean_sensor else "noisy",
            stats["reward"],
            stats["reward_std"],
            stats["progress"],
            100.0 * stats["crash_rate"],
        )


def _value_targets(dataset: Dataset, train_cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    """The accumulator that defines PPO's reward units, and the critic targets in it.

    `dataset` must be a noise-free pass: `Dataset.rewards` are what the
    *executed* action earned, so a DART dataset yields targets for
    `V^{teacher+noise}` and a `ret_rms` seed PPO will never reproduce.
    """
    gamma = train_cfg.ppo.as_kwargs(CarParams().dt)["gamma"]
    accumulator = dataset.reward_accumulator(gamma)
    scale = float(np.sqrt(accumulator.var() + REWARD_EPSILON))
    return accumulator, dataset.returns_to_go(gamma, scale, reward_clip())


def pretrain(run: Run, env_cfg: RacingEnvConfig, train_cfg: TrainConfig) -> WarmStart:
    """Stage one: clone the pure-pursuit expert into PPO's own policy shape.

    Two passes over the one teacher, unless they coincide. The actor's carries
    DART noise; the critic's carries none, so the returns it is fit to are the
    value of the policy being cloned rather than of a perturbed one.
    """
    cfg = train_cfg.regression
    dataset = collect(
        pure_pursuit_teacher(),
        env_cfg,
        cfg.pretrain_samples,
        noise_std=cfg.noise_std,
        seed=cfg.seed,
        label="pretrain",
    )
    # The critic's dataset has to be noise-free. At `noise_std == 0` the actor's
    # already is: the same expert over the same `env_cfg` from the same seed
    # collects the identical set, so a second pass would just pay for another
    # `pretrain_samples` transitions of rollout. Otherwise it is the same layout
    # pool and first spawn, diverging only where the noise did, from a fresh
    # `pure_pursuit_teacher` because its controller cache is keyed on the pooled
    # arc-length tables.
    clean = (
        dataset
        if cfg.noise_std == 0.0
        else collect(
            pure_pursuit_teacher(),
            env_cfg,
            cfg.pretrain_samples,
            noise_std=0.0,
            seed=cfg.seed,
            label="pretrain clean",
        )
    )
    accumulator, targets = _value_targets(clean, train_cfg)
    # The same bag PPO will build its policy from, so the state dict transplants
    # key for key rather than by name-matching two similar networks.
    policy_kwargs = train_cfg.policy_kwargs(warm_started=True)
    result = fit_policy(
        dataset,
        env_cfg,
        policy_kwargs,
        cfg,
        value_rows=(clean.obs, targets),
        label="pretrain",
        tb_dir=run.tb / "pretrain",
    )

    publish(run, run.pretrain, result, env_cfg, policy_kwargs, steps=0)
    score(run.pretrain, env_cfg, train_cfg.n_eval_episodes, "cloned expert")
    return WarmStart(
        state_dict=result.policy.state_dict(),
        obs_norm=result.obs_norm,
        reward_accumulator=accumulator,
    )


def distill(run: Run, env_cfg: RacingEnvConfig, train_cfg: TrainConfig) -> None:
    """Stage three: teach `student_arch` to imitate what PPO produced."""
    cfg = train_cfg.regression
    teacher = load_snapshot(run.snapshot)
    dataset = collect(
        snapshot_teacher(teacher),
        env_cfg,
        cfg.distill_samples,
        noise_std=cfg.noise_std,
        # Past the pretraining seed, so the student is not fit on the same
        # layouts and spawns the expert was sampled over.
        seed=cfg.seed + 1,
        label="distill",
    )
    # The student's critic is neither fit (no `value_rows`) nor read: this stage
    # regresses actions, `score` drives through `Snapshot.act`, and the exporter
    # takes the actor alone. Sized to match only to keep construction cheap.
    policy_kwargs = PolicyKwargs(pi_arch=cfg.student_arch, vf_arch=cfg.student_arch)
    result = fit_policy(
        dataset, env_cfg, policy_kwargs, cfg, label="distill", tb_dir=run.tb / "distill"
    )

    publish(run, run.student, result, env_cfg, policy_kwargs, steps=teacher.num_timesteps)
    score(run.student, env_cfg, train_cfg.n_eval_episodes, f"student {cfg.student_arch}")


def train(env_cfg: RacingEnvConfig, train_cfg: TrainConfig) -> Run:
    """Run every configured stage into a fresh run directory."""
    run = Run.create(train_cfg.runs_root, train_cfg.run_name)

    # The display owns the terminal for the rest of the run, so logging has to
    # be told about it before the first record: a plain StreamHandler writes
    # through the live region instead of scrolling above it.
    with progress.session() as report:
        setup_logging(run.log, console=report.console)
        save_json(config_to_dict(env_cfg, train_cfg), run.config)
        if link_latest(train_cfg.runs_root, run.root) is None:
            logger.warning(
                "%s/latest exists and is not a symlink; leaving it alone",
                train_cfg.runs_root,
            )
        logger.info("run %s, tensorboard --logdir %s", run.root, run.tb)

        cfg = train_cfg.regression
        warm_start = None
        if cfg.pretrain_samples > 0:
            warm_start = pretrain(run, env_cfg, train_cfg)

        train_ppo(run, env_cfg, train_cfg, warm_start=warm_start)
        score(run.ppo, env_cfg, train_cfg.n_eval_episodes, "trained policy")

        # Distilling a network into its own shape is not a compression: it
        # re-fits an identical architecture on its own rollouts and overwrites
        # `snapshot.pt` (the file `deploy/` reads) with a behaviour-cloned
        # copy that is strictly worse than the policy it copied.
        if cfg.distill_samples > 0:
            if tuple(cfg.student_arch) == tuple(train_cfg.pi_arch):
                logger.info(
                    "skipping distillation: student_arch %s is already pi_arch, "
                    "so PPO's own policy is what ships",
                    tuple(cfg.student_arch),
                )
            else:
                distill(run, env_cfg, train_cfg)

        logger.info("finished -> %s", run.root)
    return run


def main() -> None:
    # Raw description: the module docstring is prose the default formatter
    # would reflow into one paragraph.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_arguments(parser, RacingEnvConfig, "environment")
    add_config_arguments(parser, TrainConfig, "training")
    env_cfg, train_cfg = configs_from_args(parser.parse_args())
    train(env_cfg, train_cfg)


if __name__ == "__main__":
    main()
