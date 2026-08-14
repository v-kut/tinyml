"""The PPO stage: reinforcement learning against the simulator's reward.

A warm start is weights, statistics and a fitted critic, since the trunk is a
function of normalized observations and a good actor beside a random critic produces
advantages that say nothing. `WarmStart` carries all three.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import override

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from tinyml_racing import progress
from tinyml_racing.ml.config import CLIP_OBS, RacingEnvConfig, TrainConfig
from tinyml_racing.ml.env import RacingEnv
from tinyml_racing.ml.rl.callbacks import (
    PolicySnapshotCallback,
    ProgressCallback,
    QuietEvalCallback,
    RotatingCheckpointCallback,
    TrainingMetricsCallback,
)
from tinyml_racing.ml.rollout import eval_seeds
from tinyml_racing.ml.snapshot import ObsNorm, publish_snapshot
from tinyml_racing.sim.car import CarParams
from tinyml_racing.utils import Run

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WarmStart:
    """Everything a supervised stage hands PPO, and nothing PPO can infer.

    `reward_accumulator` replays `VecNormalize.returns`, so the reward scale
    the value head was fit in is the scale PPO starts with.
    """

    state_dict: Mapping[str, torch.Tensor]
    obs_norm: ObsNorm
    reward_accumulator: np.ndarray


def make_env(rank: int, env_cfg: RacingEnvConfig, seed: int):
    """Factory for one worker environment.

    `Monitor` is not optional: it records episode returns and lengths into
    `infos`, which is where SB3's `rollout/ep_rew_mean` comes from.
    """

    def _init():
        return Monitor(RacingEnv(config=env_cfg, seed=seed + rank))

    return _init


def worker_seed_base(seed: int) -> int:
    """The first PPO worker's env seed, for a run started at `seed`.

    Not `seed` itself: `collect` builds `RacingEnv(seed=cfg.seed)` and both config
    seeds default to 0, so worker 0 would replay the dataset's layouts. Hashed, so
    `--regression-seed` cannot collide either.
    """
    return int(np.random.SeedSequence([seed, 0x50504F]).generate_state(1)[0])


def build_train_env(env_cfg: RacingEnvConfig, train_cfg: TrainConfig, gamma: float) -> VecNormalize:
    base = worker_seed_base(train_cfg.seed)
    factories = [make_env(i, env_cfg, base) for i in range(train_cfg.n_envs)]
    # Subprocesses only pay off once there is real work to parallelize;
    # a single env in its own process just adds IPC latency per step.
    venv = SubprocVecEnv(factories) if train_cfg.n_envs > 1 else DummyVecEnv(factories)
    return VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        clip_obs=CLIP_OBS,
        # PPO's discount, not SB3's 0.99 default: `ret_rms` tracks a discounted
        # return, so a mismatch scales rewards against an unused horizon.
        gamma=gamma,
        clip_reward=reward_clip(),
    )


def reward_clip() -> float:
    """Saturation `VecNormalize` applies to each reward, in normalized units.

    `c` counts standard deviations of the discounted-return accumulator, not reward:
    20 admits the off-track penalty at its worst while clipping ~0.1% of ordinary
    steps. A function because `train._value_targets` shares the bound.
    """
    return 20.0


class FixedEpisode(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """An env whose every `reset` replays the same episode, so successive
    evaluations score the same task and `best_model.zip` is a real comparison.
    A caller-passed seed is ignored on purpose.
    """

    def __init__(self, env, seed: int):
        super().__init__(env)
        self._episode_seed = seed

    @override
    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=self._episode_seed, options=options)


def make_eval_env(eval_cfg: RacingEnvConfig, track_seed: int):
    """Factory for one evaluation worker, pinned to one held-out layout."""

    def _init():
        # `fixed_track_seed` set *after* `evaluation_variant()`, never before:
        # the variant clears it precisely so a `--fixed-track-seed` training
        # run cannot end up evaluating on the layout it trained on.
        cfg = replace(eval_cfg, fixed_track_seed=track_seed)
        return Monitor(FixedEpisode(RacingEnv(config=cfg, seed=track_seed), track_seed))

    return _init


def build_eval_env(
    env_cfg: RacingEnvConfig, train_cfg: TrainConfig, train_env: VecNormalize
) -> VecNormalize:
    """One env per evaluation episode, each pinned to a held-out seed.

    `EvalCallback` compares the mean return against the best so far, which only means
    anything over the same episodes. Seeds come from `rollout.eval_seeds`, and one env
    per episode is how `evaluate_policy` divides them evenly.

    Deliberately asymmetric with training: this sensor is the clean one, so
    `best_model.zip` and the `eval` number are selected without dropouts or range
    noise while PPO learns from whatever `env_cfg.lidar` configures.
    """
    eval_cfg = env_cfg.evaluation_variant(clean_sensor=True)
    seeds = eval_seeds(eval_cfg, train_cfg.n_eval_episodes)
    venv = VecNormalize(
        DummyVecEnv([make_eval_env(eval_cfg, seed) for seed in seeds]),
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=CLIP_OBS,
    )
    # A starting value, not a live link: assigning `train_env.obs_rms` would bind the
    # live object and track training statistics until `EvalCallback` assigns its own
    # copy. Without any seed the wrapper could be read at `count = epsilon`.
    venv.obs_rms = copy.deepcopy(train_env.obs_rms)
    return venv


# Ceiling on the sample count a warm start hands `VecNormalize`: roughly one rollout,
# so the first PPO rollout counts about as much as the teacher and the teacher's share
# halves thereafter. Freezing it would hide PPO's own drift.
WARM_START_MAX_COUNT = 20_000


def seed_normalization(venv: VecNormalize, warm: WarmStart) -> None:
    """Seed `VecNormalize` with the teacher's statistics.

    A fresh wrapper normalizes at `count = epsilon`, dividing its first batch by that
    batch's own variance, a distribution the cloned trunk has never seen. Counts travel
    with the statistics, capped at `WARM_START_MAX_COUNT`.
    """
    n = min(float(len(warm.reward_accumulator)), float(WARM_START_MAX_COUNT))
    # `VecNormalize` types these as `dict | RunningMeanStd` to cover dict
    # observation spaces; ours is a `Box`, so both are the plain accumulator.
    obs_rms, ret_rms = venv.obs_rms, venv.ret_rms
    if not isinstance(obs_rms, RunningMeanStd) or not isinstance(ret_rms, RunningMeanStd):
        raise TypeError(
            "expected Box observations: VecNormalize accumulators must be RunningMeanStd, "
            f"got {type(obs_rms).__name__} and {type(ret_rms).__name__}"
        )
    obs_rms.mean = np.asarray(warm.obs_norm.mean, dtype=np.float64)
    obs_rms.var = np.asarray(warm.obs_norm.var, dtype=np.float64)
    obs_rms.count = n
    ret_rms.mean = np.float64(warm.reward_accumulator.mean())
    ret_rms.var = np.float64(warm.reward_accumulator.var())
    ret_rms.count = n


def train_ppo(
    run: Run,
    env_cfg: RacingEnvConfig,
    train_cfg: TrainConfig,
    warm_start: WarmStart | None = None,
) -> None:
    """Run PPO into `run`, optionally starting from a cloned policy."""
    policy_kwargs = train_cfg.policy_kwargs(warm_started=warm_start is not None)
    # `dt` decides what `gamma` and `gae_lambda` are worth, and this runs before
    # the envs are built because `VecNormalize` scales the reward against the
    # same discount.
    ppo_kwargs = train_cfg.ppo.as_kwargs(CarParams().dt)
    logger.info(
        "credit: gamma %.5f (%.1f s horizon), lambda %.4f (%.1f s window) at dt=%.3f s",
        ppo_kwargs["gamma"],
        train_cfg.ppo.discount_s,
        ppo_kwargs["gae_lambda"],
        train_cfg.ppo.credit_s,
        CarParams().dt,
    )

    train_env = build_train_env(env_cfg, train_cfg, ppo_kwargs["gamma"])
    if warm_start is not None:
        seed_normalization(train_env, warm_start)
    eval_env = build_eval_env(env_cfg, train_cfg, train_env)
    logger.info(
        "reward: crash costs %.0f (%.1f s of flat-out progress), reward clip +/-%.0f sigma",
        env_cfg.off_track_seconds / CarParams().dt,
        env_cfg.off_track_seconds,
        train_env.clip_reward,
    )
    model = PPO(
        "MlpPolicy",
        train_env,
        **ppo_kwargs,
        policy_kwargs=policy_kwargs.as_kwargs(),
        tensorboard_log=str(run.tb),
        seed=train_cfg.seed,
        # CPU, always. The deployed net is a few hundred MACs at whatever
        # `--pi-arch` was trained; a GPU round-trip per forward pass costs
        # more than the forward pass.
        device="cpu",
        # Quiet: SB3's table is twenty lines per rollout and every number in
        # it already goes to TensorBoard. `ProgressCallback` keeps the two
        # that answer "is it working" on the bar.
        verbose=0,
    )
    if warm_start is not None:
        # In place, so the optimizer SB3 built over these parameters keeps pointing at
        # them. The state dicts match because the supervised stage used the same
        # `policy_kwargs`.
        model.policy.load_state_dict(warm_start.state_dict)
        logger.info(
            "warm start: cloned actor and critic loaded, exploration std %.3f",
            float(np.exp(policy_kwargs.log_std_init)),
        )

    # `CheckpointCallback` and `EvalCallback` count `_on_step` calls, one per
    # vectorized step, so the configured environment-step intervals are divided.
    # `EveryNTimesteps` already counts environment steps.
    callbacks: list[BaseCallback] = [
        TrainingMetricsCallback(),
        RotatingCheckpointCallback(
            save_freq=max(train_cfg.checkpoint_freq // train_cfg.n_envs, 1),
            save_path=str(run.checkpoints),
            name_prefix="ppo_racing",
            save_vecnormalize=True,
            keep=train_cfg.checkpoint_keep,
        ),
        EveryNTimesteps(
            train_cfg.snapshot_freq,
            PolicySnapshotCallback(run, env_cfg, policy_kwargs),
        ),
        evaluator := QuietEvalCallback(
            eval_env,
            # Both land in `training/`, under names the callback picks:
            # `best_model.zip` and `evaluations.npz`.
            best_model_save_path=str(run.training),
            log_path=str(run.training),
            n_eval_episodes=train_cfg.n_eval_episodes,
            eval_freq=max(train_cfg.eval_freq // train_cfg.n_envs, 1),
            deterministic=True,
            render=False,
            # Its own printing is what the live display replaces; one line per
            # evaluation reaches `train.log` from `QuietEvalCallback` instead.
            verbose=0,
        ),
    ]

    # `learn` collects whole rollouts, overshooting `total_steps` by up to one, so the
    # bar's target is rounded up rather than the count being wrong.
    rollout = train_cfg.ppo.n_steps * train_cfg.n_envs
    budget = math.ceil(train_cfg.total_steps / rollout) * rollout
    if rollout > train_cfg.total_steps:
        # SB3 updates once per rollout, so a rollout wider than the budget trains
        # once, after collecting `rollout` steps, and scores the policy the stage
        # started with. `--ppo-n-steps` is the buffer per worker, `--total-steps` the
        # budget.
        logger.warning(
            "rollout is %d steps (--ppo-n-steps %d x --n-envs %d) against a "
            "--total-steps budget of %d: PPO will collect %d steps, apply one "
            "update, and stop. Buffer holds %.1f GiB of observations. Raise "
            "--total-steps or lower --ppo-n-steps.",
            rollout,
            train_cfg.ppo.n_steps,
            train_cfg.n_envs,
            train_cfg.total_steps,
            rollout,
            rollout * env_cfg.obs_dim * 4 / 2**30,
        )
    with progress.stage("ppo", budget) as bar:
        callbacks.append(ProgressCallback(bar, evaluator))
        try:
            model.learn(total_timesteps=train_cfg.total_steps, callback=callbacks)
        finally:
            _finish(run, model, env_cfg, policy_kwargs, train_env, eval_env)


def _finish(run, model, env_cfg, policy_kwargs, train_env, eval_env) -> None:
    """Leave the run loadable, whatever ended it: a clean finish, the skip key
    and KeyboardInterrupt all land here, and each must leave a policy plus the
    normalization statistics that policy requires.
    """
    model.save(str(run.final_model))
    train_env.save(str(run.vecnormalize))
    # `EveryNTimesteps` fires on intervals, so without this the run's files lag by up
    # to `snapshot_freq` steps, and after an interrupt or the skip key by more.
    publish_snapshot(
        run.ppo,
        run.snapshot,
        model.policy,
        env_cfg,
        policy_kwargs,
        model.num_timesteps,
        obs_norm=ObsNorm.from_venv(train_env),
    )
    train_env.close()
    eval_env.close()
    logger.info("PPO finished at %d steps", model.num_timesteps)
