"""Configuration for the environment and the three training stages.

One tree of dataclasses: `config_io.py` serializes it, `config_cli.py` reflects it
into flags, and `RacingEnvConfig.obs_dim` is the only statement of the observation
layout later stages size themselves from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Annotated, Any

from tinyml_racing.sim.lidar import LidarConfig
from tinyml_racing.sim.track import TrackConfig
from tinyml_racing.utils import RUNS_ROOT

# Disjoint by construction rather than by convention: a policy that memorized
# its training layouts would otherwise look perfect on evaluation.
TRAIN_SEED_RANGE: tuple[int, int] = (0, 2_000_000_000)
EVAL_SEED_RANGE: tuple[int, int] = (2_000_000_000, 2**31 - 1)

# Saturation `VecNormalize` applies to normalized observations, in standard
# deviations. Part of the function the exported weights compute, so every stage
# that seeds them must agree on it.
CLIP_OBS: float = 10.0


def _floor(cfg: object, minimum: float, counts: str, *names: str) -> None:
    """Raise on the first field of `cfg` below `minimum`, naming what it counts.

    Every validation here is a count with a floor, so it is one function rather
    than a block per field; `counts` is what makes the message say why.
    """
    for name in names:
        value = getattr(cfg, name)
        if value < minimum:
            raise ValueError(f"{name} is {counts} and must be >= {minimum:g}, got {value}")


@dataclass
class RacingEnvConfig:
    """Observation layout, sensor model, episode limits, and reward shaping.

    Unfrozen on purpose: `watch.py` rebinds `fixed_track_seed` on a live env's
    config to pin the layout mid-run.
    """

    # Owned by `sim/lidar.py`, nested rather than flattened, so there is no second
    # set of defaults to disagree with that module.
    lidar: LidarConfig = field(default_factory=LidarConfig)
    # Seconds, not steps: a step count is a different fraction of a lap the moment
    # the control rate moves. Several laps, since the longest layout is 2400 m,
    # ~70 s at a pessimistic 35 m/s.
    max_episode_seconds: float = 200.0
    # Explicit override in steps, for tests and for reproducing a `config.json`
    # recorded before the budget became a duration.
    max_episode_steps: Annotated[int | None, "a step count, overriding the duration above"] = None
    # A car going nowhere is truncated after this instead of costing the full
    # budget: standing still pays zero, so nothing in the reward ends it. 0 disables.
    stall_seconds: float = 5.0

    # Drawn from a pool built once rather than generated per episode: a track costs
    # ~33 ms against ~49 us for a step.
    n_tracks: int = 64
    track_seed_range: tuple[int, int] = TRAIN_SEED_RANGE
    fixed_track_seed: Annotated[int | None, "pin every episode to one layout (debug)"] = None
    # Generator tunables, owned by `sim/track.py` for the same reason `lidar` is.
    track: TrackConfig = field(default_factory=TrackConfig)

    # Dimensionless: progress is a fraction of a flat-out step's distance and the
    # steering cost a fraction of full lock, so neither rescales with the car.
    steer_rate_penalty: float = 0.02
    # Leaving the track, in seconds of flat-out progress. It stays the larger half
    # of a crash; the other half is the forfeited future, ~500 units at this horizon.
    off_track_seconds: float = 5.0
    # Potential-based shaping on |cross_track| (Ng et al. 1999), in reward units at
    # the wall. The reward carries the potential's difference, so this moves when
    # credit arrives without moving the optimum.
    cross_track_weight: float = 5.0

    # Composition. Every block is a count, never a flag: 0 turns one off, and
    # `obs_dim` derives the layout from these alone. Sweeps are the current one then
    # `scan_history - 1` successive differences.
    scan_history: int = 2
    # Past throttle commands, newest first. 0 drops the block.
    throttle_history: int = 1
    # Signed cross-track error, newest first, as a fraction of corridor half-width.
    # Off by default: privileged racing-line state a deployed car has no map for.
    cross_track_history: int = 0

    def __post_init__(self) -> None:
        # Each is a width, so a zero or negative count hands the exporter a
        # different network than the run configured rather than a worse observation.
        _floor(self, 1, "a sweep count", "scan_history")
        _floor(self, 0, "an observation block width", "throttle_history", "cross_track_history")

    @property
    def obs_dim(self) -> int:
        """Width of the observation, and the order `_observe` builds it.

        Scan, `scan_history - 1` differences, `[vx, vy, yaw_rate, steer]`, past
        throttles, signed cross-track. The scan stays first and stays what the
        sensor produced.
        """
        return (
            self.scan_history * self.lidar.n_rays
            + 4
            + self.throttle_history
            + self.cross_track_history
        )

    def episode_steps(self, dt: float) -> int:
        """Truncation limit in steps: the override if set, else the duration at `dt`."""
        if self.max_episode_steps is not None:
            return int(self.max_episode_steps)
        return round(self.max_episode_seconds / dt)

    def evaluation_variant(self, *, clean_sensor: bool = True) -> RacingEnvConfig:
        """Same physics and optics, held-out layouts, the detector you ask for.

        `fixed_track_seed` is cleared unconditionally, so a run pinned to one
        training layout cannot report it as a held-out score.
        """
        return replace(
            self,
            track_seed_range=EVAL_SEED_RANGE,
            fixed_track_seed=None,
            lidar=self.lidar.clean() if clean_sensor else self.lidar,
        )


@dataclass(frozen=True)
class PPOConfig:
    """Pass-throughs to the `PPO(...)` constructor, resolved by `as_kwargs`.

    Named fields rather than a dict, so a mistyped hyperparameter is an argparse
    error before the run starts, not a `TypeError` from inside SB3 once the
    environments are up.
    """

    n_steps: Annotated[int, "rollout length per worker; one update per n_steps x n_envs"] = 2048
    batch_size: int = 256
    learning_rate: float = 3e-5
    ent_coef: float = 0.003
    clip_range: float = 0.2

    # Per-step factors mean a different amount of time at every control rate, so
    # these are held as the durations they stand for: value horizon, credit window.
    # See docs/findings/training-stages.md.
    discount_s: float = 10.0
    credit_s: float = 3.0
    # Per-step overrides. Set one to pin it, leave None to derive it from the
    # durations above.
    gamma: Annotated[float | None, "per-step discount, overriding --ppo-discount-s"] = None
    gae_lambda: Annotated[float | None, "per-step GAE factor, overriding --ppo-credit-s"] = None

    def as_kwargs(self, dt: float) -> dict[str, Any]:
        """The mapping `PPO(...)` is constructed with, at control interval `dt`.

        Written out rather than obtained by subtraction, which would forward
        anything new and raise from inside SB3's constructor.
        """
        return {
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "ent_coef": self.ent_coef,
            "clip_range": self.clip_range,
            "gamma": self.gamma if self.gamma is not None else math.exp(-dt / self.discount_s),
            "gae_lambda": (
                self.gae_lambda
                if self.gae_lambda is not None
                else min(max(1.0 - dt / self.credit_s, 0.0), 1.0)
            ),
        }


@dataclass(frozen=True)
class RegressionConfig:
    """The two supervised stages that bracket PPO.

    Both fit the same network to (observation, action) pairs and differ only in who
    produced the actions: `PurePursuit` before, the trained policy after. One
    config, because there is one procedure.
    """

    # Behaviour cloning, before PPO, from the pure-pursuit expert driving the
    # racing line. 0 skips the stage and PPO starts from a random policy.
    pretrain_samples: Annotated[int, "expert samples cloned before PPO (0 skips it)"] = 200_000
    # Distillation into `student_arch`, on by default because the student is what
    # ships: `pi_arch` is sized for the optimizer, `student_arch` for flash. It
    # overwrites `snapshot.pt`, so the exporter needs no wiring. 0 skips the stage,
    # and so does `student_arch == pi_arch`, which `train.train` detects and logs,
    # since fitting a net into its own shape only clones it.
    distill_samples: Annotated[int, "samples distilled into a student after PPO (0 skips it)"] = (
        200_000
    )
    student_arch: Annotated[tuple[int, ...], "hidden layers of the distilled student"] = (16, 8)

    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 1e-3
    # Held out of the fit, so the reported error is one the optimizer never saw.
    val_frac: float = 0.05
    # DART label noise as a fraction of the action range: the executed action is the
    # teacher's plus N(0, sigma), the label stays the teacher's, so the clone sees
    # the states its own errors produce.
    noise_std: float = 0.05
    # Separate from `TrainConfig.seed` because they seed separate things:
    # changing PPO's seed must not redraw the dataset it was cloned on.
    seed: int = 0

    def __post_init__(self) -> None:
        # Checked at construction, not first read: a fit is configured minutes
        # before it runs and publishes `snapshot.pt`. `epochs = 0` is not a shorter
        # fit; the loop never runs, mse stays nan, and a random policy is published
        # as PPO's warm start.
        _floor(self, 1, "a pass count over the dataset", "epochs")
        _floor(self, 1, "a row count per optimizer step", "batch_size")
        # 0 skips the stage and is meaningful; negative is a sign error `collect`
        # would turn into an empty dataset.
        _floor(self, 0, "a transition count", "pretrain_samples", "distill_samples")
        # 1.0 would hold the entire dataset out and leave the fit nothing to
        # learn from; 0.0 is legal and means no validation split.
        if not 0.0 <= self.val_frac < 1.0:
            raise ValueError(
                f"val_frac is the fraction of episodes held out of the fit and must be "
                f"in [0, 1), got {self.val_frac}"
            )


@dataclass(frozen=True)
class PolicyKwargs:
    """The `policy_kwargs=` bag SB3 spreads into its policy constructor.

    An adapter for two boundaries, `PPO(policy_kwargs=...)` and the `snapshot.pt`
    payload, so it carries no architecture defaults; `TrainConfig` chooses those.
    """

    pi_arch: tuple[int, ...]
    vf_arch: tuple[int, ...]
    log_std_init: float = 0.0

    def as_kwargs(self) -> dict[str, Any]:
        """The mapping SB3 spreads into its policy constructor, and what the
        `snapshot.pt` payload stores.

        A `{"pi": ..., "vf": ...}` dict, not a flat list, which SB3 reads as both
        heads at once.
        """
        return {
            "net_arch": {"pi": list(self.pi_arch), "vf": list(self.vf_arch)},
            "log_std_init": self.log_std_init,
        }


# Initial exploration std, as a log, for a run whose trunk arrives cloned. SB3's
# default 0.0 is std 1.0 across an action range of 2, noise as wide as the action
# space, which throws the cloning away. -1.0 is std 0.37.
WARM_START_LOG_STD = -1.0


@dataclass
class TrainConfig:
    """Run bookkeeping and the PPO hyperparameters, all as named fields."""

    n_envs: int = 16
    total_steps: Annotated[int, "PPO's environment-step budget for the stage"] = 2_000_000
    run_name: str | None = None
    runs_root: str = str(RUNS_ROOT)
    seed: int = 0

    # Sized independently: the actor ships, the critic is discarded at export, so
    # its units are free. A shipped `model.h` records the architecture it was built
    # from.
    pi_arch: Annotated[tuple[int, ...], "hidden layers of the actor, the net that ships"] = (
        64,
        64,
    )
    vf_arch: Annotated[tuple[int, ...], "hidden layers of the critic, discarded at export"] = (
        256,
        256,
    )
    # None resolves against how the run started, see `policy_kwargs`. Set it
    # to pin the exploration std regardless.
    log_std_init: Annotated[
        float | None,
        f"pin the initial exploration std; unset means 0 from scratch and "
        f"{WARM_START_LOG_STD} on a cloned policy",
    ] = None

    # Intervals below are in *environment* steps, not policy updates, and are
    # converted to callback frequencies with `n_envs` factored in.
    checkpoint_freq: int = 100_000
    # Rolling window on disk. Checkpoints are restart points, not a record:
    # `EvalCallback` keeps the best policy and training saves the last.
    checkpoint_keep: int = 20
    snapshot_freq: int = 10_000
    eval_freq: int = 100_000
    n_eval_episodes: int = 5

    ppo: PPOConfig = field(default_factory=PPOConfig)
    regression: RegressionConfig = field(default_factory=RegressionConfig)

    def __post_init__(self) -> None:
        # At startup, not first read: `checkpoint_keep` reaches its callback only
        # after the pretrain stage, so a bad value would surface minutes in.
        _floor(self, 1, "a worker-environment count", "n_envs")
        _floor(self, 1, "PPO's environment-step budget", "total_steps")
        _floor(self, 1, "the number of checkpoints kept on disk", "checkpoint_keep")
        _floor(self, 1, "an episode count per evaluation", "n_eval_episodes")

    def policy_kwargs(self, warm_started: bool = False) -> PolicyKwargs:
        """The bag PPO's policy is built from, at this run's architecture.

        The exploration std depends on what the trunk arrives as rather than on
        what the user asked for, so it resolves against `warm_started` unless
        pinned.
        """
        log_std = self.log_std_init
        if log_std is None:
            log_std = WARM_START_LOG_STD if warm_started else 0.0
        return PolicyKwargs(pi_arch=self.pi_arch, vf_arch=self.vf_arch, log_std_init=log_std)
