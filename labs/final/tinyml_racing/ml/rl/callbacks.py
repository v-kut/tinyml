"""Training-loop instrumentation: metrics, snapshots, progress bar, checkpoint
rotation. Everything runs synchronously on the training thread and nothing is
wrapped in a bare `except`, so a broken callback fails the run loudly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import override

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

from tinyml_racing.ml.config import PolicyKwargs, RacingEnvConfig
from tinyml_racing.ml.snapshot import ObsNorm, publish_snapshot, save_snapshot
from tinyml_racing.utils import Run

logger = logging.getLogger(__name__)

# Per-step diagnostics from `RacingEnv.step`'s info dict, logged as means over
# the rollout: `ep_rew_mean` says whether the run works, these say which reward
# term is doing it.
_STEP_MEANS = {
    "r_progress": "reward/progress",
    "r_steer_rate": "reward/steer_rate",
    "r_shaping": "reward/shaping",
    "cross_track": "car/cross_track_m",
    "grip_use": "car/grip_use",
    "action_rail": "action/rail_frac",
    "action_steer": "action/steer_abs",
    "action_throttle_abs": "action/throttle_abs",
}


class TrainingMetricsCallback(BaseCallback):
    """Reward decomposition, lap outcome, and behaviour, on PPO's log cadence.

    Accumulated in `_on_step`, recorded in `_on_rollout_end` because `record`
    overwrites. `reward/*` is raw env units, comparable to `RacingEnvConfig`'s
    weights; `reward/clip_frac` reports on the wrapper, so it reads the scaled one.
    """

    def __init__(self) -> None:
        super().__init__()
        self._clip: float | None = None
        self._reset()

    @override
    def _on_training_start(self) -> None:
        # `VecNormalize` is the outer wrapper, so its bound is readable here.
        self._clip = getattr(self.model.get_env(), "clip_reward", None)

    def _reset(self) -> None:
        self._steps = 0
        self._sums = dict.fromkeys(_STEP_MEANS, 0.0)
        self._speed_sum = 0.0
        # -inf rather than 0.0: a running `max` from zero reports 0 for a spun
        # or reversing policy, hiding the failure this metric exists to flag.
        # `_on_rollout_end` returns early on an empty rollout.
        self._speed_max = -np.inf
        self._saturated = 0
        self._clipped = 0
        self._laps: list[float] = []
        self._crashes: list[float] = []
        self._crash_cost: list[float] = []
        self._stalls: list[float] = []

    @override
    def _on_rollout_start(self) -> None:
        self._reset()

    @override
    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        for i, (info, done) in enumerate(
            zip(self.locals["infos"], self.locals["dones"], strict=True)
        ):
            self._steps += 1
            for key in self._sums:
                self._sums[key] += info[key]
            speed = info["speed"]
            self._speed_sum += speed
            self._speed_max = max(self._speed_max, speed)
            self._saturated += info["grip_use"] > 1.0
            # `rewards`, not `info`: `VecNormalize` divides by the running
            # return std and then clips, so this counts a normalized reward
            # against a bound stated in the same units (`ppo.reward_clip`).
            if self._clip is not None and abs(rewards[i]) >= self._clip - 1e-6:
                self._clipped += 1
            if done:
                self._laps.append(info["lap_progress"])
                self._crashes.append(float(info["off_track"]))
                self._stalls.append(float(info["stalled"]))
                self._crash_cost.append(info["r_crash"])
        return True

    @override
    def _on_rollout_end(self) -> None:
        if not self._steps:
            return
        n = self._steps
        for key, tag in _STEP_MEANS.items():
            self.logger.record(tag, self._sums[key] / n)
        self.logger.record("car/speed_mean", self._speed_sum / n)
        self.logger.record("car/speed_max", float(self._speed_max))
        self.logger.record("car/grip_saturated_frac", self._saturated / n)
        self.logger.record("reward/clip_frac", self._clipped / n)
        if not self._laps:
            return
        # Per *episode*, not per step: the fraction of a lap actually completed is
        # the task, and reward is only its proxy. A run whose return climbs while
        # this does not has found something other than driving round the track.
        self.logger.record("episode/lap_progress", float(np.mean(self._laps)))
        self.logger.record("episode/crash_rate", float(np.mean(self._crashes)))
        # Episodes that ended because the car stopped getting anywhere. Belongs
        # at zero: it is not a failure the reward names, so nothing else in this
        # table would show a policy that had settled on parking.
        self.logger.record("episode/stall_rate", float(np.mean(self._stalls)))
        self.logger.record("reward/crash_per_episode", float(np.mean(self._crash_cost)))


class PolicySnapshotCallback(BaseCallback):
    """Republish the current policy as PPO trains, to both files it owes.

    The training -> TinyML handoff: `deploy/export.py` loads the snapshot, so it
    carries the observation layout and the `VecNormalize` statistics. Also
    `training/ppo.pt`, so PPO's stage file exists mid-run. Wrap in `EveryNTimesteps`.
    """

    def __init__(self, run: Run, env_cfg: RacingEnvConfig, policy_kwargs: PolicyKwargs) -> None:
        super().__init__()
        self.run = run
        self.env_cfg = env_cfg
        self.policy_kwargs = policy_kwargs

    @override
    def _on_training_start(self) -> None:
        # Publish before the first interval elapses, otherwise anything
        # polling the run directory sits empty for minutes and looks broken.
        self._on_step()

    @override
    def _on_step(self) -> bool:
        publish_snapshot(
            self.run.ppo,
            self.run.snapshot,
            self.model.policy,
            self.env_cfg,
            self.policy_kwargs,
            self.num_timesteps,
            obs_norm=ObsNorm.from_venv(self.model.get_env()),
        )
        return True


class BestSnapshotCallback(PolicySnapshotCallback):
    """The best-scoring policy so far, in the format the rest of the run reads.

    Hangs off `EvalCallback(callback_on_new_best=...)`, so SB3 decides what
    "best" is and this only records it. Its own `best_model.zip` is not enough:
    `EvalCallback` saves the model alone, and a policy without the
    `VecNormalize` statistics it was trained against cannot be scored, distilled
    or exported. Distillation teaches from this file.
    """

    @override
    def _on_training_start(self) -> None:
        # No evaluation has happened yet, so there is no best to record. The
        # inherited hook would publish the starting policy as one.
        return

    @override
    def _on_step(self) -> bool:
        save_snapshot(
            self.run.best,
            self.model.policy,
            self.env_cfg,
            self.policy_kwargs,
            self.num_timesteps,
            obs_norm=ObsNorm.from_venv(self.model.get_env()),
        )
        return True


class QuietEvalCallback(EvalCallback):
    """`EvalCallback` without its multi-line console dump.

    SB3 prints two lines per evaluation and one more on every new best, which
    is 806 lines of history for a run whose display is five bars. The numbers
    still reach TensorBoard, `train.log` gets one line, and the newest mean is
    published for the bar to show.
    """

    last_report: str = ""

    @override
    def _on_step(self) -> bool:
        before = self.last_mean_reward
        result = super()._on_step()
        if self.last_mean_reward != before:
            self.last_report = f"eval {self.last_mean_reward:.1f}"
            logger.info(
                "eval at %d steps: %.1f mean reward over %d episodes",
                self.num_timesteps,
                self.last_mean_reward,
                self.n_eval_episodes,
            )
        return result


class ProgressCallback(BaseCallback):
    """Drive the run's progress bar, and let the skip key end learning early.

    Returning False from `_on_step` is SB3's documented way to stop `learn`, and
    it unwinds through the normal exit, so `train_ppo`'s `finally` still saves.
    """

    def __init__(self, bar, evaluator: QuietEvalCallback | None = None) -> None:
        super().__init__()
        self.bar = bar
        # Read, not driven: the evaluator owns when it runs, the bar only shows
        # its newest number beside the rollout means.
        self.evaluator = evaluator

    @override
    def _on_step(self) -> bool:
        buffer = self.model.ep_info_buffer
        parts = []
        if buffer:
            reward = float(np.mean([ep["r"] for ep in buffer]))
            length = float(np.mean([ep["l"] for ep in buffer]))
            parts.append(f"reward {reward:>8.1f} · {length:>5.0f} steps/ep")
        if self.evaluator is not None and self.evaluator.last_report:
            parts.append(self.evaluator.last_report)
        self.bar.set(self.num_timesteps, note=" · ".join(parts))
        return not self.bar.skipped


class RotatingCheckpointCallback(CheckpointCallback):
    """`CheckpointCallback`, keeping only the newest `keep` checkpoints.

    Nothing reads the older ones, the best policy is `best.pt`, the
    last is written at exit, deployment takes `snapshot.pt`, and a long run
    leaves tens of MB of them in the directory a human opens to find the model.
    """

    def __init__(self, *args, keep: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if keep < 1:
            raise ValueError(f"keep must be at least 1, got {keep}")
        self.keep = keep

    @override
    def _on_step(self) -> bool:
        due = self.n_calls % self.save_freq == 0
        result = super()._on_step()
        if due:
            self._prune()
        return result

    def _prune(self) -> None:
        # By the step count in the name, not mtime or sort order: the files are
        # `..._1000000_steps.zip`, and lexicographically that sorts before
        # `..._999999_steps.zip`.
        for pattern in ("_*_steps.zip", "_vecnormalize_*_steps.pkl"):
            found = Path(self.save_path).glob(f"{self.name_prefix}{pattern}")
            for path in sorted(found, key=lambda p: int(p.stem.split("_")[-2]))[: -self.keep]:
                path.unlink()
