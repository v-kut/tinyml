"""Deterministic rollout shared by `deploy/export`'s calibration pass and the
three scoring paths, so they cannot disagree about observation normalization.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from tinyml_racing.ml.env import RacingEnv
from tinyml_racing.sim.car import CarState

ActFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class Frame:
    """One simulation step. The fields straddle it on purpose: `obs` is what
    went *in* to `act`, while `state` and `scan` are post-step, because
    `RacingEnv.step` calls `_observe()` last.
    """

    step: int
    obs: np.ndarray  # pre-step, raw as `RacingEnv` produced it
    state: CarState  # post-step
    action: np.ndarray
    reward: float
    scan: np.ndarray  # normalized LiDAR readings, post-step
    # Gymnasium's `info` is an open-ended bag by contract, not a fixed field
    # set, so this one stays a dict where the rest of the codebase does not.
    info: dict[str, Any]
    terminated: bool
    truncated: bool


def iter_rollout(env, act: ActFn, max_steps: int, seed: int | None = None) -> Iterator[Frame]:
    """Drive `env` with `act`, yielding one `Frame` per step as it happens."""
    obs, _ = env.reset(seed=seed)
    for i in range(max_steps):
        action = np.asarray(act(obs), dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        yield Frame(
            step=i,
            obs=obs,
            state=env.state,
            action=action,
            reward=float(reward),
            scan=env.last_scan,
            info=info,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        if terminated or truncated:
            return
        obs = next_obs


@dataclass
class RolloutSummary:
    """Aggregate outcome of one episode, as `closed_loop` averages it."""

    steps: int
    total_reward: float
    # Laps completed, not position on the lap: the spawn is a random point, so
    # the raw arc length carries a U(0, 1) offset that swamps the answer.
    lap_progress: float
    max_speed: float
    crashed: bool


def run_rollout(env, act: ActFn, max_steps: int, seed: int | None = None) -> RolloutSummary:
    total = 0.0
    speed = 0.0
    progress = 0.0
    crashed = False
    steps = 0

    for frame in iter_rollout(env, act, max_steps, seed=seed):
        total += frame.reward
        speed = max(speed, float(frame.state.vx))
        progress = float(frame.info.get("lap_progress", progress))
        crashed = frame.terminated
        steps = frame.step + 1

    return RolloutSummary(
        steps=steps,
        total_reward=total,
        lap_progress=progress,
        max_speed=speed,
        crashed=crashed,
    )


def eval_seeds(env_cfg, n_tracks: int) -> list[int]:
    """Held-out track seeds, drawn the way the trainer's evaluator draws
    them, so these numbers are comparable to every previously reported run.
    """
    rng = np.random.default_rng(0)
    return [int(s) for s in rng.integers(*env_cfg.track_seed_range, size=n_tracks)]


def closed_loop(act: ActFn, env_cfg, seeds, max_steps: int, on_lap=None) -> dict[str, float]:
    """Lap `seeds` one at a time with `act`, averaged.

    The one score every stage is judged by, so a regression stage's reward and
    `deploy/evaluate.py`'s numbers mean the same measurement. A seed names the
    layout and a `SeedSequence`-derived stream the spawn, so `reward_std` spans both.
    `on_lap(i)` is called after each lap, for a caller driving a progress bar.
    """
    rewards, progress, lengths, crashes, speeds = [], [], [], [], []
    for i, seed in enumerate(seeds, start=1):
        track_seed = int(seed)
        spawn_seed = int(np.random.SeedSequence(track_seed).generate_state(1)[0])
        env = RacingEnv(config=replace(env_cfg, fixed_track_seed=track_seed), seed=track_seed)
        summary = run_rollout(env, act, max_steps=max_steps, seed=spawn_seed)
        env.close()
        rewards.append(summary.total_reward)
        progress.append(summary.lap_progress)
        lengths.append(summary.steps)
        crashes.append(float(summary.crashed))
        speeds.append(summary.max_speed)
        if on_lap is not None:
            on_lap(i)
    return {
        "reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "progress": float(np.mean(progress)),
        "steps": float(np.mean(lengths)),
        "crash_rate": float(np.mean(crashes)),
        "max_speed": float(np.mean(speeds)),
    }
