"""Gymnasium environment for a LiDAR to (steering, throttle) policy.

`RacingEnvConfig.obs_dim` states the observation layout and `_observe` builds it.
Action is [steer, throttle] in [-1, 1]. Reward is dense progress along the racing
line, minus off-track and steering-rate penalties.
"""

from __future__ import annotations

import os
from collections import deque
from itertools import pairwise
from typing import TYPE_CHECKING, override

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tinyml_racing.ml.config import RacingEnvConfig
from tinyml_racing.sim.car import CarParams, CarState, lateral_grip_usage, proprioception
from tinyml_racing.sim.car import step as car_step
from tinyml_racing.sim.expert import random_start_state
from tinyml_racing.sim.geometry import ArcLengthLUT
from tinyml_racing.sim.lidar import cast_lidar
from tinyml_racing.sim.track import Track
from tinyml_racing.sim.track_pool import TrackPool

if TYPE_CHECKING:  # `render()` imports the viewer lazily; see its comment.
    from tinyml_racing.render.viewer import PygameViewer

__all__ = ["RacingEnv"]

# "Getting somewhere" as a fraction of top speed: below this over `stall_seconds`
# the car is parked. Well under the `blend_speed` where the tire model drops out.
STALL_SPEED_FRAC = 0.01


def _after_reset[T](value: T | None, name: str) -> T:
    """Per-episode state, or an error naming the step that was skipped.

    Unreachable in a correct rollout; it replaces an `AttributeError` several
    frames deeper.
    """
    if value is None:
        raise RuntimeError(f"RacingEnv.{name} is only valid after reset()")
    return value


class RacingEnv(gym.Env[np.ndarray, np.ndarray]):
    """One racetrack episode: reset onto a layout drawn from this worker's track
    pool, then step the car until it leaves the track or the episode times out.
    """

    def __init__(
        self,
        config: RacingEnvConfig | None = None,
        car_params: CarParams | None = None,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.cfg = config or RacingEnvConfig()
        self.params = car_params or CarParams()
        # Per instance: `render()` emits one frame per `step()`, so a recorder
        # trusting a default `dt` would play the rollout back at the wrong rate.
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": round(1.0 / self.params.dt),
        }
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"unsupported render_mode {render_mode!r}; "
                f"expected one of {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode
        # Distance a flat-out step covers, so progress reward stays O(1) whatever the
        # car is.
        self._progress_ref = self.params.top_speed * self.params.dt
        # One seed, two streams: the pool from `_pool_rng`, per-episode draws from
        # `_rng`, which `reset(seed=S)` re-seeds. Sharing one would void that.
        pool_seq, episode_seq = np.random.SeedSequence(seed).spawn(2)
        self._pool_rng = np.random.default_rng(pool_seq)
        self._rng = np.random.default_rng(episode_seq)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Per-episode state, exposed through the properties below so no caller
        # reasons about the window before the first reset.
        self.pool: TrackPool | None = None
        self._track: Track | None = None
        self._progress: ArcLengthLUT | None = None
        self._state: CarState | None = None
        # Normalized, as the sensor produced it, see `_observe`.
        self.last_scan: np.ndarray | None = None
        # Newest first, refilled per episode so nothing crosses a reset. `maxlen`
        # because eviction is the policy: `scan_history == 1` must leave nothing behind.
        self._scans: deque[np.ndarray] = deque(maxlen=self.cfg.scan_history)
        self._throttles: deque[float] = deque(maxlen=self.cfg.throttle_history)
        self._crosses: deque[float] = deque(maxlen=self.cfg.cross_track_history)
        self._viewer: PygameViewer | None = None
        self._steps = 0
        self._prev_steer = 0.0
        self._s_prev = 0.0
        self._advanced = 0.0
        self._potential = 0.0
        self._half_width = 1.0
        self._stall_from = 0.0
        self._stall_steps = 0
        # Resolved against this car's `dt` once: what truncation means in lap
        # fractions must not depend on the control rate.
        self._max_steps = self.cfg.episode_steps(self.params.dt)
        # Progress pays at most 1 per `dt` seconds, so N seconds of flat-out
        # progress is N/dt reward units.
        self._off_track_cost = self.cfg.off_track_seconds / self.params.dt
        self._stall_limit = round(self.cfg.stall_seconds / self.params.dt)
        self._stall_distance = STALL_SPEED_FRAC * self.params.top_speed * self.cfg.stall_seconds

    @property
    def track(self) -> Track:
        return _after_reset(self._track, "track")

    @property
    def progress(self) -> ArcLengthLUT:
        return _after_reset(self._progress, "progress")

    @property
    def state(self) -> CarState:
        return _after_reset(self._state, "state")

    @property
    def max_steps(self) -> int:
        """Truncation limit, in steps. What a rollout should cap itself at too."""
        return self._max_steps

    def _ensure_pool(self) -> TrackPool:
        # Built on first reset, not `__init__`: `SubprocVecEnv` pickles the factory,
        # so anything built there is generated in the parent. `_pool_rng` keeps the
        # pool owned by the env rather than an episode.
        if self.pool is None:
            if self.cfg.fixed_track_seed is not None:
                self.pool = TrackPool([self.cfg.fixed_track_seed], self.cfg.track)
            else:
                self.pool = TrackPool.from_seed_range(
                    self.cfg.n_tracks,
                    self.cfg.track_seed_range,
                    self._pool_rng,
                    self.cfg.track,
                )
        return self.pool

    def warm_pool(self) -> int:
        """Generate every layout in this env's pool, returning how many there are.

        Spends the ~29 ms per layout before the first rollout rather than on a
        vectorized step's barrier, see `rl.ppo.warm_track_pools`.
        """
        pool = self._ensure_pool()
        for seed in pool.seeds:
            pool.get(seed)
        return len(pool.seeds)

    def _cross_track_potential(self, cross: float) -> float:
        """`Phi` for the shaping term: 0 on the racing line, `-w` at the wall.

        A magnitude: which side the car is on says nothing about how far off it is.
        `step` overrides this with 0 at an absorbing terminal, where Ng et al.'s
        invariance requires `Phi` to vanish.
        """
        return -self.cfg.cross_track_weight * min(abs(cross) / self._half_width, 1.0)

    def _normalized_cross(self, cross: float) -> float:
        """Signed cross-track for the observation, as a fraction of the corridor.

        Signed here, unsigned in the potential: the policy needs to know which way
        to steer back. Saturated at the corridor edge, past which the episode ends.
        """
        return float(np.clip(cross / self._half_width, -1.0, 1.0))

    def _observe(self):
        # Every block arrives normalized by its owner, so this is concatenation in
        # the order `obs_dim` states.
        scan = cast_lidar(self.state, self.track, self.cfg.lidar, self._rng)
        self.last_scan = scan
        if self._scans:
            self._scans.appendleft(scan)
        else:
            # Padded with itself, so the first frame's differences are zero rather
            # than a jump from the previous episode's spawn.
            self._scans.extend([scan] * self.cfg.scan_history)

        # Closure rate, then its rate of change. Differences rather than raw sweeps
        # so int8 spends its range on what moved.
        blocks = [scan]
        blocks += [newer - older for newer, older in pairwise(self._scans)]
        blocks.append(proprioception(self.state, self.params))
        if self._throttles:
            blocks.append(np.asarray(self._throttles, dtype=np.float32))
        if self._crosses:
            blocks.append(np.asarray(self._crosses, dtype=np.float32))

        obs = np.concatenate(blocks, dtype=np.float32)
        # Tested rather than rewritten: `nan_to_num` rebuilds all of `obs_dim` every
        # step (4.4 us of a 45 us step) to repair a fault only an upstream bug
        # produces, where the test costs a fraction of that.
        if not np.isfinite(obs).all():
            np.nan_to_num(obs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    @override
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Drawn, not generated: a track costs ~33 ms against ~49 us per step.
        # `fixed_track_seed` is re-read every reset, so a layout can be pinned mid-run.
        pool = self._ensure_pool()
        track_seed = self.cfg.fixed_track_seed
        pooled = pool.sample(self._rng) if track_seed is None else pool.get(track_seed)
        self._track = pooled.track
        self._progress = pooled.progress
        self._state = random_start_state(self.track, self._rng, params=self.params)
        self._steps = 0
        self._prev_steer = self.state.steer
        # Refilled before the closing `_observe()`: `_scans` is padded there,
        # `_crosses` below once the projection has run.
        self._scans.clear()
        self._throttles.clear()
        self._throttles.extend([0.0] * self.cfg.throttle_history)
        self._crosses.clear()
        # Narrowest half-width on the layout: the potential must bottom out no later
        # than the wall the car can hit, or it reports "inside" for a position out.
        self._half_width = float(np.min(self.track.wall_offset))
        self._s_prev, cross = self.progress.project((self.state.x, self.state.y))
        self._advanced = 0.0
        self._stall_from = 0.0
        self._stall_steps = 0
        self._potential = self._cross_track_potential(cross)
        # Padded with the spawn's own offset: a zeroed history would claim a car
        # starting 3 m off the line is on it, then jerk.
        self._crosses.extend([self._normalized_cross(cross)] * self.cfg.cross_track_history)

        # Only the env knows where an episode begins: `set_track` no-ops on a reset
        # onto the same pinned layout.
        if self._viewer is not None:
            self._viewer.begin_episode()

        return self._observe(), {}

    @override
    def step(self, action):
        # Clamped as scalars: the array round trip cost ~2 us of a ~45 us step.
        steer = min(max(float(action[0]), -1.0), 1.0)
        # A normalized drive/brake demand, not an acceleration: what it achieves
        # depends on the motor curve and the grip cornering spent.
        throttle = min(max(float(action[1]), -1.0), 1.0)
        steer_cmd = steer * self.params.max_steer
        # Recorded for the observation this step returns: `proprioception` carries
        # the *achieved* steering angle, so nothing else says the car is braking.
        self._throttles.appendleft(throttle)

        self._state = car_step(self.state, steer_cmd, throttle, self.params)
        self._steps += 1

        pos = (self.state.x, self.state.y)
        still_on_track = self.track.walls.contains(pos)

        s, cross = self.progress.project(pos)
        # Pushed before the `_observe()` this step returns, so the observation
        # describes the state the reward is about to be computed for.
        self._crosses.appendleft(self._normalized_cross(cross))
        ds = s - self._s_prev
        # Unwrapped across the start/finish line, so a completed lap is not a huge
        # negative jump.
        total_length = self.progress.total_length
        half = total_length / 2
        if ds < -half:
            ds += total_length
        elif ds > half:
            ds -= total_length
        # Distance covered since reset: `s` alone would carry the random spawn's
        # U(0, 1) offset. Starts at zero, negative for a car driving backwards.
        self._advanced += ds
        self._s_prev = s

        # The clock restarts once `_stall_distance` is covered, so this measures
        # failure to get anywhere rather than instantaneous speed: a car easing
        # through a hairpin keeps resetting it.
        self._stall_steps += 1
        if self._advanced - self._stall_from >= self._stall_distance:
            self._stall_from = self._advanced
            self._stall_steps = 0
        stalled = self._stall_limit > 0 and self._stall_steps >= self._stall_limit

        # Both terms are fractions: `ds` against a flat-out step's distance, the
        # steering increment against full lock, so the return's scale is the task's.
        progress = ds / self._progress_ref
        steer_rate_cost = (
            self.cfg.steer_rate_penalty
            * abs(self.state.steer - self._prev_steer)
            / self.params.max_steer
        )
        self._prev_steer = self.state.steer

        # The reward carries `Phi`'s difference, so shaping telescopes and cannot
        # move the optimum (Ng et al. 1999). `Phi` is 0 at the terminal, as the
        # theorem requires.
        terminated = not still_on_track
        potential = 0.0 if terminated else self._cross_track_potential(cross)
        shaping = potential - self._potential
        self._potential = potential

        reward = progress - steer_rate_cost + shaping
        if terminated:
            reward -= self._off_track_cost

        # A stall truncates: no reward attached, and PPO bootstraps the cut-off
        # value. Gated on `terminated`, since a crash on the last step is a crash.
        truncated = not terminated and (self._steps >= self._max_steps or stalled)
        info = {
            # Laps since the reset (can exceed 1.0 or go negative), and separately
            # where on the lap the car is, in [0, 1).
            "lap_progress": self._advanced / total_length,
            "track_position": s / total_length,
            "on_track": still_on_track,
            "speed": self.state.vx,
            "off_track": terminated,
            "stalled": stalled,
            # For `TrainingMetricsCallback`, in *raw* env units: a decomposition you
            # cannot compare against its own config weights cannot be tuned.
            "r_progress": progress,
            "r_steer_rate": -steer_rate_cost,
            "r_shaping": shaping,
            "r_crash": -self._off_track_cost if terminated else 0.0,
            # A magnitude: logged as a rollout mean, which averages a signed
            # quantity away to nothing.
            "cross_track": abs(cross),
            "grip_use": lateral_grip_usage(self.state, self.params),
            # `|a| > 0.99` rather than true clipping, since SB3 clips first: what is
            # left to see is a policy living on the rails, where int8 has the least
            # resolution to spare. Scalar arithmetic because
            # `np.mean(np.abs(action) > 0.99)` was 3 us of a 45 us step.
            "action_rail": 0.5 * ((abs(steer) > 0.99) + (abs(throttle) > 0.99)),
            "action_steer": abs(steer),
            # A magnitude too: alternating flat-out and hard braking would average
            # to a coast.
            "action_throttle_abs": abs(throttle),
        }
        return self._observe(), reward, terminated, truncated, info

    @override
    def render(self):
        """Offscreen RGB frame of the current state: the `rgb_array` path."""
        if self.render_mode != "rgb_array":
            return None
        # No `state is None` check: the `state` property raises a message naming the
        # unset attribute, which is more useful here.
        if self._viewer is None:
            # Imported here, not at module scope, so `SubprocVecEnv` workers running
            # physics never pay for pygame.
            from tinyml_racing.render.viewer import PygameViewer

            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            self._viewer = PygameViewer(interactive=False).open()

        self._viewer.set_track(self.track)
        # The viewer's simulation-time accumulators (distance, skid marks) advance
        # here rather than inside `draw`.
        self._viewer.observe(self.state, self.params.dt)
        self._viewer.draw(self.state, scan=self.last_scan, lidar=self.cfg.lidar)
        return self._viewer.frame_rgb()

    @override
    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
