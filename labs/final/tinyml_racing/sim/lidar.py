"""Simulated 2D LiDAR sweep. The sensor owns where the rays point (not uniformly, see
`ray_angles`), what the readings are worth (noise and dropout being properties of a
sensor rather than a curriculum) and their units: normalized, 0 at the bumper and 1 at
full scale, so nothing downstream needs `max_range`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=32)
def ray_angles(n_rays: int, fov_deg: float, ray_focus: float) -> np.ndarray:
    """Ray bearings relative to the car's heading, densest straight ahead.

    Uniform spacing spends the fan where there is nothing to learn: over 100k expert
    states with 60 uniform rays across 240 deg, every ray past +/-60 deg read a median
    6.8 m against a 6.5 m half-width, and only those inside +/-20 deg had reach.

    So the fan is warped: `phi(u) = half_fov * tan(a*u) / tan(a)` over `u` uniform on
    [-1, 1]. `tan` is what projecting an evenly spaced row of points ahead of the car
    back onto bearings gives: rays even in *distance* along a corridor, not in angle.

    `ray_focus` states that warp as something measurable: the edge/centre spacing ratio,
    `phi'(1)/phi'(0) = sec^2(a)`, so `a = arccos(ray_focus**-0.5)` and 1.0 is uniform. The
    shipped 28 rays at focus 9 put 10 inside +/-20 deg, 3.9 apart against a uniform 8.9.

    Cached: the fan is fixed for a run and this is called once per step; the array is
    read-only because every caller shares it.
    """
    if n_rays < 2:
        raise ValueError(f"n_rays must be at least 2, got {n_rays}")
    if ray_focus < 1.0:
        raise ValueError(
            f"ray_focus is an edge/centre spacing ratio and must be >= 1, got {ray_focus}"
        )

    half_fov = math.radians(fov_deg) / 2.0
    u = np.linspace(-1.0, 1.0, n_rays)
    if ray_focus == 1.0:
        angles = half_fov * u
    else:
        a = math.acos(ray_focus**-0.5)
        angles = half_fov * np.tan(a * u) / math.tan(a)
    angles.flags.writeable = False
    return angles


@dataclass(frozen=True)
class LidarConfig:
    """The sensor's whole specification.

    Sized for the car rather than for the track: 150 m of range because the GT
    car brakes from ~260 km/h and 20 m/s^2 from 72 m/s needs ~130 m of road, so
    the corner has to be visible while there is still room to stop in it.
    """

    n_rays: int = 28
    # 240, not the 180 a forward-facing scanner would give: the extra 30 deg on each
    # shoulder is the only thing that reports the wall a yawed car is being carried
    # towards, and warped rays cover it by widening the sparse end of the fan.
    fov_deg: float = 240.0
    max_range: float = 150.0
    # Ratio of outermost to innermost angular spacing, see `ray_angles`.
    ray_focus: float = 9.0

    # A fraction of full scale, which is what a range finder's error sheet quotes:
    # 0.002 is a 0.3 m sigma at 150 m. Absolute noise instead would have to be retuned
    # with `max_range`, and 0.03 m of it became 4.5 m, wider than the track.
    noise_std: float = 0.002
    dropout_prob: float = 0.01

    @property
    def angles(self) -> np.ndarray:
        return ray_angles(self.n_rays, self.fov_deg, self.ray_focus)

    def clean(self) -> LidarConfig:
        """The same optics with a perfect detector.

        Evaluation measures driving skill, not luck with a dropout pattern.
        """
        return replace(self, noise_std=0.0, dropout_prob=0.0)


def cast_lidar(state, track, cfg: LidarConfig, rng: np.random.Generator) -> np.ndarray:
    """One sweep, as `n_rays` readings in [0, 1]. 1 means nothing in range.

    The reading is the nearer of the two walls, since the sensor does not know which
    one the car is between; `track.walls` holds both, so one solve covers them.

    Angles are float64 because the walls are exact (see `WallGeometry`) and a float32
    heading would add ~1e-5 rad of error after the caster stopped making any. The
    readings come back float32, the width the observation and the device carry.
    """
    origin = np.array([state.x, state.y], dtype=np.float64)
    world_angles = np.float64(state.theta) + cfg.angles
    directions = np.stack([np.cos(world_angles), np.sin(world_angles)], axis=-1)

    # In place from here: `ray_distances` hands back an array it just allocated, and
    # the noise, dropout and clip chain used to leave four temporaries per sweep.
    reading = track.walls.ray_distances(origin, directions, cfg.max_range)
    reading /= cfg.max_range
    if cfg.noise_std > 0.0:
        reading += rng.normal(0.0, cfg.noise_std, size=reading.shape)
    if cfg.dropout_prob > 0.0:
        # A dropped return reads as "nothing out there", which is what a real
        # detector reports when the echo is too weak, not as zero, which
        # would read as a wall against the bumper.
        np.copyto(reading, 1.0, where=rng.random(reading.shape) < cfg.dropout_prob)
    np.clip(reading, 0.0, 1.0, out=reading)
    return reading.astype(np.float32)
