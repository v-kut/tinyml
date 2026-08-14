"""What one layout draws as: three polylines in world metres.

Kept in world space and projected per frame: a cache of projected copies would
be invalidated by every camera move, which in follow mode is every frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tinyml_racing.sim.track import Track


@dataclass(frozen=True)
class Scene:
    seed: int | None
    outer: np.ndarray
    inner: np.ndarray
    line: np.ndarray | None
    width_m: float

    @classmethod
    def from_track(cls, track: Track) -> Scene:
        """Convert a generated layout into the arrays a frame draws."""
        return cls(
            seed=track.seed,
            outer=np.asarray(track.outer_wall, dtype=float),
            inner=np.asarray(track.inner_wall, dtype=float),
            line=None if track.racing_line is None else np.asarray(track.racing_line, dtype=float),
            width_m=2.0 * float(track.wall_offset[0]),
        )
