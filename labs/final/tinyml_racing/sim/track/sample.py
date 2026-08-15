"""The sampled lap: the `Track` every consumer reads, and the walk that builds it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tinyml_racing.sim.geometry import WallGeometry
from tinyml_racing.sim.track.config import TrackConfig
from tinyml_racing.sim.track.fillet import _Primitives
from tinyml_racing.sim.track.outline import _signed_area


# `eq=False`: the fields are ndarrays, so a generated `__eq__` returns an array.
# Identity is what callers mean and what the per-track caches key on.
@dataclass(frozen=True, eq=False)
class Track:
    """One closed circuit, sampled at `n` evenly arc-length-spaced points.

    The corridor is symmetric by construction, both walls being the centerline offset
    by `+/- wall_offset * normals`, so one offset array describes both.
    `outer_wall`/`inner_wall` are the drawn corridor; `walls` states the same curves
    exactly, for the LiDAR. `s` and `length` have no production reader and exist for
    `tests/test_track_geometry.py`; `curvature` and `ds` feed the spawn speed cap.
    """

    centerline: np.ndarray  # (n, 2)
    tangents: np.ndarray  # (n, 2), unit, pointing along the lap
    normals: np.ndarray  # (n, 2), unit, pointing away from the infield
    curvature: np.ndarray  # (n,), signed; positive turning left along the lap
    s: np.ndarray  # (n,), arc length from the start/finish line
    ds: float
    length: float
    # (n,), the corridor half-width; every entry is equal, but it stays an array
    # because `racing_line` and `render` read the corridor bounds vectorized.
    wall_offset: np.ndarray
    outer_wall: np.ndarray  # (n, 2)
    inner_wall: np.ndarray  # (n, 2)
    walls: WallGeometry  # both walls as arcs and straights, sampled by nothing
    seed: int | None = None
    # Filled in by `racing_line.add_racing_line`, which the track pool always
    # runs; only a track built by hand (a test, the geometry viewer) has None.
    racing_line: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.centerline)


def _sampled(
    prim: _Primitives, cfg: TrackConfig, rng: np.random.Generator, seed: int | None
) -> Track:
    """Walk the primitives in `sample_spacing` steps, reading position, tangent and
    curvature off them: 0 and the edge direction on a straight, `+/-1/R` and the
    radius turned a quarter turn on an arc.

    Nothing differentiates a polyline, so the start/finish line carries no one-sided
    kink and a hairpin no smoothing error.
    """
    bound = np.concatenate([[0.0], np.cumsum(prim.segments)])
    length = float(bound[-1])
    n = cfg.n_samples or max(64, round(length / cfg.sample_spacing))
    ds = length / n
    s = np.arange(n) * ds
    k = np.searchsorted(bound, s, side="right") - 1
    local = s - bound[k]
    arc, j = k % 2 == 0, k // 2
    center = np.empty((n, 2))
    tangents = np.empty((n, 2))
    curvature = np.zeros(n)

    a = j[arc]
    r, turn = prim.radius[a], np.sign(prim.sweep[a])
    phi = prim.start[a] + turn * local[arc] / r
    radial = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    center[arc] = prim.center[a] + r[:, None] * radial
    tangents[arc] = turn[:, None] * np.stack([-radial[:, 1], radial[:, 0]], axis=1)
    curvature[arc] = turn / r

    b = j[~arc]
    center[~arc] = prim.origin[b] + local[~arc, None] * prim.heading[b]
    tangents[~arc] = prim.heading[b]

    # The lap runs counter-clockwise, so a quarter turn clockwise off the tangent
    # points out of the infield, the direction `racing_line` and `render` assume.
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=1)
    # Drawn per layout, which is where the variety a lap-wise drift would have given
    # now lives; see `TrackConfig.width_variation` for why it cannot drift.
    half_width = 0.5 * cfg.width * (1.0 + cfg.width_variation * rng.uniform(-1.0, 1.0))
    offset = np.full(n, half_width)
    # Start/finish at the middle of the longest straight, as at a real circuit.
    longest = int(np.argmax(prim.run))
    shift = round((bound[2 * longest + 1] + 0.5 * prim.run[longest]) / ds) % n
    center, tangents, normals, curvature = (
        np.roll(x, -shift, axis=0) for x in (center, tangents, normals, curvature)
    )
    wall = offset[:, None] * normals
    outer, inner = center + wall, center - wall
    if _signed_area(outer) <= _signed_area(inner):
        raise RuntimeError("_sampled: the outward normal encloses less than the inward one")
    # Unrolled on purpose: the roll above renumbers samples, and these
    # primitives are not samples. The curves are the same set either way.
    walls = WallGeometry(
        center=prim.center,
        radius=prim.radius,
        start=prim.start,
        sweep=prim.sweep,
        origin=prim.origin,
        heading=prim.heading,
        run=prim.run,
        half_width=half_width,
    )
    return Track(
        centerline=center,
        tangents=tangents,
        normals=normals,
        curvature=curvature,
        s=s,
        ds=ds,
        length=length,
        wall_offset=offset,
        outer_wall=outer,
        inner_wall=inner,
        walls=walls,
        seed=seed,
    )
