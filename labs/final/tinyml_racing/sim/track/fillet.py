"""The outline as arcs and straights.

`_Primitives` is the lap in closed form, which is what makes `Track.curvature` exact
and radii an input rather than a result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from tinyml_racing.sim.track.outline import _edge_lengths, _turn_angles


@dataclass(frozen=True)
class _Primitives:
    """The lap as alternating arcs and straights: arc `i` rounds vertex `i`, then the
    straight runs along edge `i` to the next arc.

    Everything the sampler needs is closed form here, which is what makes
    `Track.curvature` exact rather than a finite difference.
    """

    center: np.ndarray  # (m, 2), arc centres
    radius: np.ndarray  # (m,)
    start: np.ndarray  # (m,), polar angle of the arc's entry point about its centre
    sweep: np.ndarray  # (m,), signed arc angle; positive turning left
    origin: np.ndarray  # (m, 2), where the straight leaving arc `i` begins
    heading: np.ndarray  # (m, 2), unit direction of that straight
    run: np.ndarray  # (m,), its length

    @property
    def segments(self) -> np.ndarray:
        """Primitive lengths in lap order, arcs on the even indices."""
        seg = np.empty(2 * len(self.radius))
        seg[0::2] = self.radius * np.abs(self.sweep)
        seg[1::2] = self.run
        return seg

    @property
    def length(self) -> float:
        return float(self.segments.sum())

    def scaled(self, factor: float) -> _Primitives:
        """Uniform scale. Angles are invariant under it; anything with a metre in
        it is not, which is what lets lap length be set after the shape is fixed.
        """
        return replace(
            self,
            center=self.center * factor,
            radius=self.radius * factor,
            origin=self.origin * factor,
            run=self.run * factor,
        )


def _fillet(v: np.ndarray, radius: np.ndarray, min_straight: np.ndarray) -> _Primitives:
    """Round every vertex with a circular arc tangent to both of its edges.

    A turn of `d` radians at radius R eats `t = R*tan(d/2)` of each edge, so R is
    clamped to what the neighbouring edges can give up while leaving `min_straight`
    between arcs. That clamp, not a rejection loop, keeps the main straight.

    It can push a radius below anything drivable; the caller owns that check, being
    the same range check the length rescale needs.
    """
    span = _edge_lengths(v)
    heading = (np.roll(v, -1, axis=0) - v) / span[:, None]
    turn = _turn_angles(v)
    # Exterior angles of a simple CCW polygon sum to +2pi, and every downstream sign
    # convention rides on that, so it is checked rather than assumed.
    if abs(float(turn.sum()) - 2.0 * np.pi) > 1e-6:
        raise RuntimeError(
            f"_fillet: outline turns {float(turn.sum()):.3f} rad, not 2pi, it is either "
            "clockwise or no longer simple, and every sign downstream would be wrong"
        )
    # Half the slack on the shorter edge, so the corner at its other end gets its own
    # bite, floored at zero: negative slack would mean a negative radius.
    slack = np.maximum(0.5 * (span - min_straight), 0.0)
    allow = np.minimum(np.roll(slack, 1), slack)
    # A vertex left dead straight would divide by zero below. The floor goes on `turn`
    # itself, so the stored `sweep` is the angle the bite and centre came from.
    turn_floor = 2e-6
    turn = np.where(np.abs(turn) < turn_floor, np.copysign(turn_floor, turn), turn)
    half = 0.5 * np.abs(turn)
    tangent_half = np.tan(half)
    r = np.minimum(radius, allow / tangent_half)
    t = r * tangent_half
    into = np.roll(heading, 1, axis=0)
    entry = v - t[:, None] * into
    # The centre sits one radius off the incoming edge, to its left through a
    # left-hander and to its right through a right-hander.
    left = np.stack([-into[:, 1], into[:, 0]], axis=1)
    center = entry + (np.sign(turn) * r)[:, None] * left
    return _Primitives(
        center=center,
        radius=r,
        start=np.arctan2(entry[:, 1] - center[:, 1], entry[:, 0] - center[:, 0]),
        sweep=turn,
        origin=v + t[:, None] * heading,
        heading=heading,
        run=span - t - np.roll(t, -1),
    )
