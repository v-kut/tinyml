"""Racing line optimization via bound-constrained least squares. Minimizing the sum of
squared second differences of the path subject to it staying inside the corridor gives
the out-in-out shape with no tuning parameters and no iterative-smoothing instability;
it is worth ~11% of lap time over the centerline, which is why the progress reward
references it. The design matrix is banded, so it is assembled sparse and solved LSMR.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear

from tinyml_racing.sim.track import Track

__all__ = [
    "add_racing_line",
    "compute_racing_line",
]


def compute_racing_line(
    track: Track,
    wall_clearance: float = 1.0,
    centerline_pull: float = 0.02,
    tol: float = 1e-4,
) -> np.ndarray:
    """Minimum-curvature line through the track corridor.

    `wall_clearance` is the room in metres the line leaves against the wall, metres,
    not a share of the half-width, because what it has to cover is how far off the line
    a *follower* runs: the expert tracks it to a median 0.95 m, 2.0 m at the 95th.

    `centerline_pull` is the ridge that makes the solution unique on straights, and
    `tol` a convergence tolerance: tightening it to 1e-6 moves the line by at most
    3.5 cm and doubles the solve, which buys nothing anything can track.
    """
    center = np.asarray(track.centerline, dtype=float)
    normals = np.asarray(track.normals, dtype=float)
    n = len(center)
    # Signed lateral limits along `normals`, positive outward. The corridor is
    # symmetric, so one offset bounds both; the floor keeps a corridor narrower than
    # twice the clearance from collapsing onto the centerline.
    room = np.maximum(track.wall_offset - wall_clearance, 0.1 * track.wall_offset)
    lower, upper = -room, room

    index = np.arange(n)
    nxt = np.roll(index, -1)
    prv = np.roll(index, 1)
    # Second difference of the offset path, one row per coordinate, unweighted: the
    # offset path's local spacing `(1 - kappa*offset)*ds` is a function of the offset
    # being solved for, so weighting by it would turn one solve into a fixed point.
    rows = np.concatenate([index] * 3)
    cols = np.concatenate([prv, index, nxt])
    blocks = [
        sp.coo_matrix(
            (
                np.concatenate([normals[prv, axis], -2.0 * normals[:, axis], normals[nxt, axis]]),
                (rows, cols),
            ),
            shape=(n, n),
        )
        for axis in (0, 1)
    ]
    blocks.append(sp.eye(n) * centerline_pull)
    design = sp.vstack(blocks).tocsr()

    second_diff = center[nxt] - 2.0 * center + center[prv]
    rhs = np.concatenate([-second_diff[:, 0], -second_diff[:, 1], np.zeros(n)])

    offset = lsq_linear(
        design, rhs, bounds=(lower, upper), method="trf", lsq_solver="lsmr", tol=tol
    ).x
    return center + offset[:, None] * normals


def add_racing_line(track: Track) -> Track:
    return replace(track, racing_line=compute_racing_line(track))
