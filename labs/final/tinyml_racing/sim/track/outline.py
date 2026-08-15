"""The polygon stage: the closed outline that `fillet` rounds into a lap.

An elliptical cloud's convex hull, vertices pulled in past their chords so the lap
turns both ways, one edge slid apart to plant the main straight, and hooks and
chicanes jogged into the longest edges that will take one. Everything here operates
on an `(m, 2)` array of vertices in counter-clockwise order, and hands `fillet` a
simple polygon whose exterior angles sum to 2pi, which is all it asks for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from tinyml_racing.sim.track.config import TrackConfig

_CLOUD_PER_CORNER = 60  # cloud points per requested corner; the hull keeps ~n^(1/3) of them
_MAX_TURN = 2.09  # rad (120 deg): a hairpin, and the sharpest turn whose two straights clear
_CORRIDOR_GAP = 1.6  # multiples of `width` that two edges sharing no vertex must stay apart
# A jog leaves this much straight between its own arcs, instead of `min_straight`: its
# corners come in pairs a hairpin apart, which is what makes it read as one feature.
# The floor is four samples: below two, a central difference over the sampled path
# straddles both of the jog's curvature steps at once and `Track.curvature` stops
# reading as the square wave the exact primitives make it.
_JOG_STRAIGHT = (16.0, 34.0)
# Attempts per jog, longest candidate edge first. A jog that clears none of them is
# dropped rather than failing the draw: the clearance gate is what it has to satisfy.
_JOG_TRIES = 6


def _norm(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(v, axis=1)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _signed_area(v: np.ndarray) -> float:
    return 0.5 * float(_cross(v, np.roll(v, -1, axis=0)).sum())


def _edge_lengths(v: np.ndarray) -> np.ndarray:
    """Edge `i` of a closed polygon runs from vertex `i` to vertex `i+1`."""
    return _norm(np.roll(v, -1, axis=0) - v)


def _rescaled(v: np.ndarray, perimeter: float) -> np.ndarray:
    return v * (perimeter / _edge_lengths(v).sum())


def _turn_angles(v: np.ndarray) -> np.ndarray:
    """Signed exterior angle at each vertex, positive turning left. Sums to +2pi
    for any simple counter-clockwise polygon, however concave.
    """
    heading = np.roll(v, -1, axis=0) - v
    heading /= _norm(heading)[:, None]
    into = np.roll(heading, 1, axis=0)
    return np.arctan2(_cross(into, heading), np.einsum("ij,ij->i", into, heading))


def _point_gap(p0: np.ndarray, p1: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Distance from each `q` to its own segment `p0 -> p1`."""
    d = p1 - p0
    t = np.clip(np.einsum("ij,ij->i", q - p0, d) / np.einsum("ij,ij->i", d, d), 0.0, 1.0)
    return _norm(q - p0 - t[:, None] * d)


def _edge_clearance(v: np.ndarray) -> float:
    """Smallest distance between two polygon edges sharing no vertex, 0.0 if any
    such pair crosses.

    One number rejects both a folded outline and a neck too narrow for the two
    corridors through it, which would put a wall across the track elsewhere.
    """
    m = len(v)
    i, j = np.triu_indices(m, 2)
    keep = ~((i == 0) & (j == m - 1))  # the first and last edges do share a vertex
    i, j = i[keep], j[keep]
    a0, a1 = v[i], v[(i + 1) % m]
    b0, b1 = v[j], v[(j + 1) % m]
    if np.any(
        (_cross(a1 - a0, b0 - a0) * _cross(a1 - a0, b1 - a0) < 0.0)
        & (_cross(b1 - b0, a0 - b0) * _cross(b1 - b0, a1 - b0) < 0.0)
    ):
        return 0.0
    return float(
        min(
            _point_gap(a0, a1, b0).min(),
            _point_gap(a0, a1, b1).min(),
            _point_gap(b0, b1, a0).min(),
            _point_gap(b0, b1, a1).min(),
        )
    )


def _corner_polygon(
    rng: np.random.Generator, n_corners: int, perimeter: float, min_edge: float
) -> np.ndarray | None:
    """Convex counter-clockwise polygon of `n_corners` vertices, `perimeter` long.

    Elliptical cloud, not square: the hull of n uniform points in a square keeps only
    ~(8/3)*ln(n) of them against ~n^(1/3) on a smooth boundary. None means the hull
    came out coarser than asked, which is a redraw.
    """
    n_cloud = _CLOUD_PER_CORNER * n_corners
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n_cloud)
    radius = np.sqrt(rng.random(n_cloud))  # sqrt spreads the draw by area, not by radius
    aspect = rng.uniform(1.0, 1.7)  # circuits are longer than they are wide
    cloud = np.stack([radius * np.cos(angle), radius * np.sin(angle) / aspect], axis=1)
    hull = ConvexHull(cloud)
    if len(hull.vertices) < n_corners:
        return None
    v = _rescaled(cloud[hull.vertices], perimeter)  # 2-D qhull orders them CCW
    # Visvalingam: drop the vertex whose triangle with its neighbours has least area.
    # Vertices closer than `min_edge` go first whatever their area, since two corners
    # sharing an edge shorter than their fillet tangents cannot both survive.
    while len(v) > n_corners:
        prv, nxt = np.roll(v, 1, axis=0), np.roll(v, -1, axis=0)
        crowding = np.minimum(_norm(v - prv), _norm(nxt - v)) - min_edge
        area = 0.5 * np.abs(_cross(v - prv, nxt - v))
        v = np.delete(v, int(np.argmin(np.where(crowding < 0.0, crowding, area))), axis=0)
    return _rescaled(v, perimeter)


def _stretch(v: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Cut the outline at its extremes across `offset` and slide the leading chain
    along it, inserting two parallel edges of exactly `|offset|` on opposite sides of
    the infield and adding twice that to the perimeter.

    Those edges are the main straight and its counterpart. Waiting for a random hull
    to hand one out is waiting for a draw whose corners all collapsed, and the exact
    `2*|offset|` is what lets `generate_track` solve for it.
    """
    m = len(v)
    # The split points are where the outward normal turns perpendicular to `offset`.
    # Being extremal they are convex, so the cut never lands on a pulled vertex.
    across = v @ np.array([-offset[1], offset[0]])
    a, b = int(np.argmin(across)), int(np.argmax(across))
    lead = np.arange(a, a + (b - a) % m + 1) % m  # a..b, the chain that slides
    rest = np.arange(b, b + (a - b) % m + 1) % m  # b..a, the chain that stays
    return np.concatenate([v[lead] + offset, v[rest]])  # both ends duplicated: two new edges


def _pulled(rng: np.random.Generator, v: np.ndarray, clearance: float) -> np.ndarray:
    """Pull a few vertices in past the chord between their neighbours.

    Every corner of a convex lap is a left-hander and a filleted convex outline is a
    blob, so this is what makes the layout a circuit: a vertex dragged past its chord
    is a hairpin, and it sharpens its neighbours into real corners.

    A pull that folds the outline, crowds two corridors or exceeds `_MAX_TURN` is
    abandoned; only `generate_track` fails a whole draw.
    """
    m = len(v)
    budget = int(rng.integers(3, 6))
    pulled = np.zeros(m, dtype=bool)
    for j in rng.permutation(m):
        if budget == 0:
            break
        if pulled[j - 1] or pulled[(j + 1) % m]:
            continue  # two adjacent pulls make a spike, not a pair of corners
        prv, nxt = v[j - 1], v[(j + 1) % m]
        chord = nxt - prv
        foot = prv + chord * (np.dot(v[j] - prv, chord) / np.dot(chord, chord))
        height = float(np.linalg.norm(foot - v[j]))
        if height < 1e-9:
            continue
        # Past the chord by a fraction of the shorter edge, which is what turns
        # the vertex reflex however flat it started; 0.75 of it is a hairpin.
        depth = height + rng.uniform(0.30, 0.75) * min(
            float(np.linalg.norm(v[j] - prv)), float(np.linalg.norm(nxt - v[j]))
        )
        candidate = v.copy()
        candidate[j] = v[j] + (depth / height) * (foot - v[j])
        if np.abs(_turn_angles(candidate)).max() > _MAX_TURN:
            continue
        if _edge_clearance(candidate) < clearance:
            continue
        v, pulled[j], budget = candidate, True, budget - 1
    return v


@dataclass(frozen=True)
class _Outline:
    """The polygon `_fillet` will round, and what the draws owe each part of it.

    `main` and `jog` are per edge, `radius` per vertex. A jog reserves its own
    geometry when it is inserted, because the room its arcs need is what decided
    where it could go at all; `NaN` leaves the vertex to `generate_track`'s draw.
    """

    v: np.ndarray  # (m, 2)
    main: np.ndarray  # (m,) bool, the two straights `_stretch` inserted
    jog: np.ndarray  # (m,) bool, edges belonging to a hook or a chicane
    radius: np.ndarray  # (m,) fillet radius per vertex, NaN where it is free


def _jogged(
    v: np.ndarray, edge: int, depth: float, turn: float, gap: float, base: float
) -> np.ndarray | None:
    """Insert an excursion into edge `edge`, or None if the edge is too short.

    Four vertices: leave the edge at `turn` radians, run to `depth` metres inside
    it, cross `gap`, come back. The two inner turns are `-turn` and the two outer
    ones `+turn`, so the outline still sums to 2pi and stays simple, which is all
    `_fillet` asks of it. A hook is this with `turn` near a right angle and a depth
    in the hundreds of metres; a chicane is the same shape, shallow and slanted,
    which rounds into an S rather than into a pair of hairpins.

    `base` is the straight left at each end, so the corners the edge already had
    keep the room their own fillets need.
    """
    m = len(v)
    a, b = v[edge], v[(edge + 1) % m]
    span = float(np.linalg.norm(b - a))
    u = (b - a) / span
    # Left of the direction of travel is the infield, the outline being CCW.
    inward = np.array([-u[1], u[0]])
    leg = depth / np.sin(turn)
    margin = 0.5 * (span - (2.0 * depth / np.tan(turn) + gap))
    if margin < base:
        return None
    entry = np.cos(turn) * u + np.sin(turn) * inward
    leave = np.cos(turn) * u - np.sin(turn) * inward
    c0 = a + margin * u
    c1 = c0 + leg * entry
    c2 = c1 + gap * u
    c3 = c2 + leg * leave
    return np.insert(v, edge + 1, np.stack([c0, c1, c2, c3]), axis=0)


def _with_jogs(
    rng: np.random.Generator, outline: _Outline, cfg: TrackConfig, clearance: float
) -> _Outline:
    """Jog hooks and then chicanes into the longest edges that will take one.

    Longest first because a jog is mostly a length requirement, and never onto a
    main straight or onto an edge already carrying a jog: both are features the lap
    is meant to keep whole. A jog that folds the outline or crowds two corridors is
    abandoned, not repaired.
    """
    # Hook depth scales with the layout, chicane depth is metres either way. `length
    # / pi` stands in for the span: a lap is not a circle, but it is within a third
    # of one, and the clearance gate is what actually decides whether a hook fits.
    span = float(np.sum(_edge_lengths(outline.v))) / np.pi
    plan = [
        (cfg.hook_depth, cfg.hook_turn_deg, span)
        for _ in range(int(rng.integers(cfg.n_hooks[0], cfg.n_hooks[1] + 1)))
    ] + [
        (cfg.chicane_depth, cfg.chicane_turn_deg, 1.0)
        for _ in range(int(rng.integers(cfg.n_chicanes[0], cfg.n_chicanes[1] + 1)))
    ]

    for depth_range, turn_range, scale in plan:
        turn = np.radians(rng.uniform(*turn_range))
        bite = float(np.tan(0.5 * turn))
        # The tip is the tightest pair on the lap; the shoulders only have to rejoin
        # the edge, so they are drawn wider. Both are reserved rather than drawn
        # later, because the room they need is what decides where the jog can go.
        r_tip = float(rng.uniform(1.2, 3.0) * cfg.min_corner_radius)
        r_shoulder = float(rng.uniform(2.5, 6.0) * cfg.min_corner_radius)
        gap = 2.0 * r_tip * bite + rng.uniform(*_JOG_STRAIGHT)
        # Each leg carries the tip's bite at one end and the shoulder's at the other,
        # so a jog too shallow to hold its own arcs is deepened rather than dropped.
        depth = max(
            scale * rng.uniform(*depth_range),
            np.sin(turn) * ((r_tip + r_shoulder) * bite + _JOG_STRAIGHT[1]),
        )
        # What the neighbouring corners need left of the edge they already own.
        base = cfg.min_straight + 2.0 * cfg.min_corner_radius * bite

        order = np.argsort(-_edge_lengths(outline.v))
        free = [i for i in order if not outline.main[i] and not outline.jog[i]]
        for edge in free[:_JOG_TRIES]:
            v = _jogged(outline.v, edge, depth, turn, gap, base)
            if v is None:
                break  # ordered longest first, so nothing after this one fits either
            if np.abs(_turn_angles(v)).max() > _MAX_TURN:
                continue
            if _edge_clearance(v) < clearance:
                continue
            outline = _Outline(
                v=v,
                main=np.insert(outline.main, edge + 1, [False] * 4),
                jog=np.insert(outline.jog, edge + 1, [True, True, True, False]),
                radius=np.insert(outline.radius, edge + 1, [r_shoulder, r_tip, r_tip, r_shoulder]),
            )
            break
    return outline
