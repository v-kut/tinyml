"""Procedural racetrack generator: straights joined by circular arcs.

Random points in an ellipse, convex hull, vertices pulled inward so the lap turns
both ways, the outline slid apart to insert a main straight, every vertex filleted.
Curvature is exact and radii are inputs. Only the polygon needs rejecting.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial import ConvexHull

from tinyml_racing.sim.car import CarParams
from tinyml_racing.sim.geometry import WallGeometry

__all__ = [
    "Track",
    "TrackConfig",
    "default_corner_speed_radius",
    "generate_track",
    "min_steerable_corner_radius",
]

_WALL_MARGIN = 0.5  # m of curvature headroom beyond the wall, so the offset never cusps
_CLOUD_PER_CORNER = 60  # cloud points per requested corner; the hull keeps ~n^(1/3) of them
_MAX_TURN = 2.09  # rad (120 deg): a hairpin, and the sharpest turn whose two straights clear
_STRAIGHT_SHARE = 0.25  # of every edge, floor on what stays straight: at less than this the
# arcs of neighbouring corners run into each other and the lap reads as a blob, not a circuit
_TIGHT_CORNER = 40.0  # m: every lap gets a corner at least this tight, so it has a brake zone
_RADIUS_HEADROOM = 1.10  # the length rescale only grows radii, so draw them under the ceiling
_CORRIDOR_GAP = 1.6  # multiples of `width` that two edges sharing no vertex must stay apart


@dataclass(frozen=True)
class TrackConfig:
    """Generator tunables: the single source of truth for track shape."""

    width: float = 13.0  # full drivable width, metres
    # One width per layout, never drifted along the lap: `geometry.WallGeometry` is
    # the exact offset of the arc/straight chain, and only a constant offset has one.
    width_variation: float = 0.10  # fraction of `width`, either way
    length_range: tuple[float, float] = (800.0, 2400.0)
    # Derived from the car, not chosen: a literal drifted out of step with `CarParams`
    # once already. The lambda defers to a function defined further down.
    min_corner_radius: float = field(
        default_factory=lambda: min_steerable_corner_radius()  # noqa: PLW0108
    )
    max_corner_radius: float = 400.0  # a sweeper the car takes flat out
    n_corners: tuple[int, int] = (12, 24)  # inclusive draw range
    min_straight: float = 40.0  # m of straight left between consecutive corners
    main_straight_min: float = 250.0  # every layout gets one straight at least this long
    sample_spacing: float = 4.0
    n_samples: int | None = None  # overrides `sample_spacing` when set
    max_attempts: int = 200


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


def default_corner_speed_radius(params: CarParams | None = None) -> float:
    """Radius the car holds flat out at the lateral grip limit, so "slow corner"
    means slow for this car.

    The aero-assisted limit, not `mu*g`: at top speed downforce is worth more than
    the car's weight, and the mechanical figure would overstate the radius by more
    than 2x (342 m against 154 m).
    """
    p = params or CarParams()
    return float(p.top_speed**2 / p.lateral_accel(p.top_speed))


def min_steerable_corner_radius(params: CarParams | None = None) -> float:
    """Tightest corner this car can be driven through, not merely turned into.

    `min_turn_radius` is the full-lock kinematic radius, reached only at zero lateral
    acceleration; at the grip limit the car needs a further `K * ay` of steer. Twice
    the kinematic radius is that headroom.
    """
    p = params or CarParams()
    return 2.0 * p.min_turn_radius


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


def generate_track(seed: int | None = None, config: TrackConfig | None = None) -> Track:
    """Generate one closed circuit. Samples sit `config.sample_spacing` apart, so the
    count tracks lap length unless `n_samples` overrides it. Index 0 is the
    start/finish line, mid longest straight, where progress is measured from.
    """
    cfg = config or TrackConfig()
    widest = (1.0 + cfg.width_variation) * 0.5 * cfg.width + _WALL_MARGIN
    # Nested walls, asserted rather than hoped for: the offset cusps as soon as w
    # reaches the local radius of curvature. A violation is a config error.
    if cfg.min_corner_radius <= widest:
        raise ValueError(
            f"min_corner_radius {cfg.min_corner_radius:.1f} m must exceed the {widest:.2f} m "
            "widest half-corridor plus margin, or the inner wall folds through its own apex"
        )
    rng = np.random.default_rng(seed)
    clearance = _CORRIDOR_GAP * cfg.width
    rejected: Counter[str] = Counter()
    for _ in range(cfg.max_attempts):
        target = rng.uniform(*cfg.length_range)
        n_corners = int(rng.integers(cfg.n_corners[0], cfg.n_corners[1] + 1))
        # Two of the corners come from the cut below, and the straights it
        # inserts are not in the perimeter the outline is drawn to.
        main = cfg.main_straight_min * rng.uniform(1.05, 1.35)
        v = _corner_polygon(
            rng,
            n_corners - 2,
            target - 2.0 * main,
            cfg.min_straight + 2.0 * cfg.min_corner_radius,
        )
        if v is None:
            rejected["coarse"] += 1
            continue
        v = _pulled(rng, v, clearance)
        # The cut adds exactly twice its offset, so the offset that survives the
        # rescale as a `main`-long straight can be solved for.
        lean = rng.uniform(-0.35, 0.35)  # the straights run roughly along the long axis
        span = main * _edge_lengths(v).sum() / (target - 2.0 * main)
        v = _stretch(v, span * np.array([np.cos(lean), np.sin(lean)]))
        v = _rescaled(v, target)
        if _edge_clearance(v) < clearance:
            rejected["blocked"] += 1
            continue
        # The inserted pair is the only edges of length `main`; a uniform rescale
        # cannot move an edge into or out of that set, so this holds to the fillet.
        edge = _edge_lengths(v)
        straight = np.abs(edge - main) < 1e-6 * main
        if straight.sum() != 2:
            # Not the pulls: a pulled vertex is inside its neighbours' chord, so it
            # cannot be extremal. This fires only when an unrelated edge measures
            # `main` to a part in a million, one draw in many thousands.
            rejected["uncut"] += 1
            continue
        # Log-uniform: a uniform draw over 25-400 m leaves four corners in five above
        # the ~150 m this car takes flat out, a lap with nothing to brake for.
        lo, hi = np.log(cfg.min_corner_radius), np.log(cfg.max_corner_radius / _RADIUS_HEADROOM)
        radius = np.exp(rng.uniform(lo, hi, size=len(v)))
        # Heaviest braking at the end of the longest straight, as at a real
        # circuit, and that is also what guarantees the `_TIGHT_CORNER` check.
        brake = np.roll(straight, 1)  # straight `i` ends at vertex `i+1`
        radius[brake] = np.exp(
            rng.uniform(lo, np.log(_TIGHT_CORNER / _RADIUS_HEADROOM), size=int(brake.sum()))
        )
        # A share of every edge stays straight, not just the 40 m floor: the
        # difference between a circuit and one long chain of arcs that closes.
        keep_straight = np.maximum(
            np.where(straight, cfg.main_straight_min, cfg.min_straight), _STRAIGHT_SHARE * edge
        )
        prim = _fillet(v, radius, keep_straight)
        if prim.radius.min() < cfg.min_corner_radius:
            rejected["tight"] += 1
            continue
        # Corner cutting made the filleted lap shorter than the polygon, so this
        # scale lands the length in range. It only grows radii, which is why they were
        # drawn under the ceiling.
        prim = prim.scaled(target / prim.length)
        if prim.radius.max() > cfg.max_corner_radius:
            rejected["wide"] += 1
            continue
        if prim.radius.min() > _TIGHT_CORNER:
            rejected["flat"] += 1
            continue
        return _sampled(prim, cfg, rng, seed)

    why = {
        "coarse": f"hulls too coarse for {cfg.n_corners} corners",
        "blocked": f"folded over or left two corridors under {clearance:.1f} m apart",
        "uncut": "measured a second edge as long as the main straight",
        "tight": f"clamped a fillet under {cfg.min_corner_radius:.0f} m",
        "wide": f"broke {cfg.max_corner_radius:.0f} m",
        "flat": f"had no corner under {_TIGHT_CORNER:.0f} m to brake for",
    }
    detail = ", ".join(f"{count} {why[key]}" for key, count in rejected.most_common())
    raise RuntimeError(
        f"generate_track: all {cfg.max_attempts} draws rejected (seed={seed}): {detail}, "
        "widen `length_range`, lower `n_corners` or shorten `main_straight_min`"
    )
