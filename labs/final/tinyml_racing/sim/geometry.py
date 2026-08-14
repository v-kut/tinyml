"""Shared 2D primitives: ray casters, the on-track test, the curvature estimate and
the arc-length table. `WallGeometry` casts the corridor as defined, straights plus
arcs, and is what the simulator runs. The polyline trio `ray_polyline_distances`,
`point_in_polygon` and `on_track` is only its test oracle.
"""

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _cast_walls(ox, oy, u, seg_a, seg_d, arc_c, arc_r, arc_mid, arc_reach, max_range):
    """The loop behind `WallGeometry.ray_distances`. Compiled scalar loops with `best`
    doubling as the running upper bound, seeded at `max_range` so a hit past full
    scale reads as full scale.
    """
    out = np.empty(u.shape[0], dtype=np.float32)
    for i in range(u.shape[0]):
        ux = u[i, 0]
        uy = u[i, 1]
        best = max_range

        # Straights. `s`'s bound is a sign and a magnitude rather than a division:
        # an absolute epsilon on `den` would be scale-dependent.
        for j in range(seg_a.shape[0]):
            dx = seg_d[j, 0]
            dy = seg_d[j, 1]
            den = ux * dy - uy * dx
            if den == 0.0:
                continue
            fx = seg_a[j, 0] - ox
            fy = seg_a[j, 1] - oy
            s_num = fx * uy - fy * ux
            if s_num * den < 0.0 or abs(s_num) > abs(den):
                continue
            t = (fx * dy - fy * dx) / den
            if 0.0 <= t < best:
                best = t

        # Arcs, both roots: the near one can fall outside the sweep while the far one
        # lands on it, and `disc == 0` is a real hit whose roots collapse onto `-b`.
        for j in range(arc_c.shape[0]):
            cx = arc_c[j, 0]
            cy = arc_c[j, 1]
            gx = ox - cx
            gy = oy - cy
            b = ux * gx + uy * gy
            r = arc_r[j]
            disc = b * b - (gx * gx + gy * gy - r * r)
            if disc < 0.0:
                continue
            root = math.sqrt(disc)
            for t in (-b - root, -b + root):
                if t < 0.0 or t >= best:
                    continue
                # The `atan2` this bisector test avoids would cost more than
                # solving the quadratic did: one call per ray per arc, twice.
                vx = ox + t * ux - cx
                vy = oy + t * uy - cy
                if vx * arc_mid[j, 0] + vy * arc_mid[j, 1] >= arc_reach[j]:
                    best = t

        out[i] = best
    return out


def ray_polyline_distances(origin, directions, polyline, max_range):
    """Cast R rays from one origin against a closed polyline, capped at `max_range`.
    Every ray tests every segment, since two segments can sit next to each other in
    space while being far apart along the polyline.
    """
    origin = np.asarray(origin, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)
    norms = np.hypot(directions[:, 0], directions[:, 1])[:, None]
    # Normalized rather than required unit, since a non-unit direction would scale the
    # answer silently. A zero-length ray then falls through as `max_range`.
    directions = directions / np.where(norms > 0.0, norms, np.float32(1.0))
    polyline = np.asarray(polyline, dtype=np.float32)

    p3 = polyline
    d2 = np.roll(polyline, -1, axis=0) - p3  # (M, 2)
    d2x, d2y = d2[:, 0][None, :], d2[:, 1][None, :]  # (1, M)

    d1 = directions * max_range  # (R, 2)
    d1x, d1y = d1[:, 0][:, None], d1[:, 1][:, None]  # (R, 1)
    denom = d1x * d2y - d1y * d2x  # (R, M)

    diffx = p3[:, 0][None, :] - origin[0]  # (1, M)
    diffy = p3[:, 1][None, :] - origin[1]

    with np.errstate(divide="ignore", invalid="ignore"):
        t = (diffx * d2y - diffy * d2x) / denom  # (R, M)
        u = (diffx * d1y - diffy * d1x) / denom

    valid = (np.abs(denom) > 1e-9) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    min_t = np.where(valid, t, np.inf).min(axis=1)  # (R,)
    return np.where(np.isfinite(min_t), min_t * max_range, max_range)


class WallGeometry:
    """The corridor exactly: the filleted chain offset by a constant half-width.

    Offsetting is closed form, an arc giving a concentric arc and a straight a parallel
    one, so LiDAR and containment are exact over the 48-96 primitives a layout has
    rather than the ~800-1,200 sampled chords two wall polylines would cost.
    """

    def __init__(
        self,
        *,
        center: np.ndarray,
        radius: np.ndarray,
        start: np.ndarray,
        sweep: np.ndarray,
        origin: np.ndarray,
        heading: np.ndarray,
        run: np.ndarray,
        half_width: float,
    ) -> None:
        """Takes `sim/track._Primitives` spread out: arc `i` rounds vertex `i`, straight
        `i` runs along edge `i`. Both walls are built together, an arc offsetting to
        `R + w` and `R - w`, so the pair needs no sign bookkeeping.
        """
        w = np.float64(half_width)
        self.half_width = float(w)
        c = np.asarray(center, dtype=np.float64)
        r = np.asarray(radius, dtype=np.float64)
        sweep = np.asarray(sweep, dtype=np.float64)
        head = np.asarray(heading, dtype=np.float64)
        base = np.asarray(origin, dtype=np.float64)
        span = head * np.asarray(run, dtype=np.float64)[:, None]

        # Half a sweep off its own bisector is how both queries below place a
        # point on an arc, so the bisector is stored and the angles are not.
        mid = np.asarray(start, dtype=np.float64) + 0.5 * sweep
        self.arc_c = c
        self.arc_r = r
        self.arc_mid = np.stack([np.cos(mid), np.sin(mid)], axis=1)
        self.arc_cos_half = np.cos(0.5 * np.abs(sweep))
        self.seg_a = base
        self.seg_d = span
        self.seg_len2 = np.einsum("ij,ij->i", span, span)

        # Both walls, as one primitive set: the caster wants the nearer of the
        # two and has no reason to know which it found.
        wall_r = np.concatenate([r + w, r - w])
        normal = np.stack([head[:, 1], -head[:, 0]], axis=1)
        self.wall_seg_a = np.concatenate([base + w * normal, base - w * normal])
        self.wall_seg_d = np.concatenate([span, span])
        self.wall_arc_c = np.concatenate([c, c])
        self.wall_arc_r = wall_r
        self.wall_arc_mid = np.concatenate([self.arc_mid, self.arc_mid])
        # |v| is exactly `r` on a hit, so the cosine test folds the radius in.
        self.wall_arc_reach = wall_r * np.tile(self.arc_cos_half, 2)

    def ray_distances(self, origin, directions, max_range) -> np.ndarray:
        """Range to the nearest wall along each of R rays, capped at `max_range`.

        `directions` must arrive unit: `t` is metres only because |u| = 1. Structural
        at the only caller, `sim/lidar.py`, so checking it would cost the hot path.
        """
        o = np.asarray(origin, dtype=np.float64).reshape(2)
        u = np.ascontiguousarray(directions, dtype=np.float64).reshape(-1, 2)
        return _cast_walls(
            o[0],
            o[1],
            u,
            self.wall_seg_a,
            self.wall_seg_d,
            self.wall_arc_c,
            self.wall_arc_r,
            self.wall_arc_mid,
            self.wall_arc_reach,
            float(max_range),
        )

    def contains(self, point) -> bool:
        """Is `point` on the racing surface? The exact form of `on_track`, and why this
        class holds the centre chain: a point is on track iff within `half_width` of
        it, so the union of per-primitive bands is the corridor.
        """
        p = np.asarray(point, dtype=np.float64).reshape(2)
        return bool(
            _contains(
                p[0],
                p[1],
                self.half_width,
                self.seg_a,
                self.seg_d,
                self.seg_len2,
                self.arc_c,
                self.arc_r,
                self.arc_mid,
                self.arc_cos_half,
            )
        )


@njit(cache=True)
def _contains(px, py, w, seg_a, seg_d, seg_len2, arc_c, arc_r, arc_mid, arc_cos_half):
    """The sweep behind `WallGeometry.contains`. Short-circuiting, where the array
    form evaluated every band: the car is on track on all but the last step, so the
    straight it stands over answers.
    """
    w2 = w * w
    for i in range(seg_a.shape[0]):
        fx = px - seg_a[i, 0]
        fy = py - seg_a[i, 1]
        # Both comparisons are scaled by |d| rather than normalized by it, which
        # is one sqrt and two divisions per primitive not spent.
        along = fx * seg_d[i, 0] + fy * seg_d[i, 1]
        if along < 0.0 or along > seg_len2[i]:
            continue
        cross = fx * seg_d[i, 1] - fy * seg_d[i, 0]
        if cross * cross <= w2 * seg_len2[i]:
            return True

    for i in range(arc_c.shape[0]):
        vx = px - arc_c[i, 0]
        vy = py - arc_c[i, 1]
        radius = math.hypot(vx, vy)
        if abs(radius - arc_r[i]) > w:
            continue
        # Unlike a ray hit, an arbitrary point is not *on* the arc, so the
        # bisector test carries its own radius instead of a folded-in one.
        if vx * arc_mid[i, 0] + vy * arc_mid[i, 1] >= radius * arc_cos_half[i]:
            return True
    return False


def point_in_polygon(point, polygon):
    """Ray-casting point-in-polygon (even-odd rule) over all edges at once. Nothing
    in the simulator calls it (termination asks `WallGeometry.contains`) it is
    the oracle the tests hold that exact containment to.
    """
    x, y = point
    polygon = np.asarray(polygon)
    xs, ys = polygon[:, 0], polygon[:, 1]
    xs_j, ys_j = np.roll(xs, 1), np.roll(ys, 1)

    cond = (ys > y) != (ys_j > y)
    # No epsilon on the denominator: `cond` already guarantees `ys_j != ys`, and a
    # floor would perturb every intersection. `errstate` covers the masked-out lanes
    # numpy evaluates regardless.
    with np.errstate(divide="ignore", invalid="ignore"):
        x_intersect = xs + (xs_j - xs) * (y - ys) / (ys_j - ys)
    crosses = cond & (x < x_intersect)
    return bool(np.count_nonzero(crosses) % 2)


def on_track(point, inner_wall, outer_wall):
    """A point is 'on track' if it's inside the outer wall but outside the inner wall."""
    return point_in_polygon(point, outer_wall) and not point_in_polygon(point, inner_wall)


def polyline_curvature(polyline):
    """Per-point signed curvature along a closed polyline, which sets how fast the car
    may go. The exact circumscribed circle `kappa = 4 * area / (|a| |b| |c|)`, not a
    finite difference, which is biased high where corners are tightest.
    """
    p = np.asarray(polyline)

    # The chords into and out of each sample plus the long chord closing the
    # triangle through the three of them, wraparound-aware via np.roll.
    a = p - np.roll(p, 1, axis=0)
    b = np.roll(p, -1, axis=0) - p
    c = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)

    # Twice the signed area of that triangle; the sign is the turn direction.
    twice_area = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    denom = (
        np.hypot(a[:, 0], a[:, 1]) * np.hypot(b[:, 0], b[:, 1]) * np.hypot(c[:, 0], c[:, 1]) + 1e-9
    )
    return 2.0 * twice_area / denom


@njit(cache=True)
def _nearest_index(point, path):
    """The scan behind `ArcLengthLUT.nearest_index`, and the first half of
    `_project`. Compiled because the array form built two n-by-2 temporaries and an
    n-vector to take one `argmin` off, per step, per environment.
    """
    best_i = 0
    best_d2 = np.inf
    for i in range(path.shape[0]):
        dx = path[i, 0] - point[0]
        dy = path[i, 1] - point[1]
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


@njit(cache=True)
def _project(px, py, path, cum_lengths, seg_lengths):
    """The scan and the two-segment test behind `ArcLengthLUT.project`; see it
    for which two segments and why.
    """
    n = path.shape[0]
    idx = _nearest_index(np.array((px, py)), path)

    best_s = 0.0
    best_cross = np.inf
    best_side = 0.0
    for i in (idx, (idx - 1) % n):
        p0x = path[i, 0]
        p0y = path[i, 1]
        j = (i + 1) % n
        sx = path[j, 0] - p0x
        sy = path[j, 1] - p0y
        seg_len2 = sx * sx + sy * sy
        rx = px - p0x
        ry = py - p0y
        # A repeated vertex has no direction to project onto, and the foot
        # of a zero-length segment is the vertex.
        frac = 0.0 if seg_len2 < 1e-12 else min(max((rx * sx + ry * sy) / seg_len2, 0.0), 1.0)
        fx = rx - frac * sx
        fy = ry - frac * sy
        cross = math.hypot(fx, fy)
        if cross < best_cross:
            best_s = cum_lengths[i] + frac * seg_lengths[i]
            best_cross = cross
            # Which side, from the same residual: positive puts the point left of
            # the direction of travel. Free against the `hypot` already paid for.
            best_side = sx * fy - sy * fx
    return best_s, best_cross if best_side >= 0.0 else -best_cross


class ArcLengthLUT:
    """Arc-length lookup table for a closed polyline. The expert and the progress
    reward both need "how far along, and how far did it move", so that bookkeeping
    lives here once.
    """

    def __init__(self, path):
        self.path = np.ascontiguousarray(path, dtype=np.float64)
        n = len(self.path)
        diffs = np.roll(self.path, -1, axis=0) - self.path
        self.seg_lengths = np.linalg.norm(diffs, axis=1)
        self.cum_lengths = np.zeros(n)
        self.cum_lengths[1:] = np.cumsum(self.seg_lengths)[:-1]
        self.total_length = self.cum_lengths[-1] + self.seg_lengths[-1]

    def nearest_index(self, pos):
        """Index of the path vertex nearest `pos`. Full O(n) scan: a path passes close to
        itself at a hairpin, and a windowed search was seen to lock onto the wrong
        branch permanently.
        """
        return _nearest_index(np.asarray(pos, dtype=np.float64).reshape(2), self.path)

    def project(self, pos):
        """`(arc_length, signed_cross_track)` of `pos` projected onto the path.

        Projects onto the nearer of the two segments meeting the nearest vertex, both
        falling out of one O(n) scan. That is the nearest segment itself whenever the
        offset is small against the sample spacing, the regime the car drives in.
        `cross_track` is positive left of travel; callers take `abs`.

        The nearest vertex is shared by two segments and its distance does not say
        which holds the foot of the perpendicular, so both are tried, outgoing first
        with a strict tie, keeping arc length in `[0, total_length)`.
        """
        pos = np.asarray(pos, dtype=np.float64).reshape(2)
        s, cross = _project(pos[0], pos[1], self.path, self.cum_lengths, self.seg_lengths)
        return float(s), float(cross)

    def arc_length_at(self, pos):
        """Arc length of the point on the path nearest to `pos`."""
        return self.project(pos)[0]
