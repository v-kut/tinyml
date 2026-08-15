"""Geometric contracts of the generated circuit: everything downstream assumes an
annulus sampled at uniform arc length, and each assumption breaks silently, corrupt
observations, not an exception. The simulator casts against `track.walls`, the corridor
as arcs and straights, and the sampled polyline is what that is held to here. Arc length
and curvature are held to shapes with closed forms; see the block at the bottom.
"""

import itertools
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from tinyml_racing.sim.car import CarParams
from tinyml_racing.sim.geometry import (
    ArcLengthLUT,
    on_track,
    point_in_polygon,
    polyline_curvature,
    ray_polyline_distances,
)
from tinyml_racing.sim.lidar import ray_angles
from tinyml_racing.sim.racing_line import compute_racing_line
from tinyml_racing.sim.track import (
    TrackConfig,
    default_corner_speed_radius,
    generate_track,
    min_steerable_corner_radius,
)
from tinyml_racing.sim.track.outline import _signed_area


@pytest.fixture(scope="module", params=(0, 7, 42))
def track(request):
    return generate_track(seed=request.param)


def test_samples_are_evenly_spaced_by_arc_length(track):
    # The racing-line objective drops its per-sample 1/ds^2 weighting on the
    # strength of this, and the arc-length progress reward assumes it too.
    c, ds = track.centerline, track.ds
    assert track.s[0] == 0.0
    assert np.abs(np.diff(track.s) - ds).max() < 1e-6
    assert abs(track.s[-1] + ds - track.length) < 1e-6
    # Chords fall short of arcs by at most the tightest corner's sagitta; the
    # slack absorbs the dense grid the arc-length resample interpolates on.
    chord = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
    tightest = 1.0 / np.abs(track.curvature).max()
    assert chord.max() <= ds + 1e-6
    assert chord.min() >= 2 * tightest * np.sin(ds / (2 * tightest)) - 1e-4


def test_walls_nest_and_the_infield_is_a_single_hole(track):
    # What `cast_lidar` and the env's termination test need: crossed offset
    # walls make the racing surface read as off track instead of raising.
    inner, outer = track.inner_wall, track.outer_wall
    assert all(point_in_polygon(p, outer) for p in inner)
    assert not any(point_in_polygon(p, inner) for p in outer)
    assert all(on_track(p, inner, outer) for p in track.centerline)


def test_position_heading_curvature_and_normals_agree(track):
    # Four arrays from one closed form, consumed independently: spawning uses
    # `tangents`, the racing line `normals`, the speed planner `curvature`.

    # Both agreement checks below are against finite differences of the *samples*, so
    # their tolerances are discretization limits and are written as such: a round number
    # here reads as a statement about the fields and is not one.
    c, tg, nm, ds = track.centerline, track.tangents, track.normals, track.ds
    tightest = 1.0 / np.abs(track.curvature).max()
    fd = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
    fd /= np.linalg.norm(fd, axis=1)[:, None]
    # A central difference across a sample spans a chord, and on the tightest arc
    # that chord leans off the true tangent by O(ds/R). Measured 0.18*(ds/R) over
    # a decade of spacings; 0.5 is loose enough not to be a fit to one seed.
    assert np.einsum("ij,ij->i", fd, tg).min() > np.cos(0.5 * ds / tightest), (
        "tangents leave the path by more than the chord they are sampled on"
    )
    assert np.abs(np.linalg.norm(nm, axis=1) - 1.0).max() < 1e-9
    assert np.abs(np.einsum("ij,ij->i", tg, nm)).max() < 1e-9
    heading = np.arctan2(tg[:, 1], tg[:, 0])
    turn = (np.roll(heading, -1) - np.roll(heading, 1) + np.pi) % (2 * np.pi) - np.pi
    # Curvature steps at every arc/straight join, exact arcs and exact straights make
    # it a square wave, and a central difference straddling that step returns about
    # half the arc's value however fine the sampling, so `1/(2R)` is the floor here.
    assert np.abs(turn / (2 * ds) - track.curvature).max() < 1.0 / (2.0 * tightest) + 1e-9, (
        "curvature sign or roll"
    )
    # Outward is an orientation, not a distance from the middle: comparing distances to
    # the centroid only holds on a convex lap, and every layout turns both ways.

    # Signed area is the statement that survives a non-convex lap: offsetting along
    # `+normals` must enclose *more* than offsetting against it, so a flipped normal
    # fails here. Nesting is the other half, next door.
    assert _signed_area(track.outer_wall) > _signed_area(track.inner_wall), (
        "normals do not point away from the infield"
    )


def test_the_corner_floor_is_read_off_the_car_and_not_carried_separately():
    """The regression this file exists to catch, stated without a track.

    `min_corner_radius` sat at a literal 10 m while `CarParams` grew into a GT car whose
    full-lock radius is 8.6 m, so every layout contained corners it could not be steered
    through. A literal cannot notice the car changing; a derived default cannot avoid it.
    """
    assert TrackConfig().min_corner_radius == pytest.approx(min_steerable_corner_radius())
    # And the window has to stay a window: tight enough that the car must brake
    # somewhere, wide enough that it can steer the tightest thing drawn.
    assert min_steerable_corner_radius() < default_corner_speed_radius()
    # Sensitive to the car, not frozen at today's number.
    quick = replace(CarParams(), max_steer=0.6)
    assert min_steerable_corner_radius(quick) < min_steerable_corner_radius()


def test_the_whole_corridor_is_on_track(track):
    # An episode may use every metre between the walls, not just the middle.
    inner, outer = track.inner_wall, track.outer_wall
    idx = np.unique(np.linspace(0, len(track.centerline) - 1, 48).astype(int))
    # One offset bounds both walls, so the sign of `reach` is what picks a side.
    for reach in (-0.98, -0.5, 0.5, 0.98):
        lateral = (reach * track.wall_offset[idx])[:, None] * track.normals[idx]
        probe = track.centerline[idx] + lateral
        assert all(on_track(p, inner, outer) for p in probe), "racing surface leaves the walls"


def test_the_tightest_corner_stays_inside_the_grip_envelope(track):
    # The generator's own acceptance window, restated: wide enough that the
    # offset walls still nest and the car can steer it, tight enough that at
    # least one corner cannot be taken flat out.
    tightest = 1.0 / np.abs(track.curvature).max()
    half = track.wall_offset
    assert tightest > half.max(), "walls cannot nest around a corner this tight"
    assert tightest > 2.0 * CarParams().min_turn_radius, "too tight to steer"
    assert tightest <= default_corner_speed_radius() + 1e-9, "no corner needs braking"
    # One width per lap, not a drift along it: that is exactly what makes
    # `track.walls` an offset of an arc chain rather than an approximation of
    # one, so it is asserted rather than left to a caster to discover.
    assert half.min() == half.max(), "a drifting corridor is not an offset of an arc"


def test_start_finish_line_sits_on_a_straight(track):
    # Index 0 is the midpoint of the longest low-curvature run, so an episode
    # never begins mid-corner.
    kappa = np.abs(track.curvature)
    assert kappa[0] < 0.25 * kappa.max()


def test_the_racing_line_keeps_its_clearance_from_the_wall(track):
    # Inside the corridor is not enough: the line has to leave room for the error of
    # whatever follows it, and the baseline tracks it to a median ~0.95 m, so a line
    # 0.36 m off the wall puts a correctly-driving car into it.
    clearance = 1.0
    line = compute_racing_line(track, wall_clearance=clearance)
    offset = np.einsum("ij,ij->i", line - track.centerline, track.normals)
    room = np.maximum(track.wall_offset - clearance, 0.1 * track.wall_offset)
    assert np.all(np.abs(offset) <= room + 1e-9)
    assert all(on_track(p, track.inner_wall, track.outer_wall) for p in line)


def test_generation_is_deterministic_and_honours_the_sample_grid():
    a = generate_track(seed=11)
    np.testing.assert_array_equal(a.centerline, generate_track(seed=11).centerline)
    assert generate_track(seed=12).length != a.length
    assert generate_track(3, TrackConfig(sample_spacing=0.8)).ds == pytest.approx(0.8, rel=0.05)
    assert generate_track(3, TrackConfig(n_samples=512)).centerline.shape == (512, 2)


def test_the_lap_reaches_inside_its_own_ring_and_turns_both_ways():
    """The shape contract: hooks and chicanes are in the layout, not just in the config.

    A convex outline filleted into a ring passes every other test in this file and
    is not a circuit: it turns one way for a whole lap and never enters the infield.
    The 18 F1 layouts of `TUMFTM/racetrack-database` spend 0-45% of a lap more than
    120 m inside their own convex hull and integrate 3.2-5.9 laps' worth of absolute
    turning; this generator with `n_hooks=(0, 0), n_chicanes=(0, 0)` manages 1.74
    (200 seeds, 1.00-2.23) and with them 3.87 (2.58-5.78).

    So the floor sits between the two populations rather than at the corpus median:
    it fails the moment jogs stop reaching the outline, and tolerates the draw.
    """
    for seed in range(12):
        track = generate_track(seed)
        sweep = 2.0 * np.arccos(np.clip(track.walls.arc_cos_half, -1.0, 1.0))
        turning = float(np.abs(sweep).sum() / (2.0 * np.pi))
        assert 2.4 <= turning <= 6.5, f"seed {seed} turns {turning:.2f} laps' worth"

        hull = ConvexHull(track.centerline)
        # Signed distance inside every facet of the hull; positive is inside it.
        inside = -(track.centerline @ hull.equations[:, :2].T + hull.equations[:, 2])
        assert inside.min(axis=1).max() > 100.0, f"seed {seed} never leaves its perimeter"
        # Corner density, what makes a lap read as a circuit and not as a slalom.
        # Counted per arc primitive, where the corpus figure of 2.0-4.4 counts runs
        # of a smoothed curvature and merges the pairs a jog is made of.
        per_km = len(track.walls.arc_r) / (track.length / 1000.0)
        assert 2.0 <= per_km <= 8.0, f"seed {seed} has {per_km:.1f} corners per km"


def test_unsatisfiable_request_raises_rather_than_returning_junk():
    # Silently returning a layout outside the grip envelope would poison every
    # episode run on it instead of failing where the request was made.
    with pytest.raises(RuntimeError):
        generate_track(5, TrackConfig(length_range=(1.0, 2.0)))


def _poses(track, n=48, seed=0):
    """On-track poses, roughly along the lap and pointing roughly down it."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(track.n, size=n)
    lateral = rng.uniform(-0.9, 0.9, size=n) * track.wall_offset[idx]
    origins = track.centerline[idx] + lateral[:, None] * track.normals[idx]
    heading = np.arctan2(track.tangents[idx, 1], track.tangents[idx, 0])
    return origins, heading + rng.uniform(-0.3, 0.3, size=n)


def _cast_errors(track, max_range=150.0, n_rays=60):
    """Per-ray disagreement between the exact caster and the drawn polyline.

    Rays that reach neither wall are dropped: both casters return `max_range`
    there by fiat, so including them would dilute the disagreement with
    agreement neither one computed.
    """
    errs = []
    for origin, theta in zip(*_poses(track), strict=True):
        # A uniform fan: this measures the caster against the polyline, and a
        # warped one would just weight the same comparison towards the nose.
        angles = theta + ray_angles(n_rays, 240.0, 1.0)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        exact = track.walls.ray_distances(origin, directions, max_range)
        drawn = np.minimum(
            ray_polyline_distances(origin, directions, track.outer_wall, max_range),
            ray_polyline_distances(origin, directions, track.inner_wall, max_range),
        )
        reached = (exact < max_range - 1e-3) & (drawn < max_range - 1e-3)
        errs.append(np.abs(exact[reached] - drawn[reached]))
    return np.concatenate(errs)


def test_the_drawn_wall_converges_on_the_exact_one():
    """`walls.ray_distances` is the limit the polyline caster approaches.

    This is the whole claim behind casting against arcs instead of samples, and "the two
    agree closely" would not establish it. Sagitta is quadratic in chord length, so
    quartering the spacing has to cut the disagreement by about sixteen.

    On the 99th percentile, not the maximum: the maximum is the handful of grazing rays
    the polyline puts on the wrong wall entirely, which appear and vanish with the exact
    tangency rather than shrinking smoothly.
    """
    p99 = [
        float(np.percentile(_cast_errors(generate_track(7, TrackConfig(sample_spacing=h))), 99))
        for h in (4.0, 1.0, 0.25)
    ]
    assert p99[0] > 0.05, "the polyline error being replaced is smaller than reported"
    assert p99[-1] < 0.005
    for coarse, fine in itertools.pairwise(p99):
        assert 8.0 < coarse / fine < 32.0, f"not quadratic convergence: {p99}"


def test_containment_agrees_with_the_drawn_polygon_away_from_the_walls(track):
    # `walls.contains` is exact and `on_track` is the drawn polygon, so the two can only
    # disagree within a sagitta of a wall. Everywhere else they must agree, outside
    # included: containment returning True off the track ends no episode.
    idx = np.unique(np.linspace(0, track.n - 1, 64).astype(int))
    normals, offset = track.normals[idx], track.wall_offset[idx]
    for reach, expected in ((0.0, True), (0.9, True), (-0.9, True), (1.15, False), (-1.15, False)):
        probes = track.centerline[idx] + (reach * offset)[:, None] * normals
        assert all(track.walls.contains(p) is expected for p in probes)
        assert all(on_track(p, track.inner_wall, track.outer_wall) is expected for p in probes)


def test_corridor_width_is_drawn_per_layout():
    # The variety the removed within-lap drift used to provide, moved to where a
    # constant offset still admits it. A draw that never moves is a constant.
    cfg = TrackConfig()
    widths = np.array([2.0 * generate_track(s, cfg).wall_offset[0] for s in range(12)])
    band = cfg.width * cfg.width_variation
    assert np.abs(widths - cfg.width).max() <= band + 1e-9
    assert np.ptp(widths) > band, "the width draw is not exercising its range"


# Arc length and curvature, held to shapes whose answers are known exactly: the
# generated circuit cannot referee either, knowing its own arc length only through the
# cumulative sum under test. A sampled square, a polygon and a circle are closed forms.


def _sampled_square(side=10.0, spacing=1.0):
    """A CCW square sampled at uniform spacing, first vertex at the origin.

    Chosen because every quantity the LUT reports is an integer: vertex `k` sits at arc
    length `k * spacing`, the lap is `4 * side`, and a probe held `d` metres off an edge
    has cross-track exactly `d`, none of it recomputed from the table's own sum.
    """
    n = round(side / spacing)
    t = np.arange(n) * spacing
    zero, full = np.zeros(n), np.full(n, side)
    return np.concatenate(
        [
            np.column_stack([t, zero]),  # (0,0) -> (side-,0)
            np.column_stack([full, t]),  # (side,0) -> (side,side-)
            np.column_stack([side - t, full]),  # (side,side) -> (0+,side)
            np.column_stack([zero, side - t]),  # (0,side) -> (0,0+), closes
        ]
    )


def _sampled_circle(radius, n):
    ang = 2.0 * np.pi * np.arange(n) / n
    return radius * np.column_stack([np.cos(ang), np.sin(ang)])


def test_project_returns_the_exact_foot_from_the_middle_of_a_segment():
    # The easy half of the contract, and the one the old single-segment code
    # already got right: a probe abeam a segment's midpoint. Pinned first so a
    # later failure localises to the vertex handling rather than to projection.
    lut = ArcLengthLUT(_sampled_square())
    s, cross = lut.project((7.5, -0.4))
    assert s == pytest.approx(7.5, abs=1e-12)
    # Signed, and negative: the square is sampled counter-clockwise, so this edge runs
    # along +x and a probe below it sits to the *right* of travel. Asserted rather than
    # absorbed by an `abs`, because a flipped sign steers the policy the wrong way.
    assert cross == pytest.approx(-0.4, abs=1e-12)


def test_project_reaches_behind_the_nearest_vertex_instead_of_snapping_to_it():
    """The foot of the perpendicular is not always on the *outgoing* segment.

    A probe 0.05 m short of vertex 8 is nearest to that vertex, but its foot lies on the
    incoming segment 7 -> 8. Considering only `8 -> 9` clamps `frac` to zero and reports
    the vertex itself, freezing progress and inflating the cross-track error.
    """
    lut = ArcLengthLUT(_sampled_square())
    s, cross = lut.project((7.95, -0.3))
    assert s == pytest.approx(7.95, abs=1e-12)
    assert cross == pytest.approx(-0.3, abs=1e-12)


def test_project_is_strictly_monotonic_along_a_straight_run():
    """Progress along a straight must be the straight's own coordinate.

    The regression test for the vertex snap: walking a probe along one edge at a constant
    0.25 m offset, the true arc length is exactly the probe's x. Snapping to the nearest
    vertex turns that ramp into a staircase, flat over each half-segment before a vertex.

    That is what the progress reward saw, no reward for the metre before every sample,
    then a jump, so both the equality and the strict monotonicity are needed.
    """
    lut = ArcLengthLUT(_sampled_square())
    xs = np.linspace(0.5, 9.5, 401)
    got = np.array([lut.project((x, -0.25)) for x in xs])
    np.testing.assert_allclose(got[:, 0], xs, atol=1e-12)
    np.testing.assert_allclose(got[:, 1], -0.25, atol=1e-12)
    assert np.all(np.diff(got[:, 0]) > 0.0), "arc length is a staircase, not a ramp"


def test_project_approaches_total_length_from_below_across_the_closing_segment():
    """The lap must end at `total_length`, not restart at zero one sample early.

    The closing segment `n-1 -> 0` is the one place where the nearest vertex (index 0) and
    the segment holding the foot (index n-1) disagree across the seam. Reporting the
    outgoing segment there returns ~0.0 for a car that has not yet crossed the line.

    Downstream that reads as a completed lap: a full lap of backwards motion in the
    progress reward, and a lap counter that fires early.
    """
    lut = ArcLengthLUT(_sampled_square())
    total = lut.total_length
    assert total == pytest.approx(40.0, abs=1e-12)

    behind = np.array([0.5, 0.25, 0.1, 0.02, 1e-3, 1e-6])
    got = np.array([lut.project((-0.2, y))[0] for y in behind])
    np.testing.assert_allclose(got, total - behind, atol=1e-12)
    assert np.all(got < total), "arc length overshot the lap"
    assert np.all(got > total - 1.0), "arc length wrapped to the start of the lap"
    assert np.all(np.diff(got) > 0.0)


def test_arc_length_at_is_the_identity_on_every_vertex_including_the_wrap():
    # The table's defining property. The wrap vertex is the interesting one twice over:
    # `path[-1]` must report `cum_lengths[-1]` and not a full lap, and `path[0]` must
    # report 0.0 and not `total_length`, the tie the outgoing segment has to win.
    path = _sampled_square()
    lut = ArcLengthLUT(path)
    got = np.array([lut.arc_length_at(p) for p in path])
    np.testing.assert_allclose(got, np.arange(len(path), dtype=np.float64), atol=1e-12)
    assert got[0] == 0.0, "the start/finish vertex reported a whole lap"
    assert got[-1] == pytest.approx(lut.total_length - 1.0, abs=1e-12)


def test_project_holds_on_a_regular_polygon_whose_arc_length_is_known():
    # The square's segments are axis-aligned and its arc lengths integers, which a
    # projection that mixed up its components could still satisfy. On a rotated polygon
    # side `k`'s midpoint sits at `(k + 1/2) * chord`, and side `n-1` tests the closer.
    n, radius, offset = 12, 5.0, 0.35
    ang = 2.0 * np.pi * np.arange(n) / n
    lut = ArcLengthLUT(radius * np.column_stack([np.cos(ang), np.sin(ang)]))

    chord = 2.0 * radius * np.sin(np.pi / n)
    apothem = radius * np.cos(np.pi / n)
    assert lut.total_length == pytest.approx(n * chord, rel=1e-12)
    for k in range(n):
        mid = ang[k] + np.pi / n
        s, cross = lut.project((apothem + offset) * np.array([np.cos(mid), np.sin(mid)]))
        assert s == pytest.approx((k + 0.5) * chord, abs=1e-9)
        # Outside a counter-clockwise polygon is to the right of travel, so the
        # signed residual is negative at every side including the closing one.
        assert cross == pytest.approx(-offset, abs=1e-9)


def test_curvature_of_a_sampled_circle_is_one_over_the_radius_at_any_spacing():
    """Three samples of a circle of radius R circumscribe that same circle.

    So the answer is `1/R` whatever the spacing, and the tolerance below is
    spacing-independent by construction. A finite difference returns `2/(R(1 + cos phi))`
    instead (7.2% high at 12 samples) and it is the speed planner's input.
    """
    radius = 25.0
    for n in (12, 40, 400):
        kappa = polyline_curvature(_sampled_circle(radius, n))
        np.testing.assert_allclose(kappa, 1.0 / radius, rtol=1e-7)


def test_curvature_carries_no_spacing_dependent_bias():
    # Stated directly rather than inferred from the tolerance above: the same circle read
    # coarsely and finely must give the same number. Any truncated expansion in `ds`
    # fails this by definition, differing 4% of 1/R between these two spacings.
    radius = 25.0
    coarse = polyline_curvature(_sampled_circle(radius, 16))
    fine = polyline_curvature(_sampled_circle(radius, 256))
    assert abs(coarse.mean() - fine.mean()) * radius < 1e-7


def test_curvature_is_signed_by_turn_direction():
    # The speed planner takes `abs`, but the racing-line solver and the
    # observation do not: a sign that follows the sampling order rather than
    # the turn would mirror every corner.
    radius = 25.0
    ccw = _sampled_circle(radius, 64)
    np.testing.assert_allclose(polyline_curvature(ccw), 1.0 / radius, rtol=1e-7)
    np.testing.assert_allclose(polyline_curvature(ccw[::-1]), -1.0 / radius, rtol=1e-7)


def test_curvature_is_zero_on_collinear_samples_and_exact_at_a_corner():
    # Two more closed forms, and the wraparound: a straight run must read exactly zero (a
    # corner speed limit on a straight throws lap time away), and a right angle between
    # unit chords gives kappa = sqrt(2), at index 0 too, which only `np.roll` reaches.
    square = _sampled_square()
    corners = np.array([0, 10, 20, 30])
    kappa = polyline_curvature(square)
    straights = np.setdiff1d(np.arange(len(square)), corners)
    np.testing.assert_allclose(kappa[straights], 0.0, atol=1e-12)
    np.testing.assert_allclose(kappa[corners], np.sqrt(2.0), rtol=1e-8)
