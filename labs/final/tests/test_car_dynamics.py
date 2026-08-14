"""Contracts of the vehicle model that a parameter re-tune can silently break: each pins
a behaviour the model has *because* it is dynamic, not kinematic, and fails under one
plausible regression. The lapping tests work the other way, driving the pure-pursuit
baseline to catch a parameter set that is internally inconsistent. The car is GT3-class,
~72 m/s, and a threshold that would also hold for a 1/10 RC car is not testing it.
"""

import math

import numpy as np
import pytest

from tinyml_racing.sim.car import CarParams, CarState, _tire_forces, step
from tinyml_racing.sim.expert import (
    LATERAL_ACCEL_FRAC,
    SPAWN_HEADING_RAD,
    SPAWN_SPEED_FRAC,
    PurePursuit,
    random_start_state,
)
from tinyml_racing.sim.track_pool import build_track

PARAMS = CarParams()


def steps_for(seconds: float, params=PARAMS) -> int:
    """A duration in control steps. Every probe below is a manoeuvre in *seconds*, and
    writing those as step counts ties each threshold to `CarParams.dt`: a control-rate
    change then silently shortens the manoeuvre instead of failing.
    """
    return max(1, round(seconds / params.dt))


def drive(steer, throttle, seconds, state=None, params=PARAMS):
    state = state or CarState(x=0.0, y=0.0, theta=0.0)
    trace = []
    for _ in range(steps_for(seconds, params)):
        state = step(state, steer, throttle, params)
        trace.append(state)
    return state, trace


def test_top_speed_emerges_from_the_thrust_curve():
    # Top speed is where thrust equals resistance, not a clip: a clip stops the car
    # accelerating too, but with grip still spent on force that produces none. 60 s of
    # throttle, not 30, the last 0.4 m/s takes half the run as the two converge.
    final, trace = drive(0.0, 1.0, 60.0)
    speeds = np.array([s.vx for s in trace])
    assert final.vx == pytest.approx(PARAMS.top_speed, rel=1e-3)
    assert speeds.max() <= PARAMS.top_speed + 1e-6, "overshot the thrust/resistance balance"
    assert np.all(np.diff(speeds) >= -1e-9), "full throttle must not decelerate"


def test_lifting_the_throttle_slows_the_car():
    # Without rolling resistance and drag, coasting is lossless and full throttle is
    # optimal everywhere. The probe is at racing speed because drag is quadratic: a
    # slow probe would pass on rolling resistance alone with drag deleted.
    rolling, _ = drive(0.0, 0.0, 1.0, CarState(x=0.0, y=0.0, theta=0.0, vx=60.0))
    assert rolling.vx < 60.0 - 2.0


def test_a_stationary_car_stays_stationary():
    parked, _ = drive(0.0, 0.0, 2.0)
    assert parked.vx == pytest.approx(0.0)
    assert (parked.x, parked.y) == pytest.approx((0.0, 0.0))


def test_the_brakes_stop_the_car_and_hold_it():
    # A race car has a brake pedal, not an ESC sharing a channel with reverse, so zero
    # is a floor and not a crossing: without it, full brake demand backs the car out of
    # a standstill at 0.7 g, while the policy is still asking for maximum braking.
    final, trace = drive(0.0, -1.0, 10.0, CarState(x=0.0, y=0.0, theta=0.0, vx=40.0))
    speeds = [s.vx for s in trace]
    assert min(speeds) == 0.0, "the brakes drove the car backwards"
    stopped = speeds.index(0.0)
    assert all(v == 0.0 for v in speeds[stopped:]), "did not stay stopped"
    assert final.x > 0.0, "stopping moved the car backwards"


def test_resistance_brings_the_car_to_rest_without_juddering():
    # Resistance flips sign the instant velocity does. Without a clamp across
    # zero a coasting car settles into a limit cycle of alternating tiny
    # forward/backward steps instead of stopping.
    _, trace = drive(0.0, 0.0, 2.0, CarState(x=0.0, y=0.0, theta=0.0, vx=0.1))
    speeds = [s.vx for s in trace]
    assert min(speeds) == 0.0, "resistance pushed the car backwards"
    assert speeds[-1] == 0.0 and speeds[-2] == 0.0, "never came to rest"


def test_longitudinal_demand_has_priority_in_the_friction_ellipse():
    # The invariant one axle obeys: fx is granted in full up to the grip circle and fy
    # gets the Pythagorean remainder. Scaling both down together would make the brakes
    # fade mid-corner, whereas a real car keeps braking and refuses to turn.
    fz, stiffness, alpha = 8000.0, 9.0, 0.2
    grip = PARAMS.mu * fz
    assert stiffness * fz * alpha > grip, "pick a slip angle that saturates, or nothing is tested"

    free = _tire_forces(alpha, 0.0, fz, stiffness, PARAMS.mu)
    half = _tire_forces(alpha, 0.5 * grip, fz, stiffness, PARAMS.mu)
    spent = _tire_forces(alpha, grip, fz, stiffness, PARAMS.mu)
    over = _tire_forces(alpha, 2.0 * grip, fz, stiffness, PARAMS.mu)

    assert (free[0], half[0], spent[0]) == pytest.approx((0.0, 0.5 * grip, grip))
    assert over[0] == pytest.approx(grip), "longitudinal demand must clamp at the grip circle"
    assert -free[1] == pytest.approx(grip)
    assert -half[1] == pytest.approx(math.sqrt(grip**2 - (0.5 * grip) ** 2))
    assert spent[1] == pytest.approx(0.0), "all grip spent braking leaves none to turn with"


def test_power_oversteers_a_rear_wheel_drive_car():
    # Thrust goes to the rear axle alone, so opening the throttle spends the *rear*
    # lateral budget: the rear steps out, the car scrubs speed, and sideslip grows
    # against the turn. Splitting thrust across both axles would accelerate round.

    # A transient off a settled state, not the endpoint of a long run: which cornering
    # equilibrium this model settles into is a function of the substep h = dt/substeps,
    # so endpoint thresholds pinned an artifact. This moves under 0.1% across 50x dt.
    settled, _ = drive(PARAMS.max_steer, 0.35, 6.0)
    held, _ = drive(PARAMS.max_steer, 0.35, 0.6, settled)
    opened, _ = drive(PARAMS.max_steer, 1.0, 0.6, settled)
    assert opened.vx < held.vx, "full power at full lock must cost speed, not add it"
    assert abs(opened.r) > 2.0 * abs(held.r), "the rear axle did not give up grip"
    # Full lock left is a positive yaw rate, and an oversteering car's velocity
    # vector sits to the right of its nose, so tail-out is a sideslip of the
    # opposite sign to `r`, growing more negative as the rear lets go.
    assert held.r > 0.0, "positive steer must yaw the car positively"
    assert math.atan2(opened.vy, opened.vx) < math.atan2(held.vy, held.vx) < 0.0, (
        "the sideslip angle must grow against the turn as the rear axle gives up grip"
    )


def test_braking_mid_corner_rotates_the_car():
    # Load transfer unloads the rear axle under braking, so trail-braking tightens the
    # line; static axle loads cannot punish braking into a corner. 60 m is a real corner
    # for this car, an RC-scale 3 m probe would measure the lock stop instead.
    radius = 60.0
    steer = math.atan(PARAMS.wheelbase / radius)
    entry = CarState(x=0.0, y=0.0, theta=0.0, vx=0.9 * math.sqrt(PARAMS.grip_accel * radius))
    coasting, _ = drive(steer, 0.0, 0.3, entry)
    braking, _ = drive(steer, -0.5, 0.3, entry)
    assert braking.vx / braking.r < 0.9 * (coasting.vx / coasting.r)


def test_low_speed_yaw_follows_the_steering_geometry():
    # Below `blend_speed` the model relaxes onto the kinematic manifold, where
    # yaw rate is vx*tan(steer)/L. Checking the path radius rather than `r`
    # keeps the assertion about observable motion.
    creeping, _ = drive(PARAMS.max_steer, 0.05, 6.0)
    assert 0.0 < creeping.vx < PARAMS.blend_speed
    assert creeping.vx / creeping.r == pytest.approx(PARAMS.min_turn_radius, rel=0.15)


def test_lateral_dynamics_stay_bounded_under_adversarial_input():
    # The lateral/yaw eigenvalue scales as 1/vx, capped near 36 1/s by `blend_speed`, so
    # what keeps explicit Euler bounded is h*lambda with h = dt/substeps (0.045 shipped).
    # Random flailing is what PPO does first, and where smooth-input tests see nothing.
    rng = np.random.default_rng(0)
    state = CarState(x=0.0, y=0.0, theta=0.0)
    for _ in range(20_000):
        action = rng.uniform(-1.0, 1.0, 2)
        state = step(state, action[0] * PARAMS.max_steer, action[1], PARAMS)
    assert all(math.isfinite(v) for v in (state.x, state.y, state.theta))
    assert abs(state.vx) <= PARAMS.top_speed + 1e-6
    assert abs(state.r) < 4.0 * PARAMS.grip_accel / max(abs(state.vx), PARAMS.blend_speed)


def test_steering_obeys_the_servo_slew_limit():
    stepped = step(CarState(x=0.0, y=0.0, theta=0.0, vx=3.0), PARAMS.max_steer, 0.0, PARAMS)
    assert stepped.steer == pytest.approx(PARAMS.max_steer_rate * PARAMS.dt)
    assert stepped.steer < PARAMS.max_steer, "one step must not reach full lock"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_pure_pursuit_baseline_laps_without_leaving_the_track(seed):
    # A controller that follows from first principles, so a clean run is evidence about
    # the vehicle *parameters*: too little grip for the top speed it implies fails here
    # while every single-manoeuvre test above passes. The path is the centerline.

    # The distance floor is half the assertion: staying on the track is trivial for a
    # car that stops, and a grip/thrust mismatch shows up as a baseline that creeps
    # rather than one that crashes. These layouts are 2.9-3.7 km at 60+ m/s.
    track = build_track(seed).track
    controller = PurePursuit(track.centerline)
    state = CarState(
        x=track.centerline[0][0],
        y=track.centerline[0][1],
        theta=math.atan2(track.tangents[0][1], track.tangents[0][0]),
        vx=10.0,
    )
    travelled = 0.0
    for i in range(steps_for(80.0)):
        steer, throttle = controller.control(state)
        assert abs(steer) <= PARAMS.max_steer + 1e-9, "commanded beyond full lock"
        before = (state.x, state.y)
        state = step(state, steer, throttle, PARAMS)
        travelled += math.hypot(state.x - before[0], state.y - before[1])
        assert track.walls.contains((state.x, state.y)), f"left the track at step {i}"
    assert travelled > 1500.0, f"only covered {travelled:.0f} m, the baseline is crawling"


@pytest.mark.parametrize("seed", range(6))
def test_a_random_spawn_starts_on_the_track_pointing_down_it(seed):
    # Every episode begins here, so a spawn in a wall is an unearned crash the
    # policy cannot learn away, and a spawn pointing backwards teaches it to
    # drive the lap the wrong way. Both are silent: the reward just gets worse.
    track = build_track(seed % 3).track
    rng = np.random.default_rng(seed)
    for _ in range(16):
        state = random_start_state(track, rng)
        assert track.walls.contains((state.x, state.y)), "spawned off the racing surface"
        nearest = int(np.argmin(((track.centerline - [state.x, state.y]) ** 2).sum(axis=1)))
        heading = np.array([math.cos(state.theta), math.sin(state.theta)])
        # Tied to the draw, not a loose bound: the point of `SPAWN_HEADING_RAD`
        # is that it is small enough for the car to straighten out, and an
        # assertion with slack in it would not notice the draw widening again.
        off_angle = math.acos(min(1.0, heading @ track.tangents[nearest]))
        assert off_angle <= SPAWN_HEADING_RAD + 1e-9, (
            f"spawned {off_angle:.3f} rad across the track"
        )
        assert 0.0 <= state.vx <= SPAWN_SPEED_FRAC * PARAMS.top_speed


@pytest.mark.parametrize("seed", range(4))
def test_a_random_spawn_is_no_faster_than_its_corner_allows(seed):
    # The flat draw ran to 0.6*top_speed = 43 m/s while the tightest generated corner
    # holds 23, so a hairpin spawn was a crash the policy could not have avoided from
    # step 0, a termination penalty charged for the reset.
    track = build_track(seed).track
    rng = np.random.default_rng(seed)
    tangents, ahead = track.tangents, np.roll(track.tangents, -1, axis=0)
    kappa = np.abs(tangents[:, 0] * ahead[:, 1] - tangents[:, 1] * ahead[:, 0]) / track.ds
    assert kappa.max() > 0.0, "a track with no curvature cannot test a corner cap"
    for _ in range(64):
        state = random_start_state(track, rng)
        idx = int(np.argmin(((track.centerline - [state.x, state.y]) ** 2).sum(axis=1)))
        limit = PARAMS.corner_speed(kappa[idx], LATERAL_ACCEL_FRAC)
        assert state.vx <= max(limit, 1e-9) + 1e-6, (
            f"spawned at {state.vx:.1f} m/s into a corner good for {limit:.1f}"
        )


def test_pure_pursuit_previews_a_path_shorter_than_its_braking_window():
    # `preview_dist` is a braking distance from top speed (298 m) so every
    # hand-built path is a fraction of it, and a wrapped slice sized for a 3 km layout
    # then skips samples and pairs others with a distance a lap out.
    straight, radius, per = 25.0, 8.0, 16
    x = np.linspace(-straight / 2, straight / 2, per, endpoint=False)
    entry = np.linspace(math.pi / 2, -math.pi / 2, per, endpoint=False)
    exit_ = entry - math.pi
    path = np.concatenate(
        [
            np.stack([x, np.full(per, radius)], axis=1),
            np.stack([straight / 2 + radius * np.cos(entry), radius * np.sin(entry)], axis=1),
            np.stack([-x, np.full(per, -radius)], axis=1),
            np.stack([-straight / 2 + radius * np.cos(exit_), radius * np.sin(exit_)], axis=1),
        ]
    )
    controller = PurePursuit(path)
    assert controller.preview_dist > controller.arclen.total_length, "path is not short enough"
    for idx in range(len(path)):
        limits, ahead = controller._preview(idx)
        assert len(limits) == len(path), "the window is not exactly one lap of samples"
        assert ahead[0] == 0.0 and np.all(np.diff(ahead) > 0.0), "distances repeat or go backwards"
        assert ahead[-1] < controller.arclen.total_length

    # Halfway down the straight, offset towards the outside, at a speed the 8 m
    # semicircle ahead cannot hold: the command must be a real brake.
    steer, throttle = controller.control(CarState(x=-6.0, y=radius + 1.5, theta=0.0, vx=30.0))
    assert math.isfinite(steer) and math.isfinite(throttle)
    assert abs(steer) <= PARAMS.max_steer + 1e-9
    assert -1.0 <= throttle < 0.0, f"did not brake for the corner, commanded {throttle:.2f}"
