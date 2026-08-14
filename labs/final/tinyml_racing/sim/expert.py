"""The pure-pursuit baseline and the episode spawn, neither of which drives the
deployed car.

`PurePursuit` validates `CarParams` and is what `ml/regression/` clones; it reads
privileged simulator state the policy never sees. `random_start_state` draws the pose
episodes begin from. Gains here tune the baseline, not the vehicle.
"""

import math
from functools import lru_cache

import numpy as np

from tinyml_racing.sim.car import CarParams, CarState, _clamp, _drive_force, lateral_grip_usage
from tinyml_racing.sim.geometry import ArcLengthLUT, polyline_curvature

LOOKAHEAD_BASE = 8.0  # m, ~3 wheelbases
LOOKAHEAD_SPEED_GAIN = 0.4  # s, extra lookahead per m/s, the standard stabilizer
# Fraction of the available lateral grip the planner will use. The margin is
# what covers a corner-entry transient on top of the steady-state limit.
LATERAL_ACCEL_FRAC = 0.45
# Ceiling on the spawn-speed draw, as a fraction of `top_speed`, before the
# corners within braking reach tighten it further (`random_start_state`).
SPAWN_SPEED_FRAC = 0.6
# Half-width of the spawn heading draw, rad. Wide enough to cover off-angle
# states, narrow enough that the car can straighten out of every draw.
SPAWN_HEADING_RAD = 0.12
SPEED_GAIN = 2.0  # P gain, m/s^2 per m/s of speed error
# Floor under the ellipse term, as a fraction of the pedal: with no floor a car
# already at the lateral limit cannot brake at all and simply runs wide.
ELLIPSE_PEDAL_FLOOR = 0.15


def _yaw_stable_decel(vx: float, params: CarParams) -> float:
    """Largest braking deceleration that leaves the car yaw-stable at `vx`, m/s^2.

    A stability limit rather than a grip limit, and the tighter of the two: forward
    load transfer flips this car to oversteering, so at the full 20 m/s^2 the critical
    speed falls to 29 m/s.
    """
    # Stable while `Cf*Cr*L^2 > m*v^2*(lf*Cf - lr*Cr)`. Substituting the transferred
    # load `t = m*a*cg_height/L` gives a quadratic with one positive root, solved here.
    lever = params.c_stiff_front * params.c_stiff_rear * params.wheelbase**2
    load = params.mass * params.gravity + params.downforce_coeff * vx * vx
    front = load * params.lr / params.wheelbase
    rear = load * params.lf / params.wheelbase
    # Static understeer moment, positive when c_stiff_rear > c_stiff_front, and
    # the rate at which a unit of transferred load eats into it.
    margin = load * params.lf * params.lr * (params.c_stiff_rear - params.c_stiff_front)
    margin /= params.wheelbase
    rate = params.lf * params.c_stiff_front + params.lr * params.c_stiff_rear
    mv2 = params.mass * vx * vx
    b = lever * (rear - front) - mv2 * rate
    c = lever * front * rear + mv2 * margin
    transfer = (b + math.sqrt(b * b + 4.0 * lever * c)) / (2.0 * lever)
    return transfer * params.wheelbase / (params.mass * params.cg_height)


@lru_cache(maxsize=8)
def _brake_envelope(params: CarParams, steps: int = 256) -> float:
    """The deceleration a plan may assume, m/s^2: the floor of `_yaw_stable_decel`
    over the speed range, roughly half of `brake_accel_max`.

    Sizing a plan on what the brakes could produce would size it on a deceleration
    this controller forbids itself. Drag is left out, so it is margin. The floor rather
    than an average, because a plan holding at the worst speed holds everywhere.
    """
    v = np.linspace(0.0, params.top_speed, steps)
    a = np.array([_yaw_stable_decel(max(u, params.stop_speed), params) for u in v])
    return float(a.min())


class PurePursuit:
    """Pure-pursuit steering + curvature-limited speed planning along `path`.

    Curvature and the arc-length table are precomputed because both are O(N) in
    the path length and `control` runs once per control interval.
    """

    def __init__(self, path, params: CarParams | None = None):
        p = params or CarParams()
        self.params = p
        # One owner for the path: `ArcLengthLUT` already keeps it as float64.
        self.arclen = ArcLengthLUT(path)
        self.curvature = polyline_curvature(self.arclen.path)
        self.top_speed = p.top_speed
        # Corner-entry preview and the deceleration a plan inside it may assume,
        # sized from the same constant the plan uses. `_brake_envelope`'s variable-`a`
        # integral is 263 m, but `_target_speed` decelerates at the envelope's floor,
        # which needs 298 m from top speed, so the integral entered the preview 11%
        # too late and made the plan's own bound unreachable.
        self.brake_accel = _brake_envelope(p)
        self.preview_dist = p.top_speed**2 / (2.0 * self.brake_accel) + p.wheelbase
        # Every sample's steady-state corner speed, solved once. Feeding
        # `lateral_accel(state.vx)` instead makes the limit a fixed point, optimistic
        # on entry and needing distance a hairpin does not give.
        self.corner_limit = p.corner_speed(np.abs(self.curvature), LATERAL_ACCEL_FRAC)
        # Understeer compensation in metres of effective wheelbase: at the grip limit
        # the car needs a further `K * ay` of steer, and with stiffness per unit load
        # K collapses to `m * (1/Cf - 1/Cr) / Fz`.
        self.understeer = p.mass * (1.0 / p.c_stiff_front - 1.0 / p.c_stiff_rear)

    def _preview(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Corner limits within the preview window, and their arc distance ahead.

        The window covers at most one lap: on a path shorter than
        `preview_dist` a wrapped slice skips samples and duplicates others.
        """
        cum = self.arclen.cum_lengths
        total = self.arclen.total_length
        n = len(cum)
        start = cum[idx]
        if self.preview_dist >= total:
            order = (idx + np.arange(n)) % n
            return self.corner_limit[order], (cum[order] - start) % total
        preview_arc = start + self.preview_dist
        end = int(np.searchsorted(cum, preview_arc % total, side="right"))
        if preview_arc < total:
            return self.corner_limit[idx:end], cum[idx:end] - start
        return (
            np.concatenate((self.corner_limit[idx:], self.corner_limit[:end])),
            np.concatenate((cum[idx:] - start, cum[:end] + (total - start))),
        )

    def _target_speed(self, idx: int) -> float:
        """Fastest speed at sample `idx` that still brakes for the whole window.

        `min_j sqrt(v_j^2 + 2*a*d_j)`, the braking envelope rather than the tightest
        corner in it: as `max(kappa)`, a hairpin 130 m away would hold the straight
        leading to it at the hairpin's speed.
        """
        limits, ahead = self._preview(idx)
        if not limits.size:
            return self.top_speed
        return float(np.min(np.sqrt(limits * limits + 2.0 * self.brake_accel * ahead)))

    def control(self, state: CarState) -> tuple[float, float]:
        """Steering angle (rad) and normalized throttle for one step."""
        params = self.params
        cum = self.arclen.cum_lengths
        total = self.arclen.total_length
        # Full O(n) nearest scan: a windowed search is unsafe on a path that
        # passes close to itself.
        idx = self.arclen.nearest_index(np.array([state.x, state.y]))

        # Aim at a point whose lookahead grows with speed. The segment index runs to
        # n-1 and the last one's far end wraps to waypoint 0; clamped to n-2, a
        # `target_arc` past the last vertex extrapolates off the path.
        n = len(cum)
        target_arc = (cum[idx] + LOOKAHEAD_BASE + LOOKAHEAD_SPEED_GAIN * state.vx) % total
        i = max(0, min(int(np.searchsorted(cum, target_arc, side="right")) - 1, n - 1))
        frac = (target_arc - cum[i]) / (self.arclen.seg_lengths[i] + 1e-12)
        waypoints = self.arclen.path
        target = waypoints[i] + frac * (waypoints[(i + 1) % n] - waypoints[i])
        dx, dy = target[0] - state.x, target[1] - state.y
        local_x = math.cos(state.theta) * dx + math.sin(state.theta) * dy
        local_y = -math.sin(state.theta) * dx + math.cos(state.theta) * dy
        ld2 = local_x * local_x + local_y * local_y
        vertical = params.mass * params.gravity + params.downforce_coeff * state.vx * state.vx
        eff_wheelbase = params.wheelbase + self.understeer * state.vx * state.vx / vertical
        steer = 0.0 if ld2 < 1e-9 else math.atan(2.0 * local_y / ld2 * eff_wheelbase)
        steer = _clamp(steer, -params.max_steer, params.max_steer)

        target_speed = self._target_speed(idx)

        # The friction ellipse, on both pedals: grip already spent cornering is
        # available neither for braking nor for driving, and the tires resolve a
        # request that ignores it by swapping ends.
        share = min(lateral_grip_usage(state, params), 1.0)
        ellipse = math.sqrt(max(1.0 - share * share, 0.0))

        # Proportional speed tracker. The error is divided by whichever acceleration
        # the car can produce now, so one gain means the same in both directions.
        accel = SPEED_GAIN * (target_speed - state.vx)
        if accel >= 0.0:
            # No floor under the throttle, unlike the brake: rear-wheel drive puts all
            # thrust on the rear axle, so at the lateral limit throttle spins the car.
            drive_accel = _drive_force(state.vx, 1.0, params.physics) / params.mass
            return steer, _clamp(accel / drive_accel, 0.0, 1.0) * ellipse
        # Yaw stability caps the brake on top of the ellipse: full pedal above
        # ~29 m/s is yaw-divergent even in a straight line, see `_yaw_stable_decel`.
        pedal = max(ellipse, ELLIPSE_PEDAL_FLOOR) * min(
            1.0, _yaw_stable_decel(state.vx, params) / params.brake_accel_max
        )
        return steer, _clamp(accel / params.brake_accel_max, -pedal, 0.0)


def random_start_state(
    track,
    rng,
    lateral_noise: float = 0.3,
    heading_noise: float = SPAWN_HEADING_RAD,
    params: CarParams | None = None,
):
    """Spawn somewhere random on the track, roughly pointing down it.

    `lateral_noise` is a fraction of corridor half-width, `heading_noise` is radians,
    and speed is drawn from zero up to what corners within braking reach allow.
    """
    params = params or CarParams()
    centerline = track.centerline
    n = len(centerline)
    idx = int(rng.integers(0, n))

    # The generator carries exact analytic tangents, normals and curvature, so
    # all three are read rather than differenced off neighbouring samples;
    # `track._sampled` is what points the normal away from the infield.
    tangent = track.tangents[idx]
    normal = track.normals[idx]

    lateral_offset = rng.uniform(-lateral_noise, lateral_noise) * track.wall_offset[idx]
    heading_offset = rng.uniform(-heading_noise, heading_noise)
    # Capped by every corner the spawn could still be braking for, not just the one it
    # lands on: one point of a speed plan's backward pass, at the deceleration
    # `PurePursuit` permits itself.
    decel = _brake_envelope(params)
    ceiling = SPAWN_SPEED_FRAC * params.top_speed
    kappa = np.abs(track.curvature)
    # Samples within braking reach, clamped to the count: `ahead` runs 1..n-1, so the
    # window never wraps the start/finish line or re-reads the spawn sample.
    reach = min(int(ceiling * ceiling / (2.0 * decel * track.ds)) + 2, n)
    ahead = np.arange(1, reach)
    reachable = params.corner_speed(kappa[(idx + ahead) % n], LATERAL_ACCEL_FRAC)
    ceiling = min(
        ceiling,
        float(params.corner_speed(kappa[idx], LATERAL_ACCEL_FRAC)),
        float(np.sqrt(reachable**2 + 2.0 * decel * ahead * track.ds).min()),
    )
    start_speed = rng.uniform(0.0, ceiling)

    start_xy = centerline[idx] + normal * lateral_offset
    start_theta = np.arctan2(tangent[1], tangent[0]) + heading_offset
    return CarState(x=start_xy[0], y=start_xy[1], theta=start_theta, vx=start_speed, vy=0.0, r=0.0)
