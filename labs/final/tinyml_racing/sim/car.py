"""The vehicle: a single-track model with load transfer, a per-axle friction
ellipse and aero downforce, integrated in substeps.

`CarParams` is frozen, which lets `physics` and `top_speed` cache on it. The
pure-pursuit baseline that drives it is in `sim/expert.py`.
"""

import math
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np
from numba import njit


class Physics(NamedTuple):
    """`CarParams` as the flat scalars the compiled step loop reads.

    Built once per `CarParams` and cached on it; the derived quantities are
    packed here rather than recomputed eight times a step.
    """

    substeps: int
    h: float  # substep duration, dt / substeps
    max_steer: float
    max_steer_delta: float  # most one substep may slew the rack, max_steer_rate * h
    mass: float
    inv_mass: float
    inv_inertia: float
    lf: float
    lr: float
    wheelbase: float
    cg_height: float
    gravity: float
    mu: float
    c_stiff_front: float
    c_stiff_rear: float
    power: float
    max_drive_force: float
    brake_accel_max: float
    rolling_force: float  # rolling_resistance * mass * gravity
    drag_coeff: float
    downforce_coeff: float
    blend_speed: float
    stop_speed: float


@dataclass
class CarState:
    x: float
    y: float
    theta: float
    vx: float = 0.0  # forward (body-frame longitudinal) velocity, m/s
    vy: float = 0.0  # lateral (body-frame) velocity, nonzero means the car is sliding
    r: float = 0.0  # yaw rate, rad/s
    steer: float = 0.0  # actual (slew-rate-limited) front wheel angle, rad


@dataclass(frozen=True)
class CarParams:
    """Physical parameters of a GT3-class race car.

    Class-typical values rather than an identified parameter set; sim/README.md
    records where each one comes from and what it buys.
    """

    # --- Mass and geometry ---
    lf: float = 1.30  # CG to front axle, m
    lr: float = 1.35  # CG to rear axle, m
    mass: float = 1300.0  # kg, including driver and fuel
    inertia_z: float = 1800.0  # yaw moment of inertia, kg*m^2
    cg_height: float = 0.30  # m; this over `wheelbase` is the load-transfer rate

    # --- Tires ---
    mu: float = 1.55  # tire-surface friction coefficient (dry slick)
    # Per unit normal load, 1/rad: absolute front stiffness is `c_stiff_front * fzf`,
    # so load transfer and downforce scale it. Rear stiffer is understeer at the limit.
    c_stiff_front: float = 8.0  # 1/rad
    c_stiff_rear: float = 9.0  # 1/rad

    # --- Drivetrain (rear-wheel drive) ---
    power: float = 400_000.0  # W at the wheels (~540 hp)
    max_drive_force: float = 11_000.0  # N; thrust is min(this, power/v)
    brake_accel_max: float = 20.0  # m/s^2 of brake *demand*; the ellipse still decides

    # --- Resistance and aero (applied at the CG, outside the tire budget) ---
    rolling_resistance: float = 0.015  # fraction of static weight; aero is `drag_coeff`'s
    drag_coeff: float = 1.03  # N*s^2/m^2, i.e. 0.5*rho*Cd*A
    downforce_coeff: float = 3.0  # N*s^2/m^2, i.e. `fz_aero = downforce_coeff * v^2`

    # --- Steering ---
    max_steer: float = 0.30  # rad (17 deg)
    max_steer_rate: float = 4.0  # rad/s, rack slew limit; steering cannot teleport

    # --- Integration ---
    # Control interval, s: one `RacingEnv.step`, one policy evaluation, one USB round
    # trip. The rate of the whole system; every step count and display rate derives
    # from it.
    dt: float = 0.02
    substeps: int = 8  # accuracy, not stability: see sim/README.md
    blend_speed: float = 3.0  # m/s below which the kinematic model takes over
    stop_speed: float = 0.2  # m/s; braking deadband, see the clamp in `_step_kernel`
    gravity: float = 9.81

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr

    @property
    def grip_accel(self) -> float:
        """Mechanical lateral acceleration limit, mu*g, no downforce."""
        return self.mu * self.gravity

    def lateral_accel(self, v: float) -> float:
        """Aero-assisted lateral acceleration limit at speed `v`, m/s^2.

        `mu * (m*g + downforce_coeff*v^2) / m`: 1.55 g at rest, ~3.4 g at top speed.
        Anything scaling a corner to this car uses this, not `mu*g`.
        """
        return self.mu * (self.gravity + self.downforce_coeff * v * v / self.mass)

    def corner_speed(self, kappa, frac: float = 1.0):
        """Fastest speed curvature `kappa` supports, m/s, capped at `top_speed`.

        Closed form rather than a fixed point, since `lateral_accel` is affine in
        `v^2`. `kappa` may be an array; `frac` is a controller's margin, physics
        being `frac=1`.
        """
        # A corner flatter than `frac*mu*downforce_coeff/mass` (1/280 m here) gains
        # grip as fast as it spends it, so the clamp turns its non-positive
        # denominator into a speed the cap resolves.
        denom = np.asarray(kappa, dtype=float) - frac * self.mu * self.downforce_coeff / self.mass
        speeds = np.minimum(
            np.sqrt(frac * self.mu * self.gravity / np.maximum(denom, 1e-12)), self.top_speed
        )
        return float(speeds) if speeds.ndim == 0 else speeds

    @cached_property
    def top_speed(self) -> float:
        """Speed at which full-throttle thrust equals resistance, m/s.

        Newton on `drag*v^3 + rolling*m*g*v - power == 0`, which has exactly one
        positive root. Solved, not declared: nothing clips velocity to it.
        """
        roll = self.rolling_resistance * self.mass * self.gravity
        v = (self.power / self.drag_coeff) ** (1.0 / 3.0)  # drag-only root, an overestimate
        for _ in range(20):
            f = self.drag_coeff * v**3 + roll * v - self.power
            v -= f / (3.0 * self.drag_coeff * v * v + roll)
        return float(v)

    @cached_property
    def physics(self) -> Physics:
        """These parameters in the flat form `_step_kernel` reads."""
        h = self.dt / self.substeps
        return Physics(
            substeps=self.substeps,
            h=h,
            max_steer=self.max_steer,
            max_steer_delta=self.max_steer_rate * h,
            mass=self.mass,
            inv_mass=1.0 / self.mass,
            inv_inertia=1.0 / self.inertia_z,
            lf=self.lf,
            lr=self.lr,
            wheelbase=self.wheelbase,
            cg_height=self.cg_height,
            gravity=self.gravity,
            mu=self.mu,
            c_stiff_front=self.c_stiff_front,
            c_stiff_rear=self.c_stiff_rear,
            power=self.power,
            max_drive_force=self.max_drive_force,
            brake_accel_max=self.brake_accel_max,
            rolling_force=self.rolling_resistance * self.mass * self.gravity,
            drag_coeff=self.drag_coeff,
            downforce_coeff=self.downforce_coeff,
            blend_speed=self.blend_speed,
            stop_speed=self.stop_speed,
        )

    @property
    def min_turn_radius(self) -> float:
        """Kinematic turning radius at full steering lock, m."""
        return self.wheelbase / math.tan(self.max_steer)

    @property
    def yaw_rate_ref(self) -> float:
        """Fastest yaw rate the car can sustain, rad/s, and full scale for `r`.

        `sqrt(grip_accel / min_turn_radius)`, ~1.3 rad/s, so a normalized reading past
        1 means spinning rather than cornering.
        """
        return math.sqrt(self.grip_accel / self.min_turn_radius)

    @property
    def lateral_speed_ref(self) -> float:
        """Lateral speed at which the rear tire has nothing left, m/s.

        The linear tire saturates at `alpha = mu/c_stiff` whatever the load, so full
        scale for `vy` is that angle at `top_speed`, ~12.5 m/s.
        """
        return self.top_speed * math.tan(self.mu / self.c_stiff_rear)


@njit(cache=True)
def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else min(value, hi)


def lateral_grip_usage(state: CarState, params: CarParams) -> float:
    """Fraction of the lateral grip budget the car is spending.

    `|vx * r|` against `lateral_accel(vx)`. Not what `step` resolves, which the tires
    do, but the number the expert, HUD and metrics share. Above 1 the car slides.
    """
    return abs(state.vx * state.r) / params.lateral_accel(state.vx)


def proprioception(state: CarState, params: CarParams) -> np.ndarray:
    """The four scalars a single LiDAR frame cannot carry, normalized.

    Each against the largest value this car can produce in that channel, so all four
    land in roughly [-1, 1] and none rescales when the car does. Fresh array.
    """
    return np.array(
        [
            state.vx / params.top_speed,
            state.vy / params.lateral_speed_ref,
            state.r / params.yaw_rate_ref,
            state.steer / params.max_steer,
        ],
        dtype=np.float32,
    )


@njit(cache=True)
def _drive_force(vx: float, throttle: float, p: Physics) -> float:
    """Longitudinal tire force the drivetrain is asking for, in N.

    Positive throttle is torque- then power-limited, with `max(vx, 1.0)` standing in
    for the clutch. Negative is the brakes, and brakes oppose motion, so a car sliding
    backwards out of a spin is slowed by the pedal.
    """
    if throttle >= 0.0:
        return throttle * min(p.max_drive_force, p.power / max(vx, 1.0))
    if vx == 0.0:
        return 0.0
    return math.copysign(1.0, vx) * throttle * p.brake_accel_max * p.mass


@njit(cache=True)
def _axle_loads(ax: float, vx: float, p: Physics) -> tuple[float, float]:
    """Front and rear normal loads including load transfer and downforce, N.

    Downforce is split by the static weight distribution, so the aero balance matches
    the mass balance, which is what a race engineer aims for.
    """
    weight = p.mass * p.gravity
    front_frac = p.lr / p.wheelbase
    total = weight + p.downforce_coeff * vx * vx
    transfer = p.mass * ax * p.cg_height / p.wheelbase
    floor = 0.05 * weight  # an axle that reaches literally zero load has no dynamics left
    fzf = max(total * front_frac - transfer, floor)
    fzr = max(total * (1.0 - front_frac) + transfer, floor)
    return fzf, fzr


@njit(cache=True)
def _tire_forces(
    alpha: float, fx_demand: float, fz: float, c_stiff: float, mu: float
) -> tuple[float, float]:
    """One axle's (longitudinal, lateral) force under a friction ellipse, N.

    Longitudinal demand has priority and lateral force gets the rest,
    `fy_max = sqrt((mu*fz)^2 - fx^2)`, so the car keeps braking and refuses to turn.
    That is the behaviour a policy has to learn around.
    """
    grip = mu * fz
    fx = _clamp(fx_demand, -grip, grip)
    budget = math.sqrt(max(grip * grip - fx * fx, 0.0))
    fy = -c_stiff * fz * alpha  # linear tire; stiffness scales with load
    return fx, _clamp(fy, -budget, budget)


@njit(cache=True)
def _step_kernel(x, y, theta, vx, vy, r, steer, steer_cmd, throttle, p):
    """The substep loop behind `step`. Compiled and scalar: this is half the
    simulator's cost, and each substep depends on the last.
    """
    for _ in range(p.substeps):
        delta = _clamp(steer_cmd - steer, -p.max_steer_delta, p.max_steer_delta)
        steer_rate = delta / p.h
        steer += delta

        fx_drive = _drive_force(vx, throttle, p)
        # Resistance opposes the whole velocity vector and is charged to the chassis,
        # not the tires' grip budget. `resist_per_speed` is force over that speed.
        speed = math.hypot(vx, vy)
        resist_per_speed = (
            (p.rolling_force + p.drag_coeff * speed * speed) / speed if speed > 0.0 else 0.0
        )
        fx_resist = -resist_per_speed * vx
        fy_resist = -resist_per_speed * vy

        fzf, fzr = _axle_loads((fx_drive + fx_resist) * p.inv_mass, vx, p)
        if fx_drive >= 0.0:
            # Rear-wheel drive: corner-exit traction limiting then falls out of
            # the rear axle's friction ellipse, with no separate traction model.
            fxf_demand = 0.0
            fxr_demand = fx_drive
        else:
            # Ideal brake bias: split by current normal load, which grip is also
            # proportional to, so both axles reach the limit together.
            fxf_demand = fx_drive * fzf / (fzf + fzr)
            fxr_demand = fx_drive - fxf_demand

        cos_d = math.cos(steer)
        sin_d = math.sin(steer)
        # |vx|, not vx: on the signed speed every spun state fell to the
        # kinematic branch below, which assumes the car is going forwards.
        blend = _clamp(abs(vx) / p.blend_speed, 0.0, 1.0)

        vx_dot_dyn = 0.0
        vy_dot_dyn = 0.0
        r_dot_dyn = 0.0
        if blend > 0.0:
            # Slip angles are evaluated at no less than `blend_speed`, keeping the
            # sign of `vx`: with vx < 0 both atan2 calls land near +/-pi and the
            # ellipse leaves full grip opposing the slide, a spun car's recovery.
            vx_slip = math.copysign(max(abs(vx), p.blend_speed), vx)
            alpha_f = math.atan2(vy + p.lf * r, vx_slip) - steer
            alpha_r = math.atan2(vy - p.lr * r, vx_slip)
            fxf, fyf = _tire_forces(alpha_f, fxf_demand, fzf, p.c_stiff_front, p.mu)
            fxr, fyr = _tire_forces(alpha_r, fxr_demand, fzr, p.c_stiff_rear, p.mu)

            fx_long = fxf * cos_d - fyf * sin_d + fxr
            fy_lat = fyf * cos_d + fxf * sin_d + fyr
            vx_dot_dyn = fx_long * p.inv_mass + vy * r
            vy_dot_dyn = fy_lat * p.inv_mass - vx * r
            r_dot_dyn = (p.lf * (fyf * cos_d + fxf * sin_d) - p.lr * fyr) * p.inv_inertia

        # Kinematic (zero-slip) branch, for |vx| below `blend_speed` only: yaw
        # follows the steering geometry, r = vx*tan(steer)/L and vy = lr*r,
        # differentiated, these are accelerations, not rates.
        grip_total = p.mu * (fzf + fzr)
        ax_kin = _clamp(fx_drive, -grip_total, grip_total) * p.inv_mass
        tan_d = math.tan(steer)
        r_dot_kin = (ax_kin * tan_d + vx * steer_rate / (cos_d * cos_d)) / p.wheelbase
        vx_dot_kin = ax_kin
        vy_dot_kin = p.lr * r_dot_kin

        vx_dot = blend * vx_dot_dyn + (1.0 - blend) * vx_dot_kin + fx_resist * p.inv_mass
        vy_dot = blend * vy_dot_dyn + (1.0 - blend) * vy_dot_kin + fy_resist * p.inv_mass
        r_dot = blend * r_dot_dyn + (1.0 - blend) * r_dot_kin

        vx_next = vx + vx_dot * p.h
        # No reverse gear: zero is a floor rather than a crossing, and `stop_speed`
        # widens it into a deadband so the last fraction of a m/s cannot judder. A
        # spun car is left alone.
        if vx >= 0.0 and (vx_next < 0.0 or (throttle < 0.0 and vx_next < p.stop_speed)):
            vx_next = 0.0
        vy += vy_dot * p.h
        r += r_dot * p.h
        vx = vx_next

        # Re-project onto the kinematic manifold with the blend's complement: below
        # `blend_speed` tire forces are too small to correct drift, above it this is a
        # no-op.
        if blend < 1.0:
            r_kin = vx * math.tan(steer) / p.wheelbase
            r = blend * r + (1.0 - blend) * r_kin
            vy = blend * vy + (1.0 - blend) * p.lr * r_kin

        # Midpoint heading: integrating from the start-of-substep heading drifts
        # every corner outward by a term linear in r*h.
        theta_mid = theta + 0.5 * r * p.h
        x += (vx * math.cos(theta_mid) - vy * math.sin(theta_mid)) * p.h
        y += (vx * math.sin(theta_mid) + vy * math.cos(theta_mid)) * p.h
        theta += r * p.h

    return x, y, theta, vx, vy, r, steer


def step(state: CarState, steer_cmd: float, throttle: float, params: CarParams) -> CarState:
    """Advance the car by one control interval, returning a new state.

    `steer_cmd` is a commanded front-wheel angle in radians, slew-rate limited toward.
    `throttle` is a normalized demand in [-1, 1], positive driving the rear axle and
    negative braking both, deliberately not an acceleration.
    """
    p = params.physics
    x, y, theta, vx, vy, r, steer = _step_kernel(
        state.x,
        state.y,
        state.theta,
        state.vx,
        state.vy,
        state.r,
        state.steer,
        _clamp(float(steer_cmd), -p.max_steer, p.max_steer),
        _clamp(float(throttle), -1.0, 1.0),
        p,
    )
    return CarState(x=x, y=y, theta=theta, vx=vx, vy=vy, r=r, steer=steer)
