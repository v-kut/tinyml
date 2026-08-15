"""Generator tunables, and the two corner radii the car dictates.

`TrackConfig` is the single source of truth for track shape. Its defaults are shape
targets measured against a corpus of real circuits, not preferences; the package
docstring says which corpus and on what metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tinyml_racing.sim.car import CarParams


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


@dataclass(frozen=True)
class TrackConfig:
    """Generator tunables: the single source of truth for track shape."""

    width: float = 13.0  # full drivable width, metres
    # One width per layout, never drifted along the lap: `geometry.WallGeometry` is
    # the exact offset of the arc/straight chain, and only a constant offset has one.
    width_variation: float = 0.10  # fraction of `width`, either way
    # 3.0-6.0 km against the 4.3-7.0 km of the TUM circuits. Below ~3 km the longer
    # edges cannot host a hook: one needs `2*depth/tan(theta) + gap` of edge to sit on.
    length_range: tuple[float, float] = (3000.0, 6000.0)
    # Derived from the car, not chosen: a literal drifted out of step with `CarParams`
    # once already, and nothing about a corner floor is the generator's to pick.
    min_corner_radius: float = field(default_factory=min_steerable_corner_radius)
    max_corner_radius: float = 400.0  # a sweeper the car takes flat out
    # Against 9-22 corners per TUM lap. Each jog adds four of its own, so this is
    # the count before hooks and chicanes, and corner density lands at 3-4 per km.
    n_corners: tuple[int, int] = (8, 13)  # inclusive draw range
    min_straight: float = 40.0  # m of straight left between consecutive corners
    main_straight_min: float = 500.0  # every layout gets one straight at least this long
    # Infield hooks: the lap leaves the perimeter, dives inside its own ring and comes
    # back. The one feature a convex outline cannot produce, worth ~1.0 of absolute
    # turning per lap each (TUM median 4.88 against 1.94 without them), and what makes
    # a layout read like Interlagos rather than a rounded blob.
    n_hooks: tuple[int, int] = (1, 3)  # inclusive draw range
    # Depth as a fraction of the layout's span, taken as `length / pi`. TUM circuits
    # reach 0.04-0.35 of their own hull diameter inside it, median 0.18.
    hook_depth: tuple[float, float] = (0.12, 0.28)
    # Nearer a right angle costs less of the edge it sits on, which is what decides
    # whether the hook fits at all.
    hook_turn_deg: tuple[float, float] = (78.0, 90.0)
    # Chicanes: the same jog, shallow and slanted, so it reads as an S rather than as
    # an excursion. Cheaper in turning than a hook but it needs no infield room.
    n_chicanes: tuple[int, int] = (1, 3)
    chicane_depth: tuple[float, float] = (20.0, 55.0)  # metres, not a fraction
    chicane_turn_deg: tuple[float, float] = (30.0, 55.0)
    sample_spacing: float = 4.0
    n_samples: int | None = None  # overrides `sample_spacing` when set
    max_attempts: int = 200
