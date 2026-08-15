# `tinyml_racing/sim/`

The simulator: procedural circuits, exact wall geometry, a single-track vehicle
model, a LiDAR, a minimum-curvature racing line, and a pure-pursuit expert. Nothing
here knows about learning or about the device, and it is the only place a vehicle
parameter, a track dimension or a sensor characteristic is written down.

## Files

| file             | owns                                                                                                                                                                                                                                                                                                                 |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `geometry.py`    | 2D primitives: `WallGeometry` (the exact arc/straight ray caster and containment test, numba), `ArcLengthLUT` (projection onto a path, signed cross-track and arc length), `polyline_curvature`, and the sampled-polyline oracles `ray_polyline_distances` / `point_in_polygon` / `on_track`                         |
| `track/`         | the generator, one module per stage: `config.py` (`TrackConfig`, and `min_steerable_corner_radius` / `default_corner_speed_radius` read off the car), `outline.py` (hull, pulls, the main-straight cut, and the hooks and chicanes `_jogged` inserts), `fillet.py` (`_Primitives`, the arc/straight rounding), `sample.py` (`Track` and the even arc-length walk), `generate.py` (`generate_track`: the draw and its named rejections) |
| `car.py`         | `CarParams` (the vehicle), `CarState`, the substepped `step` kernel and `_tire_forces`, plus `proprioception` and `lateral_grip_usage`                                                                                                                                                                               |
| `expert.py`      | `PurePursuit` (the analytic baseline and the cloning teacher), its brake-envelope helpers, and `random_start_state`                                                                                                                                                                                                  |
| `lidar.py`       | `LidarConfig`, the warped fan `ray_angles`, and `cast_lidar` (normalized ranges, noise, dropout)                                                                                                                                                                                                                     |
| `racing_line.py` | `compute_racing_line` / `add_racing_line`: minimum curvature inside the corridor, by bound-constrained sparse least squares                                                                                                                                                                                          |
| `track_pool.py`  | `PooledTrack`, `build_track`, `TrackPool`: per-seed layouts with their racing-line LUT prebuilt, cached lazily                                                                                                                                                                                                       |

## Contracts

The lap runs counter-clockwise, normals point away from the infield and curvature is
positive turning left; `generate_track` checks that orientation rather than assuming it,
because every consumer rides on it. `Track.walls` is the exact arc-and-straight corridor
the simulator casts against, while `outer_wall` and `inner_wall` are sampled polylines
for drawing and tests, and width is constant per layout because only a constant offset
of an arc is an arc. `CarParams` is frozen, and its `dt` is the control interval every
other rate derives from; `min_corner_radius` derives from the car through a factory, so
the generator cannot draw a corner `PurePursuit` cannot lap.

## Tests

| file                                 | what it pins here                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `../../tests/test_track_geometry.py` | even arc-length sampling, wall nesting through the oracle, tangent/normal/curvature agreement at tolerances tied to `ds/R` and `1/(2R)`, the derived corner floor without building a track, corridor containment, the grip envelope, start/finish on a straight, racing-line clearance, the shape contract (infield depth, absolute turning, corner density, against the jog-free population), determinism, unsatisfiable configs raising, the quadratic polyline convergence, `contains` against `on_track`, per-layout width draw, and closed-form `ArcLengthLUT` / `polyline_curvature` cases |
| `../../tests/test_car_dynamics.py`   | top speed as a thrust/resistance balance, coast-down, stationary hold, brake-to-zero without judder, friction-ellipse priority, RWD power oversteer, trail-brake rotation, the low-speed kinematic manifold, a 20k-step adversarial boundedness run, servo slew, pure pursuit lapping four layouts for 80 s, a preview window shorter than the braking horizon, and the two spawn contracts                                                                                                                                                                                                      |
| `../../tests/test_track_pool.py`     | seed determinism and distinctness, cache keying and identity, `from_seed_range` purity and its disjointness from the eval range, over-request rejection, and the prebuilt LUT's vertex identity and closing segment                                                                                                                                                                                                                                                                                                                                                                              |

Decisions and measurements:
[docs/findings/simulator-geometry.md](../../docs/findings/simulator-geometry.md).
