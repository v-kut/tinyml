# `tinyml_racing/sim/`

The simulator: procedural circuits, exact wall geometry, a single-track vehicle
model, a LiDAR, a minimum-curvature racing line, and a pure-pursuit expert. Nothing
here knows about learning or about the device, and it is the only place a vehicle
parameter, a track dimension or a sensor characteristic is written down.

## Files

| file             | what it does                                                                        |
| ---------------- | ----------------------------------------------------------------------------------- |
| `geometry.py`    | the exact ray caster and containment test, plus arc-length projection and curvature |
| `track/`         | the generator, one module per stage: outline, fillet, sampling, and the draw        |
| `car.py`         | the vehicle parameters and the substepped single-track model                        |
| `expert.py`      | pure pursuit: the analytic baseline, the cloning teacher, and random spawns         |
| `lidar.py`       | the warped beam fan, with noise and dropout                                         |
| `racing_line.py` | minimum curvature inside the corridor, by sparse least squares                      |
| `track_pool.py`  | per-seed layouts with their arc-length tables, built lazily and cached              |

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

| test                     | what it pins here                                                                        |
| ------------------------ | ---------------------------------------------------------------------------------------- |
| `test_track_geometry.py` | track sampling, the exact wall geometry against its polyline oracle, and the racing line |
| `test_car_dynamics.py`   | the vehicle model at its limits, servo slew, and pure pursuit lapping                    |
| `test_track_pool.py`     | seed determinism, caching, and the train/eval split                                      |

Decisions and measurements:
[docs/findings/simulator-geometry.md](../../docs/findings/simulator-geometry.md).
