# `tinyml_racing/render/`

A pygame viewer that makes a policy's behaviour legible: walls and racing line as
strokes, the car, its trail, the LiDAR fan it is reading, the steering it is
commanding, and one table row per contender. Nothing in the deployment path reads
this package (`ml/env.py` imports the viewer lazily for `rgb_array`), and it is
deliberately the cheapest thing that can do the job, because it shares a thread with
the control loop.

## Files

| file          | owns                                                                                                                                                                                        |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `camera.py`   | the world <-> screen projection: `project_into` (the only statement of it), `Camera` (`fit`, `aim`, `zoom_by`, `project`, `px`, `bounds`), `fit_scale`, `clamp_zoom`, `ZOOM_LIMITS`         |
| `window.py`   | everything sized by the window: display surface, HUD font, pre-rendered glyphs, alpha overlay, car sprite and its per-shade scaled cache. Rebuilt whole on resize; the PNG decode is cached |
| `scene.py`    | `Scene`: the three polylines (outer wall, inner wall, racing line), the corridor width and the track seed, built once per layout                                                            |
| `world.py`    | the world-metre drawing pass: `draw_walls`, `draw_trail`, `draw_lidar`, `draw_car`, `draw_steer_arrow`                                                                                      |
| `hud.py`      | the pixel-space overlay: `RateMeter`, `Latency`, `Readout`, `draw_panel`, `PanelLine`, `draw_gauges` (STR/THR/BRK), `MiniMap`                                                               |
| `theme.py`    | the palette and the fixed HUD label strings, including the per-driver `CONTENDER` shades                                                                                                    |
| `viewer.py`   | `PygameViewer` (owns window, camera, scene, accumulators; pumps input, gates frames, runs the world then the HUD pass), plus `Trail` and `Ghost`                                            |
| `../watch.py` | one level up: the `tinyml-watch` front end. `POLICIES`, `Policy`, `resolve`, `Runner`, `hud_table`, `watch`, `main`. The package's only real consumer                                       |

## Contracts

`camera.project_into` is the only place metres become pixels, with `Camera.project` and
`hud.MiniMap` differing only in the box they project into, and `Window` owns everything
sized by the window, rebuilt whole on resize against the surface SDL already has.
`draw` reads state only: everything that accumulates lives in `observe`, so a dropped
frame cannot change what the run recorded, and latency is timed by whoever owns the
control loop rather than by the renderer. `viewer.pump()` is the whole input contract,
returning pygame key names for `watch.py` to bind.

## Keys

`[tab]` changes which driver the camera follows and the LiDAR fan belongs to, `[r]`
restarts the episode, `[t]` draws a new track. `tinyml-watch --policy all` puts every
driver a run can produce on the same spawn on the same layout, each in its own colour
leaving its own trail; a run that skipped a stage reports that driver as unavailable
instead of failing.

## Tests

`../../tests/test_render_math.py`, no display: the camera projection and its anchored
zoom including the clamp order, the minimap fit against the panel it must stay
inside, the frame gate, `Trail` distance sampling and its append-only contract,
viewer episode state (what `begin_episode` drops and what survives a same-seed
`set_track`), pygame teardown, `hud_table` column alignment and the followed-row
marker, the latency readout (sliding mean and worst, an unmeasured meter falsy rather
than 0.00 ms, `mcu` named only for a driver with a device, dashes for what could not
be measured), and `watch --max-steps` driven through a recording viewer.

Not covered: no test draws pixels. `world.py` and `hud.py` are exercised only by
`RacingEnv(render_mode="rgb_array")` in `../../tests/test_env_contract.py`, which
asserts the frame has many distinct colours and changes as the car moves.

Decisions and measurements:
[docs/findings/renderer-cost.md](../../docs/findings/renderer-cost.md).
