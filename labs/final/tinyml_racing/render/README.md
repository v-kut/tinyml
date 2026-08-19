# `tinyml_racing/render/`

A pygame viewer that makes a policy's behaviour legible: walls and racing line as
strokes, the car, its trail, the LiDAR fan it is reading, the steering it is
commanding, and one table row per contender. Nothing in the deployment path reads
this package (`ml/env.py` imports the viewer lazily for `rgb_array`), and it is
deliberately the cheapest thing that can do the job, because it shares a thread with
the control loop.

## Files

| file          | what it does                                                             |
| ------------- | ------------------------------------------------------------------------ |
| `camera.py`   | the world-to-pixel projection, the fit and the clamped zoom              |
| `window.py`   | everything sized by the window: surface, font, glyphs, car sprite        |
| `scene.py`    | the walls and racing line of one layout                                  |
| `world.py`    | the world-metre pass: walls, trail, LiDAR fan, car, steering arrow       |
| `hud.py`      | the pixel-space overlay: panel, gauges, latency meters, minimap          |
| `theme.py`    | the palette and the fixed HUD labels                                     |
| `viewer.py`   | `PygameViewer`: owns the window, gates frames, runs both passes          |
| `../watch.py` | one level up: the `tinyml-watch` front end, this package's only consumer |

## Contracts

`camera.project_into` is the only place metres become pixels, with `Camera.project` and
`hud.MiniMap` differing only in the box they project into, and `Window` owns everything
sized by the window, rebuilt whole on resize against the surface SDL already has.
`draw` reads state only: everything that accumulates lives in `observe`, so a dropped
frame cannot change what the run recorded, and latency is timed by whoever owns the
control loop rather than by the renderer. `Readout` is window state alone (draw rate,
simulation rate, odometer, zoom); everything per-driver reaches the panel as
`hud_table`'s rows, so no number is drawn twice. `viewer.pump()` is the whole input
contract, returning pygame key names for `watch.py` to bind.

## The contender table

`policy`, `arch`, `reward`, `laps`, `infer ms`, and `mcu us` only when something in the
field reports a device time. `arch` is what computes the action: an actor's layer widths
in `QuantModel.arch`'s format (`61-16-8-2`, from `Snapshot.arch` for a trained policy)
or, for the expert, `pure pursuit`. The two latency columns carry `mean/worst` over the
last 60 steps in one cell, dashed for a driver that has not been measured. Under the
table, the followed driver's provenance: training steps, numeric format, port.

## Keys

`[tab]` changes which driver the camera follows and the LiDAR fan belongs to, `[r]`
restarts the episode, `[t]` draws a new track. `tinyml-watch --policy all` puts every
driver a run can produce on the same spawn on the same layout, each in its own colour
leaving its own trail; a run that skipped a stage reports that driver as unavailable
instead of failing.

## Tests

| test                   | what it pins here                                                            |
| ---------------------- | ---------------------------------------------------------------------------- |
| `test_render_math.py`  | camera projection and zoom clamps, the frame gate, trails, and the HUD table |
| `test_env_contract.py` | the `rgb_array` render, the only test that draws anything                    |

Not covered: no test asserts on pixels, so `world.py` and `hud.py` are only smoke-tested
through that render.

Decisions and measurements:
[docs/findings/renderer-cost.md](../../docs/findings/renderer-cost.md).
