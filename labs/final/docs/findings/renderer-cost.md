# Why the viewer is a wireframe

`tinyml_racing/render/` shares a thread with the control loop, so every drawing decision is
a latency decision.

## Fills do not clip; strokes do

`pygame.draw.polygon` costs `bbox_height * vertices` and clips neither: a 983-vertex
outline costs 0.3 ms whole and 17 ms zoomed 50x. Filling the ground was 34 ms of a 40 ms
frame, and the viewer collapsed from 80 to 20 fps as you zoomed. `draw.lines` clips per
segment and costs 0.03 ms at any zoom, so nothing is filled and a frame is 1.1-1.4 ms flat
from 1x to 50x.

Thick strokes cannot replace fills: pygame applies width along the dominant axis, so a
200 px request draws 141 px at 45 deg. Skid marks are gone too, having been 200 polygon
calls per frame whose fill area grew with the square of the zoom, cached on a surface every
camera move invalidated.

The decorative ground cover and the cull machinery that made it affordable went on the same
argument, with the ABS/TC/SLIP lights and grip meter, which were inferred from the tire
ellipse rather than modelled. That returned a 48x48 distance field per new seed as well as
the per-frame cost.

## Three rates, kept separate

`CarParams.dt` is the control interval, one physics step, one policy evaluation, one USB
round trip, and the only rate anything else derives from. The episode budget is
`max_episode_seconds / dt`, and `render_fps` is `1/dt`, because an `rgb_array` recording
plays back in real time only at the rate its frames were produced.

The display is none of these. `tinyml-watch` paces the simulation against the wall clock
and draws when a frame is due. It used to call `viewer.tick(30)` inside the control loop,
which slept the loop to 30 Hz: a 100 Hz simulation ran at 0.3x real time and the board was
polled at 30 Hz however fast the link was. The HUD reports draw rate, achieved simulation
rate and ratio to real time as three numbers, and lateness is never repaid with catch-up
steps, since each needs its own round trip. The gate fires on control iterations, so
achievable draw rates are (control rate)/n.

## Latency is a result, not a diagnostic

`infer ms` is wall clock around the driver's own call, so a host policy reports what the
network costs in this process and the board reports the whole round trip, the number its
loop pays. `mcu us` is the device's own figure for `tinyml_infer`, and that column exists
only when something in the field reports one. Each cell is `mean/worst` over the last 60
steps.
