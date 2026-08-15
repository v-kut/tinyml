"""Procedural racetrack generator: straights joined by circular arcs.

Random points in an ellipse, convex hull, vertices pulled inward so the lap turns
both ways, the outline slid apart to insert a main straight, hooks and chicanes
jogged into its longer edges, every vertex filleted. Curvature is exact and radii
are inputs. Only the polygon needs rejecting.

Shape targets come from `TUMFTM/racetrack-database`: 18 F1 circuits, measured for
lap length, corner density, straight fraction, absolute turning per lap and how far
the lap reaches inside its own convex hull. See `sim/README.md`.

One module per stage of that sentence, each usable on its own: `config` (the
tunables and the two radii the car dictates), `outline` (the polygon, its clearance
rule, and the jogs that make hooks and chicanes), `fillet` (the outline as arcs and
straights), `sample` (`Track`, walked at even arc length), `generate` (the draw and
its named rejections). Only this file is public.
"""

from tinyml_racing.sim.track.config import (
    TrackConfig,
    default_corner_speed_radius,
    min_steerable_corner_radius,
)
from tinyml_racing.sim.track.generate import generate_track
from tinyml_racing.sim.track.sample import Track

__all__ = [
    "Track",
    "TrackConfig",
    "default_corner_speed_radius",
    "generate_track",
    "min_steerable_corner_radius",
]
