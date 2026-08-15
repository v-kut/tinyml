"""The draw: an outline, its jogs, radii, the fillet, the sample, and the rejections.

Only the polygon needs rejecting, and every rejection is named, so an unsatisfiable
`TrackConfig` says which of its own rules it could not meet.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from tinyml_racing.sim.track.config import TrackConfig
from tinyml_racing.sim.track.fillet import _fillet
from tinyml_racing.sim.track.outline import (
    _CORRIDOR_GAP,
    _JOG_STRAIGHT,
    _corner_polygon,
    _edge_clearance,
    _edge_lengths,
    _Outline,
    _pulled,
    _rescaled,
    _stretch,
    _turn_angles,
    _with_jogs,
)
from tinyml_racing.sim.track.sample import Track, _sampled

_WALL_MARGIN = 0.5  # m of curvature headroom beyond the wall, so the offset never cusps
_STRAIGHT_SHARE = 0.25  # of every edge, floor on what stays straight: at less than this the
# arcs of neighbouring corners run into each other and the lap reads as a blob, not a circuit
_TIGHT_CORNER = 40.0  # m: every lap gets a corner at least this tight, so it has a brake zone
_RADIUS_HEADROOM = 1.10  # the final rescale can only grow radii, so draw them under the ceiling
# Share of the lap the main straight takes, measured across the TUM circuits, whose
# longest straight runs 0.11-0.16 of a lap.
_MAIN_STRAIGHT_SHARE = (0.12, 0.17)
# What of the main straight survives the fillets at either end of it.
_MAIN_STRAIGHT_KEEP = 0.62
# Samples the shortest arc has to span, so the sampled curvature resolves it. Two is
# the floor a central difference needs; three leaves the field one clean sample.
_ARC_SAMPLES = 3.0


def generate_track(seed: int | None = None, config: TrackConfig | None = None) -> Track:
    """Generate one closed circuit. Samples sit `config.sample_spacing` apart, so the
    count tracks lap length unless `n_samples` overrides it. Index 0 is the
    start/finish line, mid longest straight, where progress is measured from.
    """
    cfg = config or TrackConfig()
    widest = (1.0 + cfg.width_variation) * 0.5 * cfg.width + _WALL_MARGIN
    # Nested walls, asserted rather than hoped for: the offset cusps as soon as w
    # reaches the local radius of curvature. A violation is a config error.
    if cfg.min_corner_radius <= widest:
        raise ValueError(
            f"min_corner_radius {cfg.min_corner_radius:.1f} m must exceed the {widest:.2f} m "
            "widest half-corridor plus margin, or the inner wall folds through its own apex"
        )
    rng = np.random.default_rng(seed)
    clearance = _CORRIDOR_GAP * cfg.width
    rejected: Counter[str] = Counter()
    for _ in range(cfg.max_attempts):
        target = rng.uniform(*cfg.length_range)
        n_corners = int(rng.integers(cfg.n_corners[0], cfg.n_corners[1] + 1))
        # Two of the corners come from the cut below, and the straights it
        # inserts are not in the perimeter the outline is drawn to. A share of the
        # lap rather than a fixed length: TUM's longest straight is 0.11-0.16 of its
        # circuit, and the floor only binds on the shortest laps drawn here.
        main = max(cfg.main_straight_min, target * rng.uniform(*_MAIN_STRAIGHT_SHARE))
        v = _corner_polygon(
            rng,
            n_corners - 2,
            target - 2.0 * main,
            cfg.min_straight + 2.0 * cfg.min_corner_radius,
        )
        if v is None:
            rejected["coarse"] += 1
            continue
        v = _pulled(rng, v, clearance)
        # The cut adds exactly twice its offset, so the offset that survives the
        # rescale as a `main`-long straight can be solved for.
        lean = rng.uniform(-0.35, 0.35)  # the straights run roughly along the long axis
        span = main * _edge_lengths(v).sum() / (target - 2.0 * main)
        v = _stretch(v, span * np.array([np.cos(lean), np.sin(lean)]))
        v = _rescaled(v, target)
        if _edge_clearance(v) < clearance:
            rejected["blocked"] += 1
            continue
        # The inserted pair is the only edges of length `main`; a uniform rescale
        # cannot move an edge into or out of that set, so this holds to the fillet.
        edge = _edge_lengths(v)
        straight = np.abs(edge - main) < 1e-6 * main
        if straight.sum() != 2:
            # Not the pulls: a pulled vertex is inside its neighbours' chord, so it
            # cannot be extremal. This fires only when an unrelated edge measures
            # `main` to a part in a million, one draw in many thousands.
            rejected["uncut"] += 1
            continue
        # Hooks and chicanes, after the main straights are known and before any
        # radius is drawn: a jog reserves the radii of the four corners it adds.
        outline = _with_jogs(
            rng,
            _Outline(
                v=v, main=straight, jog=np.zeros(len(v), bool), radius=np.full(len(v), np.nan)
            ),
            cfg,
            clearance,
        )
        # A jog adds `2*depth*tan(turn/2)` to the outline, so the lap comes back to
        # length here rather than in the fillet's rescale, which would otherwise be
        # shrinking every radius by whatever the hooks added. One uniform scale, so
        # the radii the jogs reserved stay the radii their own geometry can hold.
        grow = target / float(_edge_lengths(outline.v).sum())
        outline = replace(outline, v=outline.v * grow, radius=outline.radius * grow)
        v, edge = outline.v, _edge_lengths(outline.v)
        # Log-uniform: a uniform draw over the whole range leaves four corners in five
        # above the ~150 m this car takes flat out, a lap with nothing to brake for.
        # The floor is not `min_corner_radius`: the tight corners of a real lap are its
        # hairpins, which are what a hook's tip already is, and TUM's median corner is
        # 59 m. The brake corner below and the jog tips carry the tight end.
        lo = np.log(max(1.5 * cfg.min_corner_radius, _TIGHT_CORNER * 0.6))
        hi = np.log(cfg.max_corner_radius / _RADIUS_HEADROOM)
        radius = np.exp(rng.uniform(lo, hi, size=len(v)))
        # Heaviest braking at the end of the longest straight, as at a real
        # circuit, and that is also what guarantees the `_TIGHT_CORNER` check.
        brake = np.roll(outline.main, 1)  # straight `i` ends at vertex `i+1`
        radius[brake] = np.exp(
            rng.uniform(
                np.log(cfg.min_corner_radius),
                np.log(_TIGHT_CORNER / _RADIUS_HEADROOM),
                size=int(brake.sum()),
            )
        )
        radius = np.where(np.isnan(outline.radius), radius, outline.radius)
        # A vertex that turns by a degree is a gentle bend, not a corner, and at a
        # small radius it becomes an arc shorter than the sample spacing: one sample
        # then carries the whole of `1/R` while its neighbours read zero, which is a
        # spike in a field every consumer reads as continuous. Radius follows turn,
        # so the arc always spans a few samples; on a real corner the floor is metres
        # below the draw and never binds.
        span = _ARC_SAMPLES * cfg.sample_spacing / np.abs(_turn_angles(v))
        radius = np.minimum(np.maximum(radius, span), cfg.max_corner_radius / _RADIUS_HEADROOM)
        # A share of every edge stays straight, not just the 40 m floor: the
        # difference between a circuit and one long chain of arcs that closes. The
        # main straight keeps a share too rather than a length, so that the rescale
        # above can never ask for more straight than the edge has.
        keep_straight = np.maximum(
            np.where(outline.main, _MAIN_STRAIGHT_KEEP * edge, cfg.min_straight),
            _STRAIGHT_SHARE * edge,
        )
        # Inside a jog the pair of arcs is the feature, so its own edges keep only
        # what its geometry reserved; the share rule would prise them apart.
        keep_straight = np.where(outline.jog, _JOG_STRAIGHT[0], keep_straight)
        prim = _fillet(v, radius, keep_straight)
        if prim.radius.min() < cfg.min_corner_radius:
            rejected["tight"] += 1
            continue
        # Corner cutting made the filleted lap shorter than the polygon, so this
        # scale lands the length in range. It only grows radii, which is why they were
        # drawn under the ceiling.
        prim = prim.scaled(target / prim.length)
        if prim.radius.max() > cfg.max_corner_radius:
            rejected["wide"] += 1
            continue
        if prim.radius.min() > _TIGHT_CORNER:
            rejected["flat"] += 1
            continue
        return _sampled(prim, cfg, rng, seed)

    why = {
        "coarse": f"hulls too coarse for {cfg.n_corners} corners",
        "blocked": f"folded over or left two corridors under {clearance:.1f} m apart",
        "uncut": "measured a second edge as long as the main straight",
        "tight": f"clamped a fillet under {cfg.min_corner_radius:.0f} m",
        "wide": f"broke {cfg.max_corner_radius:.0f} m",
        "flat": f"had no corner under {_TIGHT_CORNER:.0f} m to brake for",
    }
    detail = ", ".join(f"{count} {why[key]}" for key, count in rejected.most_common())
    raise RuntimeError(
        f"generate_track: all {cfg.max_attempts} draws rejected (seed={seed}): {detail}, "
        "widen `length_range`, lower `n_corners` or shorten `main_straight_min`"
    )
