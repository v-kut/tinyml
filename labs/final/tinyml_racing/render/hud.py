"""The overlay pass: telemetry panel, gauge strip, minimap.

All of it is drawn in pixels against the window, so none of it costs anything
the zoom can change, `draw.rect` and `blit` clip to the surface.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, NamedTuple

import numpy as np
import pygame

from tinyml_racing.render.camera import Camera, fit_scale, project_into
from tinyml_racing.render.scene import Scene
from tinyml_racing.render.theme import (
    BRAKE,
    CAR_OUTLINE,
    GAUGE_BG,
    GAUGE_LABEL,
    GAUGE_TICK,
    HUD_DIM,
    HUD_PANEL,
    HUD_TEXT,
    MINIMAP_BG,
    MINIMAP_CAR,
    MINIMAP_TRACK,
    MINIMAP_VIEWPORT,
    STEER,
    THROTTLE,
)
from tinyml_racing.render.window import Window
from tinyml_racing.sim.car import CarParams, CarState

# Telemetry strip, bottom-left. The row height follows the HUD font rather than a
# literal, so the strip scales with the window the way every other panel does.
GAUGE_LEN = 150
GAUGE_GAP = 5
GAUGE_MARGIN = 20
GAUGE_PAD = 6

# Samples per rate estimate. 60 is about a second at any plausible rate, which
# is short enough to show a stall and long enough not to flicker.
RATE_WINDOW = 60
# Weight of the newest frame in the drawn-milliseconds average. Same job as
# `RATE_WINDOW`, one number instead of a deque.
DRAW_SMOOTH = 0.15

MINIMAP_SIZE = (160, 120)
MINIMAP_MARGIN = 20
MINIMAP_FILL = 0.8  # of the panel, so the lap never touches its border


class RateMeter:
    """Observed rate of one repeated event, averaged over its last intervals.

    One per rate: simulation, control and display each keep their own.
    """

    __slots__ = ("_stamps", "hz")

    def __init__(self, window: int = RATE_WINDOW) -> None:
        self._stamps: deque[float] = deque(maxlen=window + 1)
        self.hz = 0.0

    def hit(self) -> None:
        self._stamps.append(time.perf_counter())
        span = self._stamps[-1] - self._stamps[0]
        self.hz = (len(self._stamps) - 1) / span if span > 0 else 0.0


class Latency:
    """Mean and worst of one repeated duration, over its last samples.

    Milliseconds. `worst_ms` is kept because a control loop is late whenever a
    single step is, which a mean over a second hides.
    """

    __slots__ = ("_samples", "mean_ms", "worst_ms")

    def __init__(self, window: int = RATE_WINDOW) -> None:
        self._samples: deque[float] = deque(maxlen=window)
        self.mean_ms = 0.0
        self.worst_ms = 0.0

    def hit(self, ms: float) -> None:
        self._samples.append(ms)
        self.mean_ms = sum(self._samples) / len(self._samples)
        self.worst_ms = max(self._samples)

    def __bool__(self) -> bool:
        """False until something has been measured, so a driver with no device
        timing reads as absent rather than as 0.00 ms.
        """
        return bool(self._samples)


class Readout(NamedTuple):
    """What the panel reports that `CarState` does not carry.

    Nothing here is per-driver: a caller comparing policies owns that table and
    passes it as panel lines, so no number is drawn twice.
    """

    draw_ms: float
    sim_hz: float
    sim_dt: float
    distance_m: float
    zoom: float
    follow: bool


def _bar(surf, x, y, h, frac, color, bipolar=False, marker=None) -> None:
    """One gauge row: track, fill, and an optional second-opinion marker.

    `frac` is signed and centre-zero when `bipolar`, else 0..1 from the left.
    """
    pygame.draw.rect(surf, GAUGE_BG, (x, y, GAUGE_LEN, h))
    if bipolar:
        span = max(-1.0, min(frac, 1.0)) * GAUGE_LEN / 2
        mid = x + GAUGE_LEN / 2
        pygame.draw.rect(surf, color, (min(mid, mid + span), y, abs(span), h))
        pygame.draw.rect(surf, GAUGE_LABEL, (mid, y, 1, h))
    else:
        pygame.draw.rect(surf, color, (x, y, max(0.0, min(frac, 1.0)) * GAUGE_LEN, h))
    if marker is not None:
        at = x + (0.5 + max(-1.0, min(marker, 1.0)) / 2) * GAUGE_LEN
        pygame.draw.rect(surf, GAUGE_TICK, (at - 1, y - 2, 3, h + 4))


def draw_gauges(window: Window, state: CarState, action, params: CarParams) -> None:
    """The policy's action, made visible: steering, throttle and brake."""
    demand = steer_cmd = 0.0
    if action is not None:
        a = np.asarray(action, dtype=float).reshape(-1)
        steer_cmd = float(a[0])
        demand = float(a[1]) if a.size > 1 else 0.0
    # Bottom-up, so top-down reads steer, throttle, brake.
    rows = (
        ("BRK", max(-demand, 0.0), BRAKE, False, None),
        ("THR", max(demand, 0.0), THROTTLE, False, None),
        # Commanded steer filled, achieved steer as the tick: the gap is the
        # servo slew limit (`CarParams.max_steer_rate`) being visible.
        ("STR", steer_cmd, STEER, True, state.steer / params.max_steer),
    )
    h = window.bar_h
    bar_x = GAUGE_MARGIN + window.label_w + GAUGE_PAD
    y = window.size[1] - GAUGE_MARGIN - h
    for name, frac, color, bipolar, marker in rows:
        window.screen.blit(window.glyphs[name], (GAUGE_MARGIN, y))
        _bar(window.screen, bar_x, y, h, frac, color, bipolar, marker)
        y -= h + GAUGE_GAP


class PanelLine(NamedTuple):
    """One caller-supplied panel row, and the shade it belongs to.

    Only the swatch is coloured, so every row stays equally legible.
    """

    text: str
    shade: tuple[int, int, int] | None = None


def draw_panel(window: Window, state: CarState, extra_lines, readout: Readout, hints: bool) -> None:
    """Two lines of state, the caller's lines, then the key hints, dimmed.

    The gutter is reserved for every row once one wants it, or columns step.
    """
    speed = 3.6 * math.hypot(state.vx, state.vy)
    slip = math.degrees(math.atan2(state.vy, max(state.vx, 0.1)))
    rt = readout.sim_hz * readout.sim_dt
    rows = [
        PanelLine(
            f"{speed:5.1f} km/h  slip {slip:+5.1f}\u00b0  {readout.distance_m / 1000:5.2f} km"
        ),
        PanelLine(
            f"draw {readout.draw_ms:4.1f} ms  sim {readout.sim_hz:5.1f} Hz {rt:.2f}x  "
            f"{readout.zoom:.1f}x {'[F]' if readout.follow else '[M]'}"
        ),
        *(line if isinstance(line, PanelLine) else PanelLine(str(line)) for line in extra_lines),
    ]
    surfaces = [(window.small_font.render(r.text, True, HUD_TEXT), r.shade) for r in rows]
    if hints:
        hint = "[F]ollow  [RMB]pan  [Wheel]zoom  [Esc]quit"
        surfaces.append((window.small_font.render(hint, True, HUD_DIM), None))
    em = window.small_font.get_height()
    pad, gap = max(3, round(0.4 * em)), max(1, round(0.15 * em))
    swatch_w = max(3, round(0.35 * em)) if any(shade for _, shade in surfaces) else 0
    gutter = swatch_w + pad if swatch_w else 0
    w = max(s.get_width() for s, _ in surfaces) + gutter + 2 * pad
    h = sum(s.get_height() for s, _ in surfaces) + gap * (len(surfaces) - 1) + 2 * pad
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill(HUD_PANEL)
    window.screen.blit(panel, (pad, pad))
    y = 2 * pad
    for s, shade in surfaces:
        if shade is not None:
            # Inset a pixel top and bottom so consecutive rows read as separate
            # blocks rather than as one column of colour.
            pygame.draw.rect(window.screen, shade, (2 * pad, y + 1, swatch_w, s.get_height() - 2))
        window.screen.blit(s, (2 * pad + gutter, y))
        y += s.get_height() + gap


class MiniMap:
    """The whole lap at a fixed scale, bottom-right, with the viewport on it.

    Fitted once per layout and never zoomed, so the panel stays put.
    """

    __slots__ = ("center", "ppm")

    def __init__(self) -> None:
        self.ppm = 1.0
        self.center = np.zeros(2)

    def fit(self, outer: np.ndarray) -> None:
        """Fix the panel's scale to this layout; the same fit the camera uses."""
        self.ppm, self.center = fit_scale(outer, MINIMAP_SIZE, MINIMAP_FILL)

    def _project(self, pts: Any, size: tuple[int, int]) -> np.ndarray:
        """Panel-local pixels, moved to the panel's corner in the window."""
        origin = np.array(size) - MINIMAP_SIZE - MINIMAP_MARGIN
        return project_into(pts, self.center, self.ppm, MINIMAP_SIZE) + origin

    def draw(self, window: Window, camera: Camera, scene: Scene, state: CarState) -> None:
        size = window.size
        mw, mh = MINIMAP_SIZE
        mx = size[0] - mw - MINIMAP_MARGIN
        my = size[1] - mh - MINIMAP_MARGIN
        panel = pygame.Surface((mw, mh), pygame.SRCALPHA)
        panel.fill(MINIMAP_BG)
        pygame.draw.rect(panel, MINIMAP_TRACK, panel.get_rect(), 1)
        # The viewport rectangle, clamped rather than dropped when the view
        # reaches past the fit: the projection is a scale and a y-flip, never a
        # rotation, so a clamped corner is the corner of the intersection. It
        # goes on the panel and not on the display surface because `pygame.draw`
        # writes its colour verbatim, alpha included; only blitting the SRCALPHA
        # surface that carries it makes `MINIMAP_VIEWPORT` the intended wash
        # rather than a solid white outline.
        lo, hi = camera.bounds()
        corners = np.array([[lo[0], hi[1]], hi, [hi[0], lo[1]], lo])
        view_px = np.clip(self._project(corners, size), (mx, my), (mx + mw - 1, my + mh - 1))
        pygame.draw.lines(panel, MINIMAP_VIEWPORT, True, (view_px - (mx, my)).tolist(), 1)
        window.screen.blit(panel, (mx, my))
        for poly in (scene.outer, scene.inner):
            pygame.draw.lines(
                window.screen, MINIMAP_TRACK, True, self._project(poly, size).tolist(), 1
            )
        car = self._project((state.x, state.y), size)
        cos_t, sin_t = math.cos(state.theta), math.sin(state.theta)
        marker = 4
        nose = (car[0] + marker * cos_t, car[1] - marker * sin_t)
        tail_l = (
            car[0] - marker * 0.6 * math.cos(state.theta + 2.5),
            car[1] + marker * 0.6 * math.sin(state.theta + 2.5),
        )
        tail_r = (
            car[0] - marker * 0.6 * math.cos(state.theta - 2.5),
            car[1] + marker * 0.6 * math.sin(state.theta - 2.5),
        )
        pygame.draw.polygon(window.screen, MINIMAP_CAR, [nose, tail_l, tail_r])
        pygame.draw.polygon(window.screen, CAR_OUTLINE, [nose, tail_l, tail_r], 1)
