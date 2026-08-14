"""World-to-screen projection, and the zoom and pan that set it."""

from typing import Any

import numpy as np

FIT_MARGIN = 0.03
ZOOM_LIMITS = (0.5, 50.0)


def clamp_zoom(zoom: float) -> float:
    lo, hi = ZOOM_LIMITS
    return max(lo, min(zoom, hi))


def fit_scale(outer: np.ndarray, box: tuple[int, int], fill: float) -> tuple[float, np.ndarray]:
    """Pixels per metre and world centre that fit `outer` into `fill` of a `box`.

    Shared by `Camera.fit` and `hud.MiniMap.fit`; only the camera then zooms.
    """
    lo, hi = outer.min(axis=0), outer.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    return float(fill * np.min(np.array(box, dtype=float) / span)), (lo + hi) / 2


def project_into(pts: Any, center: np.ndarray, ppm: float, box: tuple[int, int]) -> np.ndarray:
    """Metres to pixels inside `box`, centred on `center` with y flipped.

    The only statement of the projection: `Camera.project` is this with the
    window as the box, and `hud.MiniMap` is this with its panel, offset to the
    panel's corner. Both scale and flip, neither rotates.
    """
    pts = np.asarray(pts, dtype=float)
    out = np.empty(pts.shape, dtype=float)
    out[..., 0] = box[0] / 2 + (pts[..., 0] - center[0]) * ppm
    out[..., 1] = box[1] / 2 - (pts[..., 1] - center[1]) * ppm
    return out


class Camera:
    """Metres to pixels, and where the centre of the window is in the world.

    `ppm` is derived, so no zoom, pan or resize has to remember to update it.
    """

    __slots__ = ("base_center", "base_ppm", "center", "pan_offset", "size", "zoom")

    def __init__(self, size: tuple[int, int], zoom: float = 1.0) -> None:
        self.size = size
        self.zoom = zoom
        self.base_ppm = 1.0
        self.base_center = np.zeros(2)
        self.center = np.zeros(2)
        self.pan_offset = np.zeros(2)

    @property
    def ppm(self) -> float:
        return self.base_ppm * self.zoom

    def fit(self, outer: np.ndarray) -> None:
        """Scale and centre that put the whole lap in the window; window-size
        dependent, so a resize redoes it.
        """
        self.base_ppm, self.base_center = fit_scale(outer, self.size, 1.0 - 2 * FIT_MARGIN)

    def aim(self, at: np.ndarray | None) -> None:
        """Look at `at`, or at wherever the pan has left the view.

        Following writes the pan offset, so leaving follow mode holds the view.
        """
        if at is None:
            self.center = self.base_center + self.pan_offset
        else:
            self.center = at
            self.pan_offset = at - self.base_center

    def zoom_by(self, factor: float, anchor: np.ndarray | None = None) -> None:
        """Zoom, keeping the world point under the pixel `anchor` under it.

        Without an anchor the centre stays put.
        """
        if anchor is None:
            self.zoom = clamp_zoom(self.zoom * factor)
            return
        delta = (anchor - np.array(self.size, dtype=float) / 2) * np.array([1.0, -1.0])
        world = self.center + delta / self.ppm
        self.zoom = clamp_zoom(self.zoom * factor)
        self.pan_offset = world - self.base_center - delta / self.ppm

    def pan_px(self, dx: float, dy: float) -> None:
        """Drag the view by a mouse delta in pixels."""
        self.pan_offset[0] -= dx / self.ppm
        self.pan_offset[1] += dy / self.ppm

    def project(self, pts: Any) -> np.ndarray:
        return project_into(pts, self.center, self.ppm, self.size)

    def px(self, meters: float) -> int:
        return max(1, round(meters * self.ppm))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """The view rectangle in world metres: what a cull compares against."""
        reach = np.array(self.size, dtype=float) / (2.0 * self.ppm)
        return self.center - reach, self.center + reach
