"""The world pass: walls, racing line, trail, LiDAR fan, car and steer arrow.

Drawn in world metres through the camera, under the pixel panels `hud` draws.
The LiDAR fan is the one filled polygon, and is clamped to the viewport.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pygame

from tinyml_racing.render.camera import Camera
from tinyml_racing.render.scene import Scene
from tinyml_racing.render.theme import (
    LIDAR_FAN,
    LIDAR_POINT,
    LIDAR_RAY,
    RACING_LINE,
    STEER,
    TRAIL,
    WALL,
)
from tinyml_racing.render.window import Window
from tinyml_racing.sim.car import CarParams, CarState
from tinyml_racing.sim.lidar import LidarConfig

CAR_LENGTH_PER_WHEELBASE = 1.7
CAR_WIDTH_PER_WHEELBASE = 0.72
CAR_MIN_LENGTH_PX = 12
STEER_ARROW_CAR_LENGTHS = 0.9
WALL_W_FRAC = 0.02
LINE_W_FRAC = 0.04
TRAIL_W_FRAC = 0.08  # of the corridor width
LIDAR_DOT_PX = 2


def draw_walls(window: Window, camera: Camera, scene: Scene) -> None:
    """Both walls and, if the layout solved one, the racing line."""
    wall_px = camera.px(WALL_W_FRAC * scene.width_m)
    for wall in (scene.outer, scene.inner):
        pygame.draw.lines(window.screen, WALL, True, camera.project(wall).tolist(), wall_px)
    if scene.line is not None:
        pygame.draw.lines(
            window.screen,
            RACING_LINE,
            True,
            camera.project(scene.line).tolist(),
            camera.px(LINE_W_FRAC * scene.width_m),
        )


def draw_trail(
    window: Window, camera: Camera, trail: Sequence[np.ndarray], width_m: float, color=TRAIL
) -> None:
    """The line the car actually drove, against the racing line it was meant to."""
    if len(trail) < 2:
        return
    pts = camera.project(np.array(trail)).tolist()
    pygame.draw.lines(window.screen, color, False, pts, camera.px(TRAIL_W_FRAC * width_m))


def draw_lidar(window: Window, camera: Camera, state: CarState, scan, lidar: LidarConfig) -> None:
    """The fan the policy saw, put back into metres to draw it.

    Takes the whole `LidarConfig`: the bearings are warped, not evenly spread.
    """
    scan = np.asarray(scan, dtype=float)
    if scan.size < 2:
        return
    ranges = scan * lidar.max_range
    # A no-return ray means "nothing out there", not "free space to 150 m", so
    # bound the misses to the farthest range the sweep measured. An all-miss
    # sweep has no fan at all rather than a polygon of coincident points.
    returned = ranges[scan < 1.0 - 1e-6]
    if not returned.size:
        return
    window.overlay.fill((0, 0, 0, 0))
    # `draw.polygon` costs bbox-height x vertices and clips nothing itself, so
    # clamp to the viewport diagonal: cost bounded by the window, not the zoom.
    reach = 1.1 * math.hypot(*camera.size) / camera.ppm
    ranges = np.minimum(ranges, returned.max())
    drawn = np.minimum(ranges, reach)
    angles = state.theta + lidar.angles
    ends = np.stack([state.x + drawn * np.cos(angles), state.y + drawn * np.sin(angles)], -1)
    origin = camera.project((state.x, state.y)).tolist()
    ends_px = camera.project(ends).tolist()
    pygame.draw.polygon(window.overlay, LIDAR_FAN, [origin, *ends_px])
    pygame.draw.lines(window.overlay, LIDAR_RAY, False, ends_px, 1)
    hits = (scan < 1.0 - 1e-6) & (ranges <= reach)
    for end, hit in zip(ends_px, hits.tolist(), strict=True):
        if hit:
            pygame.draw.circle(window.overlay, LIDAR_POINT, end, LIDAR_DOT_PX)
    window.screen.blit(window.overlay, (0, 0))


def car_box_px(camera: Camera, state: CarState, params: CarParams):
    """The car's screen-space frame: centre, axes, and half extents in pixels."""
    cos_t, sin_t = math.cos(state.theta), math.sin(state.theta)
    forward = np.array([cos_t, -sin_t])
    left = np.array([-sin_t, -cos_t])
    half_len = CAR_LENGTH_PER_WHEELBASE * params.wheelbase / 2 * camera.ppm
    half_wid = CAR_WIDTH_PER_WHEELBASE * params.wheelbase / 2 * camera.ppm
    inflate = max(1.0, CAR_MIN_LENGTH_PX / (2 * half_len))
    center = camera.project((state.x, state.y))
    return center, forward, left, half_len * inflate, half_wid * inflate


def draw_car(
    window: Window, camera: Camera, state: CarState, params: CarParams, shade=None
) -> None:
    """The car, as the sprite, tinted `shade` when it is one of a field.

    The artwork is white through grey, so a multiply tints it.
    """
    center, forward, _, half_len, half_wid = car_box_px(camera, state, params)
    size = max(int(2 * half_len * 1.2), int(2 * half_wid * 1.2), CAR_MIN_LENGTH_PX)
    # `forward` is already in screen axes, and pygame rotates anticlockwise
    # from "up", so the sprite's nose follows -forward.
    sprite = window.sprite(size, shade)
    rotated = pygame.transform.rotate(sprite, math.degrees(math.atan2(-forward[0], -forward[1])))
    window.screen.blit(rotated, rotated.get_rect(center=tuple(center)))


def draw_steer_arrow(
    window: Window, camera: Camera, state: CarState, params: CarParams, action
) -> None:
    """The policy's steering command off the nose, as `action[0] * CarParams.max_steer`.

    Clamped to [-1, 1] like `RacingEnv.step` does, so the arrow cannot claim a
    steer angle the simulator would never have applied.
    """
    steer_cmd = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0))
    center, forward, left, half_len, half_wid = car_box_px(camera, state, params)
    angle = steer_cmd * params.max_steer
    heading = math.cos(angle) * forward + math.sin(angle) * left
    root = center + half_len * forward
    tip = root + STEER_ARROW_CAR_LENGTHS * 2 * half_len * heading
    thickness = max(1, round(half_wid / 2 * (0.5 + 0.5 * abs(steer_cmd))))
    pygame.draw.line(window.screen, STEER, tuple(root), tuple(tip), thickness)
    if abs(steer_cmd) > 0.1:
        arrow_size = half_wid * 0.4
        perp = np.array([-heading[1], heading[0]])
        hl = tip - arrow_size * heading + arrow_size * 0.5 * perp
        hr = tip - arrow_size * heading - arrow_size * 0.5 * perp
        pygame.draw.polygon(window.screen, STEER, [tuple(tip), tuple(hl), tuple(hr)])
