"""`PygameViewer`: the window, the input, and the state the drawing passes read.

Nothing here draws. It owns the window, the camera, the scene and the
accumulators that advance on simulation time, and hands them to `world`/`hud`.
"""

from __future__ import annotations

import atexit
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
import pygame

from tinyml_racing.render.camera import Camera, clamp_zoom
from tinyml_racing.render.hud import (
    DRAW_SMOOTH,
    MiniMap,
    RateMeter,
    Readout,
    draw_gauges,
    draw_panel,
)
from tinyml_racing.render.scene import Scene
from tinyml_racing.render.theme import BACKDROP, TRAIL
from tinyml_racing.render.window import (
    ASSETS_DIR,
    DEFAULT_CAR_SPRITE,
    Window,
    clamp_size,
    open_window,
)
from tinyml_racing.render.window import resized as window_resized
from tinyml_racing.render.world import (
    draw_car,
    draw_lidar,
    draw_steer_arrow,
    draw_trail,
    draw_walls,
)
from tinyml_racing.sim.car import CarParams, CarState
from tinyml_racing.sim.lidar import LidarConfig
from tinyml_racing.sim.track import Track

TRAIL_STEP_M = 1.5  # distance between recorded points
TRAIL_POINTS = 600  # ring length, so the trail is the last 900 m


class Trail(deque[np.ndarray]):
    """The line a car actually drove, sampled by distance rather than by step.

    Same shape at any control rate; bounded to `TRAIL_POINTS * TRAIL_STEP_M`.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(maxlen=TRAIL_POINTS)

    @override
    def append(self, pos: np.ndarray) -> None:
        """Record `pos`, unless it is within `TRAIL_STEP_M` of the last point."""
        if not self or np.linalg.norm(pos - self[-1]) > TRAIL_STEP_M:
            super().append(pos)

    @override
    def extend(self, positions: Iterable[np.ndarray]) -> None:
        for pos in positions:
            self.append(pos)

    def _append_only(self, *_args) -> None:
        raise TypeError("Trail is append-only in driven order; use append or extend")

    # The inherited mutators would skip the distance test, or add points at the
    # wrong end of a deque that is read as a path.
    appendleft = extendleft = insert = rotate = _append_only


QUIT_KEYS = (pygame.K_ESCAPE, pygame.K_q)


@dataclass(frozen=True)
class Ghost:
    """One non-followed car in an overlay: its state, its shade, and its trail.

    `shade` is the same whether the car is followed or a ghost.
    """

    state: CarState
    shade: tuple[int, int, int]
    trail: Sequence[np.ndarray] = ()


class PygameViewer:
    def __init__(
        self,
        screen_size=(900, 900),
        caption="tinyml-racing",
        params: CarParams | None = None,
        interactive: bool = True,
        follow_car: bool = True,
        follow_zoom: float = 1.5,
        car_sprite_path: Path = ASSETS_DIR / DEFAULT_CAR_SPRITE,
    ):
        self.screen_size = (int(screen_size[0]), int(screen_size[1]))
        self.caption = caption
        self.params = params or CarParams()
        self.interactive = interactive
        self.follow_car = follow_car
        self.follow_zoom = follow_zoom
        self.car_sprite_path = car_sprite_path
        self.running = False
        self._window: Window | None = None
        self._scene: Scene | None = None
        self._camera = Camera(self.screen_size, follow_zoom if follow_car else 1.0)
        self._minimap = MiniMap()
        self._panning = False
        self._draw_ms = 0.0
        self._sim_rate = RateMeter()
        self._sim_dt = 0.0
        self._next_frame = 0.0
        self._distance_traveled = 0.0
        self._trail = Trail()
        self._last_position: np.ndarray | None = None

    @property
    def window(self) -> Window:
        if self._window is None:
            raise RuntimeError("PygameViewer.open() must be called before drawing")
        return self._window

    @property
    def scene(self) -> Scene:
        if self._scene is None:
            raise RuntimeError("PygameViewer.set_track() must be called before drawing")
        return self._scene

    def open(self) -> PygameViewer:
        """Build the window, refitting any layout already adopted to its size."""
        if self._window is not None:
            return self
        pygame.init()
        pygame.display.set_caption(self.caption)
        self._window = open_window(self.screen_size, str(self.car_sprite_path))
        # The compositor may grant a size other than the one requested, and the
        # camera has to be the one SDL actually gave us or the world is drawn
        # off-centre and `camera.bounds()` reports a viewport nobody sees.
        self.screen_size = self._camera.size = self._window.size
        # A close/open cycle keeps the scene but not the window, and the new
        # window may be a different size than the fit was computed for.
        if self._scene is not None:
            self._fit(self._scene.outer)
        self.running = True
        return self

    def _resize(self, size) -> None:
        """Adopt the size the compositor chose: rebuild the caches and refit.

        The size is read back off SDL's surface rather than taken from the event,
        and no `set_mode` is issued, answering a configure with a size request
        is what made a tiling compositor's maximize snap back.
        """
        if self._window is None:
            return
        # `get_surface()` is None if the display went away between the event and
        # here (a compositor closing the window mid-frame); nothing to resize.
        surface = pygame.display.get_surface()
        if surface is None:
            return
        actual = clamp_size(surface.get_size())
        if actual == self.screen_size and tuple(size) == actual:
            return
        self.screen_size = actual
        self._camera.size = actual
        self._window = window_resized(str(self.car_sprite_path))
        if self._scene is not None:
            self._fit(self._scene.outer)

    def close(self) -> None:
        """Drop this viewer's window, and nothing else. Idempotent.

        Not `pygame.quit()`: `_quit_pygame` does that once, at exit.
        """
        self.running = False
        self._window = None

    def frame_due(self, fps: float) -> bool:
        """Whether a frame is due, at most `fps` times a second.

        Returns rather than sleeps: the display never paces the simulation.
        """
        if fps <= 0:
            return True
        now = time.perf_counter()
        if now < self._next_frame:
            return False
        # Measured from `now` rather than added to the old deadline: a caller
        # that stalls (a slow control round trip, a paused simulation) must
        # not come back owing a burst of catch-up frames nobody can see.
        self._next_frame = now + 1.0 / fps
        return True

    def observe(self, state: CarState, dt: float) -> None:
        """Advance the viewer's simulation-time state by one step of `dt`.

        Call once per `env.step`, drawn or not: a skipped frame costs history.
        """
        self._sim_rate.hit()
        self._sim_dt = dt
        pos = np.array([state.x, state.y])
        if self._last_position is not None:
            self._distance_traveled += float(np.linalg.norm(pos - self._last_position))
        self._last_position = pos
        self._trail.append(pos)

    def pump(self) -> set[str]:
        """Drain SDL's events, and return the pygame key names pressed.

        Quit, follow, zoom, pan and resize act here; callers bind 'r', 't', 'tab'.
        """
        pressed: set[str] = set()
        try:
            events = pygame.event.get()
        except KeyboardInterrupt:
            self.running = False
            return pressed
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                pressed.add(pygame.key.name(event.key))
                if event.key in QUIT_KEYS:
                    self.running = False
                elif event.key == pygame.K_f:
                    self.follow_car = not self.follow_car
                    if self.follow_car:
                        self._camera.zoom = self.follow_zoom
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    self._apply_zoom(1.15)
                elif event.key == pygame.K_MINUS:
                    self._apply_zoom(1 / 1.15)
            # Both events, because which one SDL sends depends on the video
            # driver; `_resize` is a no-op when they agree.
            elif event.type == pygame.VIDEORESIZE:
                self._resize(event.size)
            elif event.type == pygame.WINDOWSIZECHANGED:
                self._resize((event.x, event.y))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                self._panning = True
                self.follow_car = False
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                self._panning = False
            elif event.type == pygame.MOUSEMOTION and self._panning:
                self._camera.pan_px(*event.rel)
            elif event.type == pygame.MOUSEWHEEL:
                self._apply_zoom(1.0 + event.y * 0.15, at_cursor=not self.follow_car)
        return pressed

    def _apply_zoom(self, factor: float, at_cursor: bool = False) -> None:
        """Zoom the camera, and remember it as the follow zoom while following.

        Follow mode keeps its own zoom across leaving and re-entering it.
        """
        if self.follow_car:
            self.follow_zoom = clamp_zoom(self.follow_zoom * factor)
            self._camera.zoom = self.follow_zoom
            return
        anchor = np.array(pygame.mouse.get_pos(), dtype=float) if at_cursor else None
        self._camera.zoom_by(factor, anchor)

    def _fit(self, outer: np.ndarray) -> None:
        """Put the whole lap in the window, and in the minimap. Both depend on
        the window size, so a resize redoes them.
        """
        self._camera.fit(outer)
        self._minimap.fit(outer)

    def set_track(self, track: Track) -> None:
        """Adopt a layout, unless it is the one already drawn.

        Called every control step: the early return keeps the episode running.
        """
        current = self._scene
        if current is not None and track.seed is not None and track.seed == current.seed:
            return
        scene = Scene.from_track(track)
        self._fit(scene.outer)
        self._camera.zoom = self.follow_zoom if self.follow_car else 1.0
        self._camera.pan_offset = np.zeros(2)
        self.begin_episode()
        self._scene = scene

    def begin_episode(self) -> None:
        """Drop the per-episode accumulators: odometer, last position, trail.

        `RacingEnv.reset`, `[r]` and `[tab]` are boundaries `set_track` misses.
        """
        self._distance_traveled = 0.0
        self._last_position = None
        self._trail.clear()

    def draw(
        self,
        state: CarState,
        scan=None,
        hud_lines=(),
        action=None,
        lidar: LidarConfig | None = None,
        ghosts: Sequence[Ghost] = (),
        trail: Sequence[np.ndarray] | None = None,
        shade: tuple[int, int, int] | None = None,
    ) -> None:
        """One frame. Reads state only, accumulators live in `observe`.

        `state` is the followed car; `ghosts` are the rest of the field.
        """
        if (scan is None) != (lidar is None):
            raise ValueError("draw() needs `scan` and `lidar` together: the bearings are warped")
        started = time.perf_counter()
        scene, window, camera = self.scene, self.window, self._camera
        camera.aim(np.array([state.x, state.y]) if self.follow_car else None)

        window.screen.fill(BACKDROP)
        draw_walls(window, camera, scene)
        if scan is not None and lidar is not None:
            draw_lidar(window, camera, state, scan, lidar)
        # Ghosts first, so the followed car is never hidden under one.
        for ghost in ghosts:
            draw_trail(window, camera, ghost.trail, scene.width_m, ghost.shade)
        for ghost in ghosts:
            draw_car(window, camera, ghost.state, self.params, ghost.shade)
        draw_trail(
            window,
            camera,
            self._trail if trail is None else trail,
            scene.width_m,
            TRAIL if shade is None else shade,
        )
        draw_car(window, camera, state, self.params, shade)
        if action is not None:
            draw_steer_arrow(window, camera, state, self.params, action)
        draw_panel(window, state, hud_lines, self._readout(), self.interactive)
        draw_gauges(window, state, action, self.params)
        self._minimap.draw(window, camera, scene, state)
        pygame.display.flip()
        # What the frame cost, not how often frames come: `frame_due` caps the
        # rate. Smoothed, because a raw per-frame number is unreadable at 60 Hz.
        ms = 1000.0 * (time.perf_counter() - started)
        self._draw_ms += DRAW_SMOOTH * (ms - self._draw_ms)

    def _readout(self) -> Readout:
        return Readout(
            draw_ms=self._draw_ms,
            sim_hz=self._sim_rate.hz,
            sim_dt=self._sim_dt,
            distance_m=self._distance_traveled,
            zoom=self._camera.zoom,
            follow=self.follow_car,
        )

    def frame_rgb(self) -> np.ndarray:
        return np.ascontiguousarray(pygame.surfarray.array3d(self.window.screen).swapaxes(0, 1))


@atexit.register
def _quit_pygame() -> None:
    """The one process-global pygame teardown, at the only moment no viewer
    can still want the display. Harmless when nothing was initialised.
    """
    pygame.quit()
