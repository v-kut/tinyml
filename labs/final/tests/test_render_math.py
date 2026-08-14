"""The arithmetic behind the window, tested without one.

Closed-form checks on projection, the zoom anchor, the minimap fit, the trail
and the HUD table. Only the two-viewer teardown test needs pygame (SDL dummy).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast, override

import numpy as np
import pytest

from tinyml_racing import watch as watch_mod
from tinyml_racing.ml.config import RacingEnvConfig
from tinyml_racing.ml.env import RacingEnv
from tinyml_racing.ml.regression.dataset import pure_pursuit_teacher
from tinyml_racing.render import viewer as viewer_mod
from tinyml_racing.render.camera import FIT_MARGIN, ZOOM_LIMITS, Camera
from tinyml_racing.render.hud import (
    MINIMAP_MARGIN,
    MINIMAP_SIZE,
    Latency,
    MiniMap,
    PanelLine,
    Readout,
)
from tinyml_racing.render.viewer import TRAIL_POINTS, TRAIL_STEP_M, PygameViewer, Trail
from tinyml_racing.sim.car import CarState
from tinyml_racing.sim.track_pool import build_track

SCREEN = (800, 600)
# A deliberately non-square world box, so a test that swapped the axes or fitted
# against the wrong one of the two ratios cannot pass by symmetry.
WORLD = np.array([[-30.0, 10.0], [70.0, 10.0], [70.0, 60.0], [-30.0, 60.0]])
WORLD_SPAN = np.array([100.0, 50.0])


@pytest.fixture
def clock(monkeypatch):
    """Hand-driven clock for `render.viewer` only; its `sleep` raises, because
    `frame_due` must gate rather than block.
    """

    class Clock:
        def __init__(self) -> None:
            self.now = 1_000.0

        def perf_counter(self) -> float:
            return self.now

        def advance(self, seconds: float) -> None:
            self.now += seconds

        def sleep(self, seconds: float) -> None:  # pragma: no cover - must not run
            raise AssertionError(f"frame_due slept {seconds}s instead of returning False")

    fake = Clock()
    monkeypatch.setattr(viewer_mod, "time", fake)
    return fake


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #


def test_project_applies_the_y_flip_once_and_the_scale_once():
    """Screen y grows downward and world y grows upward, and `ppm` is one factor.

    An unfitted camera has `base_ppm == 1`, so `ppm` is exactly the zoom.
    """
    camera = Camera(SCREEN, zoom=4.0)
    camera.aim(np.array([12.0, -7.0]))
    assert camera.ppm == 4.0

    center_px = camera.project(camera.center)
    np.testing.assert_allclose(center_px, [SCREEN[0] / 2, SCREEN[1] / 2])

    up = camera.project(camera.center + np.array([0.0, 3.0]))
    right = camera.project(camera.center + np.array([5.0, 0.0]))
    # 3 m up is 12 px up, and up is a *smaller* screen y.
    np.testing.assert_allclose(up, [SCREEN[0] / 2, SCREEN[1] / 2 - 12.0])
    assert up[1] < center_px[1]
    np.testing.assert_allclose(right, [SCREEN[0] / 2 + 20.0, SCREEN[1] / 2])


def test_project_maps_a_polyline_without_changing_its_shape():
    """Projection is a scale and a flip applied elementwise over any shape:
    `draw_walls` hands whole `(n, 2)` polylines in.
    """
    camera = Camera(SCREEN, zoom=2.5)
    camera.aim(np.array([1.0, 2.0]))
    pts = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0]])
    out = camera.project(pts)

    assert out.shape == pts.shape
    # Pairwise distances scale by exactly `ppm`: no shear, no double scaling.
    for a, b in itertools.combinations(range(3), 2):
        world_d = float(np.linalg.norm(pts[a] - pts[b]))
        screen_d = float(np.linalg.norm(out[a] - out[b]))
        assert screen_d == pytest.approx(world_d * camera.ppm)


def test_fit_puts_the_whole_layout_in_the_viewport_with_its_margin():
    """`fit` leaves `FIT_MARGIN` of the window clear on the tight axis, with one
    scale for both axes.
    """
    camera = Camera(SCREEN)
    camera.fit(WORLD)
    camera.aim(None)

    corners = camera.project(WORLD)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    extent = hi - lo
    usable = np.array(SCREEN, dtype=float) * (1.0 - 2.0 * FIT_MARGIN)

    # 800/100 = 8 px/m against 600/50 = 12 px/m, so x limits and fills 0.94*800,
    # and the other axis follows at that same one scale: 50 m * 7.52 px/m.
    assert extent[0] == pytest.approx(usable[0])
    assert extent[1] == pytest.approx(WORLD_SPAN[1] * usable[0] / WORLD_SPAN[0])
    assert extent[1] < usable[1]
    # Nothing hangs outside the window, and the margin is really there.
    np.testing.assert_array_less(np.array(SCREEN) * FIT_MARGIN - 1e-9, lo)
    np.testing.assert_array_less(hi, np.array(SCREEN) * (1.0 - FIT_MARGIN) + 1e-9)
    # The fit is centred on the layout, not on its corner.
    np.testing.assert_allclose(camera.center, WORLD.mean(axis=0))


def test_bounds_are_exactly_the_rectangle_project_maps_onto_the_window():
    """`bounds` is the wall cull's rectangle, so it must agree with `project`:
    the low corner maps to the bottom-left pixel, the high corner to the top-right.
    """
    camera = Camera(SCREEN, zoom=3.0)
    camera.fit(WORLD)
    camera.aim(np.array([5.0, 40.0]))

    lo, hi = camera.bounds()
    np.testing.assert_allclose(camera.project(lo), [0.0, SCREEN[1]], atol=1e-9)
    np.testing.assert_allclose(camera.project(hi), [SCREEN[0], 0.0], atol=1e-9)
    np.testing.assert_allclose(hi - lo, np.array(SCREEN) / camera.ppm)
    assert lo[0] < camera.center[0] < hi[0]
    assert lo[1] < camera.center[1] < hi[1]


@pytest.mark.parametrize("factor", [0.2, 0.5, 0.9, 1.0, 1.1, 2.0, 8.0])
def test_anchored_zoom_leaves_the_world_point_under_the_cursor(factor):
    """A wheel zoom magnifies what the cursor points at, not the centre: the
    invariant holds after `aim`, which is when the projection is read.
    """
    camera = Camera(SCREEN, zoom=4.0)
    camera.fit(WORLD)
    camera.pan_offset = np.array([13.0, -8.0])
    camera.aim(None)

    # Some world point well off centre; the cursor is wherever it happens to be.
    world = np.array([46.0, 21.0])
    anchor = camera.project(world)
    assert not np.allclose(anchor, np.array(SCREEN) / 2)

    camera.zoom_by(factor, anchor)
    camera.aim(None)

    assert camera.zoom == pytest.approx(4.0 * factor)
    np.testing.assert_allclose(camera.project(world), anchor, atol=1e-9)


def test_anchored_zoom_holds_even_when_the_zoom_clamps():
    """Hitting `ZOOM_LIMITS` must not tear the view off the cursor: the anchor
    correction uses the zoom that was applied, not the one requested.
    """
    camera = Camera(SCREEN, zoom=4.0)
    camera.fit(WORLD)
    camera.aim(None)

    world = np.array([-11.0, 55.0])
    anchor = camera.project(world)
    camera.zoom_by(1_000.0, anchor)
    camera.aim(None)

    assert camera.zoom == ZOOM_LIMITS[1]
    np.testing.assert_allclose(camera.project(world), anchor, atol=1e-9)


def test_unanchored_zoom_keeps_the_centre_where_it_was():
    """Keyboard `+`/`-` zoom about the middle of the window: with no anchor,
    `pan_offset` is left alone.
    """
    camera = Camera(SCREEN, zoom=2.0)
    camera.fit(WORLD)
    camera.pan_offset = np.array([4.0, 9.0])
    camera.aim(None)
    centered_on = camera.center.copy()

    camera.zoom_by(1.15)
    camera.aim(None)

    np.testing.assert_allclose(camera.center, centered_on)
    np.testing.assert_allclose(
        camera.project(centered_on), np.array(SCREEN, dtype=float) / 2, atol=1e-9
    )


def test_pan_px_moves_the_view_by_the_mouse_delta_it_was_given():
    """Dragging moves the world one pixel per pixel under the cursor; the y row
    needs the opposite sign to x, for the same reason `project` flips.
    """
    camera = Camera(SCREEN, zoom=5.0)
    camera.fit(WORLD)
    camera.aim(None)
    world = np.array([20.0, 30.0])
    before = camera.project(world)

    camera.pan_px(17.0, -9.0)
    camera.aim(None)

    np.testing.assert_allclose(camera.project(world) - before, [17.0, -9.0], atol=1e-9)


# --------------------------------------------------------------------------- #
# MiniMap
# --------------------------------------------------------------------------- #


def test_minimap_fit_keeps_the_lap_inside_the_panel_at_its_fill_fraction():
    """The lap fits the fixed minimap panel at `MINIMAP_FILL`: the wall
    polylines are drawn unclipped, so an overflow scribbles on the telemetry.
    """
    minimap = MiniMap()
    minimap.fit(WORLD)
    projected = minimap._project(WORLD, SCREEN)

    mw, mh = MINIMAP_SIZE
    panel_lo = np.array([SCREEN[0] - mw - MINIMAP_MARGIN, SCREEN[1] - mh - MINIMAP_MARGIN])
    panel_hi = panel_lo + np.array([mw, mh])

    lo, hi = projected.min(axis=0), projected.max(axis=0)
    np.testing.assert_array_less(panel_lo - 1e-9, lo)
    np.testing.assert_array_less(hi, panel_hi + 1e-9)

    # 160/100 = 1.6 px/m against 120/50 = 2.4, so x limits and fills 0.8 of 160.
    extent = hi - lo
    assert extent[0] == pytest.approx(0.8 * mw)
    assert extent[1] == pytest.approx(0.8 * mw * (50.0 / 100.0))
    # Centred in the panel, so the free space is shared rather than banked.
    np.testing.assert_allclose((lo + hi) / 2, (panel_lo + panel_hi) / 2, atol=1e-9)


# --------------------------------------------------------------------------- #
# frame_due
# --------------------------------------------------------------------------- #


def test_frame_due_gates_on_the_interval_and_returns_rather_than_sleeping(clock):
    """The render gate skips frames and never waits: called every control step,
    a gate that slept would pace the physics off the display.
    """
    viewer = PygameViewer(interactive=False)

    assert viewer.frame_due(60.0) is True  # nothing drawn yet: always due
    clock.advance(0.5 / 60.0)
    assert viewer.frame_due(60.0) is False
    clock.advance(0.4 / 60.0)
    assert viewer.frame_due(60.0) is False
    clock.advance(0.2 / 60.0)  # now 1.1/60 s past the last frame
    assert viewer.frame_due(60.0) is True
    assert viewer.frame_due(60.0) is False  # and the deadline moved with it


def test_frame_due_is_unconditional_at_a_non_positive_rate(clock):
    """`fps <= 0` means "draw every call" (`--render-fps 0`), and must not reach
    the interval arithmetic, where `1/fps` divides by zero.
    """
    viewer = PygameViewer(interactive=False)
    for fps in (0.0, -1.0, -60.0):
        assert all(viewer.frame_due(fps) is True for _ in range(5))
    # And the gate is not left holding a deadline from those calls.
    assert viewer.frame_due(30.0) is True


def test_frame_due_does_not_repay_a_stall_as_a_burst_of_frames(clock):
    """After a stall exactly one frame is due, not the hundred that were missed:
    the deadline is measured from now, not advanced by `+= 1/fps`.
    """
    viewer = PygameViewer(interactive=False)
    assert viewer.frame_due(60.0) is True

    clock.advance(10.0)  # a slow serial round trip, or a paused simulation
    due = [viewer.frame_due(60.0) for _ in range(200)]
    assert due.count(True) == 1
    assert due[0] is True

    clock.advance(0.9 / 60.0)
    assert viewer.frame_due(60.0) is False
    clock.advance(0.2 / 60.0)
    assert viewer.frame_due(60.0) is True


# --------------------------------------------------------------------------- #
# Trail
# --------------------------------------------------------------------------- #


def test_trail_samples_by_distance_and_drops_points_inside_the_threshold():
    """The trail is sampled by metres driven, not steps taken, on a strict
    threshold: a tie at exactly `TRAIL_STEP_M` is rejected.
    """
    trail = Trail()
    for i in range(11):  # 0.0 m .. 5.0 m in 0.5 m steps
        trail.append(np.array([0.5 * i, 0.0]))

    assert TRAIL_STEP_M == 1.5
    np.testing.assert_allclose(np.array(trail), [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])


def test_trail_costs_nothing_while_the_car_stands_still():
    """A stationary car must not grow the trail, however long it sits there:
    otherwise a stall flushes the ring and loses the line that was driven.
    """
    trail = Trail()
    for _ in range(500):
        trail.append(np.array([3.0, -4.0]))
    assert len(trail) == 1


def test_trail_is_a_bounded_sequence_the_drawing_code_can_consume():
    """`draw_trail` consumes a bounded `Sequence[np.ndarray]`: the `maxlen` ring
    must drop the oldest points, never the newest.
    """
    trail = Trail()
    assert isinstance(trail, Sequence)
    assert trail.maxlen == TRAIL_POINTS

    n = TRAIL_POINTS + 100
    for i in range(n):
        trail.append(np.array([10.0 * i, 0.0]))

    assert len(trail) == TRAIL_POINTS
    as_array = np.array(trail)
    assert as_array.shape == (TRAIL_POINTS, 2)
    # The oldest 100 fell off the back, not the newest.
    assert as_array[0, 0] == pytest.approx(10.0 * (n - TRAIL_POINTS))
    assert as_array[-1, 0] == pytest.approx(10.0 * (n - 1))

    trail.clear()
    assert len(trail) == 0
    assert np.array(trail).size == 0


# --------------------------------------------------------------------------- #
# Episode boundaries
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def track():
    """One generated layout, shared: generating it is ~30 ms and nothing mutates it."""
    return build_track(7).track


def _at(x: float, y: float) -> CarState:
    return CarState(x=x, y=y, theta=0.0)


def test_set_track_on_the_layout_it_already_drew_keeps_the_episode_running(track):
    """`set_track` is called every control step, so the same seed must be a
    no-op: it must not restart the episode under the caller.
    """
    viewer = PygameViewer(interactive=False)
    viewer.set_track(track)
    scene = viewer.scene

    viewer.observe(_at(0.0, 0.0), 0.05)
    viewer.observe(_at(10.0, 0.0), 0.05)
    assert viewer._distance_traveled == pytest.approx(10.0)

    viewer.set_track(track)

    assert viewer.scene is scene  # not rebuilt
    assert viewer._distance_traveled == pytest.approx(10.0)
    assert viewer._last_position is not None
    assert len(viewer._trail) == 2


def test_begin_episode_drops_the_odometer_the_trail_and_the_last_position(track):
    """An episode boundary on a pinned layout must still reset the accumulators,
    because `set_track` early-returns there.
    """
    viewer = PygameViewer(interactive=False)
    viewer.set_track(track)
    viewer.observe(_at(0.0, 0.0), 0.05)
    viewer.observe(_at(10.0, 0.0), 0.05)

    viewer.begin_episode()

    assert viewer._distance_traveled == 0.0
    assert viewer._last_position is None
    assert len(viewer._trail) == 0
    assert viewer.scene is not None  # the layout survives; only the episode does not
    assert viewer._readout().distance_m == 0.0


def test_begin_episode_stops_the_next_observe_integrating_a_teleport(track):
    """The `[tab]` regression: `begin_episode` drops `_last_position`, so the
    first `observe` after switching car adds nothing to the odometer.
    """
    viewer = PygameViewer(interactive=False)
    viewer.set_track(track)
    viewer.observe(_at(0.0, 0.0), 0.05)
    viewer.observe(_at(10.0, 0.0), 0.05)

    viewer.begin_episode()
    viewer.observe(_at(410.0, -260.0), 0.05)  # a different car, far away

    assert viewer._distance_traveled == 0.0
    assert len(viewer._trail) == 1
    np.testing.assert_allclose(np.array(viewer._trail), [[410.0, -260.0]])

    viewer.observe(_at(410.0, -258.0), 0.05)
    assert viewer._distance_traveled == pytest.approx(2.0)


def test_set_track_on_a_new_layout_begins_an_episode(track):
    """A layout change is always an episode boundary: carrying a trail anchored
    in the old layout's coordinates across `[t]` would draw a line off track.
    """
    viewer = PygameViewer(interactive=False)
    viewer.set_track(track)
    viewer.observe(_at(0.0, 0.0), 0.05)
    viewer.observe(_at(10.0, 0.0), 0.05)

    other = build_track(8).track
    assert other.seed != track.seed
    viewer.set_track(other)

    assert viewer.scene.seed == other.seed
    assert viewer._distance_traveled == 0.0
    assert viewer._last_position is None
    assert len(viewer._trail) == 0


# --------------------------------------------------------------------------- #
# close()
# --------------------------------------------------------------------------- #


@pytest.fixture
def sdl_dummy(monkeypatch):
    """Ask SDL for the headless video driver, the way `RacingEnv.render` does."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")


def test_close_leaves_the_display_up_for_every_other_viewer(sdl_dummy):
    """Closing one viewer must not uninitialise pygame for the whole process:
    `watch` owns one while `RacingEnv(render_mode="rgb_array")` builds another.
    """
    import pygame

    first = PygameViewer(screen_size=(320, 240), interactive=False).open()
    second = PygameViewer(screen_size=(320, 240), interactive=False).open()
    try:
        first.close()

        assert pygame.display.get_init() is True
        assert first.running is False
        with pytest.raises(RuntimeError):
            _ = first.window

        frame = second.frame_rgb()
        assert frame.shape == (240, 320, 3)
    finally:
        second.close()


def test_close_is_idempotent_and_safe_on_a_viewer_that_never_opened(sdl_dummy):
    """`close` runs from an `ExitStack` callback on paths that may never have
    opened, and is called twice when both the stack and the caller tidy up.
    """
    import pygame

    never_opened = PygameViewer(interactive=False)
    never_opened.close()
    never_opened.close()
    assert never_opened.running is False

    opened = PygameViewer(screen_size=(320, 240), interactive=False).open()
    opened.close()
    opened.close()
    assert pygame.display.get_init() is True


def test_a_compositor_resize_rebuilds_the_window_without_asking_for_a_size(sdl_dummy, monkeypatch):
    """A resize must never call `set_mode` again.

    Answering a compositor's configure with a size request reads as "the client
    wants its own geometry": under niri that dropped the maximized state, so
    `maximize-column` snapped straight back to the tiled width. Measured with
    the old code: set_mode(925x1064) -> set_mode(1858x1064) -> forced back to
    925x1064. The size is also read off the surface rather than the event, so
    `Window.size` cannot disagree with what is being drawn on.
    """
    import pygame

    viewer = PygameViewer(screen_size=(320, 240), interactive=False).open()
    try:
        calls: list[tuple[Any, ...]] = []
        monkeypatch.setattr(pygame.display, "set_mode", lambda *a, **k: calls.append(a))
        # What the compositor did: the surface is already the new size.
        grown = pygame.Surface((640, 480))
        monkeypatch.setattr(pygame.display, "get_surface", lambda: grown)

        viewer._resize((640, 480))

        assert calls == []
        assert viewer.window.size == (640, 480)
        assert viewer.screen_size == (640, 480)
        assert viewer._camera.size == (640, 480)
        assert viewer.window.screen is grown
    finally:
        viewer.close()


# --------------------------------------------------------------------------- #
# watch.hud_table
# --------------------------------------------------------------------------- #


def _runner(name: str, arch: str, detail: str, reward: float, laps: float, **flags):
    """A `Runner` with only the fields `hud_table` reads: the table is pure
    formatting, so it needs no environment.
    """
    return watch_mod.Runner(
        policy=watch_mod.Policy(name, lambda env, obs: np.zeros(2), arch, detail),
        # A stand-in, not an env: `hud_table` reads nothing off it, and `Runner`
        # requires one.
        env=cast("RacingEnv", SimpleNamespace(state=CarState(x=0.0, y=0.0, theta=0.0))),
        shade=(0, 0, 0),
        reward=reward,
        laps=laps,
        **flags,
    )


TABLE_RUNNERS = [
    _runner("expert", "pure pursuit", "privileged state", 1234.0, 2.5),
    _runner("pretrain", "61-64-64-2", "clone of expert, 0 steps", -5.0, 0.0, done=True),
    _runner("int8", "61-16-8-2", "int8 emulated on host", 0.0, 12.25, done=True, crashed=True),
]


def _measured(*ms: float) -> Latency:
    meter = Latency()
    for sample in ms:
        meter.hit(sample)
    return meter


def test_latency_reports_the_mean_and_the_worst_of_its_window():
    """The mean says what a step costs; the worst is why a loop missed its
    deadline, which a mean over sixty steps hides.
    """
    meter = Latency(window=3)
    for sample in (1.0, 2.0, 9.0):
        meter.hit(sample)
    assert meter.mean_ms == pytest.approx(4.0)
    assert meter.worst_ms == pytest.approx(9.0)

    # The window slides, so a spike leaves both numbers once it ages out.
    meter.hit(3.0)
    assert meter.mean_ms == pytest.approx(14.0 / 3)
    assert meter.worst_ms == pytest.approx(9.0)
    meter.hit(3.0)
    meter.hit(3.0)
    assert meter.worst_ms == pytest.approx(3.0)


def test_an_unmeasured_latency_is_falsy_rather_than_zero():
    """A driver with no device timing must read as absent: 0.00 ms would claim
    the board answered instantly.
    """
    assert not Latency()
    assert _measured(0.0)


def test_the_panel_reports_nothing_the_contender_table_already_carries():
    """The per-driver numbers live in the table, once each. The viewer's own
    readout is window state, so it cannot grow a second copy of `infer`/`mcu`.
    """
    assert set(Readout._fields) == {
        "draw_ms",
        "sim_hz",
        "sim_dt",
        "distance_m",
        "zoom",
        "follow",
    }
    # And the table is not asked for a speed the panel already prints.
    assert [title for title, _, _ in watch_mod._COLUMNS] == [
        "policy",
        "arch",
        "reward",
        "laps",
        "infer ms",
        "mcu us",
    ]


def test_the_mcu_column_exists_only_when_a_driver_reports_a_device_time():
    """Without a board every cell of it is a dash, which is width spent saying
    nothing; the column comes back the moment one is in the field.
    """
    header = watch_mod.hud_table(TABLE_RUNNERS, 0)[0].text
    assert "infer ms" in header and "mcu us" not in header

    with_board = [*TABLE_RUNNERS, _runner("board", "61-16-8-2", "int8 on tty", 0.0, 0.0)]
    with_board[-1].device.hit(0.5)
    assert "mcu us" in watch_mod.hud_table(with_board, 0)[0].text


def test_the_table_names_what_computes_each_action():
    """`inference` says nothing; the actor's layer widths say which net is
    driving, in the same format the deployed header records.
    """
    rows = [row.text for row in watch_mod.hud_table(TABLE_RUNNERS, 0)]
    header, *body = rows[: 1 + len(TABLE_RUNNERS)]

    arch_at = header.index("arch")
    for row, runner in zip(body, TABLE_RUNNERS, strict=True):
        arch = runner.policy.arch
        assert row[arch_at : arch_at + len(arch)] == arch
        # Left-aligned in its own column: the next column starts after a gap.
        assert row[arch_at - 1] == " "


def test_the_table_shows_a_latency_per_contender_and_dashes_what_was_not_measured():
    """One column each, mean and worst in the cell: a row says what its own
    driver costs, the emulator in this process against the board over USB.
    """
    runners = [
        _runner("int8", "61-16-8-2", "host emulator", 0.0, 0.0, infer=_measured(0.21)),
        _runner(
            "board",
            "61-16-8-2",
            "int8 on /dev/ttyACM0",
            0.0,
            0.0,
            infer=_measured(7.1, 7.3),
            device=_measured(0.523, 0.771),
        ),
        _runner("expert", "pure pursuit", "privileged state", 0.0, 0.0),
    ]
    rows = [row.text for row in watch_mod.hud_table(runners, 0)]
    header, emulator, board, expert = rows[0], rows[1], rows[2], rows[3]

    def cell(row: str, label: str) -> str:
        end = header.index(label) + len(label)
        return row[:end].rsplit("  ", 1)[-1].strip()

    assert cell(emulator, "infer ms") == "0.21/0.21"
    # The mean is over the window, the worst is the single late step in it.
    assert cell(board, "infer ms") == "7.20/7.30"
    assert cell(board, "mcu us") == "647/771"
    # Only the board reports a device time; the rest run here.
    assert cell(emulator, "mcu us") == "--"
    # A driver that has not stepped yet has nothing to report either.
    assert cell(expert, "infer ms") == "--"


@pytest.mark.parametrize("focus", [0, 1, 2])
def test_hud_table_columns_line_up_under_names_of_different_lengths(focus):
    """Every value must sit under its own header, whatever the policy is called:
    asserted against the header's column edges, not the format string re-typed.
    """
    rows = watch_mod.hud_table(TABLE_RUNNERS, focus)
    header, *body = [row.text for row in rows[: 1 + len(TABLE_RUNNERS)]]

    assert len(rows) == len(TABLE_RUNNERS) + 3
    name_at = header.index("policy")
    status_at = header.index("status")
    right_edge = {label: header.index(label) + len(label) for label in ("reward", "laps")}

    # Reward to whole units, laps to three decimals.
    expected = [
        {"reward": "1234", "laps": "2.500", "status": "on"},
        {"reward": "-5", "laps": "0.000", "status": "out"},
        {"reward": "0", "laps": "12.250", "status": "OFF"},
    ]
    for row, runner, want in zip(body, TABLE_RUNNERS, expected, strict=True):
        assert row[name_at : name_at + len(runner.policy.name)] == runner.policy.name
        assert row[name_at - 1] == " "
        for label, text in ((k, v) for k, v in want.items() if k != "status"):
            end = right_edge[label]
            assert row[end - len(text) : end] == text, f"{label} not right-aligned in {row!r}"
            assert row[end - len(text) - 1] == " ", f"{label} overflows its column in {row!r}"
        assert row[status_at:] == want["status"]
        assert runner.status == want["status"]


@pytest.mark.parametrize("focus", [0, 1, 2])
def test_hud_table_marks_exactly_the_followed_runner(focus):
    """The `>` says which car the camera, the gauges and the lidar rays belong
    to, and it occupies its own two columns.
    """
    rows = watch_mod.hud_table(TABLE_RUNNERS, focus)
    body = [row.text for row in rows[1 : 1 + len(TABLE_RUNNERS)]]

    assert [row[0] for row in body] == [">" if i == focus else " " for i in range(len(body))]
    assert [row[:2] for row in body].count("> ") == 1
    assert rows[0].text[:2] == "  "

    # The swatch, not the text, carries the contender's colour.
    assert [row.shade for row in rows[1 : 1 + len(TABLE_RUNNERS)]] == [
        r.shade for r in TABLE_RUNNERS
    ]

    followed = TABLE_RUNNERS[focus].policy
    assert rows[-2].text == f"[tab] {followed.name}: {followed.detail}"
    assert rows[-1].text == "[r]estart  [t]rack"


def test_hud_table_of_a_single_runner_still_marks_it():
    """`--policy expert` on its own is the common case, and focus is still 0."""
    rows = watch_mod.hud_table(TABLE_RUNNERS[:1], 0)
    assert len(rows) == 4
    assert rows[1].text[0] == ">"
    assert rows[-2].text.startswith("[tab] expert:")


# --------------------------------------------------------------------------- #
# watch(--max-steps)
# --------------------------------------------------------------------------- #

# Long enough that pure pursuit is still lapping, and well past the ~55 steps a
# car at full lock and full throttle survives from this spawn.
MAX_STEPS = 100
WATCH_SEED = 17


class _RecordingViewer(PygameViewer):
    """`PygameViewer` with the window and the pixels taken out; everything
    `watch` drives it with (`frame_due`, `observe`, `set_track`) is real.
    """

    latest: _RecordingViewer | None = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hud_history: list[list[PanelLine]] = []
        self.observed = 0
        type(self).latest = self

    @override
    def open(self) -> _RecordingViewer:
        self.running = True
        return self

    @override
    def pump(self) -> set[str]:
        return set()

    @override
    def observe(self, state, dt) -> None:
        self.observed += 1
        super().observe(state, dt)

    # Signature mirrors `PygameViewer.draw` so the override stays substitutable.
    @override
    def draw(
        self,
        state,
        scan=None,
        hud_lines=(),
        action=None,
        lidar=None,
        ghosts=(),
        trail=None,
        shade=None,
    ) -> None:
        self.hud_history.append(list(hud_lines))


class _SecondEpisodeError(Exception):
    """Raised from the second episode's first reset to leave `watch`'s outer loop."""


def test_max_steps_stops_the_loop_even_after_the_first_runner_is_out(monkeypatch):
    """`--max-steps` counts the loop's own steps, not `runners[0].steps`:
    `Runner.step` returns early once that runner is done, freezing its counter.
    """
    monkeypatch.setattr(watch_mod, "PygameViewer", _RecordingViewer)

    def full_lock(env, obs):
        return np.array([1.0, 1.0], dtype=np.float32)

    policies = [
        watch_mod.Policy("crasher", full_lock, "constant", "full lock, full throttle"),
        watch_mod.Policy("expert", pure_pursuit_teacher(), "pure pursuit", "privileged state"),
    ]

    # `watch`'s outer loop restarts the episode forever; leave it from the first
    # reset of the second episode, once the episode under test has finished.
    resets = itertools.count(1)
    original_reset = watch_mod.Runner.reset

    def counted_reset(self, seed):
        if next(resets) > len(policies):
            raise _SecondEpisodeError
        original_reset(self, seed)

    monkeypatch.setattr(watch_mod.Runner, "reset", counted_reset)

    cfg = RacingEnvConfig(n_tracks=1, max_episode_steps=400, stall_seconds=0.0)
    with pytest.raises(_SecondEpisodeError):
        watch_mod.watch(
            policies,
            replace(cfg),
            track_seed=WATCH_SEED,
            render_fps=0.0,  # every step draws, so the HUD history is per-step
            sim_speed=0.0,  # no wall-clock pacing: the test must not sleep
            max_steps=MAX_STEPS,
        )

    viewer = _RecordingViewer.latest
    assert viewer is not None
    # The contract: exactly `MAX_STEPS` control steps, then out.
    assert viewer.observed == MAX_STEPS
    assert len(viewer.hud_history) == MAX_STEPS

    status_at = viewer.hud_history[0][0].text.index("status")
    crasher = [rows[1].text[status_at:] for rows in viewer.hud_history]
    expert = [rows[2].text[status_at:] for rows in viewer.hud_history]

    # The premise: the first-listed runner really did freeze well before the
    # limit, so a limit read off its own counter could never have been reached.
    out_at = next(i for i, s in enumerate(crasher) if s != "on")
    assert out_at < MAX_STEPS - 10
    assert crasher[-1] == "OFF"
    # ...while the second was still going, so nothing else ended the episode.
    assert set(expert) == {"on"}
