"""Everything whose size is the window's: the surface, the font, and the caches
keyed to them.

`set_mode` is called once, by `open_window`. A resize rebuilds the caches
against the surface SDL already has, see `Window.resized`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pygame

from tinyml_racing.render.theme import GAUGE_LABEL, GAUGE_LABELS

HUD_FONT_FRAC = 0.016
HUD_FONT_MIN_PX = 10
MIN_WINDOW_PX = 240  # floor on either axis, so a compositor cannot collapse the view
ASSETS_DIR = Path(__file__).parent / "assets"
DEFAULT_CAR_SPRITE = "car.png"


@dataclass
class Window:
    size: tuple[int, int]
    screen: pygame.Surface
    # One font: the HUD panel used to render its own rows twice this size.
    small_font: pygame.font.Font
    overlay: pygame.Surface
    car_sprite: pygame.Surface
    glyphs: dict[str, pygame.Surface]
    label_w: int  # the widest gauge label, which is what the bars start after
    bar_h: int
    # One entry per shade, holding the size it was built at. Keyed by shade, not
    # by size: a field of six cars needs six surfaces, while the continuously
    # changing zoom would grow a size-keyed cache without bound.
    _scaled: dict[tuple[int, int, int] | None, tuple[int, pygame.Surface]] = field(
        default_factory=dict, repr=False
    )

    def sprite(self, size: int, shade: tuple[int, int, int] | None = None) -> pygame.Surface:
        """The car sprite at `size` pixels on its longest side, in `shade`.

        `shade` is normalized to its brightest channel; scaled once per size.
        """
        cached = self._scaled.get(shade)
        if cached is not None and cached[0] == size:
            return cached[1]
        orig_w, orig_h = self.car_sprite.get_size()
        longest = max(orig_w, orig_h)
        scaled = pygame.transform.smoothscale(
            self.car_sprite,
            (max(1, round(size * orig_w / longest)), max(1, round(size * orig_h / longest))),
        )
        if shade is not None:
            gain = 255.0 / max(*shade, 1)
            tint = tuple(min(255, round(c * gain)) for c in shade)
            scaled.fill((*tint, 255), special_flags=pygame.BLEND_RGB_MULT)
        self._scaled[shade] = (size, scaled)
        return scaled


def clamp_size(size) -> tuple[int, int]:
    """A size the camera arithmetic can divide by. Never a request to the
    compositor: a window manager owns the geometry, and asking it for a size in
    answer to its own configure is how a tiling one un-maximizes the window.
    """
    return (max(MIN_WINDOW_PX, int(size[0])), max(MIN_WINDOW_PX, int(size[1])))


def _build(size: tuple[int, int], screen: pygame.Surface, sprite_path: str) -> Window:
    """A `Window` for `screen`, with everything derived from `size`."""
    hud_px = max(HUD_FONT_MIN_PX, round(min(size) * HUD_FONT_FRAC))
    small = pygame.font.SysFont("monospace", hud_px)
    glyphs = {name: small.render(name, True, GAUGE_LABEL) for name in GAUGE_LABELS}
    return Window(
        size=size,
        screen=screen,
        small_font=small,
        overlay=pygame.Surface(size, pygame.SRCALPHA),
        # Needs a display to `convert_alpha` against, so never before `set_mode`.
        car_sprite=_load_car_sprite(sprite_path),
        glyphs=glyphs,
        label_w=max(glyphs[name].get_width() for name in GAUGE_LABELS),
        bar_h=small.get_height(),
    )


def open_window(size: tuple[int, int], sprite_path: str) -> Window:
    """The display surface, the fonts and the glyph cache. Calls `set_mode`.

    `RESIZABLE` is what lets the window manager own the geometry, its own
    maximize and fullscreen included; `size` is only the initial request, and
    the compositor is free to answer with another, which SDL reports back.
    """
    with warnings.catch_warnings():
        # A tiling compositor answers the initial request with its tile size, and
        # that is the size we adopt, pygame's warning about it is not news.
        warnings.filterwarnings("ignore", "Requested window was forcibly resized", RuntimeWarning)
        screen = pygame.display.set_mode(size, pygame.RESIZABLE)
    return _build(screen.get_size(), screen, sprite_path)


def resized(sprite_path: str) -> Window:
    """Rebuild for the surface SDL now has, after the compositor resized it.

    Deliberately no `set_mode`: the surface is already the new size, and calling
    it again would re-request a geometry the window manager just chose. Under a
    tiling compositor that re-request reads as "the client wants its own size",
    which drops the maximized state and snaps the window back.
    """
    screen = pygame.display.get_surface()
    if screen is None:
        raise RuntimeError("resized() before open_window(): there is no display surface")
    return _build(screen.get_size(), screen, sprite_path)


# Decoded PNGs, keyed by path. Every SDL resize event rebuilds the window, and
# re-reading the artwork from disk each time is the one cost that would scale
# with dragging a window edge; the decode is size-independent.
_decoded: dict[str, pygame.Surface] = {}


def _load_car_sprite(path: str) -> pygame.Surface:
    """The car artwork converted for the current display, or a `FileNotFoundError`.

    The sprite ships in `render/assets`; there is no fallback car renderer.
    """
    image = _decoded.get(path)
    if image is None:
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(
                f"car sprite {resolved} is missing; it ships in {ASSETS_DIR} and there is no "
                "fallback renderer for the car"
            )
        image = pygame.image.load(str(resolved))
        _decoded[path] = image
    return image.convert_alpha()
