"""The palette, and the fixed strings drawn in it.

One module because the passes borrow from each other, the minimap's track is
the racing line's blue, and a split palette drifts.
"""

BACKDROP = (30, 40, 33)
WALL = (96, 100, 108)
RACING_LINE = (86, 170, 255)
TRAIL = (255, 214, 92)
LIDAR_RAY = (128, 224, 255, 90)
LIDAR_POINT = (198, 244, 255)
LIDAR_FAN = (110, 210, 255, 40)
CAR_OUTLINE = (250, 240, 235)
STEER = (255, 206, 62)

# One shade per contender: the colour its car, its trail and its panel row are
# all drawn in. The minimap is not one of them, it draws the followed car
# alone, in `MINIMAP_CAR`.
CONTENDER = (
    (86, 170, 255),
    (255, 206, 62),
    (110, 226, 140),
    (240, 120, 220),
    (255, 140, 60),
    (170, 150, 255),
)

HUD_TEXT = (232, 234, 238)
HUD_DIM = (140, 146, 156)
HUD_PANEL = (18, 19, 23, 196)
MINIMAP_BG = (18, 19, 23, 220)
MINIMAP_TRACK = (86, 170, 255)
MINIMAP_CAR = (222, 46, 46)
MINIMAP_VIEWPORT = (255, 255, 255, 100)
GAUGE_BG = (52, 56, 64)
GAUGE_LABEL = (150, 155, 165)
GAUGE_TICK = (200, 205, 215)
THROTTLE = (86, 214, 108)
BRAKE = (232, 72, 72)

# The gauge strip's text is three fixed labels, so every glyph is rendered once
# at `open_window` and blitted thereafter. They live here with their colour
# because `window` renders them and `hud` lays them out, and neither owns both.
GAUGE_LABELS = ("THR", "BRK", "STR")
