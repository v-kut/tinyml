"""Top-down pygame viewer: walls, racing line, car, LiDAR fan, trail, HUD.

A wireframe on a flat backdrop: it shares a thread with the control loop, so
per-frame cost must not scale with the zoom.
"""
