"""Clipper2 wrapper demo: offset a polyline and hatch a closed region.

Run with::

    python examples/example_clipper2_offset_hatch.py

Builds two things from one L-shaped outline:

* a set of concentric INWARD offsets (typical CNC contour-clearing rings), and
* a 45-degree HATCH of parallel lines clipped to the region (e.g. a pocket
  fill / raster pass).

If ``compas_viewer`` is installed the result is shown; otherwise a short text
summary is printed so the example still runs headless.
"""

import math

from compas.geometry import Polyline

from compas_cnc import hatch
from compas_cnc import offset_polyline

# An L-shaped outline (units = mm), closed (first point repeated).
outline = Polyline(
    [
        [0, 0, 0],
        [80, 0, 0],
        [80, 30, 0],
        [35, 30, 0],
        [35, 60, 0],
        [0, 60, 0],
        [0, 0, 0],
    ]
)

# Concentric inward offsets: peel 4 mm off repeatedly until nothing is left.
TOOL_RADIUS = 4.0
contours = []
distance = -TOOL_RADIUS
while True:
    rings = offset_polyline(outline, distance, join_type="round")
    if not rings:
        break
    contours.extend(rings)
    distance -= TOOL_RADIUS

# A 45-degree hatch fill, lines 3 mm apart, clipped to the outline.
hatch_lines = hatch(outline, spacing=3.0, angle=math.radians(45))

print(f"outline      : {len(outline.points)} points")
print(f"offsets      : {len(contours)} contour(s) at {TOOL_RADIUS} mm steps")
print(f"hatch (45deg): {len(hatch_lines)} clipped lines at 3 mm spacing")

# ------------------------------------------------------------------ #
# Optional visualisation.
# ------------------------------------------------------------------ #
try:
    from compas_viewer import Viewer

    viewer = Viewer()
    viewer.scene.add(outline, linecolor=(0.1, 0.1, 0.1), name="outline")
    for i, contour in enumerate(contours):
        viewer.scene.add(contour, linecolor=(0.2, 0.4, 0.9), name=f"offset_{i}")
    for i, line in enumerate(hatch_lines):
        viewer.scene.add(line, linecolor=(0.1, 0.7, 0.2), name=f"hatch_{i}")
    viewer.show()
except ImportError:
    print("(compas_viewer not installed -- skipping the 3D view)")
