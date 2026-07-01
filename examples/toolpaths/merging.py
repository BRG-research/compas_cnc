from compas_viewer import Viewer

from compas.geometry import Line
from compas.geometry import Polyline
from compas.geometry import Vector
from compas_cnc import toolpath_2d_drill
from compas_cnc import toolpath_2d_ramp
from compas_cnc import toolpath_2d_surfacing
from compas_cnc import toolpath_merge
from compas_cnc.tools import FLAT_3MM
from compas_cnc.tools import add_tool_slider

tool = FLAT_3MM

stock = Polyline([[0, 0, 0], [80, 0, 0], [80, 50, 0], [0, 50, 0], [0, 0, 0]])

surface = toolpath_2d_surfacing.from_quad(stock, radius=tool.radius, stepover=tool.diameter, safe_z=20.0)
slot = toolpath_2d_ramp(Line([20, 25, 0], [60, 25, 0]), Vector(0, 0, -10), step=2.0, safe_z=20.0)
drill = toolpath_2d_drill(Line([40, 25, 20], [40, 25, 0]), hole_radius=6.0, tool_diameter=tool.diameter, safe_z=20.0)

path = toolpath_merge(
    surface,     # any number of tool-paths (objects with .path) or raw Polylines, in cutting order
    slot,
    drill,
    home=True,   # True | False -- end the merged path back at the global start, at safe Z
)
print(f"merged: {len(path.points)} points from surfacing + slot + drill")

# =============================================================================
# Visualization
# =============================================================================

viewer = Viewer()
viewer.scene.add(stock, linecolor=(0.6, 0.6, 0.6), name="stock")
viewer.scene.add(path, linecolor=(1.0, 0.5, 0.0), name="toolpath")
add_tool_slider(viewer, tool, path, facecolor=(0.7, 0.7, 0.75), opacity=0.6)
viewer.show()
