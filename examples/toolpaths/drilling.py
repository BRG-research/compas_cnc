import math

from compas_viewer import Viewer

from compas.geometry import Circle
from compas.geometry import Frame
from compas.geometry import Line
from compas_cnc import toolpath_2d_drill
from compas_cnc.tools import FLAT_3_175MM
from compas_cnc.tools import add_tool_slider

tool = FLAT_3_175MM

axis = Line([40, 25, 20], [40, 25, 0])
hole_radius = 2.5

drill = toolpath_2d_drill(
    axis,                            # Line, top -> bottom; its length is the drill depth
    hole_radius=hole_radius,         # float > 0 (mm) -- radius of the hole to bore
    tool_diameter=tool.diameter,     # float > 0 (mm) -- cutting tool diameter
    ramp_angle=math.radians(15.0),   # None (default) | float radians -- helix descent angle
    bottom_pass=True,                # True | False -- finishing circle at the floor
    floor=None,                      # None (no limit) | float world-Z -- never cut below this
    safe_z=None,                     # None (auto: 10 above) | float world-Z -- approach/retract height
    direction=None,                  # None | "climb" | "conventional"
)
print(drill)

# =============================================================================
# Visualization
# =============================================================================

viewer = Viewer()
viewer.scene.add(axis, name="axis")
viewer.scene.add(Circle(hole_radius, Frame(axis.end)), name="hole")
viewer.scene.add(drill.path, linecolor=(1.0, 0.5, 0.0), name="toolpath")
add_tool_slider(viewer, tool, drill.path, facecolor=(0.7, 0.7, 0.75), opacity=0.6)
viewer.show()
