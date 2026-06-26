from compas_viewer import Viewer

from compas.geometry import Polyline
from compas_cnc import toolpath_2d_surfacing
from compas_cnc.tools import FLAT_3MM
from compas_cnc.tools import add_tool_slider

tool = FLAT_3MM

face = Polyline([[0, 0, 0], [80, 0, 0], [80, 50, 10], [0, 50, 10], [0, 0, 0]])  # inclined face, z 0 -> 10

surfacing = toolpath_2d_surfacing.from_quad(
    face,                    # 4 corners or a closed polyline (closing corner dropped)
    radius=tool.radius,      # float > 0 (mm) -- tool radius: boundary inset + incline comp
    safe_z=None,             # None (auto: 10 above) | float -- plunge/retract height
    stepover=tool.diameter,  # None (= radius) | float > 0 -- max distance between passes
    flip=True,               # True | False -- zig-zag direction (False = sweep the longer side)
    incline=True,            # True | False -- tilt the path so a flat tool rides a sloped face
    direction=None,          # None (fast zig-zag) | "climb" | "conventional" (one-directional)
)
print(surfacing)

# =============================================================================
# Visualization
# =============================================================================

viewer = Viewer()
viewer.scene.add(face, linecolor=(0.6, 0.6, 0.6), name="face")
viewer.scene.add(surfacing.path, linecolor=(1.0, 0.5, 0.0), name="toolpath")
add_tool_slider(viewer, tool, surfacing.path, facecolor=(0.7, 0.7, 0.75), opacity=0.6)
viewer.show()
