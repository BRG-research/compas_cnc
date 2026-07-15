from compas_viewer import Viewer

from compas.geometry import Polyline
from compas.geometry import Vector
from compas_cnc import toolpath_2d_ramp
from compas_cnc.tools import VBIT_3_175MM
from compas_cnc.tools import add_tool_slider

tool = VBIT_3_175MM

stock = Polyline([[0, 0, 0], [80, 0, 0], [80, 50, 0], [0, 50, 0], [0, 0, 0]])
path = Polyline([[10, 10, 15], [40, 10, 15], [40, 40, 15], [70, 40, 15], [50, 50, 15]])

ramp = toolpath_2d_ramp(
    path,                # Polyline (open) or Line -- path to ramp along, at the mouth (top)
    Vector(0, 0, -15),   # Vector -- descent mouth -> floor (length = cut depth)
    step=2.0,            # None | float > 0 -- vertical drop per pass (overrides ramp_angle)
    ramp_angle=None,     # None | float radians -- used only when step is None
    bottom_pass=True,    # True | False -- flat finishing pass at full depth
    safe_z=None,         # None (auto: 10 above) | float -- plunge/retract height
    offset=tool.radius,  # float -- planar offset before ramping (0 = none; sign picks the side)
    notch=tool.radius,   # float -- dogbone corner-relief radius (0 = off)
    notch_flip=False,    # True | False -- which corner handedness gets notched
    direction=None,      # None | "climb" | "conventional" (closed loops only)
    pocket=True,         # True | False -- closed loop: pocket (cleared inside) vs island/profile
)
print(ramp)

# =============================================================================
# Visualization
# =============================================================================

viewer = Viewer()
viewer.scene.add(stock, linecolor=(0.6, 0.6, 0.6), name="stock")
viewer.scene.add(path, linecolor=(0.2, 0.4, 0.9), name="path")
viewer.scene.add(ramp.path, linecolor=(1.0, 0.5, 0.0), name="toolpath")
for corner, tip in ramp.notches:
    viewer.scene.add(Polyline([corner, tip]), linecolor=(0.1, 0.8, 0.3), name="notch")
add_tool_slider(viewer, tool, ramp.path, facecolor=(0.7, 0.7, 0.75), opacity=0.6)
viewer.show()
