import pathlib

from compas_viewer import Viewer

from compas.geometry import Line
from compas.geometry import Polyline
from compas.geometry import Vector
from compas_cnc import Postprocessor
from compas_cnc import toolpath_2d_drill
from compas_cnc import toolpath_2d_ramp
from compas_cnc import toolpath_2d_surfacing
from compas_cnc import toolpath_merge
from compas_cnc.tools import FLAT_3_175MM
from compas_cnc.tools import add_tool_slider

tool = FLAT_3_175MM

stock = Polyline([[0, 0, 0], [80, 0, 0], [80, 50, 0], [0, 50, 0], [0, 0, 0]])

surface = toolpath_2d_surfacing.from_quad(stock, radius=tool.radius, stepover=tool.diameter, safe_z=20.0)
slot = toolpath_2d_ramp(Line([20, 25, 0], [60, 25, 0]), Vector(0, 0, -10), step=2.0, safe_z=20.0)
drill = toolpath_2d_drill(Line([40, 25, 20], [40, 25, 0]), hole_radius=6.0, tool_diameter=tool.diameter, safe_z=20.0)
path = toolpath_merge(surface, slot, drill)

post = Postprocessor(
    tool=tool,                # Tool | None -- header + diameter
    tool_number=4,            # int -- T-number for the tool change
    feed=200,                 # float -- cutting feed rate (mm/min)
    spindle_speed=12000,      # int -- spindle speed (rpm)
    coolant="mist",           # "mist"/"air" (M7) | "flood" (M8) | None/False (off)
    rapid_z=None,             # None (= path max Z) | float -- rapid/clearance Z
    material="Aluminum",      # str -- stock material (header note)
    stock_size=(80, 50, 20),  # (x, y, z) -- stock bounding box for the travel check
    program="2D Contour",     # str -- program name in the header
    precision=4,              # int 0..8 -- decimal places in the G-code
    margin=0.0,               # float -- extra keep-out from the machine travel limits
    on_exceed="raise",        # "raise" | "warn" | "ignore" -- when the path leaves the work area
)  # travel defaults to the Carvera Air work area
print(post)
print("within Carvera Air travel:", not post.check_limits(path))
nc = post.write(pathlib.Path(__file__).parent / "postprocessing.nc", path)
print(f"wrote {nc.name}: {len(path.points)} points -> Carvera Air G-code")

# =============================================================================
# Visualization
# =============================================================================

viewer = Viewer()
viewer.scene.add(stock, linecolor=(0.6, 0.6, 0.6), name="stock")
viewer.scene.add(path, linecolor=(1.0, 0.5, 0.0), name="toolpath")
add_tool_slider(viewer, tool, path, facecolor=(0.7, 0.7, 0.75), opacity=0.6)
viewer.show()
