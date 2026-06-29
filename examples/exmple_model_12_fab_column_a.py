import pathlib

import compas
from compas.datastructures import Mesh
from compas.geometry import Frame
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import Scale
from compas.geometry import Transformation
from compas.geometry import Translation
from compas.geometry import Vector
from compas_model.models import Model
from compas_tf.column import ColumnElement
from compas_tf.viewer import TeeScene
from compas_tf.viewer import add_tool_simulation
from compas_tf.viewer import dump_bundle
from compas_tf.viewer import make_viewer
from compas_tf.viewer import triangulated

from compas_cnc import Postprocessor
from compas_cnc import clip
from compas_cnc import offset_polyline
from compas_cnc import outline
from compas_cnc import toolpath_2d_drill
from compas_cnc import toolpath_2d_ramp
from compas_cnc import toolpath_2d_surfacing
from compas_cnc import toolpath_merge
from compas_cnc.dxf import load_dxf
from compas_cnc.tools import Tool

# Setup A: the column laid flat at 1:10, machining everything reachable straight
# down -- the silhouette end cut, the vertical drills and the up-facing cuts. The
# side features are left for setup B (exmple_model_12_fab_column_b.py).

GREY = (0.85, 0.85, 0.85)
RED = (0.9, 0.2, 0.2)
BLUE = (0.2, 0.4, 0.9)
GREEN = (0.2, 0.7, 0.3)

SCALE = 0.1
XO, YO = 5, 15

RADIUS = 1.5  # tool radius (mm)
TOOL_DIAMETER = RADIUS * 2
TOOL = Tool(TOOL_DIAMETER, 30.0, name="flat_3mm")
STEPOVER = TOOL_DIAMETER / 4  # surfacing pass spacing
DOC = 2.0  # depth of cut per pass (ramp stepdown)
RECT_OFFSET = 5.0  # grow the top-surface rectangle outwards (mm)
RECT_Z = 36.0  # top-surface cut height (mm) -- raised 3mm above the 33 model top
DIRECTION = "climb"  # one-directional milling following the CW (M3) cutter; None = faster bidirectional zig-zag
OFFSET = 0.0  # contour offset from the silhouette (default RADIUS to ride outside; 0 = on the edge)
Z_SAFE = 60.0
DRILL_LENGTH = 30.0
DRILL_FLOOR = -1.0
END_CAP = 5.0

data_dir = pathlib.Path(__file__).parent.parent / "data"


def cylinder_axis_radius(mesh):
    """``(axis Line, radius)`` of a capped-cylinder cutter mesh, else ``None``."""
    degree = {v: len(list(mesh.vertex_neighbors(v))) for v in mesh.vertices()}
    peak = max(degree.values())
    centers = [v for v, d in degree.items() if d == peak]
    if peak < 6 or len(centers) != 2:
        return None
    a = Point(*mesh.vertex_coordinates(centers[0]))
    b = Point(*mesh.vertex_coordinates(centers[1]))
    ring = [Point(*mesh.vertex_coordinates(n)) for n in mesh.vertex_neighbors(centers[0])]
    return Line(a, b), sum(a.distance_to_point(p) for p in ring) / len(ring)


def silhouette(geometry, offset):
    """Part outline on the table, offset outward by ``offset`` (0 = on the edge).

    ``outline`` only unions the faces when grown by a positive distance, so grow
    by the tool RADIUS and then offset the result back to the requested distance.
    """
    grown = outline(geometry, RADIUS, z=0.0)
    return offset_polyline(grown[0], offset - RADIUS, join_type="miter")


model: Model = compas.json_load(data_dir / "cantilevers_model.json")
column: ColumnElement = model.find_element_with_name("column_0")

# Lay the column flat at a real table spot, then shrink it 1:10 in place.
cnc_frame = Frame([XO, YO, 0], [0, 1, 0], [0, 0, 1])
fab = Transformation.from_frame_to_frame(Frame.worldXY().translated((-column.width * 0.5, -column.depth * 0.5, 0)), cnc_frame)
xform = Scale.from_factors([SCALE, SCALE, SCALE], frame=cnc_frame) * fab

uncut = column.compute_elementgeometry(types=["ColumnAddFeature"]).transformed(xform)
final_geometry = column.compute_elementgeometry().transformed(xform)

beam_z = [final_geometry.vertex_coordinates(v)[2] for v in final_geometry.vertices()]
top_z, beam_height = max(beam_z), max(beam_z) - min(beam_z)

profile = silhouette(final_geometry, OFFSET)

# End cut: ramp the silhouette cap at the start of the beam straight down. Like
# the clip ramps it follows the contour, so the path is offset by the tool radius
# to ride on the cut side (the slot ramp below stays on its centreline).
end_cut = toolpath_2d_ramp.from_outline(profile[0], depth=beam_height, top_z=top_z, end="start", cap=END_CAP, step=DOC, safe_z=Z_SAFE, offset=-RADIUS)

# Surfacing of a manual rectangle, and descending ramps along the silhouette
# clipped to that rectangle.
rectangle = Polyline([[0, 0, RECT_Z], [90, 0, RECT_Z], [90, 44, RECT_Z], [0, 44, RECT_Z], [0, 0, RECT_Z]])
rectangle.transform(Translation.from_vector([285 - 85 + XO, YO, 0]))
# Surface a copy grown RECT_OFFSET outwards so the top face is cleared past its edges.
o = RECT_OFFSET
surf_rect = Polyline([[-o, -o, RECT_Z], [90 + o, -o, RECT_Z], [90 + o, 44 + o, RECT_Z], [-o, 44 + o, RECT_Z], [-o, -o, RECT_Z]])
surf_rect.transform(Translation.from_vector([285 - 85 + XO, YO, 0]))
surfacing_rect = toolpath_2d_surfacing.from_quad(surf_rect, RADIUS, safe_z=Z_SAFE, stepover=STEPOVER, direction=DIRECTION)
clip_ramps = [
    toolpath_2d_ramp(part.transformed(Translation.from_vector([0, 0, top_z])), Vector(0, 0, -beam_height), step=DOC, safe_z=Z_SAFE, offset=-RADIUS)
    for part in clip(profile[0], rectangle, z=0.0)
]

cuts = [mesh.transformed(xform) for feature in column.get_features(["ColumnCutFeature"]) for mesh in feature.meshes]

# Helical drilling for the holes that are vertical in this orientation.
drills = []
for solid in cuts:
    found = cylinder_axis_radius(solid)
    if found is None:
        continue
    axis, hole_radius = found
    top, bottom = Point(*axis.start), Point(*axis.end)
    if bottom[2] > top[2]:
        top, bottom = bottom, top
    up = (top - bottom).unitized()
    if up[2] < 0.87:  # not vertical -- drilled in setup B after the roll
        continue
    drills.append(toolpath_2d_drill(Line(bottom + up * DRILL_LENGTH, bottom), hole_radius, TOOL_DIAMETER, floor=DRILL_FLOOR, safe_z=Z_SAFE))

# Zig-zag surfacing: the manual rectangle first, then the up-facing cut plates.
# START is the outline corner id (0-3) each sweep begins at; None starts at the
# plate's highest corner (top-down).
PLATES = [2, 1, 5, 4]
START = [3, 2, 2, None]
plate_surfacing = [surfacing_rect]
for index, start in zip(PLATES, START):
    tp = toolpath_2d_surfacing.from_plate(cuts[index], RADIUS, safe_z=Z_SAFE, flip=True, incline=True, stepover=STEPOVER, start=start, direction=DIRECTION)
    if tp is not None:
        plate_surfacing.append(tp)

# Ramp the narrow slot (cut_9) down its centreline.
slot_ramp = toolpath_2d_ramp.from_box(cuts[9], step=DOC, safe_z=Z_SAFE)

# Surfacing first, then the drills and the ramps last.
toolpaths = [*plate_surfacing, *drills, end_cut, *clip_ramps]
if slot_ramp is not None:
    toolpaths.append(slot_ramp)

# `toolpaths` is an ordered list (toolpath_0, toolpath_1, ...). To machine only some,
# pick them before merging, e.g.  toolpaths = toolpaths[:6]  or  [toolpaths[i] for i in (0, 5, 8)].
job = toolpath_merge(*toolpaths)
# Post-process the chosen tool-paths into one Carvera Air program and write the .nc.
post = Postprocessor(tool=TOOL, feed=300, spindle_speed=10000, coolant="air", material="Wood", program="Column setup A")
post.write(data_dir / "column_fab_a.nc", job)

# ------------------------------------------------------------------ #
# Viewer
# ------------------------------------------------------------------ #

viewer = make_viewer(data_dir)
scene = TeeScene(viewer.scene)
scene.add(final_geometry, hide_coplanaredges=True)

stock = scene.add_group(f"{column.name}__stock_and_cuts")
stock.add(triangulated(uncut), name=f"{column.name}_uncut", hide_coplanaredges=True, color=GREY)
# Red cut-solid plates hidden -- uncomment to preview the removed material blocks.
# cutters = stock.add_group("cut_solids")
# for index, solid in enumerate(cuts):
#     cutters.add(triangulated(solid), name=f"cut_{index}", color=RED, hide_coplanaredges=True)

table_group = scene.add_group("cnc_table")
for index, curve in enumerate(load_dxf(data_dir / "cnc_table.dxf")):
    table_group.add(curve, name=f"table_{index}", color=BLUE)

clamp_group = scene.add_group("clamp_1")
clamp_group.add(triangulated(Mesh.from_obj(data_dir / "clamp_1.obj")), name="clamp_1", color=GREEN, hide_coplanaredges=True)

dump_bundle(scene, data_dir / "column_fab_a_rhino.json")
add_tool_simulation(viewer, [tp.path for tp in toolpaths], radius=RADIUS, height=30.0)
viewer.show()
