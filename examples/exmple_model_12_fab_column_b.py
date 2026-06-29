import math
import pathlib

import compas
from compas.datastructures import Mesh
from compas.geometry import Frame
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Rotation
from compas.geometry import Scale
from compas.geometry import Transformation
from compas_model.models import Model
from compas_tf.column import ColumnElement
from compas_tf.viewer import TeeScene
from compas_tf.viewer import add_tool_simulation
from compas_tf.viewer import dump_bundle
from compas_tf.viewer import make_viewer
from compas_tf.viewer import triangulated

from compas_cnc import Postprocessor
from compas_cnc import toolpath_2d_drill
from compas_cnc import toolpath_2d_ramp
from compas_cnc import toolpath_merge
from compas_cnc.dxf import load_dxf
from compas_cnc.tools import Tool

# Setup B: the same column rolled 90 degrees about its long axis (pivoting on the
# base centre so the base stays on the table and the head swings round), then
# machining only what setup A could not reach -- the two cross holes and the side
# slot, both vertical after the roll.

GREY = (0.85, 0.85, 0.85)
RED = (0.9, 0.2, 0.2)
BLUE = (0.2, 0.4, 0.9)
GREEN = (0.2, 0.7, 0.3)

SCALE = 0.1
XO, YO = 5, 15

RADIUS = 1.5  # tool radius (mm)
TOOL_DIAMETER = RADIUS * 2
TOOL = Tool(TOOL_DIAMETER, 30.0, name="flat_3mm")
DOC = 2.0  # depth of cut per pass (ramp stepdown)
Z_SAFE = 60.0
DRILL_LENGTH = 30.0
DRILL_FLOOR = -1.0

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


model: Model = compas.json_load(data_dir / "cantilevers_model.json")
column: ColumnElement = model.find_element_with_name("column_0")

cnc_frame = Frame([XO, YO, 0], [0, 1, 0], [0, 0, 1])
fab = Transformation.from_frame_to_frame(Frame.worldXY().translated((-column.width * 0.5, -column.depth * 0.5, 0)), cnc_frame)
xform = Scale.from_factors([SCALE, SCALE, SCALE], frame=cnc_frame) * fab

# Roll 90 degrees about the long axis through the base end-face centre (on the
# centreline), so the base stays put and the body rotates in place.
laid = column.compute_elementgeometry().transformed(xform)
base_x = min(laid.vertex_coordinates(v)[0] for v in laid.vertices())
base = [laid.vertex_coordinates(v) for v in laid.vertices() if laid.vertex_coordinates(v)[0] < base_x + 0.5]
base_y = sum(p[1] for p in base) / len(base)
base_z = sum(p[2] for p in base) / len(base)
xform = Rotation.from_axis_and_angle([1, 0, 0], math.radians(90), point=[0, base_y, base_z]) * xform

uncut = column.compute_elementgeometry(types=["ColumnAddFeature"]).transformed(xform)
final_geometry = column.compute_elementgeometry().transformed(xform)
cuts = [mesh.transformed(xform) for feature in column.get_features(["ColumnCutFeature"]) for mesh in feature.meshes]

# Helical drilling for the cross holes, now vertical after the roll.
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
    if up[2] < 0.87:  # not vertical after the roll -- drilled in setup A
        continue
    drills.append(toolpath_2d_drill(Line(bottom + up * DRILL_LENGTH, bottom), hole_radius, TOOL_DIAMETER, floor=DRILL_FLOOR, safe_z=Z_SAFE))

# Ramp the side slot (cut_6), which opens straight up after the roll.
slot_ramp = toolpath_2d_ramp.from_box(cuts[6], step=DOC, safe_z=Z_SAFE)

toolpaths = list(drills)
if slot_ramp is not None:
    toolpaths.append(slot_ramp)

# `toolpaths` is an ordered list (toolpath_0, toolpath_1, ...). To machine only some,
# pick them before merging, e.g.  toolpaths = toolpaths[:1]  or  [toolpaths[i] for i in (0, 2)].
job = toolpath_merge(*toolpaths)
# Post-process the chosen tool-paths into one Carvera Air program and write the .nc.
post = Postprocessor(tool=TOOL, feed=300, spindle_speed=10000, coolant="air", material="Wood", program="Column setup B")
post.write(data_dir / "column_fab_b.nc", job)

# ------------------------------------------------------------------ #
# Viewer
# ------------------------------------------------------------------ #

viewer = make_viewer(data_dir)
scene = TeeScene(viewer.scene)
scene.add(final_geometry, hide_coplanaredges=True)

stock = scene.add_group(f"{column.name}__stock_and_cuts")
stock.add(triangulated(uncut), name=f"{column.name}_uncut", hide_coplanaredges=True, color=GREY)
cutters = stock.add_group("cut_solids")
for index, solid in enumerate(cuts):
    cutters.add(triangulated(solid), name=f"cut_{index}", color=RED, hide_coplanaredges=True)

table_group = scene.add_group("cnc_table")
for index, curve in enumerate(load_dxf(data_dir / "cnc_table.dxf")):
    table_group.add(curve, name=f"table_{index}", color=BLUE)

clamp_group = scene.add_group("clamp_1")
clamp_group.add(triangulated(Mesh.from_obj(data_dir / "clamp_1.obj")), name="clamp_1", color=GREEN, hide_coplanaredges=True)

dump_bundle(scene, data_dir / "column_fab_b_rhino.json")
add_tool_simulation(viewer, [tp.path for tp in toolpaths], radius=RADIUS, height=30.0)
viewer.show()
