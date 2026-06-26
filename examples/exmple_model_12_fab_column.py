import math
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

from compas_cnc import clip
from compas_cnc import outline
from compas_cnc.dxf import load_dxf
from compas_cnc.toolpath_2d_drill import toolpath_2d_drill
from compas_cnc.toolpath_2d_ramp import toolpath_2d_ramp
from compas_cnc.toolpath_2d_ramp import toolpath_2d_ramp
from compas_cnc.toolpath_2d_surfacing import toolpath_2d_surfacing

data_dir = pathlib.Path(__file__).parent.parent / "data"

GREY = (0.85, 0.85, 0.85)
RED = (0.9, 0.2, 0.2)
BLUE = (0.2, 0.4, 0.9)  # cnc table outline
GREEN = (0.2, 0.7, 0.3)  # clamp fixture

SCALE = 0.1  # 1:10 model -- the column and its cut features are scaled down 10x.

XO = 5
YO = 15

RADIUS = 3.0  # mm -- milling tool radius (boundary inset + max pass spacing)
Z_SAFE = 60.0  # plunge/retract height (kept >= 10 above the tool-path plane)
TOOL_DIAMETER = 3.0  # mm -- end-mill diameter used to helically bore the round holes
DRILL_LENGTH = 30.0  # mm -- extend each recovered hole axis to this length (tool starts above the stock)
DRILL_FLOOR = -1.0  # mm -- the drill must not go below this Z; it stops there
RAMP_STEP = 2.0  # mm -- vertical descent per pass when ramping down a narrow slot
END_CAP = 5.0  # mm -- how far in from the beam end the end-cut segment reaches


def cylinder_axis_radius(mesh):
    """Recover ``(axis, radius)`` from a capped-cylinder cutter mesh.

    A cylinder cutter has exactly two high-valence vertices -- the cap centres,
    each joined to a whole end ring -- while the box / wedge cutters do not. The
    axis runs between those two centres and the radius is their average distance
    to the ring. Returns ``None`` for non-cylinder cutters.
    """
    degree = {v: len(list(mesh.vertex_neighbors(v))) for v in mesh.vertices()}
    peak = max(degree.values())
    centers = [v for v, d in degree.items() if d == peak]
    if peak < 6 or len(centers) != 2:
        return None
    a = Point(*mesh.vertex_coordinates(centers[0]))
    b = Point(*mesh.vertex_coordinates(centers[1]))
    ring = [Point(*mesh.vertex_coordinates(n)) for n in mesh.vertex_neighbors(centers[0])]
    radius = sum(a.distance_to_point(p) for p in ring) / len(ring)
    return Line(a, b), radius


def write_obj(path, meshes=(), polylines=()):
    """Write meshes (as v/f) and tool-paths (as v + polyline ``l``) into one OBJ,
    so the part and every tool-path can be inspected together in another program.
    """
    out = []
    index = 1  # OBJ vertex indices are 1-based and global
    for mesh in meshes:
        verts, faces = mesh.to_vertices_and_faces()
        out += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
        out += ["f " + " ".join(str(index + i) for i in face) for face in faces]
        index += len(verts)
    for polyline in polylines:
        pts = [list(p) for p in polyline]
        out += [f"v {p[0]:.6f} {p[1]:.6f} {(p[2] if len(p) > 2 else 0.0):.6f}" for p in pts]
        out.append("l " + " ".join(str(index + i) for i in range(len(pts))))
        index += len(pts)
    pathlib.Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ #
# Deserialize the cantilevers model written by example_model_8, pick a column.
# ------------------------------------------------------------------ #

model: Model = compas.json_load(data_dir / "cantilevers_model.json")
column: ColumnElement = model.find_element_with_name("column_0")

# Lay the column flat at a REAL table spot, then shrink it to 1:10 IN PLACE: the
# scale is taken about cnc_frame (not the world origin), so the part gets smaller
# but its position is NOT also multiplied -- that's why this uses (5, 15) and not
# (50, 150). (Whole-model scaling about the origin lives in exmple_model_18.)
cnc_frame = Frame([XO, YO, 0], [0, 1, 0], [0, 0, 1])  # the real spot on the table
fab = Transformation.from_frame_to_frame(Frame.worldXY().translated((-column.width * 0.5, -column.depth * 0.5, 0)), cnc_frame)
scale = Scale.from_factors([SCALE, SCALE, SCALE], frame=cnc_frame)  # shrink in place at cnc_frame
xform = scale * fab

# ------------------------------------------------------------------ #
# Uncut stock: apply ONLY the additive capitel, skip the cuts.
# ------------------------------------------------------------------ #

uncut = column.compute_elementgeometry(types=["ColumnAddFeature"]).transformed(xform)
final_geometry = column.compute_elementgeometry().transformed(xform)

# ------------------------------------------------------------------ #
# Profile (perimeter) tool-path: the 2D outline of the finished part flattened
# onto the table, grown outward by the tool RADIUS so the cutter rides just
# OUTSIDE the silhouette. The offset is the tool RADIUS -- half the TOOL_DIAMETER,
# not the diameter -- so the cutting edge (not the centre) grazes the silhouette.
# `outline` projects every face to XY, unions them and offsets in one Clipper2
# pass. Placed at z=0 (table level) for a through cut.
# ------------------------------------------------------------------ #

tool_radius = TOOL_DIAMETER / 2.0
profile = outline(final_geometry, tool_radius, z=0.0)
print(f"profile outline: {len(profile)} contour(s) offset {tool_radius} mm (tool radius) outside the silhouette")

# ------------------------------------------------------------------ #
# End-cut tool-path: cut off the START end of the beam. The full profile is a
# good path, but here we only want the SEGMENT around the end -- a line across
# the end face -- and instead of staying flat we ramp it down gradually from the
# top of the beam to the table, sweeping back and forth (like the slot ramp).
# `from_outline` slices the cap within END_CAP of the chosen end out of the
# silhouette and ramps it straight down by the beam height.
# ------------------------------------------------------------------ #

beam_z = [final_geometry.vertex_coordinates(v)[2] for v in final_geometry.vertices()]
top_z, beam_height = max(beam_z), max(beam_z) - min(beam_z)
end_cut = toolpath_2d_ramp.from_outline(
    profile[0], depth=beam_height, top_z=top_z, end="start", cap=END_CAP, step=RAMP_STEP, safe_z=Z_SAFE
)
print(f"end cut (start): {end_cut.passes} passes, {end_cut.step:.2f} mm/pass, {math.degrees(end_cut.ramp_angle):.1f} deg, depth {end_cut.depth:.1f} mm")

# ------------------------------------------------------------------ #
# Solids that carve the uncut stock, recovered by type for fabrication.
# ------------------------------------------------------------------ #

debug = []

# ------------------------------------------------------------------ #
# A separate, manual rectangular cutting tool-path (a test rectangle).
# ------------------------------------------------------------------ #

polyline0 = Polyline([
    [0, 0, 33],
    [90, 0, 33],
    [90, 44, 33],
    [0, 44, 33],
    [0, 0, 33]])
polyline0.transform(Translation.from_vector([285 - 85 + XO, YO, 0]))  # same translation as the column

# Use the two SHORT edges so the zigzag sweeps ALONG the 80 mm length and steps
# across the 44 mm width (the other orientation).
line0 = Line(polyline0[0], polyline0[3])  # left  edge (the 44 mm side)
line1 = Line(polyline0[1], polyline0[2])  # right edge (the 44 mm side)
toolpath = toolpath_2d_surfacing(line0, line1, RADIUS, safe_z=Z_SAFE)
debug.append(polyline0)
debug.append(toolpath.path)

# Boolean-intersect the silhouette (offset profile) with polyline0: keep only the
# parts of the silhouette outline that lie INSIDE the rectangle you gave.
profile_clipped = clip(profile[0], polyline0, z=0.0)
print(f"silhouette clipped to polyline0: {len(profile_clipped)} open part(s)")

# Turn each open clipped path into a zig that gradually descends along it: lift it
# to the top of the part, then ramp straight down by the part height, sweeping the
# path back and forth (like the end cut, but following this clipped outline).
for part in profile_clipped:
    mouth = part.transformed(Translation.from_vector([0, 0, top_z]))
    ramp_profile = toolpath_2d_ramp(mouth, Vector(0, 0, -beam_height), step=RAMP_STEP, safe_z=Z_SAFE)
    debug.append(ramp_profile.path)
    print(f"profile ramp: {ramp_profile.passes} passes, {ramp_profile.step:.2f} mm/pass")

# ------------------------------------------------------------------ #
# The cut solids that carve it, recovered by type for fabrication.
# ------------------------------------------------------------------ #

cuts = []
for feature in column.get_features(["ColumnCutFeature"]):
    for mesh in feature.meshes:
        cuts.append(mesh.transformed(xform))  # SAME xform as uncut -> consistent 1:10

# ------------------------------------------------------------------ #
# Helical drilling tool-paths for the CYLINDRICAL cut features. A 3 mm end mill
# is too narrow to clear these holes by plunging straight, so it spirals down
# each hole axis instead: the tool CENTRE rides a helix of radius
# (hole_radius - tool_radius) so the cutting edge just reaches the hole wall,
# descending one tool width per turn -- a typical helical-boring drill cycle.
# ------------------------------------------------------------------ #

drills = []
for solid in cuts:
    cylinder = cylinder_axis_radius(solid)
    if cylinder is None:
        continue  # box / wedge cutter -- not a round hole
    axis, hole_radius = cylinder
    # The drill length comes straight from the axis line now: extend the recovered
    # hole axis to DRILL_LENGTH (anchored at the bottom) so the tool starts above
    # the stock.
    top, bottom = Point(*axis.start), Point(*axis.end)
    if bottom[2] > top[2]:
        top, bottom = bottom, top
    up = (top - bottom).unitized()
    axis = Line(bottom + up * DRILL_LENGTH, bottom)
    drills.append(toolpath_2d_drill(axis, hole_radius, TOOL_DIAMETER, floor=DRILL_FLOOR, safe_z=Z_SAFE))
print(f"helical drills: {len(drills)} cylindrical hole(s), tool diameter {TOOL_DIAMETER} mm, length {DRILL_LENGTH} mm, floor z={DRILL_FLOOR}")

# ------------------------------------------------------------------ #
# Zigzag milling tool-path from a CUT plate's big bottom face (triangulation-proof
# -- it merges the cutter's coplanar triangles back into quads first).
# ------------------------------------------------------------------ #

indices = [2, 1, 4, 5]  # pick which cut solids to mill
flips = [False, True, True, True]
for index, flip in zip(indices, flips):  # pick which cut solids to mill
    # flip=False/True chooses the zigzag direction; incline=True shifts the path by
    # the tool radius up-slope so the flat bit rides the tilted surface (not gouge).
    tp = toolpath_2d_surfacing.from_plate(cuts[index], RADIUS, safe_z=Z_SAFE, flip=flip, incline=True)
    if tp is None:
        print(f"cut_{index}: face too small for the {RADIUS} mm tool -- skipped")
        continue
    debug.append(tp.path)

# ------------------------------------------------------------------ #
# Ramp tool-path for the NARROW slot (cut_9): its thin side is only as wide as
# the tool, so there is no room to clear it sideways. The tool follows the slot's
# single centreline and ramps down gradually -- back and forth, RAMP_STEP mm per
# pass -- until it reaches the floor, then one flat finishing pass cleans it out.
# ------------------------------------------------------------------ #

ramp = toolpath_2d_ramp.from_box(cuts[9], step=RAMP_STEP, safe_z=Z_SAFE)
if ramp is None:
    print("cut_9: not a clean box -- ramp skipped")
else:
    print(f"slot ramp (cut_9): {ramp.passes} passes, {ramp.step:.2f} mm/pass, {math.degrees(ramp.ramp_angle):.1f} deg, depth {ramp.depth:.1f} mm")


# ------------------------------------------------------------------ #
#  View - the uncut stock (grey) with the cut solids (red) that carve it.
# ------------------------------------------------------------------ #

viewer = make_viewer(data_dir)

scene = TeeScene(viewer.scene)  # draw to the viewer AND record a Rhino bundle

for o in debug:
    scene.add(o)  # yellow debug lines
scene.add(final_geometry, hide_coplanaredges=True)  # yellow final geometry

# Helical drilling spirals down each cylindrical hole.
for index, drill in enumerate(drills):
    scene.add(drill.path, name=f"drill_{index}")

# Ramp: the narrow slot (cut_9) descended gradually along one line.
if ramp is not None:
    scene.add(ramp.path, name="ramp_cut_9")

# Profile outline, CLIPPED to polyline0 (only the part inside the rectangle).
for index, contour in enumerate(profile_clipped):
    scene.add(contour, name=f"profile_clip_{index}")

# End-cut: the start end of the beam ramped down gradually.
scene.add(end_cut.path, name="end_cut_start")

stock = scene.add_group(f"{column.name}__stock_and_cuts")
stock.add(triangulated(uncut), name=f"{column.name}_uncut", hide_coplanaredges=True, color=GREY)

cutters = stock.add_group("cut_solids")
for index, solid in enumerate(cuts):
    cutters.add(triangulated(solid), name=f"cut_{index}", color=RED, hide_coplanaredges=True)

# ------------------------------------------------------------------ #
#  CNC table loaded from DXF (mm, drawn at the 1:10 model scale), with the
#  1:10 column laid on it. ezdxf reads DXF, not DWG -- export DWG as ASCII DXF.
# ------------------------------------------------------------------ #
table_path = data_dir / "cnc_table.dxf"
try:
    table = load_dxf(table_path)
    table_group = scene.add_group("cnc_table")
    for index, curve in enumerate(table):
        table_group.add(curve, name=f"table_{index}", color=BLUE)
    print(f"cnc table: {len(table)} curves from {table_path.name}")
except Exception as exc:
    print(f"[cnc table] could not load {table_path.name}: {exc}")
    print("[cnc table] ezdxf reads DXF, not DWG -- export the table as ASCII DXF.")

# ------------------------------------------------------------------ #
#  Clamp fixture loaded from OBJ (clamp_1.obj -- same basename as clamp_1.stp,
#  meshed externally). compas reads the triangle mesh straight from OBJ, so no
#  OpenCASCADE/compas_occ is needed to bring the STEP geometry into the scene.
# ------------------------------------------------------------------ #
clamp_path = data_dir / "clamp_1.obj"
try:
    clamp = Mesh.from_obj(clamp_path)
    clamp_group = scene.add_group("clamp_1")
    clamp_group.add(triangulated(clamp), name="clamp_1", color=GREEN, hide_coplanaredges=True)
    print(f"clamp: {clamp.number_of_faces()} faces from {clamp_path.name}")
except Exception as exc:
    print(f"[clamp] could not load {clamp_path.name}: {exc}")

# Write the Rhino bundle (plain, already-computed geometry) next to the viewer
# scene, so it can be loaded into Rhino with no recompute - see RHINO block below.
dump_bundle(scene, data_dir / "column_fab_rhino.json")

# ------------------------------------------------------------------ #
#  Export the final geometry + every tool-path to OBJ for external inspection.
# ------------------------------------------------------------------ #
toolpaths = list(debug) + [drill.path for drill in drills] + list(profile_clipped)
obj_path = data_dir / "column_fab.obj"
write_obj(obj_path, meshes=[final_geometry], polylines=toolpaths)
print(f"wrote {obj_path.name}: 1 mesh + {len(toolpaths)} tool-path polylines")

# ------------------------------------------------------------------ #
#  Tool simulation: two sliders drive a milling-tool cylinder (radius = the tool
#  RADIUS, height 30) over the tool-path POLYLINES -- "polyline" picks which
#  tool-path (its value is the polyline id) and "along %" slides the tool along
#  that polyline. The helper handles BOTH viewer modes -- it builds the sliders
#  directly on a live viewer, or records the paths for the watch viewer to build
#  them in its own window, so the simulation works in the watcher workflow.
# ------------------------------------------------------------------ #
sim_paths = list(debug) + [drill.path for drill in drills]
if ramp is not None:
    sim_paths.append(ramp.path)
sim_paths += list(profile_clipped) + [end_cut.path]
add_tool_simulation(viewer, sim_paths, radius=TOOL_DIAMETER / 2.0, height=30.0)  # 1.5 mm = actual end-mill radius

viewer.show()