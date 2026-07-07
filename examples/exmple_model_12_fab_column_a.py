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
from compas_cnc.tools import add_toolpath_slider

# Setup A: the column laid flat at 1:10, machining everything reachable straight
# down -- the silhouette end cut, the vertical drills and the up-facing cuts. The
# side features are left for setup B (exmple_model_12_fab_column_b.py).

GREY = (0.85, 0.85, 0.85)
RED = (0.9, 0.2, 0.2)
BLUE = (0.2, 0.4, 0.9)
GREEN = (0.2, 0.7, 0.3)
ORANGE = (0.95, 0.55, 0.10)  # 6mm tool-paths
PURPLE = (0.60, 0.20, 0.90)  # 3mm tool-paths

SCALE = 0.1
XO, YO = 5, 15

RADIUS = 3.0  # 6mm tool radius (mm) -- surfacing + big ramps (most toolpaths)
TOOL_DIAMETER = RADIUS * 2
TOOL = Tool(TOOL_DIAMETER, 30.0, name="flat_6mm")
STEPOVER = TOOL_DIAMETER / 4  # surfacing pass spacing (6mm tool)
R3 = 1.5  # 3mm tool radius -- drills + narrow slot
TOOL_3 = Tool(R3 * 2, 30.0, name="flat_3mm")
DOC = 2.0  # depth of cut per pass (ramp stepdown)
RECT_OFFSET = 5.0  # grow the top-surface rectangle outwards (mm)
RECT_Z = 35.0  # top-surface cut height (mm) -- raised 3mm above the 33 model top
DIRECTION = "climb"  # one-directional milling following the CW (M3) cutter; None = faster bidirectional zig-zag
OFFSET = RADIUS  # grow the silhouette outward by the tool radius (Clipper2) so the ramps ride the cut side; the ramps then take offset=0 (0 here = on the edge). Doing the offset on the CLOSED contour avoids the naive single-sided open-path offset, which self-intersects at the boolean-difference corners.
ARC_TOL = 0.01  # chord tolerance (mm) for the rounded tool-radius corners of the silhouette offset -- SMALLER = smoother round corners (more points). 0.2 looks faceted; 0.01 is a clean round. This is the only knob for corner smoothness (clip + the ramp faithfully keep whatever points the offset emits).
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
    ``ARC_TOL`` is the CHORD tolerance for the rounded (tool-radius) corners: the tool
    centre rides an arc there, and since the post emits only straight G1 moves the arc
    is linearised into segments deviating at most ``ARC_TOL`` from the true arc. This is
    the ONLY control over corner smoothness -- clip() and the ramp preserve exactly the
    points the offset emits, so a coarse value here is what makes corners look faceted.
    """
    grown = outline(geometry, RADIUS, z=0.0, arc_tolerance=ARC_TOL)
    return offset_polyline(grown[0], offset - RADIUS, join_type="miter")


def trim_sweep_high_x(line0, line1, distance):
    """Pull a surfacing sweep's +X (right-hand) end in by ``distance`` mm.

    Used to keep the up-facing plate cuts within the machine's X travel. A sweep
    reaches into +X either along one whole rail (passes step across into +X) or at
    the rails' shared end (passes run along +X); this detects which and shortens the
    rails there, keeping their orientation so the climb/conventional winding is
    unchanged. Returns the trimmed ``(line0, line1)`` to rebuild the tool-path from.
    """
    ends = {"0s": line0.start, "0e": line0.end, "1s": line1.start, "1e": line1.end}
    hi = set(sorted(ends, key=lambda k: ends[k][0], reverse=True)[:2])
    if hi in ({"0s", "1s"}, {"0e", "1e"}):  # a rail END is the +X side -> shorten both rails there
        at_start = hi == {"0s", "1s"}

        def shorten(line):
            u = line.vector.unitized()
            return Line(line.start + u * distance, line.end) if at_start else Line(line.start, line.end - u * distance)

        return shorten(line0), shorten(line1)
    # a whole rail is the +X side -> slide it across toward the other rail
    far, near = (line1, line0) if ends["1s"][0] + ends["1e"][0] > ends["0s"][0] + ends["0e"][0] else (line0, line1)
    slid = Line(
        far.start + (near.start - far.start).unitized() * distance,
        far.end + (near.end - far.end).unitized() * distance,
    )
    return (slid, line1) if far is line0 else (line0, slid)


def extend_sweep_y(line0, line1, distance):
    """Lengthen both rails by ``distance`` mm at their +Y (far) end, so the sweep
    reaches further along Y. Extends the higher-Y endpoint of each rail along the rail
    direction (keeping its Z, so the inclined plane just continues), leaving the sweep
    orientation -- and so the climb winding -- unchanged.
    """
    def grow(line):
        u = line.vector.unitized()
        if line.end[1] >= line.start[1]:  # END is the +Y end -> push it out
            return Line(line.start, line.end + u * distance)
        return Line(line.start - u * distance, line.end)

    return grow(line0), grow(line1)


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

# End cut: ramp the silhouette cap at the start of the beam straight down. The
# profile is already grown outward by the tool radius (OFFSET=RADIUS), so it rides
# the cut side with offset=0 -- no per-ramp offset (the slot ramp below stays on its
# centreline).
end_cut = toolpath_2d_ramp.from_outline(profile[0], depth=beam_height, top_z=top_z, end="start", cap=END_CAP, step=DOC, safe_z=Z_SAFE, offset=0.0)

# Surfacing of a manual rectangle, and descending ramps along the silhouette
# clipped to that rectangle.
rectangle = Polyline([[0, 0, RECT_Z], [90, 0, RECT_Z], [90, 44, RECT_Z], [0, 44, RECT_Z], [0, 0, RECT_Z]])
rectangle.transform(Translation.from_vector([285 - 85 + XO, YO, 0]))
# Surface a copy grown RECT_OFFSET outwards so the top face is cleared past its edges.
o = RECT_OFFSET
surf_rect = Polyline([[-o, -o, RECT_Z], [90 + o, -o, RECT_Z], [90 + o, 44 + o, RECT_Z], [-o, 44 + o, RECT_Z], [-o, -o, RECT_Z]])
surf_rect.transform(Translation.from_vector([285 - 85 + XO, YO, 0]))
surfacing_rect = toolpath_2d_surfacing.from_quad(surf_rect, RADIUS, safe_z=Z_SAFE, stepover=STEPOVER, direction=DIRECTION)
# Column-head rectangle: a second flat surfacing pass, swept longitudinally (along
# its long X side) and started at its first control point so the tool clears the
# head from that corner.

head_rect = Polyline([[210.653113, 69.763286, 18], [294.945943, 57.266757, 18], [294.945943, 37.266757, 18], [210.653113, 49.763286, 18], [210.653113, 69.763286, 18]])
head_surfacing = toolpath_2d_surfacing.from_quad(head_rect, RADIUS, safe_z=Z_SAFE, stepover=STEPOVER, direction=DIRECTION, start=0)
clip_ramps = [
    toolpath_2d_ramp(part.transformed(Translation.from_vector([0, 0, top_z])), Vector(0, 0, -beam_height), step=DOC, safe_z=Z_SAFE, offset=0.0)
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
    drills.append(toolpath_2d_drill(Line(bottom + up * DRILL_LENGTH, bottom), hole_radius, R3 * 2, floor=DRILL_FLOOR, safe_z=Z_SAFE))

# Zig-zag surfacing: the manual rectangle first, then the up-facing cut plates.
# START is the outline corner id (0-3) each sweep begins at (None = highest corner).
# FLIP sets the sweep axis: False = passes along the LONG side (longitudinal),
# True = across. The one-directional cut direction follows the start corner -- the
# two corners on one side share a direction, the opposite side flips it -- so to
# keep climb while moving the start, pick a corner on the same side.
PLATES = [2, 1, 5, 4]
START = [3, 2, 2, None]
FLIP = [False, True, True, True]
TRIM = [10.0, 10.0, 0.0, 0.0]  # shorten the sweep's +X (right) end by this many mm -- cuts[2]/cuts[1] otherwise reach past the CNC X travel
CONTOUR = [True, True, True, False]  # cuts[4] (toolpath_5): skip the finishing perimeter lap and retract straight up at the end
EXTEND_Y = [15.0, 0.0, 0.0, 0.0]  # cuts[2] (toolpath_2): lengthen the sweep's +Y (far) end so it covers the top of the plate (part reaches Y~47, the raw sweep fell short)
plate_surfacing = [surfacing_rect, head_surfacing]
for index, start, flip, trim, contour, extend_y in zip(PLATES, START, FLIP, TRIM, CONTOUR, EXTEND_Y):
    tp = toolpath_2d_surfacing.from_plate(cuts[index], RADIUS, safe_z=Z_SAFE, flip=flip, incline=True, stepover=STEPOVER, start=start, direction=DIRECTION, contour=contour)
    if tp is None:
        continue
    line0, line1 = tp.line0, tp.line1
    if trim:  # shorten the right end so it stays inside the machine
        line0, line1 = trim_sweep_high_x(line0, line1, trim)
    if extend_y:  # lengthen the far (+Y) end to cover more of the plate
        line0, line1 = extend_sweep_y(line0, line1, extend_y)
    if trim or extend_y:  # rebuild from the adjusted rails
        tp = toolpath_2d_surfacing(line0, line1, RADIUS, safe_z=Z_SAFE, stepover=STEPOVER, incline=True, direction=DIRECTION, contour=contour)
    plate_surfacing.append(tp)

# Ramp the narrow slot (cut_9) down its centreline.
slot_ramp = toolpath_2d_ramp.from_box(cuts[9], step=DOC, safe_z=Z_SAFE)

# Two tools, one program (one .nc) each -- run the 6mm, swap the bit, run the 3mm:
#   6mm -> surfacing + big contour ramps ;  3mm -> narrow slot, then the drills LAST.
# To machine only some, drop items from these lists before merging.
group_6mm = [*plate_surfacing, end_cut, *clip_ramps]
group_3mm = ([slot_ramp] if slot_ramp is not None else []) + [*drills]

# Combined machining order, also the viewer/animation order (toolpath_0, 1, ...):
# the 6mm group first, then the 3mm group with the drills at the very end.
toolpaths = group_6mm + group_3mm

post_6mm = Postprocessor(tool=TOOL, tool_number=1, feed=300, spindle_speed=10000, coolant="air", material="Wood", program="Column setup A (6mm)")
post_6mm.write(data_dir / "column_fab_a_6mm.nc", toolpath_merge(*group_6mm))

post_3mm = Postprocessor(tool=TOOL_3, tool_number=2, feed=300, spindle_speed=10000, coolant="air", material="Wood", program="Column setup A (3mm)")
post_3mm.write(data_dir / "column_fab_a_3mm.nc", toolpath_merge(*group_3mm))

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
for index, curve in enumerate(load_dxf(data_dir / "cnc_table_holes.dxf")):
    table_group.add(curve, name=f"table_{index}", color=BLUE)

clamp_group = scene.add_group("clamp_1")
clamp_group.add(triangulated(Mesh.from_obj(data_dir / "clamp_1.obj")), name="clamp_1", color=GREEN, hide_coplanaredges=True)

# The tool-centre paths as polylines -- orange for the 6mm group, purple for the 3mm.
# Keep the LIVE objects so the selected one can turn red; record each into the bundle.
paths = scene.add_group("toolpaths")
six = {id(tp) for tp in group_6mm}
path_objs, path_colors = [], []
for index, tp in enumerate(toolpaths):
    color = ORANGE if id(tp) in six else PURPLE
    path_objs.append(paths._live.add(tp.path, name=f"path_{index}", color=color))
    paths._rec.add(tp.path, name=f"path_{index}", color=color)
    path_colors.append(color)

dump_bundle(scene, data_dir / "column_fab_a_rhino.json")

# Two sliders (live viewer only): 'toolpath' selects a path (turning it RED), 'position'
# scrubs the cutter along it -- 6mm on the surfacing/ramps, 3mm on the slot/drills.
sim_entries = [(TOOL, tp.path) for tp in group_6mm] + [(TOOL_3, tp.path) for tp in group_3mm]
if hasattr(viewer, "ui"):
    add_toolpath_slider(viewer, sim_entries, path_objs, path_colors)
viewer.show()
