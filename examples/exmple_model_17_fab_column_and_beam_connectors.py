import math
import pathlib

from compas.datastructures import Mesh
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import Translation
from compas.geometry import Vector
from compas_tf.viewer import TeeScene
from compas_tf.viewer import dump_bundle
from compas_tf.viewer import make_viewer
from compas_tf.viewer import triangulated

from compas_cnc import Postprocessor
from compas_cnc import offset_polyline
from compas_cnc import toolpath_2d_drill
from compas_cnc import toolpath_2d_ramp
from compas_cnc import toolpath_2d_surfacing
from compas_cnc import toolpath_merge
from compas_cnc.tools import Tool
from compas_cnc.tools import add_toolpath_slider

# Column-and-beam connectors, machined flat from a 3mm plate with TWO tools:
#   * a 6mm-diameter flat mill FACES the top rectangles (the stepped top faces at
#     Z = 1 / 2 / 3), and
#   * a 2mm-diameter mill RAMPS every contour through the stock.
# The cut curves come as Rhino NURBS-curve OBJs: Mill_6mm.obj (the rectangles) and
# Ramp_2mm.obj (the contours). In Ramp_2mm the degree-2 (rational) curves are the
# CIRCLES = holes and the degree-1 curves are the outer polygons -- and that split
# sets the offset SIDE of the tool centre: holes are cut from the INSIDE (helical
# drill, centre inset toward the hole axis) and outer polygons from the OUTSIDE
# (contour grown outward by the tool radius). geometry.stp (in metres) is the real
# solids, imported only for context.

GREY = (0.80, 0.80, 0.80)
RED = (0.90, 0.20, 0.20)
BLUE = (0.20, 0.40, 0.90)
GREEN = (0.20, 0.70, 0.30)
ORANGE = (0.95, 0.55, 0.10)  # 6mm facing tool-paths
PURPLE = (0.60, 0.20, 0.90)  # 2mm contour/hole tool-paths

RADIUS_MILL = 3.0  # 6mm-DIAMETER facing tool -> radius (the offset) is half = 3.0
RADIUS_RAMP = 1.0  # 2mm-DIAMETER contour/hole tool -> radius (the offset) is half = 1.0
STEPOVER = RADIUS_MILL * 2 / 4  # facing pass spacing
STOCK_TOP = 3.0  # plate top (mm) -- the contours sit at Z=0, cut down FROM here
OVERCUT = 0.5  # cut this far past the bottom so parts release cleanly
THROUGH = STOCK_TOP + OVERCUT  # ramp/drill descent depth
DOC = 1.0  # depth of cut per pass (ramp/helix stepdown)
DIRECTION = "climb"  # one-directional milling (CW / M3 cutter)
MITER_LIMIT = 4.0  # convex corners offset to SHARP mitered points (no arcs); bevels past this length ratio
Z_SAFE = 25.0

data_dir = pathlib.Path(__file__).parent.parent / "data"
conn_dir = data_dir / "Custom_Cutting_Connectors"

TOOL_MILL = Tool(RADIUS_MILL * 2, 30.0, name="mill_6mm")
TOOL_RAMP = Tool(RADIUS_RAMP * 2, 30.0, name="ramp_2mm")


def parse_obj_curves(path):
    """Rhino NURBS-curve OBJ -> ``[(degree, [Point, ...]), ...]``.

    Each ``curv`` references global 1-based vertex indices (``v`` lines, whose 4th
    value, if present, is a rational weight we drop). ``deg 1`` curves are polylines,
    ``deg 2`` rational curves are circles. The repeated closing control point is
    dropped, so each curve comes back as its distinct control points in order.
    """
    verts = []
    curves = []
    degree = None
    with open(path) as handle:
        for raw in handle:
            parts = raw.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "deg":
                degree = int(parts[1])
            elif parts[0] == "curv":
                idx = [int(i) for i in parts[3:]]  # skip the two u-range floats
                pts = [Point(*verts[i - 1]) for i in idx]
                if len(pts) >= 2 and pts[0].distance_to_point(pts[-1]) < 1e-6:
                    pts = pts[:-1]
                curves.append((degree, pts))
    return curves


def fit_circle(pts):
    """Centre and radius of a control polygon that samples a circle.

    The control points are symmetric about the centre, so their mean is the centre;
    the on-circle control points are the nearest ones, so the min radius is the true
    circle radius (the corner control points sit farther out)."""
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    r = min(math.hypot(p[0] - cx, p[1] - cy) for p in pts)
    return Point(cx, cy, 0.0), r


def closed(pts, z=0.0):
    """A closed :class:`compas.geometry.Polyline` at height ``z`` from corner points."""
    ring = [[p[0], p[1], z] for p in pts]
    return Polyline(ring + [ring[0]])


def load_step_meshes(path, scale=1000.0):
    """Load the STEP solids as COMPAS meshes, scaled from metres to millimetres.

    Uses trimesh (OCCT via cascadio) if available; returns ``[]`` otherwise so the
    tool-paths still build without the visualisation."""
    try:
        import trimesh
    except Exception:
        print("[step] trimesh not available -- skipping geometry.stp visualisation")
        return []
    scene = trimesh.load(str(path))
    geometries = scene.geometry.values() if hasattr(scene, "geometry") else [scene]
    meshes = []
    for geometry in geometries:
        if len(geometry.vertices) < 3 or len(geometry.faces) == 0:
            continue
        vertices = [[v[0] * scale, v[1] * scale, v[2] * scale] for v in geometry.vertices]
        faces = [[int(i) for i in face] for face in geometry.faces]
        meshes.append(Mesh.from_vertices_and_faces(vertices, faces))
    return meshes


# ------------------------------------------------------------------ #
# Read the cut curves and sort them by what machines them.
# ------------------------------------------------------------------ #
rectangles = [pts for _deg, pts in parse_obj_curves(conn_dir / "Mill_6mm.obj")]  # 6mm facing quads
ramp_curves = parse_obj_curves(conn_dir / "Ramp_2mm.obj")
outer_polys = [pts for deg, pts in ramp_curves if deg == 1]  # outer polygons -> offset OUTWARD
holes = [fit_circle(pts) for deg, pts in ramp_curves if deg == 2]  # circles -> holes, cut from inside

# ------------------------------------------------------------------ #
# 6mm tool: face each top rectangle flat at its own Z.
# ------------------------------------------------------------------ #
surfacings = []
for rect in rectangles:
    tp = toolpath_2d_surfacing.from_quad(rect, RADIUS_MILL, safe_z=Z_SAFE, stepover=STEPOVER, direction=DIRECTION)
    if tp is not None:
        surfacings.append(tp)

# ------------------------------------------------------------------ #
# 2mm tool: ramp every contour through the plate. Outer polygons are grown OUTWARD
# by the tool radius (Clipper2, MITER join -- convex corners come out as SHARP
# polygonal points, not faceted arcs; a round tool traces a sharp external corner
# fine, so the toolpath needs no rounding) and the tool rides the waste side; the
# profile is lifted to the stock top and ramped straight down through the material.
# A round tool still cannot reach INTO an inside (concave) corner, so ``notch=`` cuts
# a dogbone overcut there -- ``notch_flip=True`` targets those concave corners of the
# part outline (an island), not the convex ones -- so a mating square tab still seats.
# ------------------------------------------------------------------ #
contour_ramps = []
for poly in outer_polys:
    grown = offset_polyline(closed(poly), RADIUS_RAMP, join_type="miter", miter_limit=MITER_LIMIT)[0]
    grown = grown.transformed(Translation.from_vector([0.0, 0.0, STOCK_TOP]))  # to the mouth
    contour_ramps.append(
        toolpath_2d_ramp(grown, Vector(0.0, 0.0, -THROUGH), step=DOC, safe_z=Z_SAFE, offset=0.0, direction=DIRECTION, pocket=False, notch=RADIUS_RAMP, notch_flip=True)
    )

# Holes: a flat mill helical-drills each one, its centre orbiting inset toward the
# axis by the tool radius so the edge just reaches the wall (inside-offset).
hole_drills = []
for center, radius in holes:
    axis = Line(Point(center[0], center[1], STOCK_TOP), Point(center[0], center[1], -OVERCUT))
    hole_drills.append(toolpath_2d_drill(axis, radius, RADIUS_RAMP * 2, floor=-OVERCUT, safe_z=Z_SAFE))

# ------------------------------------------------------------------ #
# Two .nc files, one per tool: SETUP A = the 6mm facing, SETUP B = the 2mm contours
# + holes (run A, swap the bit, run B).
# ------------------------------------------------------------------ #
group_6mm = surfacings
group_2mm = contour_ramps + hole_drills
toolpaths = group_6mm + group_2mm

post_a = Postprocessor(tool=TOOL_MILL, tool_number=1, feed=300, spindle_speed=10000, coolant="air", material="Wood", program="Connectors setup A (6mm facing)")
post_a.write(conn_dir / "connectors_a_6mm.nc", toolpath_merge(*group_6mm))

post_b = Postprocessor(tool=TOOL_RAMP, tool_number=2, feed=250, spindle_speed=10000, coolant="air", material="Wood", program="Connectors setup B (2mm contours + holes)")
post_b.write(conn_dir / "connectors_b_2mm.nc", toolpath_merge(*group_2mm))

# ------------------------------------------------------------------ #
# Viewer
# ------------------------------------------------------------------ #
viewer = make_viewer(data_dir)
scene = TeeScene(viewer.scene)

solids = scene.add_group("connectors_geometry")
for index, mesh in enumerate(load_step_meshes(conn_dir / "geometry.stp")):
    solids.add(triangulated(mesh), name=f"solid_{index}", color=GREY, hide_coplanaredges=True)

curves = scene.add_group("cut_curves")
for index, rect in enumerate(rectangles):
    curves.add(closed(rect, z=STOCK_TOP), name=f"mill_rect_{index}", color=BLUE)
for index, poly in enumerate(outer_polys):
    curves.add(closed(poly), name=f"outer_{index}", color=GREEN)
for index, (center, radius) in enumerate(holes):
    ring = [Point(center[0] + radius * math.cos(t), center[1] + radius * math.sin(t), 0.0) for t in [i / 48 * 2 * math.pi for i in range(49)]]
    curves.add(Polyline(ring), name=f"hole_{index}", color=RED)

# The actual tool-centre paths as polylines -- orange for the 6mm facing, purple for
# the 2mm contours/holes. Keep the LIVE objects so the selected one can turn red; also
# record each into the bundle.
paths = scene.add_group("toolpaths")
path_objs, path_colors = [], []
for index, tp in enumerate(toolpaths):
    color = ORANGE if index < len(group_6mm) else PURPLE
    path_objs.append(paths._live.add(tp.path, name=f"path_{index}", color=color))
    paths._rec.add(tp.path, name=f"path_{index}", color=color)
    path_colors.append(color)

dump_bundle(scene, data_dir / "connectors_fab_rhino.json")

# Two sliders (live viewer only): one selects the tool-path by id (turning it RED),
# the other scrubs the cutter along it. 6mm mill for the facing paths, 2mm for the rest.
sim_entries = [(TOOL_MILL, tp.path) for tp in group_6mm] + [(TOOL_RAMP, tp.path) for tp in group_2mm]
if hasattr(viewer, "ui"):
    add_toolpath_slider(viewer, sim_entries, path_objs, path_colors)
viewer.show()
