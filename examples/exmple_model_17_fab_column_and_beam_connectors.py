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
from compas_cnc.dxf import load_dxf
from compas_cnc.tools import Tool
from compas_cnc.tools import add_toolpath_slider

# Column-and-beam connectors, machined flat from stock in TWO ordered steps with a
# SINGLE 2mm mill (data/custom_tool_path_connectors/):
#   1. the 2mm mill SURFACES the MATERIAL (`2_surfacing_6mm.obj`, quads at Z=0 and
#      Z=1) -- the stepped top faces of the stock.
#   2. the same 2mm mill CUTS the parts (`3_ramp_with_stop_points.obj`): the degree-1
#      curves are the outer contours (RAMPED through the stock, tool on the waste side)
#      and the degree-2 rational curves are the circles = holes (helical-DRILLED). That
#      file also carries standalone `p` STOP-POINTS -- markers where the ramp must NOT
#      cut all the way through: the tool lifts a little there, leaving ~0.5mm uncut
#      HOLD-DOWN TABS so the freed part cannot fly off the table.
# `4_geometry.obj` is the trimmed-NURBS solids (no mesh faces), so the viewer instead
# pulls the render meshes out of the sibling `.3dm` for context.

GREY = (0.80, 0.80, 0.80)
RED = (0.90, 0.20, 0.20)
BLUE = (0.20, 0.40, 0.90)
GREEN = (0.20, 0.70, 0.30)
ORANGE = (0.95, 0.55, 0.10)  # facing tool-paths (2mm)
PURPLE = (0.60, 0.20, 0.90)  # contour/hole tool-paths (2mm)
YELLOW = (0.98, 0.85, 0.10)  # hold-down tab markers

RADIUS_MILL = 1.0  # 2mm-DIAMETER facing tool -> radius (the offset) is half = 1.0
RADIUS_RAMP = 1.0  # 2mm-DIAMETER contour/hole tool -> radius (the offset) is half = 1.0
STEPOVER = RADIUS_MILL * 2 / 4  # facing pass spacing

# Z model of the stock (mm), taken from the exported geometry: the parts span Z=0
# (bottom == the table / contour plane) up to Z=+2 (the tallest tops; the stepped top
# faces sit at Z=1 and Z=2). The cut fills exactly that range -- start at the stock TOP
# and go down to the part BOTTOM at Z=0, and NEVER deeper (nothing cuts below zero into
# the table), so the machined depth matches the geometry.
STOCK_TOP = 2.0  # tallest part top -- ramp/drill start here (air above shorter parts is harmless)
PART_BOTTOM = 0.0  # contour plane -> part bottom == table top (Z=0); nothing cuts below this
OVERCUT = 0.0  # keep the floor AT the geometry bottom -- do not cut below it
THROUGH = (STOCK_TOP - PART_BOTTOM) + OVERCUT  # ramp/drill descent depth (2.0)
FLOOR = STOCK_TOP - THROUGH  # deepest world-Z reached (0.0 == part bottom == table top)

TAB_BRIDGE = 1.0  # bridge thickness from the floor up -- top reaches mid-thickness (2mm stock)
TAB_LIFT = OVERCUT + TAB_BRIDGE  # tab height ABOVE the floor: skip the overcut + leave the bridge
TAB_WIDTH = 3.0  # flat span of each tab along the cut

DOC = 0.5  # depth of cut per pass (ramp/helix stepdown) -- small, the 2mm tool is weak
DIRECTION = "climb"  # one-directional milling (CW / M3 cutter)
MITER_LIMIT = 4.0  # convex corners offset to SHARP mitered points (no arcs); bevels past this length ratio
Z_SAFE = 25.0

data_dir = pathlib.Path(__file__).parent.parent / "data"
conn_dir = data_dir / "custom_tool_path_connectors"

TOOL = Tool(RADIUS_RAMP * 2, 30.0, name="mill_2mm")  # single 2mm mill for every operation


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


def parse_obj_points(path):
    """Standalone ``p`` points of a Rhino OBJ -> ``[Point, ...]``.

    ``p`` lines reference global 1-based vertex indices, the same ``v`` table the
    ``curv`` lines use. These are the ramp STOP-POINTS (tab markers).
    """
    verts = []
    points = []
    with open(path) as handle:
        for raw in handle:
            parts = raw.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "p":
                for i in parts[1:]:
                    points.append(Point(*verts[int(i) - 1]))
    return points


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


def _pt_seg_dist(p, a, b):
    """XY distance from point ``p`` to segment ``a``-``b``."""
    ax, ay = a[0], a[1]
    dx, dy = b[0] - ax, b[1] - ay
    ll = dx * dx + dy * dy
    if ll < 1e-18:
        return math.hypot(p[0] - ax, p[1] - ay)
    u = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / ll))
    return math.hypot(p[0] - (ax + u * dx), p[1] - (ay + u * dy))


def nearest_contour(point, contours):
    """Index of the contour (list of corner points) whose outline ``point`` lies on."""
    def poly_dist(poly):
        ring = list(poly) + [poly[0]]
        return min(_pt_seg_dist(point, ring[i], ring[i + 1]) for i in range(len(ring) - 1))

    return min(range(len(contours)), key=lambda i: poly_dist(contours[i]))


def marker(pt, size=1.2):
    """A small ``+`` polyline centred on ``pt`` (for viewer tab markers)."""
    x, y, z = pt[0], pt[1], pt[2]
    return Polyline([[x - size, y, z], [x + size, y, z], [x, y, z], [x, y - size, z], [x, y + size, z]])


def load_geometry_meshes(dm_path):
    """Pull the part solids out of the Rhino ``.3dm`` as COMPAS meshes.

    ``4_geometry.obj`` is a trimmed-NURBS OBJ (no ``f`` faces) so it cannot be meshed
    directly. The sibling ``.3dm`` stores the same Breps WITH render meshes, so read
    those via ``rhino3dm`` and merge them, dropping the machine/table (it spans
    +/-3000 and up to Z=3014 -- far outside the part region). Returns ``[]`` if
    ``rhino3dm`` is missing or the read fails, so the tool-paths still build."""
    try:
        import rhino3dm
    except Exception:
        print("[geometry] rhino3dm not available -- skipping solids")
        return []
    try:
        model = rhino3dm.File3dm.Read(str(dm_path))
    except Exception as exc:
        print(f"[geometry] could not read {dm_path}: {exc}")
        return []
    mt = rhino3dm.MeshType.Any
    vertices, faces = [], []
    for obj in model.Objects:
        geom = obj.Geometry
        if type(geom).__name__ != "Brep":
            continue
        bv, bf = [], []
        for face in geom.Faces:
            rm = face.GetMesh(mt)
            if rm is None or len(rm.Faces) == 0:
                continue
            base = len(bv)
            for i in range(len(rm.Vertices)):
                p = rm.Vertices[i]
                bv.append((p.X, p.Y, p.Z))
            for j in range(len(rm.Faces)):
                a, b, c, d = rm.Faces[j]
                bf.append([base + a, base + b, base + c] if c == d else [base + a, base + b, base + c, base + d])
        if not bf:
            continue
        xs = [p[0] for p in bv]
        ys = [p[1] for p in bv]
        zs = [p[2] for p in bv]
        if min(xs) < -50 or max(xs) > 400 or min(ys) < -50 or max(ys) > 400 or min(zs) < -5 or max(zs) > 20:
            continue  # machine / table -- not a part
        off = len(vertices)
        vertices.extend(bv)
        faces.extend([[off + i for i in f] for f in bf])
    if not faces:
        return []
    try:
        return [Mesh.from_vertices_and_faces(vertices, faces)]
    except Exception as exc:
        print(f"[geometry] mesh build failed: {exc}")
        return []


# ------------------------------------------------------------------ #
# Read the three tool-path files.
# ------------------------------------------------------------------ #
# `len(pts) >= 3` drops any degenerate 1-2 point curve a messy export might sneak in
# (a quad/contour needs >= 3 control points); the real curves are unaffected.
mat_quads = [pts for _deg, pts in parse_obj_curves(conn_dir / "2_surfacing_6mm.obj") if len(pts) >= 3]

ramp_curves = parse_obj_curves(conn_dir / "3_ramp_with_stop_points.obj")
outer_polys = [pts for deg, pts in ramp_curves if deg == 1 and len(pts) >= 3]  # outer contours -> ramp
holes = [fit_circle(pts) for deg, pts in ramp_curves if deg == 2]  # circles -> drill
# The export carries an extra 0,0,0 REFERENCE point -- drop it; the rest are tab markers.
stops = [p for p in parse_obj_points(conn_dir / "3_ramp_with_stop_points.obj") if p.distance_to_point(Point(0.0, 0.0, 0.0)) > 1e-6]

# Each stop-point sits on one contour's outline -> group by nearest contour.
tabs_per_contour = [[] for _ in outer_polys]
for stop in stops:
    tabs_per_contour[nearest_contour(stop, outer_polys)].append(stop)

# ------------------------------------------------------------------ #
# 2mm tool, step 1: face each top rectangle of the MATERIAL at its own Z.
# ------------------------------------------------------------------ #
mat_surfacings = []
for quad in mat_quads:
    tp = toolpath_2d_surfacing.from_quad(quad, RADIUS_MILL, safe_z=Z_SAFE, stepover=STEPOVER, direction=DIRECTION)
    if tp is not None:
        mat_surfacings.append(tp)

# ------------------------------------------------------------------ #
# 2mm tool, step 2a: ramp every contour through the stock. Contours are grown OUTWARD
# by the tool radius (Clipper2, MITER join -> sharp polygonal corners; a round tool
# traces a sharp external corner fine) so the tool rides the waste side; the profile
# is lifted to STOCK_TOP and ramped straight down to FLOOR. A round tool cannot reach
# INTO a concave corner, so ``notch=`` cuts a dogbone there (``notch_flip=True`` targets
# the part outline's concave corners). At each STOP-POINT the ramp leaves a ~0.5mm
# hold-down TAB (``tabs=``) so the freed part stays put.
# ------------------------------------------------------------------ #
contour_ramps = []
for index, poly in enumerate(outer_polys):
    grown = offset_polyline(closed(poly), RADIUS_RAMP, join_type="miter", miter_limit=MITER_LIMIT)[0]
    grown = grown.transformed(Translation.from_vector([0.0, 0.0, STOCK_TOP]))  # to the mouth
    contour_ramps.append(
        toolpath_2d_ramp(
            grown,
            Vector(0.0, 0.0, -THROUGH),
            step=DOC,
            safe_z=Z_SAFE,
            offset=0.0,
            direction=DIRECTION,
            pocket=False,
            notch=RADIUS_RAMP,
            notch_flip=True,
            tabs=tabs_per_contour[index],
            tab_height=TAB_LIFT,
            tab_width=TAB_WIDTH,
        )
    )

# 2mm tool, step 2b: helical-drill each hole, the tool centre orbiting inset toward the
# axis by the tool radius so the edge just reaches the wall (inside-offset).
hole_drills = []
for center, radius in holes:
    axis = Line(Point(center[0], center[1], STOCK_TOP), Point(center[0], center[1], FLOOR))
    hole_drills.append(toolpath_2d_drill(axis, radius, RADIUS_RAMP * 2, floor=FLOOR, safe_z=Z_SAFE))

# ------------------------------------------------------------------ #
# ONE .nc, one 2mm tool, one merged toolpath: material facing first, then the contours
# + holes -- a single program, no bit change, run start to finish in one go.
# ------------------------------------------------------------------ #
group_surface = mat_surfacings
group_cut = contour_ramps + hole_drills
toolpaths = group_surface + group_cut

post = Postprocessor(tool=TOOL, tool_number=1, feed=400, spindle_speed=10000, coolant="air", material="Wood", program="Connectors (2mm: surfacing + contours + holes, hold-down tabs)")
post.write(conn_dir / "connectors_2mm.nc", toolpath_merge(*toolpaths))

# ------------------------------------------------------------------ #
# Viewer
# ------------------------------------------------------------------ #
viewer = make_viewer(data_dir)
scene = TeeScene(viewer.scene)

solids = scene.add_group("connectors_geometry")
for index, mesh in enumerate(load_geometry_meshes(conn_dir / "custom_tool_path_connectors.3dm")):
    solids.add(triangulated(mesh), name=f"solid_{index}", color=GREY, hide_coplanaredges=True)

# The CNC table (Carvera Air 300x200 work area, outline + 57 mounting holes) from the
# DXF, for context. The circles are stored with a flipped OCS extrusion (0,0,-1) -- the
# raw file reads them at negative X -- but ezdxf resolves OCS->WCS on load, so they come
# back correctly in +X (7.5..295), seated under the parts (X 100..200, Y 31..125). The
# table plane and the part BOTTOM are both Z=0, so no drop is needed (PART_BOTTOM=0.0).
DROP = Translation.from_vector([0.0, 0.0, PART_BOTTOM])  # seat the table on the part-bottom plane
table_group = scene.add_group("cnc_table")
for index, curve in enumerate(load_dxf(data_dir / "cnc_table_holes.dxf")):
    table_group.add(curve.transformed(DROP), name=f"table_{index}", color=BLUE)

curves = scene.add_group("cut_curves")
for index, quad in enumerate(mat_quads):
    curves.add(closed(quad, z=quad[0][2]), name=f"mill_quad_{index}", color=BLUE)
for index, poly in enumerate(outer_polys):
    curves.add(closed(poly, z=PART_BOTTOM), name=f"contour_{index}", color=GREEN)
for index, (center, radius) in enumerate(holes):
    ring = [Point(center[0] + radius * math.cos(t), center[1] + radius * math.sin(t), PART_BOTTOM) for t in [i / 48 * 2 * math.pi for i in range(49)]]
    curves.add(Polyline(ring), name=f"hole_{index}", color=RED)

# Hold-down tab markers (the ramp snaps each stop-point onto its own centre-path).
tab_group = scene.add_group("hold_down_tabs")
tab_id = 0
for tp in contour_ramps:
    for tab in tp.tabs:
        tab_group.add(marker(tab), name=f"tab_{tab_id}", color=YELLOW)
        tab_id += 1

# The actual tool-centre paths as polylines -- orange for the facing, purple for the
# contours/holes (all 2mm). Keep the LIVE objects so the selected one can turn red;
# also record each into the bundle.
paths = scene.add_group("toolpaths")
path_objs, path_colors = [], []
for index, tp in enumerate(toolpaths):
    color = ORANGE if index < len(group_surface) else PURPLE
    path_objs.append(paths._live.add(tp.path, name=f"path_{index}", color=color))
    paths._rec.add(tp.path, name=f"path_{index}", color=color)
    path_colors.append(color)

dump_bundle(scene, data_dir / "custom_tool_path_connectors.json")

# Two sliders (live viewer only): one selects the tool-path by id (turning it RED),
# the other scrubs the cutter along it. A single 2mm mill drives every path.
sim_entries = [(TOOL, tp.path) for tp in toolpaths]
if hasattr(viewer, "ui"):
    add_toolpath_slider(viewer, sim_entries, path_objs, path_colors)
viewer.show()
