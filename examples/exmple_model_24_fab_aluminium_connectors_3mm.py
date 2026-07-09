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
from compas_cnc import toolpath_merge
from compas_cnc.dxf import load_dxf
from compas_cnc.tools import Tool
from compas_cnc.tools import add_toolpath_slider

# Connectors machined flat from 3mm ALUMINIUM sheet with a SINGLE 3mm mill
# (data/custom_tool_path_connectors_aluminium_3mm/). ONE combined tool-path file
# (`custom_tool_path_connectors_aluminium_3mm.obj`) carries three kinds of entity:
#   * degree-1 curves  -> the outer part CONTOURS, RAMPED through the 3mm sheet with the
#     tool on the waste side.
#   * degree-2 circles -> HOLES, helical-(spiral-)DRILLED: the tool centre spirals down a
#     helix so a 3mm mill can bore a 5mm hole it could not plunge straight into.
#   * standalone `p` points -> hold-down TAB markers where the ramp lifts a little,
#     leaving an uncut bridge so the freed part stays on the bed.
# Aluminium tuning: DOC (stepdown per pass) is 0.05mm -- for the ramps that is the
# vertical descent per pass, and for the spiral holes it is the helix PITCH (drop per
# revolution), so every plunging cut takes a very light 0.05mm bite.
# `*_geometry.obj` is trimmed-NURBS (no mesh faces); a `.stp` STEP solid ships alongside
# and is rendered IF a STEP backend (compas_occ) is installed, else the cut curves stand
# in for the parts.

GREY = (0.80, 0.80, 0.80)
RED = (0.90, 0.20, 0.20)
BLUE = (0.20, 0.40, 0.90)
GREEN = (0.20, 0.70, 0.30)
PURPLE = (0.60, 0.20, 0.90)  # contour/hole tool-paths
YELLOW = (0.98, 0.85, 0.10)  # hold-down tab markers

TOOL_DIAMETER = 3.0  # 3mm mill for the 3mm aluminium
RADIUS = TOOL_DIAMETER / 2.0  # 1.5 -- the offset every operation insets/grows by

# Z model of the stock (mm). The part contours and hole circles sit on the Z=0 plane
# (bottom == bed == contour plane); the 3mm aluminium sheet rises to Z=+3 (STOCK_TOP).
# Every cut fills exactly that range -- start at the stock TOP and go down to the part
# BOTTOM at Z=0, and NEVER below (nothing cuts into the bed), so the machined depth == the
# sheet.
THICKNESS = 3.0  # 3mm aluminium
STOCK_TOP = THICKNESS  # ramp/drill start here (top of the sheet)
PART_BOTTOM = 0.0  # contour plane -> part bottom == bed top (Z=0); nothing cuts below this
OVERCUT = 0.0  # keep the floor AT the geometry bottom -- do not cut into the bed
THROUGH = (STOCK_TOP - PART_BOTTOM) + OVERCUT  # ramp/drill descent depth (3.0)
FLOOR = STOCK_TOP - THROUGH  # deepest world-Z reached (0.0 == part bottom == bed top)

TAB_BRIDGE = 0.5  # uncut bridge thickness left at each tab (thin, so it snaps off cleanly)
TAB_LIFT = OVERCUT + TAB_BRIDGE  # tab-top height ABOVE the floor
TAB_WIDTH = 3.0  # flat span of each tab along the cut

DOC = 0.05  # depth of cut per pass -- ramp stepdown AND helix pitch (fine aluminium bite)
DIRECTION = "climb"  # one-directional milling (CW / M3 cutter)
MITER_LIMIT = 4.0  # convex corners offset to SHARP mitered points (no arcs)
Z_SAFE = 25.0

data_dir = pathlib.Path(__file__).parent.parent / "data"
conn_dir = data_dir / "custom_tool_path_connectors_aluminium_3mm"
toolpath_obj = conn_dir / "custom_tool_path_connectors_aluminium_3mm.obj"
geometry_dm = conn_dir / "custom_tool_path_connectors_aluminium_3mm.3dm"  # rendered if present
geometry_stp = conn_dir / "custom_tool_path_connectors_aluminium_3mm.stp"  # rendered if OCC present

TOOL = Tool(TOOL_DIAMETER, 30.0, name="mill_3mm")  # single mill for every operation


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


def helix_ramp_angle(hole_radius, tool_diameter, pitch):
    """Helix descent angle (radians) that drops ``pitch`` mm per revolution.

    The tool centre orbits a helix of radius ``hole_radius - tool_radius``; per turn it
    descends ``2*pi*helix_radius * tan(angle)``. Solving for a target vertical PITCH per
    turn gives ``atan2(pitch, circumference)`` -- so the helix "steps down" ``pitch`` mm
    each revolution, the spiral analogue of the ramp's per-pass stepdown."""
    helix_radius = max(1e-6, hole_radius - tool_diameter / 2.0)
    circumference = 2.0 * math.pi * helix_radius
    return math.atan2(pitch, circumference)


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


def load_geometry_meshes(dm_path, stp_path):
    """Part solids as COMPAS meshes: from a Rhino ``.3dm``, else a STEP ``.stp``.

    The shipped ``*_geometry.obj`` is trimmed-NURBS (no ``f`` faces) so it cannot be
    meshed directly. A ``.3dm`` (via ``rhino3dm``) carries render meshes; a ``.stp`` (via
    a compas Brep backend such as ``compas_occ``) can be tessellated. Both are best-effort
    -- if neither the file nor its reader is present this returns ``[]`` and the viewer
    falls back to the cut curves."""
    meshes = _meshes_from_3dm(dm_path)
    if meshes:
        return meshes
    return _meshes_from_step(stp_path)


def _meshes_from_3dm(dm_path):
    if not dm_path.exists():
        return []
    try:
        import rhino3dm
    except Exception:
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


def _meshes_from_step(stp_path):
    if not stp_path.exists():
        return []
    try:
        from compas.geometry import Brep

        brep = Brep.from_step(str(stp_path))
        vertices, faces = brep.to_tesselation()  # (Mesh) or (vertices, faces) per backend
        if isinstance(vertices, Mesh):
            return [vertices]
        return [Mesh.from_vertices_and_faces(vertices, faces)]
    except Exception as exc:
        print(f"[geometry] STEP not rendered ({type(exc).__name__}); install compas_occ to see solids")
        return []


# ------------------------------------------------------------------ #
# Read the single combined tool-path file: contours, holes, and tab markers.
# ------------------------------------------------------------------ #
# `len(pts) >= 3` drops any degenerate 1-2 point curve a messy export might sneak in
# (a contour needs >= 3 control points); the real curves are unaffected.
curves_in = parse_obj_curves(toolpath_obj)
outer_polys = [pts for deg, pts in curves_in if deg == 1 and len(pts) >= 3]  # outer contours -> ramp
holes = [fit_circle(pts) for deg, pts in curves_in if deg == 2]  # circles -> spiral drill
# The export carries an extra 0,0,0 REFERENCE point -- drop it; the rest are tab markers.
stops = [p for p in parse_obj_points(toolpath_obj) if p.distance_to_point(Point(0.0, 0.0, 0.0)) > 1e-6]

# Each stop-point sits on one contour's outline -> group by nearest contour.
tabs_per_contour = [[] for _ in outer_polys]
for stop in stops:
    tabs_per_contour[nearest_contour(stop, outer_polys)].append(stop)

# ------------------------------------------------------------------ #
# Machined LAST (frees the parts, so it must follow the holes): ramp every contour
# through the 3mm sheet. Contours grow OUTWARD by the tool radius (Clipper2, MITER join
# -> sharp mitered corners) so the tool rides the waste side; the profile is lifted to
# STOCK_TOP and ramped straight down to FLOOR in 0.05mm steps. A round tool cannot reach
# INTO a concave corner, so ``notch=`` cuts a dogbone there (``notch_flip=True`` targets
# the part outline's concave corners). At each STOP-POINT the ramp leaves an uncut
# hold-down TAB (``tabs=``).
# ------------------------------------------------------------------ #
contour_ramps = []
for index, poly in enumerate(outer_polys):
    grown = offset_polyline(closed(poly), RADIUS, join_type="miter", miter_limit=MITER_LIMIT)[0]
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
            notch=RADIUS,
            notch_flip=True,
            tabs=tabs_per_contour[index],
            tab_height=TAB_LIFT,
            tab_width=TAB_WIDTH,
        )
    )

# Machined FIRST (while the parts are still solidly fixed to the sheet, before any
# contour frees them): spiral-drill each hole. The tool centre orbits a helix inset
# toward the axis by the tool radius so the edge just reaches the wall, descending at a
# 0.05mm PITCH per revolution (``ramp_angle`` solved from that pitch).
hole_drills = []
for center, radius in holes:
    axis = Line(Point(center[0], center[1], STOCK_TOP), Point(center[0], center[1], FLOOR))
    ramp_angle = helix_ramp_angle(radius, TOOL_DIAMETER, DOC)
    hole_drills.append(toolpath_2d_drill(axis, radius, TOOL_DIAMETER, ramp_angle=ramp_angle, floor=FLOOR, safe_z=Z_SAFE, direction=DIRECTION))

# ------------------------------------------------------------------ #
# ONE .nc, one 3mm tool, one merged toolpath, in machining order: spiral-DRILL the holes
# while every part is still fixed, then RAMP the freeing contours LAST so no part comes
# loose before its holes are bored -- a single program, no bit change.
# ------------------------------------------------------------------ #
group_cut = hole_drills + contour_ramps  # holes FIRST, part-freeing contours LAST
toolpaths = group_cut

post = Postprocessor(
    tool=TOOL,
    tool_number=1,
    feed=400,
    spindle_speed=10000,
    coolant="air",
    material="Aluminum",
    program="Aluminium connectors 3mm (3mm: spiral holes + contours, hold-down tabs)",
)
post.write(conn_dir / "connectors_aluminium_3mm.nc", toolpath_merge(*toolpaths))

# ------------------------------------------------------------------ #
# Viewer
# ------------------------------------------------------------------ #
viewer = make_viewer(data_dir)
scene = TeeScene(viewer.scene)

solids = scene.add_group("connectors_geometry")
for index, mesh in enumerate(load_geometry_meshes(geometry_dm, geometry_stp)):
    solids.add(triangulated(mesh), name=f"solid_{index}", color=GREY, hide_coplanaredges=True)

# The CNC table (Carvera Air 300x200 work area) from the DXF, for context, seated on the
# part-bottom plane (Z=0).
DROP = Translation.from_vector([0.0, 0.0, PART_BOTTOM])
table_group = scene.add_group("cnc_table")
for index, curve in enumerate(load_dxf(data_dir / "cnc_table_holes.dxf")):
    table_group.add(curve.transformed(DROP), name=f"table_{index}", color=BLUE)

curves = scene.add_group("cut_curves")
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

# The actual tool-centre paths as polylines (purple -- holes and contours, all 3mm). Keep
# the LIVE objects so the selected one can turn red; also record each into the bundle.
paths = scene.add_group("toolpaths")
path_objs, path_colors = [], []
for index, tp in enumerate(toolpaths):
    color = PURPLE
    path_objs.append(paths._live.add(tp.path, name=f"path_{index}", color=color))
    paths._rec.add(tp.path, name=f"path_{index}", color=color)
    path_colors.append(color)

dump_bundle(scene, conn_dir / "custom_tool_path_connectors_aluminium_3mm.json")

# Two sliders (live viewer only): one selects the tool-path by id (turning it RED), the
# other scrubs the cutter along it. A single 3mm mill drives every path.
sim_entries = [(TOOL, tp.path) for tp in toolpaths]
if hasattr(viewer, "ui"):
    add_toolpath_slider(viewer, sim_entries, path_objs, path_colors)
viewer.show()
