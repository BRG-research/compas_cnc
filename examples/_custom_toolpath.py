"""Shared helpers for the custom-toolpath fabrication examples (models 18-22).

Every ``data/custom_toolpath_*`` folder ships the same kind of Rhino OBJ exports as
``exmple_model_17`` -- degree-1 polylines (surfacing quads / ramp contours), degree-2
rational circles (drill holes), a ``p`` origin marker at 0,0,0 -- plus a ``.stp`` solid
that ``_convert_stp_to_mesh.py`` has already turned into ``*_geometry.obj``. The OBJ
FILENAMES encode the tool and operation, e.g. ``..._6mm_surfacing.obj`` (Ø6 mm) or
``..._3_175mm_ramp.obj`` (Ø3.175 mm). This module centralises the OBJ parsing, the
tool-diameter sniffing, the toolpath builders, and the viewer/NC boilerplate so each
example only has to describe the parts specific to its folder.
"""

import math
import os
import pathlib
import re

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
from compas_cnc import toolpath_2d_hatch
from compas_cnc import toolpath_2d_ramp
from compas_cnc import toolpath_2d_surfacing
from compas_cnc import toolpath_merge
from compas_cnc.dxf import load_dxf
from compas_cnc.tools import Tool
from compas_cnc.tools import add_toolpath_slider

# ------------------------------------------------------------------ #
# Palette (shared with exmple_model_17 / _12).
# ------------------------------------------------------------------ #
GREY = (0.80, 0.80, 0.80)
RED = (0.90, 0.20, 0.20)
BLUE = (0.20, 0.40, 0.90)
GREEN = (0.20, 0.70, 0.30)
ORANGE = (0.95, 0.55, 0.10)  # 6mm tool-paths
PURPLE = (0.60, 0.20, 0.90)  # 3.175mm tool-paths
YELLOW = (0.98, 0.85, 0.10)  # hold-down tab markers

# ------------------------------------------------------------------ #
# Machining constants (same regime as exmple_model_17: weak small tools, climb,
# sharp mitered offsets, a modest safe height).
# ------------------------------------------------------------------ #
DOC = 0.5  # depth of cut per ramp/helix stepdown (mm) -- small, the 3.175mm tool is weak
DIRECTION = "climb"  # one-directional milling (CW / M3 cutter)
TAB_HEIGHT = 1.0  # uncut bridge left at each tab, measured up from the floor (mm) -> 1mm tabs
TAB_WIDTH = 3.0  # flat span of each tab along the cut (mm)
MITER_LIMIT = 4.0  # convex corners offset to sharp mitered points
Z_SAFE = 25.0
PART_BOTTOM = 0.0  # the ramp/drill OBJs sit at Z=0 -- the table / contour plane
FEED = 400
SPINDLE = 10000

data_dir = pathlib.Path(__file__).parent.parent / "data"


# ================================================================== #
# OBJ parsing (verbatim from exmple_model_17).
# ================================================================== #
def parse_obj_curves(path):
    """Rhino NURBS-curve OBJ -> ``[(degree, [Point, ...]), ...]``.

    ``deg 1`` curves are polylines, ``deg 2`` rational curves are circles. The
    repeated closing control point is dropped.
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
    """Standalone ``p`` points of a Rhino OBJ -> ``[Point, ...]`` (the origin marker)."""
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
    """Centre and radius of a control polygon that samples a circle (see exmple_model_17)."""
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    r = min(math.hypot(p[0] - cx, p[1] - cy) for p in pts)
    return Point(cx, cy, 0.0), r


def closed(pts, z=0.0):
    """A closed :class:`compas.geometry.Polyline` at height ``z`` from corner points."""
    ring = [[p[0], p[1], z] for p in pts]
    return Polyline(ring + [ring[0]])


def poly_at(pts, z=None):
    """Closed polyline through ``pts`` at their own Z (or forced ``z``) for the viewer."""
    if z is None:
        ring = [[p[0], p[1], p[2]] for p in pts]
    else:
        ring = [[p[0], p[1], z] for p in pts]
    return Polyline(ring + [ring[0]])


def marker(pt, size=1.2):
    """A small ``+`` polyline centred on ``pt`` (viewer origin / tab marker)."""
    x, y, z = pt[0], pt[1], pt[2]
    return Polyline([[x - size, y, z], [x + size, y, z], [x, y, z], [x, y - size, z], [x, y + size, z]])


# ================================================================== #
# Folder / filename introspection.
# ================================================================== #
_OPERATIONS = ("paired_ramped", "surfacing", "ramp", "drill")


def tool_from_name(name):
    """Parse the tool from an OBJ filename -> :class:`Tool`.

    ``6mm`` -> Ø6, ``3_175mm`` / ``3.175mm`` -> Ø3.175. Returns ``(Tool, radius)``.
    """
    # Anchor to a segment boundary (start or ``_``) so ``part1_6mm`` reads as Ø6, not
    # Ø1.6, while ``3_175mm`` / ``3.175mm`` still read as Ø3.175 (``_`` = decimal point).
    match = re.search(r"(?:^|_)(\d+(?:[._]\d+)?)mm", name)
    if not match:
        raise ValueError(f"no tool diameter in {name!r}")
    diameter = float(match.group(1).replace("_", "."))
    label = f"flat_{match.group(1).replace('.', '_')}mm"
    return Tool(diameter, 30.0, name=label), diameter / 2.0


def classify_objs(folder):
    """Group a folder's toolpath OBJs by operation.

    The folder name is NOT trusted for the file prefix (e.g. ``inner_beams`` ships
    ``custom_toolpath_inner_ribs_*`` files), so classify purely by the operation
    suffix. Returns ``{operation: [pathlib.Path, ...]}`` (skips ``*_geometry.obj``).
    """
    groups = {op: [] for op in _OPERATIONS}
    for path in sorted((data_dir / folder).glob("*.obj")):
        stem = path.stem
        if stem.endswith("_geometry"):
            continue
        for op in _OPERATIONS:  # paired_ramped before ramp so it wins
            if stem.endswith(op):
                groups[op].append(path)
                break
    return groups


def geometry_mesh(folder):
    """Load the pre-converted STP mesh (``*_geometry.obj``) as a COMPAS ``Mesh``.

    Returns ``(mesh, z_max)`` or ``(None, 0.0)`` if it is missing (run
    ``_convert_stp_to_mesh.py`` with the ``occ`` env to (re)generate it).
    """
    candidates = sorted((data_dir / folder).glob("*_geometry.obj"))
    if not candidates:
        print(f"[geometry] {folder}: no *_geometry.obj -- run _convert_stp_to_mesh.py")
        return None, 0.0
    mesh = Mesh.from_obj(candidates[0])
    z_max = max(mesh.vertex_coordinates(v)[2] for v in mesh.vertices())
    return mesh, z_max


# ================================================================== #
# Toolpath builders.
# ================================================================== #
def surfacing_toolpaths(obj_paths, safe_z=Z_SAFE, direction=DIRECTION, start=None):
    """Build surfacing tool-paths that clear the INSIDE of each polygon.

    Degree-1 4-corner faces -> :meth:`toolpath_2d_surfacing.from_quad` (``incline=True``
    when the corners differ in Z, so the flat tool rides a tilted face). Any other
    polygon (an N-gon, e.g. outer_ribs part1) -> :class:`toolpath_2d_hatch`, a raster
    pocket inset by the tool radius. ``start`` chooses where each quad sweep BEGINS:
    ``None`` = the highest corner; an ``(x, y, z)`` point = the corner nearest it; a
    CALLABLE ``f(pts) -> point`` = computed per face from that face's OWN corners (e.g.
    its farthest-from-centre corner, so each face starts at its own outer end and faces on
    opposite sides sweep opposite ways). Returns ``(toolpaths, polygons)``.
    """
    toolpaths = []
    polygons = []
    for path in obj_paths:
        _tool, radius = tool_from_name(path.name)
        stepover = radius * 2 / 4  # quarter-diameter pass spacing
        for degree, pts in parse_obj_curves(path):
            if degree != 1 or len(pts) < 3:
                continue
            polygons.append(poly_at(pts))
            face_start = start(pts) if callable(start) else start
            if len(pts) == 4:
                z_range = max(p[2] for p in pts) - min(p[2] for p in pts)
                incline = z_range > 1e-6
                tp = toolpath_2d_surfacing.from_quad(
                    [list(p) for p in pts],
                    radius,
                    safe_z=safe_z,
                    stepover=stepover,
                    incline=incline,
                    # Tilted faces sweep the SHORTER direction (flip) so the passes run the
                    # short way and the tool does continuous inclined cutting instead of
                    # leaving stair-steps -- the same choice as exmple_model_12's plates.
                    flip=incline,
                    # Begin each sweep at `face_start` (per-face corner) -- else highest.
                    start=None if face_start is None else list(face_start),
                    direction=direction,
                )
            else:
                z = sum(p[2] for p in pts) / len(pts)
                tp = toolpath_2d_hatch(
                    closed(pts, z=z),
                    stepover,
                    radius=radius,
                    z=z,
                    safe_z=safe_z,
                    direction=direction,
                )
            if tp is not None:
                toolpaths.append(tp)
            else:
                print(f"[surfacing] {path.name}: a face was too small for the tool -- skipped")
    return toolpaths, polygons


def ramp_toolpaths(obj_paths, stock_top, safe_z=Z_SAFE, grow=True, tabs=0):
    """Ramp every degree-1 contour straight down through the stock.

    Each contour is grown OUTWARD by the tool radius (Clipper2 miter -> sharp corners)
    so the cutter rides the WASTE side and the finished part keeps nominal size, lifted
    to ``stock_top``, then ramped down to :data:`PART_BOTTOM`. The per-pass Z stepdown is
    the TOOL RADIUS. Concave corners get a dogbone (``notch``). ``grow=False`` skips the
    outward offset. ``tabs`` is the NUMBER of hold-down tabs to auto-place per contour
    (``0`` = none, the default; the ramp toolpath places them on the longest edges and
    leaves a :data:`TAB_HEIGHT`-mm uncut bridge). Returns ``(toolpaths, contours)`` --
    closed polylines at Z=0 for the viewer.
    """
    through = stock_top - PART_BOTTOM
    toolpaths = []
    contours = []
    for path in obj_paths:
        _tool, radius = tool_from_name(path.name)
        step = radius  # Z descent per ramp pass = tool RADIUS (half the diameter)
        for degree, pts in parse_obj_curves(path):
            if degree != 1 or len(pts) < 3:
                continue
            contours.append(closed(pts, z=PART_BOTTOM))
            if grow:
                ring = offset_polyline(closed(pts), radius, join_type="miter", miter_limit=MITER_LIMIT)[0]
            else:
                ring = closed(pts)
            ring = ring.transformed(Translation.from_vector([0.0, 0.0, stock_top]))
            toolpaths.append(
                toolpath_2d_ramp(
                    ring,
                    Vector(0.0, 0.0, -through),
                    step=step,
                    safe_z=safe_z,
                    offset=0.0,
                    direction=DIRECTION,
                    pocket=False,
                    notch=radius,
                    notch_flip=True,
                    # `tabs` is a COUNT -> the ramp auto-places that many hold-down tabs
                    # on the longest edges, each leaving a TAB_HEIGHT-mm uncut bridge.
                    tabs=tabs,
                    tab_height=TAB_HEIGHT,
                    tab_width=TAB_WIDTH,
                )
            )
    return toolpaths, contours


def drill_toolpaths(obj_paths, stock_top, safe_z=Z_SAFE):
    """Helical-drill every degree-2 circle from ``stock_top`` down to :data:`PART_BOTTOM`.

    Returns ``(toolpaths, holes)`` -- ``holes`` are ``(centre, radius)`` for the viewer.
    """
    toolpaths = []
    holes = []
    for path in obj_paths:
        tool, _radius = tool_from_name(path.name)
        for degree, pts in parse_obj_curves(path):
            if degree != 2 or len(pts) < 3:
                continue
            centre, hole_radius = fit_circle(pts)
            holes.append((centre, hole_radius))
            axis = Line(Point(centre[0], centre[1], stock_top), Point(centre[0], centre[1], PART_BOTTOM))
            toolpaths.append(toolpath_2d_drill(axis, hole_radius, tool.diameter, floor=PART_BOTTOM, safe_z=safe_z))
    return toolpaths, holes


# ================================================================== #
# Paired ramp -- inclined faces given as matched top/bottom rail loops (T-sections).
# ================================================================== #
class _Path:
    """Minimal tool-path wrapper exposing ``.path`` (a Polyline), so it merges and
    animates exactly like the built-in ``toolpath_2d_*`` objects."""

    def __init__(self, path):
        self.path = path


def _signed_area_xy(pts):
    """Signed XY area of a closed ring of points (sign = winding direction)."""
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i][0], pts[i][1]
        x2, y2 = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _centroid(pts):
    n = len(pts)
    return Point(sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n, sum(p[2] for p in pts) / n)


def _match_winding(top, bot):
    """Reorder ``bot`` so ``bot[i]`` corresponds to ``top[i]``.

    Flip ``bot`` if it winds the opposite way to ``top`` (so both traverse the loop the
    same sense), then roll its start index to the rotation that minimises the summed
    point-to-point distance -- the closest-point correspondence.
    """
    if _signed_area_xy(top) * _signed_area_xy(bot) < 0:
        bot = list(reversed(bot))
    n = len(bot)
    best_k, best = 0, float("inf")
    for k in range(n):
        s = sum(top[i].distance_to_point(bot[(i + k) % n]) for i in range(n))
        if s < best:
            best, best_k = s, k
    return [bot[(i + best_k) % n] for i in range(n)]


def _outward_offset(pts, dist):
    """Grow a closed ring OUTWARD by ``dist`` with a per-vertex MITER offset.

    Same effect as the ``offset_polyline(..., join_type="miter")`` the other ramps use to
    ride the waste side, but done per vertex so the point COUNT (and thus the top/bottom
    correspondence needed for the loft) is preserved. Each vertex slides along the
    bisector of its two edge outward-normals, scaled so the perpendicular stand-off from
    each edge is exactly ``dist`` (a true miter). Winding is read from the signed area, so
    "outward" is always away from the enclosed region regardless of CW/CCW.
    """
    n = len(pts)
    sign = 1.0 if _signed_area_xy(pts) > 0 else -1.0  # +1 for CCW (outward = right normal)

    def edge_normal(a, b):
        ex, ey = b[0] - a[0], b[1] - a[1]
        nx, ny = ey * sign, -ex * sign  # outward normal of edge a->b
        length = math.hypot(nx, ny)
        return (nx / length, ny / length) if length > 1e-12 else (0.0, 0.0)

    out = []
    for i in range(n):
        a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        n1 = edge_normal(a, b)
        n2 = edge_normal(b, c)
        bx, by = n1[0] + n2[0], n1[1] + n2[1]
        blen = math.hypot(bx, by)
        if blen < 1e-9:  # 180deg reversal -- fall back to one edge normal
            bx, by, blen = n1[0], n1[1], 1.0
        ux, uy = bx / blen, by / blen
        cos_half = ux * n1[0] + uy * n1[1]  # bisector . edge-normal = cos(half corner angle)
        scale = dist / cos_half if cos_half > 1e-3 else dist  # miter stand-off = dist
        out.append(Point(b[0] + ux * scale, b[1] + uy * scale, b[2]))
    return out


def _paired_ramp_helix(top, bot, top_z, bot_z, radius, safe_z, step):
    """Helical descent between two winding-matched rail loops, offset OUTWARD by ``radius``.

    Both rail loops are first grown OUTWARD by the tool radius (:func:`_outward_offset`),
    exactly like every other ramp, so the cutter rides the WASTE side of the inclined
    face and the finished part keeps nominal size. The tool then spirals down while the
    outline MORPHS from the grown ``top`` (at ``top_z``) into the grown ``bot`` (at
    ``bot_z``) -- riding the inclined ruled surface between the two waste-side edges.
    """
    n = len(top)
    otop = _outward_offset(top, radius)  # waste-side, point-for-point with `top`
    obot = _outward_offset(bot, radius)
    layers = max(1, int(math.ceil((top_z - bot_z) / step)))
    samples = layers * n  # one full loop per descent layer

    def ring_point(t, i):
        return Point(
            otop[i][0] * (1 - t) + obot[i][0] * t,
            otop[i][1] * (1 - t) + obot[i][1] * t,
            top_z * (1 - t) + bot_z * t,
        )

    start = ring_point(0.0, 0)
    pts = [Point(start[0], start[1], safe_z), start]
    for s in range(1, samples + 1):  # helix: descend a layer per loop
        pts.append(ring_point(s / samples, s % n))
    for i in range(n + 1):  # finishing lap around the bottom outline
        pts.append(ring_point(1.0, i % n))
    tail = pts[-1]
    pts.append(Point(tail[0], tail[1], safe_z))  # retract straight up
    return Polyline(pts)


def paired_ramp_toolpaths(obj_paths, safe_z=Z_SAFE, step=DOC):
    """Ramp inclined faces given as PAIRED top/bottom rail loops (the T-sections case).

    Each ``*_paired_ramped.obj`` holds closed rail loops at two heights: the TOP edge of
    an inclined face (higher Z) and its BOTTOM edge (Z=0). Every top loop is paired with
    the nearest bottom loop by centroid, their windings are matched point-to-point
    (:func:`_match_winding`), and the tool descends the OUTWARD-offset helix from
    :func:`_paired_ramp_helix` (both rails grown by the tool radius to the waste side,
    like every other ramp). Returns ``(toolpaths, contours)`` -- ``contours`` are the raw
    top and bottom rail loops for the viewer.
    """
    toolpaths = []
    contours = []
    for path in obj_paths:
        _tool, radius = tool_from_name(path.name)
        loops = [pts for degree, pts in parse_obj_curves(path) if degree == 1 and len(pts) >= 3]
        if not loops:
            continue
        zmean = [sum(p[2] for p in loop) / len(loop) for loop in loops]
        top_z, bot_z = max(zmean), min(zmean)
        tops = [loop for loop, z in zip(loops, zmean) if abs(z - top_z) < abs(z - bot_z)]
        bots = [loop for loop, z in zip(loops, zmean) if abs(z - top_z) >= abs(z - bot_z)]
        used = set()
        for top in tops:
            tc = _centroid(top)
            choices = [k for k in range(len(bots)) if k not in used and len(bots[k]) == len(top)]
            if not choices:
                print(f"[paired_ramp] {path.name}: no size-matching bottom loop for a top -- skipped")
                continue
            j = min(choices, key=lambda k: _centroid(bots[k]).distance_to_point(tc))
            used.add(j)
            bot = _match_winding(top, bots[j])
            toolpaths.append(_Path(_paired_ramp_helix(top, bot, top_z, bot_z, radius, safe_z, step)))
            contours.append(closed(top, z=top_z))
            contours.append(closed(bot, z=bot_z))
    return toolpaths, contours


# ================================================================== #
# NC output + viewer (one call from each example).
# ================================================================== #
def circle_polyline(centre, radius, z=PART_BOTTOM, n=48):
    """A closed circle polyline for the viewer."""
    ring = [Point(centre[0] + radius * math.cos(t), centre[1] + radius * math.sin(t), z) for t in [i / n * 2 * math.pi for i in range(n + 1)]]
    return Polyline(ring)


def finish(folder, program, groups, curves, mesh=None, markers=None):
    """Write one ``.nc`` per tool group, dump the viewer bundle, and show the viewer.

    Parameters
    ----------
    folder : str
        ``custom_toolpath_*`` folder name (also the ``.nc`` / ``.json`` basename).
    program : str
        Human-readable program name for the post-processor header.
    groups : list[tuple]
        Ordered ``(tool, suffix, [toolpaths], color)`` per tool. Each non-empty group
        writes ``<folder>_<suffix>.nc``; ``color`` paints its paths in the viewer.
    curves : list[tuple]
        ``(name, geometry, color)`` context curves (surfacing polys, contours, circles).
    mesh : :class:`compas.datastructures.Mesh`, optional
        The STP geometry mesh for solid context.
    markers : list[:class:`compas.geometry.Point`], optional
        Extra ``+`` markers (e.g. the origin) to draw.
    """
    out = data_dir / folder
    tool_number = 0
    ordered = []  # (tool, toolpath, color) in machining/animation order
    for tool, suffix, toolpaths, color in groups:
        if not toolpaths:
            continue
        tool_number += 1
        post = Postprocessor(
            tool=tool,
            tool_number=tool_number,
            feed=FEED,
            spindle_speed=SPINDLE,
            coolant="air",
            material="Wood",
            program=f"{program} ({suffix})",
        )
        post.write(out / f"{folder}_{suffix}.nc", toolpath_merge(*toolpaths))
        for tp in toolpaths:
            ordered.append((tool, tp, color))
    print(f"[{folder}] wrote {tool_number} .nc file(s), {len(ordered)} tool-paths")

    viewer = make_viewer(data_dir)
    scene = TeeScene(viewer.scene)

    if mesh is not None:
        solids = scene.add_group(f"{folder}_geometry")
        solids.add(triangulated(mesh), name="solid", color=GREY, hide_coplanaredges=True)

    # The CNC table (Carvera work area + mounting holes), for context -- seated on the
    # part-bottom plane (Z=0), same as exmple_model_17.
    drop = Translation.from_vector([0.0, 0.0, PART_BOTTOM])
    table_group = scene.add_group("cnc_table")
    for index, curve in enumerate(load_dxf(data_dir / "cnc_table_holes.dxf")):
        table_group.add(curve.transformed(drop), name=f"table_{index}", color=BLUE)

    cut_group = scene.add_group("cut_curves")
    for index, (name, geometry, color) in enumerate(curves):
        cut_group.add(geometry, name=f"{name}_{index}", color=color)

    if markers:
        marker_group = scene.add_group("origin")
        for index, pt in enumerate(markers):
            marker_group.add(marker(pt), name=f"marker_{index}", color=RED)

    # Hold-down tabs (the ramp snapped each auto-tab onto its centre-path).
    tab_pts = [tab for _tool, suffix, tps, _c in groups for tp in tps for tab in getattr(tp, "tabs", [])]
    if tab_pts:
        tab_group = scene.add_group("hold_down_tabs")
        for index, tab in enumerate(tab_pts):
            tab_group.add(marker(tab), name=f"tab_{index}", color=YELLOW)

    paths = scene.add_group("toolpaths")
    path_objs, path_colors = [], []
    for index, (_tool, tp, color) in enumerate(ordered):
        path_objs.append(paths._live.add(tp.path, name=f"path_{index}", color=color))
        paths._rec.add(tp.path, name=f"path_{index}", color=color)
        path_colors.append(color)

    dump_bundle(scene, data_dir / f"{folder}.json")

    # Headless check (CNC_HEADLESS=1): the .nc files and the .json bundle are written;
    # skip the OpenGL slider + window so the build can be verified without a display.
    if os.environ.get("CNC_HEADLESS"):
        print(f"[{folder}] headless: wrote {folder}.json, skipping viewer")
        return ordered

    sim_entries = [(tool, tp.path) for tool, tp, _color in ordered]
    if hasattr(viewer, "ui"):
        add_toolpath_slider(viewer, sim_entries, path_objs, path_colors)
    viewer.show()


# ================================================================== #
# Job -- the high-level, readable API each example is written against.
# ================================================================== #
def _dia_tag(diameter):
    """``6.0 -> '6'``, ``3.175 -> '3_175'`` -- the tool tag used in tool names / .nc suffixes."""
    return str(int(diameter)) if float(diameter).is_integer() else str(diameter).replace(".", "_")


class Job:
    """One folder's fabrication job, built by chaining operations then ``.run()``.

    A folder ships Rhino OBJ toolpaths (their filenames encode the tool Ø and operation)
    plus a converted ``*_geometry.obj`` solid. Each method consumes the OBJs of one
    operation, builds the tool-paths, and files them under their own tool -- so the
    ``.run()`` at the end writes ONE ``.nc`` per distinct tool (Ø6 before Ø3.175) and
    opens the viewer. Example::

        Job("custom_toolpath_inner_beams", "Inner beams").surface().ramp(tabs=4).drill().run()

    Call ``surface()`` before ``ramp()``/``drill()`` -- surfacing sets the stock top the
    ramps and drills descend from.
    """

    def __init__(self, folder, program):
        self.folder = folder
        self.program = program
        self.objs = classify_objs(folder)
        self.mesh, self._mesh_top = geometry_mesh(folder)
        self._surf_top = 0.0
        self._by_tool = {}  # tool diameter -> [toolpaths] in build order
        self._curves = []  # (name, geometry, color) for the viewer

    @property
    def stock_top(self):
        """Tallest material: the STP mesh top, raised to any higher surfacing plane."""
        return max(self._mesh_top, self._surf_top)

    def _file(self, path, toolpaths):
        self._by_tool.setdefault(tool_from_name(path.name)[0].diameter, []).extend(toolpaths)

    def _center(self):
        """Centre of the global bounding box of ALL this folder's geometry."""
        if self.mesh is not None:
            xyz = [self.mesh.vertex_coordinates(v) for v in self.mesh.vertices()]
        else:
            xyz = [p for path in self.objs["surfacing"] for _d, pts in parse_obj_curves(path) for p in pts]
        lo = [min(p[i] for p in xyz) for i in range(3)]
        hi = [max(p[i] for p in xyz) for i in range(3)]
        return Point(*[(lo[i] + hi[i]) / 2 for i in range(3)])

    def surface(self, start=None):
        """Mill the INSIDE of every ``*_surfacing`` face (quad, tilted, or N-gon).

        ``start="outer"`` begins EACH face at its own corner FARTHEST from the whole
        part's centre -- so every face starts at its own outer end and faces on opposite
        sides (e.g. wedge part1 vs part2) sweep from opposite directions. Or pass an
        explicit ``(x, y, z)`` point (same start corner for all faces).
        """
        if start == "outer":
            centre = self._center()
            ref = lambda pts: max(pts, key=lambda p: (p[0] - centre[0]) ** 2 + (p[1] - centre[1]) ** 2 + (p[2] - centre[2]) ** 2)  # noqa: E731
        else:
            ref = start
        for path in self.objs["surfacing"]:
            toolpaths, polygons = surfacing_toolpaths([path], start=ref)
            self._file(path, toolpaths)
            for poly in polygons:
                self._surf_top = max(self._surf_top, max(p[2] for p in poly))
                self._curves.append(("surfacing", poly, BLUE))
        return self

    def ramp(self, tabs=0):
        """Ramp every ``*_ramp`` contour through the stock. ``tabs`` = hold-down tab count."""
        for path in self.objs["ramp"]:
            toolpaths, contours = ramp_toolpaths([path], self.stock_top, tabs=tabs)
            self._file(path, toolpaths)
            self._curves += [("contour", c, GREEN) for c in contours]
        return self

    def paired_ramp(self):
        """Ramp inclined faces given as paired top/bottom rails (``*_paired_ramped``)."""
        for path in self.objs["paired_ramped"]:
            toolpaths, contours = paired_ramp_toolpaths([path])
            self._file(path, toolpaths)
            self._curves += [("contour", c, GREEN) for c in contours]
        return self

    def drill(self):
        """Helical-drill (or plunge) every ``*_drill`` circle through the stock."""
        for path in self.objs["drill"]:
            toolpaths, holes = drill_toolpaths([path], self.stock_top)
            self._file(path, toolpaths)
            self._curves += [("hole", circle_polyline(c, r), RED) for c, r in holes]
        return self

    def run(self):
        """Write one ``.nc`` per tool (largest first) and open the viewer."""
        groups = []
        for diameter in sorted(self._by_tool, reverse=True):  # Ø6 before Ø3.175
            tag = _dia_tag(diameter)
            tool = Tool(diameter, 30.0, name=f"flat_{tag}mm")
            color = ORANGE if diameter >= 6.0 else PURPLE
            groups.append((tool, f"{tag}mm", self._by_tool[diameter], color))
        finish(self.folder, self.program, groups, self._curves, mesh=self.mesh)
