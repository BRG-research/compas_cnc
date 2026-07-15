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
ROUGH_STEPDOWN = 5.0  # axial depth per constant-Z terrace when roughing a slope (mm)
DIRECTION = "climb"  # one-directional milling (CW / M3 cutter)
TAB_HEIGHT = 2.0  # uncut bridge THICKNESS left at each tab, up from the ramp floor (mm)
# TAB_WIDTH is the real UNCUT BRIDGE length that must SURVIVE, not the tool's lift zone. A
# round tool riding the contour eats a full RADIUS into the lift zone from each side on the
# deep passes, so the builders widen the lift zone to (tool diameter + TAB_WIDTH) to leave
# this much actual bridge -- otherwise a tab narrower than the tool leaves nothing at all.
TAB_WIDTH = 6.0  # real uncut bridge length along the cut (mm)
MITER_LIMIT = 4.0  # convex corners offset to sharp mitered points
SAFE_CLEARANCE = 40.0  # rapid-travel height ABOVE the stock top (mm); each Job computes
#                        safe_z = stock_top + SAFE_CLEARANCE, so a taller part gets a
#                        proportionally higher safe plane instead of a fixed one.
Z_SAFE = 25.0  # fallback safe height for standalone builder calls (a Job overrides it)
PART_BOTTOM = 0.0  # the ramp/drill OBJs sit at Z=0 -- the table / contour plane
RAMP_OVERCUT = 0.0  # ramp ends exactly at the contour polyline's Z (no dip below it), so the
#                     tab tops sit the full TAB_HEIGHT (2 mm) above that polyline
DRILL_OVERCUT = 0.5  # drill deliberately below PART_BOTTOM so through-holes break through
FEED = 400
FIRST_CUT_FEED_FACTOR = 0.5  # first cutting move of every tool-path runs at half FEED --
#                              the initial engagement bites deepest, so ease it in
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


def fit_circle(pts):
    """Centre and radius of a control polygon that samples a circle (see exmple_model_17)."""
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    cz = sum(p[2] for p in pts) / len(pts)
    r = min(math.hypot(p[0] - cx, p[1] - cy) for p in pts)
    return Point(cx, cy, cz), r


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


def ring_at_z(pts, z):
    """Closed XY ring of ``pts`` forced to world height ``z`` (drops duplicate close point)."""
    ring = [[p[0], p[1], z] for p in pts]
    if len(ring) >= 2 and Point(*ring[0]).distance_to_point(Point(*ring[-1])) < 1e-6:
        ring = ring[:-1]
    return Polyline(ring + [ring[0]])


def marker(pt, size=1.2):
    """A small ``+`` polyline centred on ``pt`` (viewer origin / tab marker)."""
    x, y, z = pt[0], pt[1], pt[2]
    return Polyline([[x - size, y, z], [x + size, y, z], [x, y, z], [x, y - size, z], [x, y + size, z]])


# ================================================================== #
# Folder / filename introspection.
# ================================================================== #
_OPERATIONS = ("paired_ramped", "surfacing", "ramp", "drill")

# Filename suffixes -> operation bucket. A folder may label its paired top/bottom rail
# file either ``*_paired_ramped.obj`` (T-sections) or ``*_pairs.obj`` (inner ribs) --
# both feed the paired-ramp builder. Ordered so the paired suffixes are tested before
# the plain ``ramp`` suffix.
_SUFFIX_TO_OP = {
    "paired_ramped": "paired_ramped",
    "pairs": "paired_ramped",
    "surfacing": "surfacing",
    "ramp": "ramp",
    "drill": "drill",
}


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
        for suffix, op in _SUFFIX_TO_OP.items():  # paired_ramped / pairs before ramp so they win
            if stem.endswith(suffix):
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
# ------------------------------------------------------------------ #
# Slope roughing -- constant-Z terraces above an inclined finish face.
# ------------------------------------------------------------------ #
def _plane_coeffs(pts):
    """Plane ``z = A*x + B*y + C`` through the (planar) face corners, or ``None`` if the
    face is vertical (no single-valued Z, so nothing to terrace)."""
    p0, p1, p2 = pts[0], pts[1], pts[2]
    ux, uy, uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    vx, vy, vz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    if abs(nz) < 1e-9:
        return None
    a = -nx / nz
    b = -ny / nz
    return a, b, p0[2] - a * p0[0] - b * p0[1]


def _clip_halfplane(poly_xy, a, b, rhs):
    """Sutherland-Hodgman clip of a convex XY polygon to the half-plane ``a*x + b*y <= rhs``."""
    out = []
    n = len(poly_xy)
    for i in range(n):
        cur, nxt = poly_xy[i], poly_xy[(i + 1) % n]
        dcur = a * cur[0] + b * cur[1] - rhs
        dnxt = a * nxt[0] + b * nxt[1] - rhs
        if dcur <= 0:
            out.append(cur)
        if (dcur <= 0) != (dnxt <= 0):  # edge crosses the clip line -> add the intersection
            t = dcur / (dcur - dnxt)
            out.append((cur[0] + (nxt[0] - cur[0]) * t, cur[1] + (nxt[1] - cur[1]) * t))
    return out


def incline_terraces(pts, radius, ceiling, stepdown=ROUGH_STEPDOWN, stepover=None, safe_z=Z_SAFE):
    """Rough the material ABOVE an inclined finish face as ONE continuous stepped path.

    The face is a planar quad; the raw stock fills from its plane up to ``ceiling`` (the
    stock top). Instead of the finish pass hogging that whole wedge in one inclined
    sweep (a fully-buried tool), the bulk is removed in flat levels ``stepdown`` apart:
    at each level ``z_c`` the footprint is clipped to where the final plane sits BELOW
    ``z_c`` (:func:`_clip_halfplane`) -- the only area still holding material at that
    height -- and that region is raster-filled BIDIRECTIONALLY (a both-ways boustrophedon
    snake, tool-radius compensated so it clears to the footprint edge and stays off the
    un-cut up-slope side).

    Because each lower region NESTS inside the one above (``{plane <= z_c}`` only shrinks
    as ``z_c`` drops), the levels are stitched into a SINGLE continuous path: plunge once
    at the top, snake a terrace, then step straight DOWN ``stepdown`` into the next level
    -- the link to it runs over floor already cleared at the higher level -- and retract
    to ``safe_z`` only at the very end. So one slope = one uninterrupted landscape-milling
    descent, not one lift-and-replunge per level. The leftover staircase is skimmed by the
    inclined finish pass that follows. Returns ``[_Path]`` (one continuous tool-path), or
    ``[]`` for a vertical / already-flat face or when there is no material above the plane.
    """
    coeffs = _plane_coeffs(pts)
    if coeffs is None:
        return []
    a, b, c = coeffs
    footprint = [(p[0], p[1]) for p in pts]
    z_lo = min(p[2] for p in pts)
    z_hi = max(p[2] for p in pts)
    if ceiling <= z_lo + 1e-6:
        return []  # no material above the low edge -- nothing to rough
    stepover = radius if stepover is None else stepover  # roughing can step coarser than the finish

    # One bidirectional snake per terrace level, top to bottom.
    snakes = []
    z_c = ceiling - stepdown
    while z_c > z_lo + 1e-6:
        region = footprint if z_c >= z_hi else _clip_halfplane(footprint, a, b, z_c - c)
        if len(region) >= 3:
            ring = Polyline([[x, y, z_c] for x, y in region] + [[region[0][0], region[0][1], z_c]])
            tp = toolpath_2d_hatch(ring, stepover, radius=radius, z=z_c, safe_z=safe_z, direction=None, contour=False)
            snake = list(tp.fill.points)  # continuous both-ways raster over the (convex) region
            if len(snake) >= 2:
                snakes.append(snake)
        z_c -= stepdown
    if not snakes:
        return []

    # Stitch the levels into one descending path (plunge once, retract once).
    path = [Point(snakes[0][0][0], snakes[0][0][1], safe_z)]  # lead-in over the first start
    prev_end = None
    for snake in snakes:
        if prev_end is not None:
            if prev_end.distance_to_point(snake[-1]) < prev_end.distance_to_point(snake[0]):
                snake = snake[::-1]  # enter this level by its nearer end -> minimal link
            # cross the higher (cleared) floor to the next entry, then the snake's first
            # point plunges straight down `stepdown` into this level.
            path.append(Point(snake[0][0], snake[0][1], prev_end[2]))
        path.extend(snake)
        prev_end = snake[-1]
    path.append(Point(prev_end[0], prev_end[1], safe_z))  # retract once at the end
    return [_Path(Polyline(path))]


def _flat_levels(ceiling, face_z, stepdown):
    """Z levels for a FLAT surfacing face cut in several stepped passes.

    From just below ``ceiling`` down to ``face_z`` in steps of at most ``stepdown``, the
    last level landing exactly on ``face_z``. Returns ``[face_z]`` (a single pass, the
    default behaviour) when no ``stepdown`` is asked for or the ceiling is at/below the
    face -- e.g. the topmost surfacing plane, which has nothing above it to step through.
    """
    if not stepdown or ceiling is None or ceiling <= face_z + 1e-6:
        return [face_z]
    # `- 1e-6` so a depth that is (bar float noise) an exact multiple of the stepdown does
    # not round UP to a spurious extra near-duplicate pass; the last level is then pinned
    # to the face itself so the final skim lands exactly on it.
    passes = max(1, int(math.ceil((ceiling - face_z) / stepdown - 1e-6)))
    levels = [max(face_z, ceiling - stepdown * k) for k in range(1, passes + 1)]
    levels[-1] = face_z
    return levels


def surfacing_toolpaths(obj_paths, safe_z=Z_SAFE, direction=DIRECTION, start=None, rough=False, ceiling=None, stepdown=ROUGH_STEPDOWN, flip_faces=frozenset(), flat_stepdown=None, flat_ceiling=None):
    """Build surfacing tool-paths that clear the INSIDE of each polygon.

    Degree-1 4-corner faces -> :meth:`toolpath_2d_surfacing.from_quad` (``incline=True``
    when the corners differ in Z, so the flat tool rides a tilted face). Any other
    polygon (an N-gon, e.g. outer_ribs part1) -> :class:`toolpath_2d_hatch`, a raster
    pocket inset by the tool radius. ``start`` chooses where each quad sweep BEGINS:
    ``None`` = the highest corner; an ``(x, y, z)`` point = the corner nearest it; a
    CALLABLE ``f(pts) -> point`` = computed per face from that face's OWN corners (e.g.
    its farthest-from-centre corner, so each face starts at its own outer end and faces on
    opposite sides sweep opposite ways).

    ``rough`` -- when ``True`` (and a stock ``ceiling`` is given), each INCLINED quad is
    first roughed with constant-Z terraces from the ceiling down to its low edge
    (:func:`incline_terraces`, ``stepdown`` mm per level) BEFORE its inclined finishing
    pass, so the tool removes the wedge of material in light flat steps instead of a
    single fully-buried inclined sweep. Flat faces are untouched by ``rough``.

    ``flat_stepdown`` -- max depth of cut for a FLAT face that sits below ``flat_ceiling``:
    the raster is repeated at stepped Z levels from the ceiling down to the face (see
    :func:`_flat_levels`) instead of hogging the whole depth in one buried pass, so a deep
    recessed pocket is cleared in light layers. ``None`` (the default) keeps the single
    finish pass at the face's own Z. A face already at ``flat_ceiling`` is untouched.

    ``flip_faces`` -- indices (0-based, in the order faces are met here) whose sweep is
    ROTATED 90 degrees: a quad takes the other rail pair (``flip`` toggled -- shorter vs
    longer side), an N-gon hatch runs at 90 degrees instead of 0. Use it to make one face
    sweep across its short side rather than the default long side. Returns
    ``(toolpaths, polygons)``.
    """
    toolpaths = []
    polygons = []
    face_i = -1

    def _rotate_quad(tp, make):
        # Turn a quad's sweep 90 degrees. The climb pair-toggle can otherwise defeat a
        # plain `flip`, so LOCK the pair and pick whichever locked variant runs
        # perpendicular to the default sweep. Returns the (possibly rotated) tool-path.
        if tp is None:
            return tp
        z0 = tp.zigzag.points
        d0 = (z0[1][0] - z0[0][0], z0[1][1] - z0[0][1])
        for fv in (True, False):
            cand = make(fv, lock=True)
            if cand is None:
                continue
            z1 = cand.zigzag.points
            d1 = (z1[1][0] - z1[0][0], z1[1][1] - z1[0][1])
            dot = abs(d0[0] * d1[0] + d0[1] * d1[1])
            norm = math.hypot(*d0) * math.hypot(*d1)
            if norm > 1e-9 and dot / norm < 0.3:  # ~perpendicular to the default
                return cand
        return tp

    for path in obj_paths:
        _tool, radius = tool_from_name(path.name)
        stepover = radius * 2 / 4  # quarter-diameter pass spacing
        for degree, pts in parse_obj_curves(path):
            if degree != 1 or len(pts) < 3:
                continue
            face_i += 1
            rotate = face_i in flip_faces  # this face's sweep is turned 90 degrees
            polygons.append(poly_at(pts))
            face_start = start(pts) if callable(start) else start
            face_tps = []  # this face's tool-path(s) -- one per stepped Z pass when flat
            if len(pts) == 4:
                z_range = max(p[2] for p in pts) - min(p[2] for p in pts)
                incline = z_range > 1e-6
                start_arg = None if face_start is None else list(face_start)
                if incline:
                    if rough and ceiling is not None:
                        # Rough the wedge above the slope in flat BIDIRECTIONAL terraces
                        # first (landscape step milling -- both ways, no climb lift), THEN
                        # finish.
                        toolpaths.extend(incline_terraces(pts, radius, ceiling, stepdown=stepdown, safe_z=safe_z))
                    quad = [list(p) for p in pts]

                    def _quad(flip, lock=False, q=quad):
                        # Tilted faces sweep the SHORTER direction (flip) for continuous
                        # inclined cutting; the finishing contour + raster honour `direction`.
                        return toolpath_2d_surfacing.from_quad(q, radius, safe_z=safe_z, stepover=stepover, incline=True, flip=flip, start=start_arg, direction=direction, lock_flip=lock)

                    tp = _quad(True)  # default sweep (climb may pick the pair)
                    if rotate:
                        tp = _rotate_quad(tp, _quad)
                    face_tps.append(tp)
                else:
                    # FLAT quad: skim at the face Z, or step down to it in `flat_stepdown`
                    # layers when it is recessed below `flat_ceiling`.
                    face_z = min(p[2] for p in pts)
                    for lz in _flat_levels(flat_ceiling, face_z, flat_stepdown):
                        quad = [[p[0], p[1], lz] for p in pts]

                        def _quad(flip, lock=False, q=quad):
                            return toolpath_2d_surfacing.from_quad(q, radius, safe_z=safe_z, stepover=stepover, incline=False, flip=flip, start=start_arg, direction=direction, lock_flip=lock)

                        tp = _quad(False)
                        if rotate:
                            tp = _rotate_quad(tp, _quad)
                        face_tps.append(tp)
            else:
                # N-gon pocket: one raster per stepped Z level (single level by default).
                face_z = sum(p[2] for p in pts) / len(pts)
                for lz in _flat_levels(flat_ceiling, face_z, flat_stepdown):
                    face_tps.append(
                        toolpath_2d_hatch(
                            closed(pts, z=lz),
                            stepover,
                            radius=radius,
                            z=lz,
                            safe_z=safe_z,
                            direction=direction,
                            angle=math.pi / 2 if rotate else 0.0,  # rotate the raster 90 degrees
                        )
                    )
            for tp in face_tps:
                if tp is not None:
                    toolpaths.append(tp)
                else:
                    print(f"[surfacing] {path.name}: a face was too small for the tool -- skipped")
    return toolpaths, polygons


def ramp_toolpaths(obj_paths, stock_top, safe_z=Z_SAFE, grow=True, tabs=0, overcut=RAMP_OVERCUT, notch=True, step_divisions=1, tab_height=TAB_HEIGHT, tab_width=TAB_WIDTH):
    """Ramp every degree-1 contour straight down through the stock.

    Each contour is grown OUTWARD by the tool radius (Clipper2 miter -> sharp corners)
    so the cutter rides the WASTE side and the finished part keeps nominal size. The
    contour's own OBJ Z is preserved: the profile is lifted to ``stock_top`` when needed,
    then ramped down to that contour plane and below it by ``overcut``.
    The per-pass Z stepdown is the TOOL RADIUS. Concave corners
    get a dogbone (``notch``). ``grow=False`` skips the
    outward offset. ``tabs`` is the NUMBER of hold-down tabs to auto-place per contour
    (``0`` = none, the default; the ramp toolpath places them on the longest edges and
    leaves a :data:`TAB_HEIGHT`-mm uncut bridge). ``notch`` enables concave-corner
    dogbones for a round cutter. ``step_divisions`` subdivides the default stepdown
    (tool radius): ``4`` means four times smaller Z steps (gentler descent).
    Returns ``(toolpaths, contours)`` --
    closed polylines at Z=0 for the viewer.
    """
    toolpaths = []
    contours = []
    for path in obj_paths:
        _tool, radius = tool_from_name(path.name)
        divisions = max(1.0, float(step_divisions))
        step = radius / divisions  # smaller step => more passes / gentler descent
        for degree, pts in parse_obj_curves(path):
            if degree != 1 or len(pts) < 3:
                continue
            contour_z = sum(p[2] for p in pts) / len(pts)
            contours.append(closed(pts, z=contour_z))
            top_z = max(stock_top, contour_z)
            through = (top_z - contour_z) + float(overcut)
            if grow:
                grown = offset_polyline(closed(pts, z=contour_z), radius, join_type="miter", miter_limit=MITER_LIMIT)[0]
                ring = ring_at_z(grown, contour_z)
            else:
                ring = closed(pts, z=contour_z)
            ring = ring.transformed(Translation.from_vector([0.0, 0.0, top_z - contour_z]))
            toolpaths.append(
                toolpath_2d_ramp(
                    ring,
                    Vector(0.0, 0.0, -through),
                    step=step,
                    safe_z=safe_z,
                    offset=0.0,
                    direction=DIRECTION,
                    pocket=False,
                    notch=radius if notch else 0.0,
                    notch_flip=True,
                    # `tabs` is a COUNT -> the ramp auto-places that many hold-down tabs
                    # on the longest edges, each leaving a TAB_HEIGHT-mm uncut bridge.
                    tabs=tabs,
                    tab_height=tab_height,
                    # Widen the lift zone by the tool DIAMETER so `tab_width` of real
                    # uncut bridge survives (the tool eats a radius from each side).
                    tab_width=2.0 * radius + tab_width,
                )
            )
    return toolpaths, contours


def drill_toolpaths(obj_paths, stock_top, safe_z=Z_SAFE):
    """Helical-drill every degree-2 circle from ``stock_top`` down below each circle's OBJ Z.

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
            floor = centre[2] - DRILL_OVERCUT
            top_z = max(stock_top, centre[2])
            axis = Line(Point(centre[0], centre[1], top_z), Point(centre[0], centre[1], floor))
            toolpaths.append(toolpath_2d_drill(axis, hole_radius, tool.diameter, floor=floor, safe_z=safe_z))
    return toolpaths, holes


# ================================================================== #
# Paired ramp -- inclined faces given as matched top/bottom rail loops (T-sections).
# ================================================================== #
class _Path:
    """Minimal tool-path wrapper exposing ``.path`` (a Polyline) and ``.tabs`` (bridge
    centres), so it merges, animates and draws its hold-down tabs exactly like the
    built-in ``toolpath_2d_*`` objects."""

    def __init__(self, path, tabs=None):
        self.path = path
        self.tabs = list(tabs) if tabs else []


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


def _auto_tab_points(ring, count, lift_width):
    """Hold-down tab centres on a closed ``ring``, each CENTRED on a straight edge and kept
    clear of every corner.

    Same rule as :meth:`toolpath_2d_ramp._auto_tab_points`: a tab whose lift zone (width
    ``lift_width``) straddles a corner leaves its bridge wrapped around a sharp vertex,
    where it snaps off -- so tabs are only placed on the INTERIOR of an edge, at least half
    a lift zone (plus a comfort clearance) from either end, and the ``count`` tabs are
    spread EVENLY BY the remaining valid arc-length (the longest edges get proportionally
    more, none land in or near a corner). The clearance relaxes toward the hard minimum
    only if the part is too small; a ring with no edge a lift zone long falls back to its
    longest-edge midpoints. Drops a duplicated closing point first.
    """
    pts = list(ring)
    if count < 1 or len(pts) < 3:
        return []
    if pts[0].distance_to_point(pts[-1]) < 1e-9:
        pts = pts[:-1]
    n = len(pts)
    if n < 3:
        return []
    seg_len = [pts[i].distance_to_point(pts[(i + 1) % n]) for i in range(n)]
    if sum(seg_len) < 1e-9:
        return []

    def edge_point(i, t):
        a, b = pts[i], pts[(i + 1) % n]
        return Point(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)

    def valid_arc(margin):
        intervals = []
        acc = total = 0.0
        for i in range(n):
            if seg_len[i] > 2.0 * margin + 1e-9:
                intervals.append((acc + margin, acc + seg_len[i] - margin, i, acc))
                total += seg_len[i] - 2.0 * margin
            acc += seg_len[i]
        return intervals, total

    r = 0.5 * lift_width  # the lift zone must sit fully on one edge -> min corner keep-out
    best = None
    for clearance in (lift_width, 0.5 * lift_width, 0.25 * lift_width, 0.0):
        intervals, total = valid_arc(r + clearance)
        if total > 1e-9:
            best = (intervals, total)  # keep the roomiest clearance that still fits the tabs
            if total >= count * lift_width or len(intervals) >= count:
                break
    if best is None:  # no edge even as long as a lift zone -> best-effort longest-edge midpoints
        order = sorted(range(n), key=lambda i: seg_len[i], reverse=True)
        return [edge_point(i, 0.5) for i in order[:count]]

    intervals, total = best
    step = total / count
    tabs = []
    for k in range(count):
        target = (k + 0.5) * step  # position along the concatenated VALID (corner-free) arc
        acc_v = 0.0
        for vs, ve, i, es in intervals:
            span = ve - vs
            if acc_v + span >= target - 1e-9:
                tabs.append(edge_point(i, (vs + (target - acc_v) - es) / seg_len[i]))
                break
            acc_v += span
    return tabs


def _apply_helix_tabs(points, disks, floor_z, tab_height, tab_width):
    """Lift a descending path over hold-down TAB disks (port of ``toolpath_2d_ramp``).

    ``disks`` are ``(cx, cy)`` bridge centres on the bottom outline; each is a disk of
    radius ``tab_width / 2``. Because the helix MORPHS toward the bottom rail, every disk
    is crossed once per descent loop: on the upper loops the cut is already above the
    bridge top (``floor_z + tab_height``) so nothing changes, but on the lowest loops and
    the finishing lap the tool walls straight UP onto the bridge where it ENTERS a disk,
    rides flat across, and walls straight DOWN where it LEAVES -- leaving a ``tab_height``
    bridge of uncut stock. Returns ``(points, tab_centres)``.
    """
    if not disks:
        return points, []
    r = 0.5 * tab_width
    tab_top_z = floor_z + tab_height

    def inside(x, y):
        for cx, cy in disks:
            if math.hypot(x - cx, y - cy) <= r + 1e-9:
                return True
        return False

    out = []
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        ax, ay, az = a[0], a[1], a[2]
        dx, dy, dz = b[0] - ax, b[1] - ay, b[2] - az
        out.append(Point(ax, ay, max(az, tab_top_z) if inside(ax, ay) else az))
        aa = dx * dx + dy * dy
        if aa < 1e-18:
            continue
        crossings = []
        for cx, cy in disks:
            fx, fy = ax - cx, ay - cy
            bb = 2.0 * (fx * dx + fy * dy)
            cc = fx * fx + fy * fy - r * r
            disc = bb * bb - 4.0 * aa * cc
            if disc <= 0.0:
                continue
            sq = math.sqrt(disc)
            for tt in ((-bb - sq) / (2.0 * aa), (-bb + sq) / (2.0 * aa)):
                if 1e-9 < tt < 1.0 - 1e-9:
                    crossings.append(tt)
        for tt in sorted(crossings):
            xt, yt, zt = ax + tt * dx, ay + tt * dy, az + tt * dz
            eps = 1e-6
            before = inside(ax + (tt - eps) * dx, ay + (tt - eps) * dy)
            after = inside(ax + (tt + eps) * dx, ay + (tt + eps) * dy)
            if (not before) and after:  # entering a tab -> wall UP onto the bridge
                out.append(Point(xt, yt, zt))
                if zt < tab_top_z:
                    out.append(Point(xt, yt, tab_top_z))
            elif before and (not after):  # leaving a tab -> wall DOWN to the floor
                if zt < tab_top_z:
                    out.append(Point(xt, yt, tab_top_z))
                out.append(Point(xt, yt, zt))
            else:  # boundary between overlapping disks -- stay on the bridge
                out.append(Point(xt, yt, max(zt, tab_top_z)))
    lp = points[-1]
    out.append(Point(lp[0], lp[1], max(lp[2], tab_top_z) if inside(lp[0], lp[1]) else lp[2]))
    return out, [Point(cx, cy, tab_top_z) for cx, cy in disks]


def _paired_ramp_helix(top, bot, top_z, bot_z, radius, safe_z, step, tabs=0, tab_height=TAB_HEIGHT, tab_width=TAB_WIDTH):
    """Helical descent between two winding-matched rail loops, offset OUTWARD by ``radius``.

    Both rail loops are first grown OUTWARD by the tool radius (:func:`_outward_offset`),
    exactly like every other ramp, so the cutter rides the WASTE side of the inclined
    face and the finished part keeps nominal size. The tool then spirals down while the
    outline MORPHS from the grown ``top`` (at ``top_z``) into the grown ``bot`` (at
    ``bot_z``) -- riding the inclined ruled surface between the two waste-side edges.
    ``tabs`` (a COUNT) auto-places that many hold-down bridges on the bottom outline, each
    leaving a ``tab_height``-mm uncut bridge (:func:`_apply_helix_tabs`). Returns
    ``(Polyline, tab_centres)``.
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
    body = [start]
    for s in range(1, samples + 1):  # helix: descend a layer per loop
        body.append(ring_point(s / samples, s % n))
    for i in range(n + 1):  # finishing lap around the bottom outline
        body.append(ring_point(1.0, i % n))

    # HOLD-DOWN TABS: place the bridges on the grown BOTTOM outline (where the finishing
    # lap rides) so the disk-clamp lands exactly on the path, then lift the descent over
    # them. `bot_z` is the deepest cut, so the bridge top sits at bot_z + tab_height.
    tab_centres = []
    if tabs and int(tabs) > 0:
        # Same widened lift zone the bridges use below, so tabs are kept a full lift zone
        # off every corner (a bridge wrapped round a vertex snaps off).
        lift_width = 2.0 * radius + tab_width
        markers = _auto_tab_points([Point(*p) for p in obot], int(tabs), lift_width)
        disks = [(float(m[0]), float(m[1])) for m in markers]
        # Widen the lift disk by the tool DIAMETER so `tab_width` of real uncut bridge
        # survives -- the helix eats a radius from each side, like the ramp does.
        body, tab_centres = _apply_helix_tabs(body, disks, bot_z, tab_height, 2.0 * radius + tab_width)

    lead_in = Point(body[0][0], body[0][1], safe_z)
    tail = body[-1]
    lead_out = Point(tail[0], tail[1], safe_z)  # retract straight up
    return Polyline([lead_in] + body + [lead_out]), tab_centres


def paired_ramp_toolpaths(obj_paths, safe_z=Z_SAFE, step=DOC, tabs=0, tab_height=TAB_HEIGHT, tab_width=TAB_WIDTH):
    """Ramp inclined faces given as PAIRED top/bottom rail loops (the T-sections case).

    Each ``*_paired_ramped.obj`` / ``*_pairs.obj`` holds closed rail loops at two heights:
    the TOP edge of an inclined face (higher Z) and its BOTTOM edge (lower Z). Every top
    loop is paired with the nearest bottom loop by centroid, their windings are matched
    point-to-point (:func:`_match_winding`), and the tool descends the OUTWARD-offset
    helix from :func:`_paired_ramp_helix` (both rails grown by the tool radius to the
    waste side, like every other ramp). ``tabs`` (a COUNT) auto-places that many hold-down
    bridges per pair on its bottom outline. Returns ``(toolpaths, contours)`` --
    ``contours`` are the raw top and bottom rail loops for the viewer.
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
            helix, tab_centres = _paired_ramp_helix(top, bot, top_z, bot_z, radius, safe_z, step, tabs=tabs, tab_height=tab_height, tab_width=tab_width)
            toolpaths.append(_Path(helix, tabs=tab_centres))
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


def finish(folder, program, groups, curves, mesh=None, markers=None, feeds=None):
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
    # One Carvera tool-slot number PER DISTINCT DIAMETER (largest = T1), so a tool that
    # appears in several .nc files -- e.g. Ø3.175 surfacing AND a separate paired-ramp
    # re-run file -- carries the SAME T-number in every header (a re-run loads the same
    # physical tool). One .nc is still written per group.
    numbers = {diameter: index + 1 for index, diameter in enumerate(sorted({tool.diameter for tool, _s, tps, _c in groups if tps}, reverse=True))}
    files = 0
    ordered = []  # (tool, toolpath, color) in machining/animation order
    for tool, suffix, toolpaths, color in groups:
        if not toolpaths:
            continue
        files += 1
        post = Postprocessor(
            tool=tool,
            tool_number=numbers[tool.diameter],
            feed=(feeds or {}).get(tool.diameter, FEED),  # per-tool override (see Job.feed), else FEED
            first_cut_feed_factor=FIRST_CUT_FEED_FACTOR,
            spindle_speed=SPINDLE,
            coolant="air",
            material="Wood",
            program=f"{program} ({suffix})",
        )
        post.write(out / f"{folder}_{suffix}.nc", toolpath_merge(*toolpaths))
        for tp in toolpaths:
            ordered.append((tool, tp, color))
    print(f"[{folder}] wrote {files} .nc file(s), {len(ordered)} tool-paths")

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
        self._separate = []  # (diameter, suffix, [toolpaths], color) forced into their OWN .nc
        self._curves = []  # (name, geometry, color) for the viewer
        self._feeds = {}  # tool diameter -> cutting feed override (mm/min); others use FEED

    @property
    def stock_top(self):
        """Tallest material: the STP mesh top, raised to any higher surfacing plane."""
        return max(self._mesh_top, self._surf_top)

    @property
    def safe_z(self):
        """Rapid-travel height: :data:`SAFE_CLEARANCE` mm above this job's stock top.

        Tracks the stock so a short part (like the inner ribs, whose stock top is the
        Z=0 datum) gets a safe plane at ~40 mm, while a taller folder's paths
        automatically travel higher.
        """
        return self.stock_top + SAFE_CLEARANCE

    def _ceiling(self):
        """Top of the raw stock = the tallest surfacing face, raised to the mesh top.

        This is the height slope roughing terraces down from -- the same one-block stock
        model the ramp assumes (it cuts every profile from :attr:`stock_top`). Computed
        over ALL surfacing faces up front so it is stable regardless of processing order.
        """
        z = self._mesh_top
        for path in self.objs["surfacing"]:
            for degree, pts in parse_obj_curves(path):
                if degree == 1 and len(pts) >= 3:
                    z = max(z, max(p[2] for p in pts))
        return z

    def _file(self, path, toolpaths):
        self._by_tool.setdefault(tool_from_name(path.name)[0].diameter, []).extend(toolpaths)

    def _file_separate(self, path, toolpaths, operation):
        """File ``toolpaths`` into their OWN .nc (``<folder>_<tag>mm_<operation>.nc``)
        instead of merging them into their tool's main file, so this one operation can be
        posted -- and re-run on the machine -- on its own."""
        diameter = tool_from_name(path.name)[0].diameter
        color = ORANGE if diameter >= 6.0 else PURPLE
        self._separate.append((diameter, f"{_dia_tag(diameter)}mm_{operation}", list(toolpaths), color))

    def _center(self):
        """Centre of the global bounding box of ALL this folder's geometry."""
        if self.mesh is not None:
            xyz = [self.mesh.vertex_coordinates(v) for v in self.mesh.vertices()]
        else:
            xyz = [p for path in self.objs["surfacing"] for _d, pts in parse_obj_curves(path) for p in pts]
        lo = [min(p[i] for p in xyz) for i in range(3)]
        hi = [max(p[i] for p in xyz) for i in range(3)]
        return Point(*[(lo[i] + hi[i]) / 2 for i in range(3)])

    def _surf_ceiling(self):
        """Highest ``*_surfacing`` face Z -- the plane the earlier, larger-tool surfacing
        already cleared, so it is the ceiling a later tool steps DOWN from when clearing a
        recessed flat face (independent of the STP mesh, which may be absent)."""
        z = None
        for path in self.objs["surfacing"]:
            for degree, pts in parse_obj_curves(path):
                if degree == 1 and len(pts) >= 3:
                    top = max(p[2] for p in pts)
                    z = top if z is None else max(z, top)
        return z

    def surface(self, start=None, rough=False, stepdown=ROUGH_STEPDOWN, flip=None, flat_stepdown=None):
        """Mill the INSIDE of every ``*_surfacing`` face (quad, tilted, or N-gon).

        ``start="outer"`` begins EACH face at its own corner FARTHEST from the whole
        part's centre -- so every face starts at its own outer end and faces on opposite
        sides (e.g. wedge part1 vs part2) sweep from opposite directions. Or pass an
        explicit ``(x, y, z)`` point (same start corner for all faces).

        ``rough`` -- when ``True`` each INCLINED face is first roughed with constant-Z
        terraces ``stepdown`` mm apart (from the stock top down to the slope's low edge)
        BEFORE its inclined finishing pass, so a tall wedge is hogged out in light flat
        steps instead of one fully-buried inclined sweep. Flat faces are unaffected.

        ``flat_stepdown`` -- max depth of cut (mm) for a recessed FLAT face: it is cleared
        in stepped Z passes from the plane the earlier surfacing cleared
        (:meth:`_surf_ceiling`) down to the face, instead of hogging the whole depth in one
        buried pass. ``None`` (default) keeps the single skim at the face's own Z. E.g. for
        the inner ribs (top faced at -3, pockets at -6) ``flat_stepdown=1.5`` cuts each
        pocket in two 1.5 mm passes (-4.5, -6); the -3 faces stay single-pass.

        ``flip`` -- a surfacing-face index (or an iterable of them) whose sweep is turned
        90 degrees, so it runs across the SHORTER side instead of the default longer one.
        The index matches the viewer's toolpath slider for surfacing paths (0-based, in
        machining order), e.g. ``flip=5`` rotates the 6th surfacing path.
        """
        if start == "outer":
            centre = self._center()
            ref = lambda pts: max(pts, key=lambda p: (p[0] - centre[0]) ** 2 + (p[1] - centre[1]) ** 2 + (p[2] - centre[2]) ** 2)  # noqa: E731
        else:
            ref = start
        flip_set = {int(flip)} if isinstance(flip, int) else set(int(i) for i in (flip or []))
        ceiling = self._ceiling() if rough else None
        flat_ceiling = self._surf_ceiling() if flat_stepdown else None
        face_base = 0  # running surfacing-face index across files, to match the slider order
        for path in self.objs["surfacing"]:
            nfaces = sum(1 for degree, pts in parse_obj_curves(path) if degree == 1 and len(pts) >= 3)
            local_flips = {i - face_base for i in flip_set if face_base <= i < face_base + nfaces}
            toolpaths, polygons = surfacing_toolpaths([path], safe_z=self.safe_z, start=ref, rough=rough, ceiling=ceiling, stepdown=stepdown, flip_faces=local_flips, flat_stepdown=flat_stepdown, flat_ceiling=flat_ceiling)
            face_base += nfaces
            self._file(path, toolpaths)
            for poly in polygons:
                self._surf_top = max(self._surf_top, max(p[2] for p in poly))
                self._curves.append(("surfacing", poly, BLUE))
        return self

    def ramp(self, tabs=0, overcut=RAMP_OVERCUT, notch=True, step_divisions=1, tab_height=TAB_HEIGHT, tab_width=TAB_WIDTH):
        """Ramp every ``*_ramp`` contour through the stock.

        Parameters
        ----------
        tabs : int, optional
            Hold-down tab count.
        overcut : float, optional
            Extra depth below each contour's own OBJ Z plane. Set ``0.0`` (the default)
            to stop exactly on the contour plane -- so the ramp never crosses below it.
        notch : bool, optional
            Add dogbone notches in concave corners (recommended for end mills).
        step_divisions : float, optional
            Divide the default ramp step (tool radius) by this value. ``4`` means
            four times smaller Z steps.
        tab_height : float, optional
            Uncut bridge height above the ramp floor (contour plane).
        tab_width : float, optional
            Flat span of each tab along the cut.
        """
        for path in self.objs["ramp"]:
            toolpaths, contours = ramp_toolpaths([path], self.stock_top, safe_z=self.safe_z, tabs=tabs, overcut=overcut, notch=notch, step_divisions=step_divisions, tab_height=tab_height, tab_width=tab_width)
            self._file(path, toolpaths)
            self._curves += [("contour", c, GREEN) for c in contours]
        return self

    def paired_ramp(self, tabs=0, tab_height=TAB_HEIGHT, tab_width=TAB_WIDTH, separate=False):
        """Ramp inclined faces given as paired top/bottom rails (``*_paired_ramped`` /
        ``*_pairs``), the tool interpolating each pair as it descends.

        Parameters
        ----------
        tabs : int, optional
            Hold-down tab count per pair (``0`` = none). Auto-placed by arc-length on
            each pair's bottom outline, leaving a ``tab_height``-mm uncut bridge.
        tab_height : float, optional
            Bridge height above the deepest cut (bottom-rail Z).
        tab_width : float, optional
            Flat span of each tab along the path.
        separate : bool, optional
            Write these rib-wall paths to their OWN ``<folder>_<tag>mm_paired_ramp.nc``
            instead of merging them into the tool's main surfacing file, so this
            operation can be re-run on the machine on its own. Defaults to ``False``.
        """
        for path in self.objs["paired_ramped"]:
            toolpaths, contours = paired_ramp_toolpaths([path], safe_z=self.safe_z, tabs=tabs, tab_height=tab_height, tab_width=tab_width)
            if separate:
                self._file_separate(path, toolpaths, "paired_ramp")
            else:
                self._file(path, toolpaths)
            self._curves += [("contour", c, GREEN) for c in contours]
        return self

    def drill(self):
        """Helical-drill (or plunge) every ``*_drill`` circle through the stock."""
        for path in self.objs["drill"]:
            toolpaths, holes = drill_toolpaths([path], self.stock_top, safe_z=self.safe_z)
            self._file(path, toolpaths)
            self._curves += [("hole", circle_polyline(c, r, z=c[2]), RED) for c, r in holes]
        return self

    def feed(self, diameter, feed):
        """Override the cutting feed (mm/min) for ONE tool diameter; the rest keep
        :data:`FEED`. Applies to every operation that tool runs (surface/drill/ramp). E.g.
        ``.feed(3.175, 2 * ct.FEED)`` runs the Ø3.175 tool twice as fast."""
        self._feeds[float(diameter)] = float(feed)
        return self

    def run(self):
        """Write one ``.nc`` per tool (largest first), then any ``separate`` operations in
        their own files, and open the viewer."""
        groups = []
        for diameter in sorted(self._by_tool, reverse=True):  # Ø6 before Ø3.175
            tag = _dia_tag(diameter)
            tool = Tool(diameter, 30.0, name=f"flat_{tag}mm")
            color = ORANGE if diameter >= 6.0 else PURPLE
            groups.append((tool, f"{tag}mm", self._by_tool[diameter], color))
        # Separated operations come AFTER the per-tool files, in machining order (their
        # surfacing has already run); each keeps its tool's slot number (set in finish).
        for diameter, suffix, toolpaths, color in self._separate:
            tool = Tool(diameter, 30.0, name=f"flat_{_dia_tag(diameter)}mm")
            groups.append((tool, suffix, toolpaths, color))
        finish(self.folder, self.program, groups, self._curves, mesh=self.mesh, feeds=self._feeds)
