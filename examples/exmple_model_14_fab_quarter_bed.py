import math
import os
import pathlib
from collections import defaultdict

import compas
from compas.geometry import Frame
from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import Rotation
from compas.geometry import Scale
from compas.geometry import Transformation
from compas.geometry import Translation
from compas.geometry import Vector
from compas.geometry import angle_vectors_signed
from compas.geometry import convex_hull_xy
from compas_model.elements.group import Group
from compas_model.models import Model

from compas_tf.plate import PlateElement
from compas_tf.viewer import TeeScene
from compas_tf.viewer import dump_bundle
from compas_tf.viewer import make_viewer

from compas_cnc import Postprocessor
from compas_cnc import offset_polyline
from compas_cnc import toolpath_2d_ramp
from compas_cnc import toolpath_merge
from compas_cnc.tools import Tool

import _custom_toolpath as ct  # reuse the paired top/bottom-rail helix (models 18-22) to
#                               bevel the one split edge -- see beveled_cut_toolpath below

data_dir = pathlib.Path(__file__).parent.parent / "data"

# ------------------------------------------------------------------ #
# Palette.
# ------------------------------------------------------------------ #
GREY = (0.85, 0.85, 0.85)  # sheet rectangle (300x200 work area)
BLACK = (0.0, 0.0, 0.0)  # part outline (the flattened triangular bed)
BLUE = (0.10, 0.45, 0.95)  # inner fold edges (engraved 2 mm)
GREEN = (0.20, 0.70, 0.30)  # engrave tool-path
PURPLE = (0.60, 0.20, 0.90)  # boundary cut tool-path
YELLOW = (0.98, 0.85, 0.10)  # hold-down tab markers

# ------------------------------------------------------------------ #
# Fabrication constants.
#
# Each unrolled bed is ~2.7 m across, so it is shrunk 1:10 onto the Carvera Air
# 300x200 bed - one bed per sheet. Each bed is ONE combined .nc program run as a single
# job on the machine (so you probe the stock once, then it runs unattended):
#   1) ENGRAVE - Tool 1 scores every inner fold line 2 mm deep across the top face.
#   2) tool change to Tool 2 (the Carvera auto-calibrates the new tool length on its setter)
#   3) CUT     - Tool 2 profiles the outer boundary through the 3 mm sheet (waste-side,
#                with a few hold-down tabs so the part stays put), no inner edges.
#
# Z convention: the TOP face of the sheet is the work Z-zero -- everything is cut
# DOWNWARD from there, so STOCK_TOP=0 and the sheet bottom sits at -3 (all machining
# happens in negative Z). Probe/zero the tool on the top of the stock.
# ------------------------------------------------------------------ #
SCALE = 0.1  # 1:10 - the user-chosen scale; beds_1_0 is ~317 mm long at this scale
#              (it overruns the 300 mm bed / 302 mm travel by ~17 mm, so its post
#              is written with on_exceed="warn" and flagged, not raised).
BED_X, BED_Y = 300.0, 200.0  # Carvera Air rated work area (mm) - one sheet per bed
SHEET_GAP = 60.0  # viewer-only: stack the three sheets this far apart in +Y
ENGRAVE_TOOL_DIAMETER = 2.0  # Ø2 mm bit in slot T1 -- scores the fold lines (centre-line, no offset)
CUT_TOOL_DIAMETER = 2.0  # Ø2 mm flat end mill in slot T2 -- profiles the boundary; its radius sets
#                          the waste-side offset, so change this if the cutting tool differs
STOCK_THICKNESS = 3.0  # 3 mm sheet (e.g. plywood) the outline is cut through
ENGRAVE_DEPTH = 2.0  # inner fold lines scored this deep below the top face
CUT_OVERCUT = 0.2  # boundary cut dips this far past the sheet bottom so it breaks fully through
STOCK_TOP = 0.0  # top face of the sheet = work Z-zero (probe the top; cut downward)
PART_BOTTOM = STOCK_TOP - STOCK_THICKNESS  # sheet bottom 3 mm below zero -> cut in -Z
ENGRAVE_FLOOR = STOCK_TOP - ENGRAVE_DEPTH  # score 2 mm into the top face
SAFE_Z = STOCK_TOP + 10.0  # rapid-travel plane (>= MIN_CLEARANCE above the top)
TABS = 4  # hold-down tabs auto-placed around each boundary
TAB_HEIGHT = 1.0  # uncut bridge left at each tab, measured up from the cut floor
TAB_WIDTH = 4.0  # flat span of each tab along the cut
MITER_LIMIT = 4.0  # sharp mitered corners on the outward (waste-side) offset
SAFE_MARGIN = 2.0  # keep-out so the whole cut path (incl. mitered corner spikes) stays this
#                    far off the origin on BOTH X and Y -- the machine origin is the NEAR
#                    corner, so a part touching 0 would drive the tool into negative travel
SHEET_MARGIN = CUT_TOOL_DIAMETER / 2.0 + SAFE_MARGIN  # left/bottom inset of the PART: the waste
#                                                   side cut (grown out one tool radius) then
#                                                   starts SAFE_MARGIN in from the (0,0) origin
NEST_GAP = 6.0  # gap between the body and a split-off tail piece nested on the same sheet
FEED = 600  # cutting feed for BOTH tool-paths (engrave + cut); first cut eased to half
FIRST_CUT_FEED_FACTOR = 0.5  # ease the first (deepest) engagement of every path in
SPINDLE = 10000

# ------------------------------------------------------------------ #
# Lift "quarter_model_0" out of the cantilevers model and UNROLL its three bed
# strips flat, then lay each developed strip on its own 300x200 CNC sheet and
# generate its engrave + cut tool-paths.
#
# A quarter has three beds (groups ``beds_0_0``, ``beds_1_0``, ``beds_2_0``),
# each a CHAIN of six plates that share an edge with the next one and curve up
# off the floor. "Unrolling" a bed hinges every plate about its shared edge into
# its neighbour's plane (``hinge_unfold``), so the six plates land co-planar,
# still touching, in the order they were assembled - the developed strip. Each
# strip develops to a roughly triangular fan: five shared edges (the fold lines)
# fan across it and a many-segment outer boundary wraps it.
#
# As in the other fab examples, each plate's full placement is first folded into
# its OWN transformation (and every group's zeroed), so a plate's transformation
# IS its world placement - base_frame, fabrication_polylines and modelgeometry
# then all agree, and the unroll move can simply be composed onto it.
# ------------------------------------------------------------------ #

model: Model = compas.json_load(data_dir / "cantilevers_model.json")
quarter: Model = model.find_group_with_name("quarter_model_0")

BED_GROUPS = ["beds_0_0", "beds_1_0", "beds_2_0"]


def find_node(node, name):
    """The tree node whose element is named ``name``, searched depth-first."""
    element = getattr(node, "element", None)
    if element is not None and getattr(element, "name", None) == name:
        return node
    for child in getattr(node, "children", []) or []:
        found = find_node(child, name)
        if found is not None:
            return found
    return None


def plates_in(node):
    """Every PlateElement at or below ``node``."""
    out = []
    element = getattr(node, "element", None)
    if isinstance(element, PlateElement):
        out.append(element)
    for child in getattr(node, "children", []) or []:
        out.extend(plates_in(child))
    return out


def bed_strip(name):
    """The plates of bed group ``name``, ordered along the chain.

    Names are ``beds_<strip>_<chain>_<quarter>`` - token 2 is the position of the
    plate within its strip, which is the unroll order.
    """
    plates = plates_in(find_node(quarter.tree.root, name))
    plates.sort(key=lambda plate: int(plate.name.split("_")[2]))
    return plates


# ------------------------------------------------------------------ #
# Fold every plate's placement into its own transformation, zero the groups.
# ------------------------------------------------------------------ #

all_plates = [element for element in quarter.elements() if isinstance(element, PlateElement)]
placements = {plate: plate.modeltransformation for plate in all_plates}
for node in quarter.tree.nodes:
    element = getattr(node, "element", None)
    if isinstance(element, Group):
        element.transformation = Transformation()
quarter.transformation = Transformation()
for plate, placement in placements.items():
    plate.transformation = placement


# ------------------------------------------------------------------ #
# Edge-unfolding: hinge each plate about its shared edge into one plane.
# ------------------------------------------------------------------ #


def _plate_edges(plate):
    pts = list(plate.face_polylines["bottom"].points[:-1])
    n = len(pts)
    return [(pts[i], pts[(i + 1) % n]) for i in range(n)]


def shared_edge(p, q, tol=5.0):
    """The edge (two world points) shared by plates ``p`` and ``q``, or None."""
    for a0, a1 in _plate_edges(p):
        for b0, b1 in _plate_edges(q):
            if (a0.distance_to_point(b0) < tol and a1.distance_to_point(b1) < tol) or (a0.distance_to_point(b1) < tol and a1.distance_to_point(b0) < tol):
                return a0, a1
    return None


def hinge_unfold(plates):
    """One transform per plate (world -> flat) that unfolds the chain to the ground.

    Builds the shared-edge adjacency, spans it with a BFS tree from ``plates[0]``,
    and rotates every plate about its hinge edge into its parent's plane
    (``A[child] = A[parent] * R_hinge``). The whole assembly is then dropped onto
    the ground oriented by the group's common axis - the chain direction through
    the plates' base-frame points, projected to z=0 - so the developed strip keeps
    the orientation and footprint the bed had in 3D. Returns ``None`` if the
    plates are not all edge-connected.
    """
    n = len(plates)
    adjacency = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            edge = shared_edge(plates[i], plates[j])
            if edge is not None:
                adjacency[i].append((j, edge))
                adjacency[j].append((i, edge))

    placed = [None] * n
    placed[0] = Transformation()
    queue = [0]
    while queue:
        i = queue.pop(0)
        for j, (e0, e1) in adjacency[i]:
            if placed[j] is not None:
                continue
            axis = Vector.from_start_end(e0, e1).unitized()
            theta = angle_vectors_signed(plates[j].base_frame.zaxis, plates[i].base_frame.zaxis, axis)
            placed[j] = placed[i] * Rotation.from_axis_and_angle(axis, theta, point=e0)
            queue.append(j)

    if any(transform is None for transform in placed):
        return None

    # The group's common axis is the chain direction running through the plates'
    # base-frame points. Map the FLATTENED strip's own axis (source) onto that
    # same 3D axis PROJECTED to the ground (target): the developed strip lands on
    # z=0 pointing the way the bed runs in 3D, anchored at its plan footprint.
    base_points = [plate.base_frame.point for plate in plates]
    flat_points = [point.transformed(transform) for point, transform in zip(base_points, placed)]

    source_x = Vector.from_start_end(flat_points[0], flat_points[-1]).unitized()
    source_z = plates[0].base_frame.zaxis
    source = Frame(flat_points[0], source_x, source_z.cross(source_x))

    axis = Vector(base_points[-1].x - base_points[0].x, base_points[-1].y - base_points[0].y, 0.0)
    if axis.length < 1e-9:  # bed runs straight up in plan - fall back to flat axis
        axis = Vector(source_x.x, source_x.y, 0.0)
    target_x = axis.unitized()
    target = Frame(Point(base_points[0].x, base_points[0].y, 0.0), target_x, Vector(0, 0, 1).cross(target_x))

    flatten = Transformation.from_frame_to_frame(source, target)
    return [flatten * transform for transform in placed]


def unroll_strip(plates):
    """Unroll a bed chain flat onto the ground, keeping its 3D plan footprint."""
    transforms = hinge_unfold(plates)
    for plate, transform in zip(plates, transforms):
        plate.transformation = transform * plate.transformation


# ------------------------------------------------------------------ #
# Split the developed strip into INNER fold edges and the OUTER boundary.
#
# Every plate contributes its bottom-outline edges. An undirected edge SHARED by
# two plates is a fold line (inner); an edge belonging to only ONE plate is on
# the outer boundary. The boundary edges are then chained into one closed loop.
# ------------------------------------------------------------------ #


def _key(point):
    return (round(point[0], 2), round(point[1], 2))


def classify_edges(plates):
    """Return ``(inner_segments, boundary_loop)`` for a developed bed.

    ``inner_segments`` is a list of ``(Point, Point)`` fold lines (shared by two
    plates); ``boundary_loop`` is an ordered list of ``Point`` corners of the
    single outer contour (not closed - the caller closes it).
    """
    count = defaultdict(int)
    rep = {}  # undirected edge key -> one representative (Point, Point)
    for plate in plates:
        bottom, _top = plate.fabrication_polylines()
        pts = list(bottom.points[:-1])
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            key = tuple(sorted([_key(a), _key(b)]))
            count[key] += 1
            rep[key] = (a, b)

    inner = [rep[key] for key, c in count.items() if c >= 2]
    boundary = [rep[key] for key, c in count.items() if c == 1]
    return inner, _order_loop(boundary)


def _order_loop(segments):
    """Chain undirected boundary ``segments`` [(Point, Point), ...] into an ordered
    loop of Points, walking shared endpoints until it returns to the start."""
    adjacency = defaultdict(list)
    point_of = {}
    for a, b in segments:
        ka, kb = _key(a), _key(b)
        point_of[ka], point_of[kb] = a, b
        adjacency[ka].append(kb)
        adjacency[kb].append(ka)

    start = _key(segments[0][0])
    loop = [start]
    previous, current = None, start
    while True:
        nexts = [k for k in adjacency[current] if k != previous]
        if not nexts:
            break
        nxt = nexts[0]
        if nxt == start:
            break
        loop.append(nxt)
        previous, current = current, nxt
        if len(loop) > 10000:  # guard against a broken (non-manifold) boundary
            break
    return [point_of[k] for k in loop]


# ------------------------------------------------------------------ #
# Orient each developed bed onto its own 300x200 sheet.
#
# The tightest (minimum-area) oriented bounding box of the boundary gives the
# rotation that squares the bed to the rectangle; its long side is laid along X
# (the 300 axis) and short side along Y (the 200 axis), then the whole bed is
# scaled 1:10 and centred in the sheet.
# ------------------------------------------------------------------ #


def oriented_box(points):
    """Minimum-area bounding box of 2D ``points`` via rotating calipers.

    Returns ``(long_dir, center, long_len, short_len)`` - the unit vector of the
    box's LONGER side, the box centre (x, y), and the two side lengths.
    """
    hull = convex_hull_xy([[p[0], p[1]] for p in points])
    n = len(hull)
    best = None
    for i in range(n):
        ax = hull[(i + 1) % n][0] - hull[i][0]
        ay = hull[(i + 1) % n][1] - hull[i][1]
        length = math.hypot(ax, ay)
        if length < 1e-9:
            continue
        ux, uy = ax / length, ay / length  # edge direction
        vx, vy = -uy, ux  # its perpendicular
        us = [h[0] * ux + h[1] * uy for h in hull]
        vs = [h[0] * vx + h[1] * vy for h in hull]
        umin, umax, vmin, vmax = min(us), max(us), min(vs), max(vs)
        w, h = umax - umin, vmax - vmin
        area = w * h
        if best is None or area < best[0]:
            uc, vc = (umin + umax) / 2, (vmin + vmax) / 2
            center = (uc * ux + vc * vx, uc * uy + vc * vy)
            best = (area, (ux, uy), center, w, h)

    _area, (ux, uy), center, w, h = best
    if w >= h:
        return (ux, uy), center, w, h
    return (-uy, ux), center, h, w  # rotate so the LONGER side is the reported axis


def sheet_transform(long_dir, center, sheet_center):
    """World -> sheet transform: put ``center`` at ``sheet_center``, turn ``long_dir``
    onto +X, and scale by :data:`SCALE`. (Rotate/scale about the origin after the bed
    is centred there, then translate onto the sheet.)"""
    angle = math.atan2(long_dir[1], long_dir[0])
    to_origin = Translation.from_vector([-center[0], -center[1], 0.0])
    align = Rotation.from_axis_and_angle([0.0, 0.0, 1.0], -angle, point=[0.0, 0.0, 0.0])
    shrink = Scale.from_factors([SCALE, SCALE, SCALE])
    onto_sheet = Translation.from_vector([sheet_center[0], sheet_center[1], 0.0])
    return onto_sheet * shrink * align * to_origin


# ------------------------------------------------------------------ #
# Tool-path builders.
# ------------------------------------------------------------------ #


def engrave_toolpath(segments):
    """Score each inner fold LINE 1 mm into the top face.

    For every ``(a, b)`` segment the tool approaches at :data:`SAFE_Z`, plunges to
    :data:`ENGRAVE_FLOOR`, cuts straight across to the far end, and retracts - the
    pieces are joined (rapids at safe Z between them) into one path. Returns a
    :class:`compas.geometry.Polyline`.
    """
    pieces = []
    for a, b in segments:
        pieces.append(
            Polyline(
                [
                    Point(a[0], a[1], SAFE_Z),
                    Point(a[0], a[1], ENGRAVE_FLOOR),
                    Point(b[0], b[1], ENGRAVE_FLOOR),
                    Point(b[0], b[1], SAFE_Z),
                ]
            )
        )
    return toolpath_merge(*pieces)


def cut_toolpath(loop, radius):
    """Profile the outer boundary through the sheet, riding the waste side with tabs.

    The closed boundary is grown OUTWARD by the tool radius (Clipper2 miter -> sharp
    corners) so the cutter runs on the WASTE side and the part keeps nominal size,
    lifted to the stock top, then ramped straight down through the sheet (and
    :data:`CUT_OVERCUT` past its bottom so it breaks through), leaving
    :data:`TABS` auto-placed hold-down bridges. Returns a ``toolpath_2d_ramp``.
    """
    ring = Polyline([Point(p[0], p[1], PART_BOTTOM) for p in loop] + [Point(loop[0][0], loop[0][1], PART_BOTTOM)])
    grown = offset_polyline(ring, radius, join_type="miter", miter_limit=MITER_LIMIT)[0]
    mouth = Polyline([Point(p[0], p[1], STOCK_TOP) for p in grown.points])
    through = (STOCK_TOP - PART_BOTTOM) + CUT_OVERCUT
    return toolpath_2d_ramp(
        mouth,
        Vector(0.0, 0.0, -through),
        step=radius,
        safe_z=SAFE_Z,
        offset=0.0,
        direction="climb",
        pocket=False,  # profiling a kept part (island), not clearing a pocket
        notch=0.0,
        tabs=TABS,
        tab_height=TAB_HEIGHT,
        tab_width=TAB_WIDTH,
    )


def beveled_cut_toolpath(boundary, owner, split_edge_world, place, radius):
    """Profile cut where ONLY the split (former-fold) edge is milled as an INCLINED wall.

    When a bed is split, the fold that was scored-and-folded becomes a real cut, and a
    square vertical cut there would only kiss its mate at one corner (the plates meet at a
    dihedral). So the two endpoints of the split edge get a TOP rail taken from the owning
    plate's TOP face outline -- which the unroll leaves shifted in-plane from the bottom by
    the plate's real bevel (~11 deg) -- while every other vertex keeps top == bottom (a
    plain vertical wall). The tool then spirals from the sheet top down through the sheet,
    morphing between the two rails, so the split edge leans to the plate's true angle and
    the free edges stay square. ``place`` maps the owner's model-space outlines into the
    same sheet-local frame as ``boundary``. Returns a ``.path`` / ``.tabs`` wrapper.
    """
    bottom = list(owner.fabrication_polylines()[0].points[:-1])
    top = list(owner.fabrication_polylines()[1].points[:-1])
    overrides = {}
    for corner in split_edge_world:  # the two endpoints of the fold, in model space
        k = min(range(len(bottom)), key=lambda j: Point(*bottom[j]).distance_to_point(Point(*corner)))
        placed_bottom = Point(*bottom[k]).transformed(place)
        placed_top = Point(*top[k]).transformed(place)  # same vertex on the top face -> the bevel
        m = min(range(len(boundary)), key=lambda j: Point(*boundary[j]).distance_to_point(placed_bottom))
        overrides[m] = placed_top

    bot_rail = [Point(*p) for p in boundary]
    top_rail = [overrides.get(i, bot_rail[i]) for i in range(len(bot_rail))]
    # Spiral from the stock top through the sheet (+CUT_OVERCUT breakout), morphing rails.
    helix, tab_centres = ct._paired_ramp_helix(
        top_rail, bot_rail, STOCK_TOP, PART_BOTTOM - CUT_OVERCUT, radius, SAFE_Z, radius, tabs=TABS, tab_height=TAB_HEIGHT, tab_width=TAB_WIDTH
    )
    return ct._Path(helix, tabs=tab_centres)


def developed(plate_subset):
    """``(inner_folds, boundary)`` of a plate subset plus its min-area box.

    A thin wrapper over :func:`classify_edges` + :func:`oriented_box` so the SAME
    code can orient a whole strip or a single split-off plate. Returns
    ``(inner, boundary, long_dir, center, long_len, short_len)``.
    """
    inner, boundary = classify_edges(plate_subset)
    long_dir, center, long_len, short_len = oriented_box(boundary)
    return inner, boundary, long_dir, center, long_len, short_len


def place_on_sheet(dev, min_corner):
    """Place a developed strip long-side along X, scaled 1:10, its min (left/bottom)
    corner dropped at ``min_corner``. Returns placed ``(boundary, inner, (w, h))`` in
    sheet-local coordinates."""
    inner, boundary, long_dir, center, long_len, short_len = dev
    width, height = long_len * SCALE, short_len * SCALE
    target = (min_corner[0] + width / 2.0, min_corner[1] + height / 2.0)
    place = sheet_transform(long_dir, center, target)
    placed_boundary = [Point(*p).transformed(place) for p in boundary]
    placed_inner = [(Point(*a).transformed(place), Point(*b).transformed(place)) for a, b in inner]
    return placed_boundary, placed_inner, (width, height), place


def far_end_plate(plates, long_dir):
    """Split the chain at whichever END plate sits at the +``long_dir`` end -- the tip
    that lands at higher X once placed, i.e. the one that overruns. Returns
    ``(body_plates, tail_plate)`` (removing an end plate keeps the body one connected
    strip). Projection onto ``long_dir`` orders the plates the way X will after the
    sheet transform rotates ``long_dir`` onto +X."""

    def projection(plate):
        pts = list(plate.fabrication_polylines()[0].points[:-1])
        return sum(p[0] * long_dir[0] + p[1] * long_dir[1] for p in pts) / len(pts)

    if projection(plates[-1]) >= projection(plates[0]):
        return plates[:-1], plates[-1]
    return plates[1:], plates[0]


# ------------------------------------------------------------------ #
# Build the three beds: unroll, orient onto a sheet, cut two tool-paths, post NC.
# ------------------------------------------------------------------ #

strips = [bed_strip(name) for name in BED_GROUPS]
for plates in strips:
    unroll_strip(plates)

# Two physical tools, one slot each: T1 engraves the fold lines, T2 profiles the boundary.
# Only the CUT tool's radius shapes geometry (the waste-side offset); the engrave rides the
# fold-line centre, so its diameter is cosmetic here.
ENGRAVE_TOOL = Tool(ENGRAVE_TOOL_DIAMETER, 30.0, name=f"engrave_{str(ENGRAVE_TOOL_DIAMETER).replace('.', '_')}mm")
CUT_TOOL = Tool(CUT_TOOL_DIAMETER, 30.0, name=f"cut_{str(CUT_TOOL_DIAMETER).replace('.', '_')}mm")
out_dir = data_dir / "quarter_bed_toolpaths"
out_dir.mkdir(exist_ok=True)

limit_probe = Postprocessor(on_exceed="ignore")  # only reads the travel envelope for a fit test

beds = []  # per-bed dict: name, index, placed pieces, engrave + cut tool-paths (sheet-local)
for index, (name, plates) in enumerate(zip(BED_GROUPS, strips)):
    # Orient the whole strip long-side along X, scaled 1:10, then LEFT/BOTTOM aligned to the
    # origin (plus SAFE_MARGIN) so all three beds share one datum and nothing reaches negative
    # travel. Everything below is in sheet-local coordinates.
    dev = developed(plates)
    boundary, inner, (width, height), place = place_on_sheet(dev, (SHEET_MARGIN, SHEET_MARGIN))
    # per piece: placed geometry + (owner plate, split-fold edge, its placement) so a split
    # piece can bevel its one former-fold edge; ``owner=None`` -> a plain vertical cut.
    pieces = [{"boundary": boundary, "inner": inner, "owner": None, "split": None, "place": place}]

    # Does the whole strip fit the machine? beds_1_0 is ~317 mm at 1:10 and overruns X. If so,
    # peel its overrunning END plate off and NEST it as its own closed piece in the sheet's
    # slack (+Y) space, so the long axis stops overrunning. Only that one plate is split off;
    # the fold it shared with the body becomes a real cut on both pieces (re-joined on assembly),
    # so that ONE edge is milled inclined to the plate's true dihedral on each piece.
    if limit_probe.check_limits(cut_toolpath(boundary, CUT_TOOL.radius)):
        body_plates, tail_plate = far_end_plate(plates, dev[2])
        neighbor, split_world = None, None  # the body plate still touching the tail + the shared fold
        for candidate in body_plates:
            edge = shared_edge(tail_plate, candidate)
            if edge is not None:
                neighbor, split_world = candidate, edge
                break
        boundary, inner, (width, height), body_place = place_on_sheet(developed(body_plates), (SHEET_MARGIN, SHEET_MARGIN))
        tail_corner = (SHEET_MARGIN, SHEET_MARGIN + height + NEST_GAP)
        tail_boundary, tail_inner, _, tail_place = place_on_sheet(developed([tail_plate]), tail_corner)
        pieces = [
            {"boundary": boundary, "inner": inner, "owner": neighbor, "split": split_world, "place": body_place},
            {"boundary": tail_boundary, "inner": tail_inner, "owner": tail_plate, "split": split_world, "place": tail_place},
        ]

    # One engrave path (all fold lines over every piece; only multi-plate bodies have folds)
    # and one cut PROGRAM that lists each piece's closed profile in turn -- a split piece
    # bevels its former-fold edge, everything else is a plain vertical profile.
    engrave = engrave_toolpath([segment for piece in pieces for segment in piece["inner"]])
    cuts = []
    for piece in pieces:
        if piece["owner"] is not None and piece["split"] is not None:
            cuts.append(beveled_cut_toolpath(piece["boundary"], piece["owner"], piece["split"], piece["place"], CUT_TOOL.radius))
        else:
            cuts.append(cut_toolpath(piece["boundary"], CUT_TOOL.radius))

    cut_pts = [pt for cut in cuts for pt in cut.path.points]
    footprint = (max(p[0] for p in cut_pts) - min(p[0] for p in cut_pts), max(p[1] for p in cut_pts) - min(p[1] for p in cut_pts))

    # Post ONE combined .nc for the whole bed: T1 engraves every fold line, then a tool
    # change to T2 (the Carvera auto-calibrates its length on the setter) and T2 profiles
    # the boundary. So the machine is loaded/probed once and runs both operations. A nested
    # bed still writes a single file; its cut section simply holds two closed profiles.
    # Overruns warn (don't raise) and report.
    post = Postprocessor(
        feed=FEED,
        first_cut_feed_factor=FIRST_CUT_FEED_FACTOR,
        spindle_speed=SPINDLE,
        coolant="air",
        material="Plywood",
        stock_size=(BED_X, BED_Y, STOCK_THICKNESS),
        on_exceed="warn",
    )
    sections = [
        (ENGRAVE_TOOL, 1, f"quarter bed {index} - engrave inner fold lines {ENGRAVE_DEPTH:g}mm", [engrave]),
        (CUT_TOOL, 2, f"quarter bed {index} - cut outer boundary through {STOCK_THICKNESS:g}mm", cuts),
    ]
    target = out_dir / f"quarter_bed_{index}.nc"
    post.write_program(target, sections)
    violations = post.check_limits(engrave, *cuts)
    note = "" if not violations else "  [OVER: " + ", ".join(f"{v['axis']} {v['span']:.1f}>{v['limit']:.0f}mm" for v in violations) + "]"
    split = "  [split: tail nested]" if len(pieces) > 1 else ""
    print(f"[{name}] {target.name}  (T1 engrave + T2 cut, {footprint[0]:.1f} x {footprint[1]:.1f} mm){note}{split}")

    beds.append({"name": name, "index": index, "pieces": pieces, "engrave": engrave, "cuts": cuts})


# ------------------------------------------------------------------ #
# Viewer: draw the three sheets stacked in +Y, each with its 300x200 outline, the
# flattened bed, its inner fold lines, and the two tool-paths (+ tab markers).
# The .nc files hold each bed in its OWN sheet-local frame; here they are merely
# offset in Y for a side-by-side review.
# ------------------------------------------------------------------ #


def sheet_rectangle(y0):
    return Polyline([[0, y0, 0], [BED_X, y0, 0], [BED_X, y0 + BED_Y, 0], [0, y0 + BED_Y, 0], [0, y0, 0]])


viewer = make_viewer(data_dir)
scene = TeeScene(viewer.scene)

for bed in beds:
    ty = bed["index"] * (BED_Y + SHEET_GAP)
    shift = Translation.from_vector([0.0, ty, 0.0])
    group = scene.add_group(bed["name"])

    group.add(sheet_rectangle(ty), name=f"{bed['name']}_sheet", linecolor=GREY, linewidth=1)

    # Each piece (usually one; a nested bed has a body + a split-off tail): outline + folds.
    for pi, piece in enumerate(bed["pieces"]):
        boundary = piece["boundary"]
        outline = Polyline([Point(p[0], p[1], STOCK_TOP) for p in boundary] + [Point(boundary[0][0], boundary[0][1], STOCK_TOP)])
        group.add(outline.transformed(shift), name=f"{bed['name']}_p{pi}_outline", linecolor=BLACK, linewidth=3)
        for i, (a, b) in enumerate(piece["inner"]):
            fold = Polyline([Point(a[0], a[1], STOCK_TOP), Point(b[0], b[1], STOCK_TOP)])
            group.add(fold.transformed(shift), name=f"{bed['name']}_p{pi}_fold_{i}", linecolor=BLUE, linewidth=2)

    group.add(bed["engrave"].transformed(shift), name=f"{bed['name']}_engrave_path", linecolor=GREEN, linewidth=1)
    for ci, cut in enumerate(bed["cuts"]):
        group.add(cut.path.transformed(shift), name=f"{bed['name']}_cut{ci}_path", linecolor=PURPLE, linewidth=1)
        for i, tab in enumerate(getattr(cut, "tabs", [])):
            size = 2.0
            cross = Polyline([[tab[0] - size, tab[1], tab[2]], [tab[0] + size, tab[1], tab[2]], [tab[0], tab[1], tab[2]], [tab[0], tab[1] - size, tab[2]], [tab[0], tab[1] + size, tab[2]]])
            group.add(cross.transformed(shift), name=f"{bed['name']}_cut{ci}_tab_{i}", linecolor=YELLOW, linewidth=3)

dump_bundle(scene, data_dir / "quarter_bed_toolpaths.json")

if os.environ.get("CNC_HEADLESS"):
    print("[quarter_bed_toolpaths] headless: wrote 3 combined .nc files + quarter_bed_toolpaths.json, skipping viewer")
else:
    viewer.show()


# ====================================================================== #
#  RHINO  -  copy the code between the triple quotes into the Rhino 8
#  ScriptEditor (Python 3) and Run it to add THIS example's geometry to the
#  active Rhino document (named layers, per-object colour). Needs only the
#  installed compas_tf (see install steps); recomputes nothing.
# ====================================================================== #
RHINO = r'''
from compas_tf.rhino import draw_bundle
draw_bundle(r"C:\brg\compas_cnc\data\quarter_bed_toolpaths.json")
'''
