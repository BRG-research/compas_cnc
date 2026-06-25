import pathlib

import compas
from compas.geometry import Frame
from compas.geometry import Point
from compas.geometry import Transformation
from compas.geometry import Translation
from compas.geometry import Vector
from compas.geometry import convex_hull_xy
from compas_model.elements.group import Group
from compas_model.models import Model

from compas_tf.plate import PlateElement
from compas_tf.viewer import TeeScene
from compas_tf.viewer import dump_bundle
from compas_tf.viewer import frame_rectangle
from compas_tf.viewer import make_viewer
from compas_tf.viewer import triangulated

data_dir = pathlib.Path(__file__).parent.parent / "data"

GREY = (0.85, 0.85, 0.85)  # plate mesh
BLACK = (0.0, 0.0, 0.0)  # plate bottom/top boundary polylines
PLANE = (0.95, 0.6, 0.1)  # plate top plane + normal (the face laid on the ground)
CUTMIN = (0.1, 0.5, 1.0)  # minimal cut geometry (box outlines + dowel center lines)
CUT = (0.9, 0.2, 0.2)  # cut feature solids (box slot + dowels)
SPREAD = 2.0  # explode factor: scale each beam's distance from the group plan centre
MARGIN = 30.0  # min gap left between laid-flat beam footprints by the de-overlap pass

# ------------------------------------------------------------------ #
# Lift "quarter_model_0" out of the cantilevers model and lay its FRAME beams
# flat on the ground for fabrication - the tsections, outer/inner ribs, inner
# beams and their wedges. The beds (group "beds_0") are NOT touched here: they
# are a developable chain, unrolled separately in
# exmple_model_14_fab_quarter_beds_ribs_tsections_outer_beams_wedges.py.
#
# Like the oculus fab example, each plate's full placement is first folded into
# its OWN transformation (every group's zeroed), so a plate's transformation IS
# its world placement - base_frame, fabrication_polylines and modelgeometry then
# all agree. These frame beams are fabricated lying on their TOP face, so lay_flat
# maps each beam's top_frame (the OTHER polyline, not the bottom base_frame) onto
# the ground at its own plan position (body up), so the beams fan out like the
# quarter seen from above, each resting on its top face.
# ------------------------------------------------------------------ #

model: Model = compas.json_load(data_dir / "cantilevers_model.json")
quarter: Model = model.find_group_with_name("quarter_model_0")


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


# The frame beams = every plate in the quarter EXCEPT the ones under "beds_0".
bed_plates = set(plates_in(find_node(quarter.tree.root, "beds_0")))
beams = [plate for plate in quarter.elements() if isinstance(plate, PlateElement) and plate not in bed_plates]

# ------------------------------------------------------------------ #
# Fold every plate's placement into its own transformation; zero the groups.
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


def lay_flat(plate, offset=0.0):
    """Lay the plate flat on worldXY at its plan position, resting on its TOP
    face, body UP (+Z).

    These frame beams are fabricated lying on their top face, so the reference is
    the plate's (placed) top_frame, NOT its base_frame (the bottom). top_frame's
    z-axis runs bottom->top, so the body sits on its -z side; the target ground
    frame's z-axis is therefore -Z (down), so once top_frame is mapped onto it the
    top face lands on the ground and the body extrudes UP. The x-axis is the
    plate's long edge projected to the ground. The world-space move is composed
    onto the plate's current placement (built on, not dropped). ``offset`` nudges
    it sideways."""
    top = plate.top_frame
    x2d = Vector(top.xaxis[0], top.xaxis[1], 0.0)
    if x2d.length < 1e-9:  # long edge runs vertical - use the top y
        x2d = Vector(top.yaxis[0], top.yaxis[1], 0.0)
    xaxis = x2d.unitized()
    yaxis = Vector(0, 0, -1).cross(xaxis)  # target z = xaxis x yaxis = -Z (down) -> body UP
    point = Point(top.point[0] + yaxis[0] * offset, top.point[1] + yaxis[1] * offset, 0.0)
    target = Frame(point, xaxis, yaxis)
    return Transformation.from_frame_to_frame(top, target) * plate.transformation


def explode_from_centre(plates, factor):
    """Push every laid-flat beam radially away from the group's plan centre.

    Scales each beam's distance from the shared plan centre by ``factor`` (along
    the direction centre -> beam, on the ground), opening proportional gaps
    between all beams so their footprints and dowel cylinders stop overlapping.
    ``factor=1`` leaves them put; larger spreads them further apart. The move is a
    world-space translation composed onto each plate's current placement."""
    points = [Point(*plate.top_frame.point) for plate in plates]
    cx = sum(point.x for point in points) / len(points)
    cy = sum(point.y for point in points) / len(points)
    for plate, point in zip(plates, points):
        direction = Vector(point.x - cx, point.y - cy, 0.0)
        if direction.length < 1e-9:  # a beam sitting on the centre stays put
            continue
        move = Translation.from_vector(direction * (factor - 1.0))
        plate.transformation = move * plate.transformation


def _footprint_hull(plate):
    """Convex hull (XY, on the ground) of the beam's laid-flat footprint.

    Uses both boundary polylines so a non-prismatic beam's full plan projection is
    covered. The hull slightly over-estimates a T-shaped tsection (it fills the
    notch), which only ever leaves a touch more clearance - fine for fabrication."""
    points = []
    for polyline in (plate.top_polyline, plate.bottom_polyline):
        if polyline is not None:
            points.extend([[point.x, point.y] for point in polyline.points])
    hull = convex_hull_xy(points)
    return [Point(h[0], h[1], 0.0) for h in hull]


def _separation(a, b, margin):
    """Smallest translation (a Vector applied to ``b``) that leaves convex hulls
    ``a`` and ``b`` at least ``margin`` apart, or ``None`` if they already are.

    Separating-Axis Theorem: project both hulls onto every edge normal. If any
    axis shows a gap >= margin the hulls are clear; otherwise the minimum push
    over all axes is the translation that just separates them."""
    best = None
    best_mag = None
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            p0, p1 = poly[i], poly[(i + 1) % n]
            nx, ny = -(p1.y - p0.y), (p1.x - p0.x)
            length = (nx * nx + ny * ny) ** 0.5
            if length < 1e-9:
                continue
            nx, ny = nx / length, ny / length
            a_proj = [nx * p.x + ny * p.y for p in a]
            b_proj = [nx * p.x + ny * p.y for p in b]
            gap_pos = min(b_proj) - max(a_proj)  # b on +normal side
            gap_neg = min(a_proj) - max(b_proj)  # b on -normal side
            if gap_pos >= margin or gap_neg >= margin:
                return None  # separating axis found -> already clear
            push_pos = margin - gap_pos
            push_neg = margin - gap_neg
            if push_pos <= push_neg:
                mag, dx, dy = push_pos, nx * push_pos, ny * push_pos
            else:
                mag, dx, dy = push_neg, -nx * push_neg, -ny * push_neg
            if best_mag is None or mag < best_mag:
                best_mag, best = mag, (dx, dy)
    return Vector(best[0], best[1], 0.0) if best is not None else None


def _hull_centroid(hull):
    """Mean (x, y) of a hull's points."""
    return (sum(point.x for point in hull) / len(hull), sum(point.y for point in hull) / len(hull))


def deoverlap(plates, margin, iterations=3000):
    """Nudge laid-flat beams apart until no two footprints overlap (with ``margin``).

    The radial explode fans the beams out but, scaling everything uniformly, it
    cannot separate crossing parts, and the radial beams' inner ends all pile up at
    the hub - a dense cluster that pairwise 50/50 pushes only shuffle in place. So
    each pass does two things, applied immediately (Gauss-Seidel) by shifting each
    beam's cached hull: (1) resolve every overlapping pair by its minimum
    separating translation, split evenly between the two beams; (2) give every beam
    that overlapped this pass a small OUTWARD nudge from the layout's plan centre,
    which expands the crowded hub and breaks the gridlock that a pure pairwise pass
    deadlocks on. It stops as soon as a pass is clean, so the fan grows only as
    much as it must. Net per-beam moves are accumulated and composed onto each
    placement once at the end - pure ground-plane moves, so the beams stay flat on
    their top face."""
    hulls = [_footprint_hull(plate) for plate in plates]
    accumulated = [[0.0, 0.0] for _ in plates]
    # Fixed plan centre that crowded beams are pushed away from, and the outward
    # nudge per pass (a fraction of the gap - large enough to break the gridlock,
    # small enough that the final fan stays compact).
    centroids = [_hull_centroid(hull) for hull in hulls]
    cx = sum(c[0] for c in centroids) / len(centroids)
    cy = sum(c[1] for c in centroids) / len(centroids)
    bias = margin * 0.2
    for _ in range(iterations):
        crowded = set()
        for i in range(len(plates)):
            for j in range(i + 1, len(plates)):
                sep = _separation(hulls[i], hulls[j], margin)
                if sep is None:
                    continue
                crowded.add(i)
                crowded.add(j)
                hx, hy = sep.x * 0.5, sep.y * 0.5
                for point in hulls[i]:
                    point.x -= hx
                    point.y -= hy
                for point in hulls[j]:
                    point.x += hx
                    point.y += hy
                accumulated[i][0] -= hx
                accumulated[i][1] -= hy
                accumulated[j][0] += hx
                accumulated[j][1] += hy
        if not crowded:
            break
        for k in crowded:
            ccx, ccy = _hull_centroid(hulls[k])
            dx, dy = ccx - cx, ccy - cy
            length = (dx * dx + dy * dy) ** 0.5
            if length < 1e-9:
                continue
            ux, uy = dx / length * bias, dy / length * bias
            for point in hulls[k]:
                point.x += ux
                point.y += uy
            accumulated[k][0] += ux
            accumulated[k][1] += uy
    for plate, (dx, dy) in zip(plates, accumulated):
        if dx * dx + dy * dy > 1e-12:
            plate.transformation = Translation.from_vector(Vector(dx, dy, 0.0)) * plate.transformation


def draw_plate(plate, parent):
    """Draw a plate and everything tied to it - in ONE group named after the
    plate: carved mesh (grey), bottom/top boundary polylines (black, co-wound),
    top plane + normal (orange, the face laid on the ground), cut feature SOLIDS
    (red: box slot + dowels) and the cut feature outlines + dowel center lines
    (blue, minimal geometry)."""
    group = parent.add_group(plate.name)
    group.add(triangulated(plate.modelgeometry), name=plate.name, hide_coplanaredges=True, color=GREY)

    bottom, top = plate.fabrication_polylines()  # co-wound (same winding)
    group.add(bottom, name=f"{plate.name}_bottom", linecolor=BLACK, linewidth=3)
    group.add(top, name=f"{plate.name}_top", linecolor=BLACK, linewidth=3)

    # Draw the top plane with its normal flipped to point INTO the plate body (the
    # mesh), not outward: top_frame.zaxis runs bottom->top, so at the top face it
    # points away from the body - negating the frame's y-axis flips its z-axis
    # (the rectangle is symmetric, so it looks the same).
    top = plate.top_frame
    inward = Frame(top.point, top.xaxis, top.yaxis * -1)
    rectangle, normal = frame_rectangle(inward, scale=150)
    group.add(rectangle, name=f"{plate.name}_top_plane", facecolor=PLANE, opacity=0.3)
    group.add(normal, name=f"{plate.name}_top_normal", linecolor=PLANE, linewidth=2)

    # Cut feature solids (box slot + dowels), placed in the plate's current frame.
    for feature in plate.get_features():
        for mesh in getattr(feature, "meshes", None) or []:
            group.add(triangulated(mesh), name=f"{plate.name}__{feature.name or 'cut'}", color=CUT, hide_coplanaredges=True)

    # Minimal drilling geometry: box (bottom, top) outlines + dowel center lines.
    for index, geometry in enumerate(plate.get_features(minimal=True)):
        group.add(geometry, name=f"{plate.name}_cutline_{index}", linecolor=CUTMIN, linewidth=3)


viewer = make_viewer(data_dir)
scene = TeeScene(viewer.scene)  # draw to the viewer AND record a Rhino bundle

# 1) The frame beams as assembled (before laying flat).
assembled = scene.add_group("frame_assembled")
for plate in beams:
    draw_plate(plate, assembled)

# 2) Lay each beam flat at its plan position, explode them radially apart from the
#    group's plan centre for the overall fan-out, then run a footprint de-overlap
#    pass to clear the remaining crossings (tsections over ribs) by the minimum
#    amount - so the layout is as close as possible with no intersections. Then
#    write the fab layout and draw it.
for plate in beams:
    plate.transformation = lay_flat(plate)

explode_from_centre(beams, SPREAD)
deoverlap(beams, MARGIN)

compas.json_dump(quarter, data_dir / "quarter_frame_fab_model.json")

flat = scene.add_group("frame_flat")
for plate in beams:
    draw_plate(plate, flat)

# Plain, already-computed geometry for Rhino (no recompute on load) - see RHINO below.
dump_bundle(scene, data_dir / "quarter_frame_fab_rhino.json")

viewer.show()


# ====================================================================== #
#  RHINO  -  copy the code between the triple quotes into the Rhino 8
#  ScriptEditor (Python 3) and Run it to add THIS example's geometry to the
#  active Rhino document (named layers, per-object colour). Needs only the
#  installed compas_tf (see install steps); recomputes nothing.
# ====================================================================== #
RHINO = r'''
from compas_tf.rhino import draw_bundle
draw_bundle(r"C:\brg\compas_tf\data\quarter_frame_fab_rhino.json")
'''
