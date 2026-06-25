import pathlib

import compas
from compas.geometry import Frame
from compas.geometry import Point
from compas.geometry import Rotation
from compas.geometry import Transformation
from compas.geometry import Translation
from compas.geometry import Vector
from compas.geometry import angle_vectors_signed
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
PLANE = (0.95, 0.6, 0.1)  # plate base plane + normal
CUTMIN = (0.1, 0.5, 1.0)  # minimal cut geometry (dowel lines + wedge polylines)
SPREAD = 500.0  # explode distance: push each strip this far from the group centre

# ------------------------------------------------------------------ #
# Lift "quarter_model_0" out of the cantilevers model and UNROLL its three bed
# strips flat on the ground for fabrication.
#
# A quarter has three beds (groups ``beds_0_0``, ``beds_1_0``, ``beds_2_0``),
# each a CHAIN of six plates that share an edge with the next one and curve up
# off the floor. "Unrolling" a bed hinges every plate about its shared edge into
# its neighbour's plane (``hinge_unfold``), so the six plates land co-planar,
# still touching, in the order they were assembled - the developed strip.
#
# The developed strip is dropped onto the ground oriented by the GROUP'S COMMON
# AXIS - the chain direction running through the plates' base-frame points,
# projected to z=0 - NOT re-anchored to worldXY. So each bed keeps the
# orientation and footprint it had in 3D and the three strips fan out exactly
# like the assembled quarter seen from above - the ground layout closely
# resembles the 3D version.
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
# Beds get unrolled below; the rest of the quarter keeps its assembled position.
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
    """Unroll a bed chain flat onto the ground, keeping its 3D plan footprint.

    Composes the unfold move onto each plate's transformation (which already IS
    its world placement), so the plate lands developed on the ground in the same
    orientation and position it had in plan.
    """
    transforms = hinge_unfold(plates)
    for plate, transform in zip(plates, transforms):
        plate.transformation = transform * plate.transformation


def strip_centroid(plates):
    """Plan centroid of a strip, averaged over its plates' base points."""
    points = [plate.base_frame.point for plate in plates]
    return Point(sum(p.x for p in points) / len(points), sum(p.y for p in points) / len(points), 0.0)


def explode_from_centre(strips, distance):
    """Push every strip ``distance`` away from the whole group's plan centre.

    Each strip slides radially along the direction from the group's centre to the
    strip's own centroid, opening clear gaps between ALL strips (not just the
    outer two) so their inner ends stop crowding at the column corner.
    """
    centroids = [strip_centroid(plates) for plates in strips]
    cx = sum(c.x for c in centroids) / len(centroids)
    cy = sum(c.y for c in centroids) / len(centroids)
    for plates, centroid in zip(strips, centroids):
        direction = Vector(centroid.x - cx, centroid.y - cy, 0.0)
        if direction.length < 1e-9:
            continue
        offset = Translation.from_vector(direction.unitized() * distance)
        for plate in plates:
            plate.transformation = offset * plate.transformation


def draw_plate(plate, parent):
    """Draw a plate and everything tied to it in ONE group named after the plate:
    the carved mesh (grey), its bottom/top boundary polylines (black), its base
    plane + normal (orange), and its minimal cut geometry (blue).
    """
    group = parent.add_group(plate.name)
    group.add(triangulated(plate.modelgeometry), name=plate.name, hide_coplanaredges=True, color=GREY)

    bottom, top = plate.fabrication_polylines()  # co-wound (same winding)
    group.add(bottom, name=f"{plate.name}_bottom", linecolor=BLACK, linewidth=3)
    group.add(top, name=f"{plate.name}_top", linecolor=BLACK, linewidth=3)

    rectangle, normal = frame_rectangle(plate.base_frame, scale=150)
    group.add(rectangle, name=f"{plate.name}_base_plane", facecolor=PLANE, opacity=0.3)
    group.add(normal, name=f"{plate.name}_base_normal", linecolor=PLANE, linewidth=2)

    for index, geometry in enumerate(plate.get_features(minimal=True)):
        group.add(geometry, name=f"{plate.name}_cut_{index}", linecolor=CUTMIN, linewidth=3)


viewer = make_viewer(data_dir)
scene = TeeScene(viewer.scene)  # draw to the viewer AND record a Rhino bundle

# 1) The three bed strips as assembled (before unrolling).
assembled = scene.add_group("beds_assembled")
strips = [bed_strip(name) for name in BED_GROUPS]
for name, plates in zip(BED_GROUPS, strips):
    group = assembled.add_group(name)
    for plate in plates:
        draw_plate(plate, group)

# 2) Unroll each strip flat, each keeping its own 3D plan footprint, so the three
#    developed strips fan out just like the assembled quarter seen from above.
for plates in strips:
    unroll_strip(plates)

# 2b) Explode the strips apart: push each away from the whole group's plan centre
#     so their inner ends (which share the column corner) stop crowding/colliding.
explode_from_centre(strips, SPREAD)

compas.json_dump(quarter, data_dir / "quarter_fab_model.json")

# 3) Draw the unrolled layout.
flat = scene.add_group("beds_unrolled")
for name, plates in zip(BED_GROUPS, strips):
    group = flat.add_group(name)
    for plate in plates:
        draw_plate(plate, group)

# Plain, already-computed geometry for Rhino (no recompute on load) - see RHINO below.
dump_bundle(scene, data_dir / "quarter_bed_fab_rhino.json")

viewer.show()


# ====================================================================== #
#  RHINO  -  copy the code between the triple quotes into the Rhino 8
#  ScriptEditor (Python 3) and Run it to add THIS example's geometry to the
#  active Rhino document (named layers, per-object colour). Needs only the
#  installed compas_tf (see install steps); recomputes nothing.
# ====================================================================== #
RHINO = r'''
from compas_tf.rhino import draw_bundle
draw_bundle(r"C:\brg\compas_tf\data\quarter_bed_fab_rhino.json")
'''
