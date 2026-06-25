import pathlib

import compas
from compas.geometry import Frame
from compas.geometry import Point
from compas.geometry import Transformation
from compas.geometry import Vector
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
CUTMIN = (0.1, 0.5, 1.0)  # minimal cut geometry (dowel center lines)
CUT = (0.9, 0.2, 0.2)  # cut feature solids (box slot + dowels)

# ------------------------------------------------------------------ #
# Lift the "oculus" group out of the cantilevers model. Fold each plate's full
# placement into its OWN transformation (and zero the group's), so a plate's
# transformation IS its world placement - base_frame, fabrication_polylines and
# modelgeometry then all agree, and each plate can be re-placed on its own.
# ------------------------------------------------------------------ #

model: Model = compas.json_load(data_dir / "cantilevers_model.json")
oculus: Model = model.find_group_with_name("oculus")
plates = [element for element in oculus.elements() if isinstance(element, PlateElement)]

placements = [plate.modeltransformation for plate in plates]
oculus.transformation = Transformation()
for plate, placement in zip(plates, placements):
    plate.transformation = placement


def lay_flat(plate, offset=0.0):
    """Transformation that lays the plate flat on worldXY at its plan position,
    its body extruding UP (+Z), pushed out by ``offset``.

    The move is built from the plate's current (placed) base_frame and then
    COMPOSED onto its current transformation - ``from_frame_to_frame(base,
    target)`` is a world-space move, so assigning it directly would only be
    correct for an un-placed plate. The target frame's z-axis is +Z, so the
    plate lands on the ground with its body above it (not below)."""
    base = plate.base_frame
    xaxis = Vector(base.xaxis[0], base.xaxis[1], 0).unitized()  # plate long edge, horizontal
    yaxis = Vector(0, 0, 1).cross(xaxis)  # zaxis = xaxis x yaxis = +Z (up)
    point = Point(base.point[0] + yaxis[0] * offset, base.point[1] + yaxis[1] * offset, 0)
    target = Frame(point, xaxis, yaxis)
    return Transformation.from_frame_to_frame(base, target) * plate.transformation


def draw_plate(plate, parent):
    """Draw a plate and everything tied to it - in ONE group named after the
    plate, so the scene tree keeps them together:

    - the carved plate mesh (grey),
    - its bottom/top boundary polylines (black, co-wound),
    - its base plane + normal (orange),
    - its cut feature SOLIDS (red: the box slot + dowel cylinders),
    - the dowel center lines (blue, minimal geometry).

    Everything is named after the plate, so the scene tree groups by plate.
    """
    group = parent.add_group(plate.name)
    group.add(triangulated(plate.modelgeometry), name=plate.name, hide_coplanaredges=True, color=GREY)

    bottom, top = plate.fabrication_polylines()  # co-wound (same winding)
    group.add(bottom, name=f"{plate.name}_bottom", linecolor=BLACK, linewidth=3)
    group.add(top, name=f"{plate.name}_top", linecolor=BLACK, linewidth=3)

    rectangle, normal = frame_rectangle(plate.base_frame, scale=150)
    group.add(rectangle, name=f"{plate.name}_base_plane", facecolor=PLANE, opacity=0.3)
    group.add(normal, name=f"{plate.name}_base_normal", linecolor=PLANE, linewidth=2)

    # The boolean cut features carving this plate - box slot + dowels - as solids,
    # placed in the plate's (current) frame. get_features() returns them already
    # transformed, so they follow the plate whether assembled or laid flat.
    for feature in plate.get_features():
        for mesh in getattr(feature, "meshes", None) or []:
            group.add(triangulated(mesh), name=f"{plate.name}__{feature.name or 'cut'}", color=CUT, hide_coplanaredges=True)

    # The dowel center lines (parametric minimal geometry; the box has none).
    for index, geometry in enumerate(plate.get_features(minimal=True)):
        group.add(geometry, name=f"{plate.name}_cutline_{index}", linecolor=CUTMIN, linewidth=3)


viewer = make_viewer(data_dir)
scene = TeeScene(viewer.scene)  # draw to the viewer AND record a Rhino bundle

# 1) The oculus as assembled (before laying flat).
assembled = scene.add_group("oculus_assembled")
for plate in plates:
    draw_plate(plate, assembled)

# 2) Lay each plate flat on the ground - the boundary plates (0..3) pushed out
#    further than the wedges (4..7); the inner plate (8) stays at the origin -
#    then write that fabrication layout to its OWN file (NOT oculus_model.json,
#    which example_model_5 owns) and draw it.
offsets = [200, 200, 200, 200, 50, 50, 50, 50, 0]
for plate, offset in zip(plates, offsets):
    plate.transformation = lay_flat(plate, offset)

compas.json_dump(oculus, data_dir / "oculus_fab_model.json")

flat = scene.add_group("oculus_flat")
for plate in plates:
    draw_plate(plate, flat)

# Plain, already-computed geometry for Rhino (no recompute on load) - see RHINO below.
dump_bundle(scene, data_dir / "oculus_fab_rhino.json")

viewer.show()


# ====================================================================== #
#  RHINO  -  copy the code between the triple quotes into the Rhino 8
#  ScriptEditor (Python 3) and Run it to add THIS example's geometry to the
#  active Rhino document (named layers, per-object colour). Needs only the
#  installed compas_tf (see install steps); recomputes nothing.
# ====================================================================== #
RHINO = r'''
from compas_tf.rhino import draw_bundle
draw_bundle(r"C:\brg\compas_tf\data\oculus_fab_rhino.json")
'''
