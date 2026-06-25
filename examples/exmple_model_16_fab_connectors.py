import pathlib

import compas
from compas.geometry import Frame
from compas.geometry import Point
from compas.geometry import Transformation
from compas.geometry import Vector
from compas_model.models import Model

from compas_tf.connectors import ConnectorElement
from compas_tf.connectors import ConnectorWedgeElement
from compas_tf.viewer import make_viewer
from compas_tf.viewer import triangulated

data_dir = pathlib.Path(__file__).parent.parent / "data"

GREY = (0.85, 0.85, 0.85)  # connector solids (the box shows its drilled-through holes)
RED = (0.9, 0.2, 0.2)  # box connector drill-hole cutters (the dowel cylinders)
GREEN = (0.2, 0.7, 0.3)  # wedge dowel cylinders
GAP = 250.0  # spacing left between laid-flat connectors along +X

# ------------------------------------------------------------------ #
# The connectors of ONE quarter (quarter_model_0), laid out flat for fabrication:
#
# - the THREE wedge connectors the quarter shares with the oculus ring: one on
#   each of its two inner edges (shared with the neighbouring quarters) plus the
#   inner-corner wedge (connector_wedge_0 / _1 / _2);
# - ONE column <-> rib box connector PLATE (connector_0), which carries its four
#   dowel drill holes as a boolean MeshCutFeature - so its modelgeometry is the
#   plate already drilled through.
#
# Each connector is lifted out of the cantilevers model into a fresh model with
# its full world placement folded into its OWN transformation (no parent groups),
# so its transformation IS its world placement - modelgeometry, create_cylinders()
# and the drilled holes then all agree, and each part can be re-oriented on its
# own. orient_to_xy then lays each one flat on worldXY on its broadest face (its
# shortest bounding-box dimension pointing up), spread out in a row along +X.
# ------------------------------------------------------------------ #

WEDGE_NAMES = ["connector_wedge_0", "connector_wedge_1", "connector_wedge_2"]
BOX_NAMES = ["connector_0"]

model: Model = compas.json_load(data_dir / "cantilevers_model.json")
by_name = {element.name: element for element in model.elements()}

fab = Model(name="quarter_connectors_fab_model")
wedges_group = fab.add_group("wedges")
boxes_group = fab.add_group("box_connectors")


def lift(name, parent):
    """Copy connector ``name`` into the fab model with its world placement folded
    into its own transformation (so transformation == world placement)."""
    source = by_name[name]
    part = source.copy()
    part.transformation = source.modeltransformation
    fab.add_element(part, parent=parent)
    return part


wedges = [lift(name, wedges_group) for name in WEDGE_NAMES]
boxes = [lift(name, boxes_group) for name in BOX_NAMES]
connectors = wedges + boxes


def obb_of(mesh):
    """The mesh's oriented bounding box (``Mesh.obb`` is a method here, a property
    in some compas versions - call it either way)."""
    obb = mesh.obb
    return obb() if callable(obb) else obb


def orient_to_xy(mesh, ox, oy=0.0):
    """Transformation that lays a world-space mesh flat on worldXY, resting on its
    broadest face, centred at ``(ox, oy)``.

    The part's oriented bounding box gives three axes; the SHORTEST becomes the
    layout up (+Z) and the LONGEST becomes +X, so the largest face sits on the
    ground (z = 0) and the body extrudes up. The move is returned as a world-space
    transformation to compose onto the part's current placement."""
    obb = obb_of(mesh)
    axes = sorted(
        [
            (obb.xsize, Vector(*obb.frame.xaxis)),
            (obb.ysize, Vector(*obb.frame.yaxis)),
            (obb.zsize, Vector(*obb.frame.zaxis)),
        ],
        key=lambda item: item[0],
    )
    up_size, up = axes[0]
    long_axis = axes[2][1]
    source = Frame(Point(*obb.frame.point), long_axis, up.cross(long_axis))  # z-axis = up
    target = Frame(Point(ox, oy, 0.5 * up_size), Vector(1, 0, 0), Vector(0, 1, 0))
    return Transformation.from_frame_to_frame(source, target)


def draw_connector(part, parent):
    """Draw a connector and its drilling in ONE group named after it: the carved
    solid (grey), plus - for a wedge - its dowel cylinders (green), or - for a box
    connector - its dowel drill-hole cutters (red), placed in the part's current
    frame so they follow it whether assembled or laid flat."""
    group = parent.add_group(part.name)
    group.add(triangulated(part.modelgeometry), name=part.name, hide_coplanaredges=True, color=GREY)

    if isinstance(part, ConnectorWedgeElement):
        for index, cylinder in enumerate(part.create_cylinders()):
            group.add(triangulated(cylinder.boolean_geometry), name=f"{part.name}_dowel_{index}", color=GREEN, hide_coplanaredges=True)
    elif isinstance(part, ConnectorElement):
        for feature in part._features:
            for index, mesh in enumerate(getattr(feature, "meshes", None) or []):
                placed = mesh.transformed(part.modeltransformation)
                group.add(triangulated(placed), name=f"{part.name}_drill_{index}", color=RED, hide_coplanaredges=True)


viewer = make_viewer(data_dir)

# 1) The connectors as assembled (before orienting to 2D).
assembled = viewer.scene.add_group("connectors_assembled")
for part in connectors:
    draw_connector(part, assembled)

# 2) Orient each connector to 2D, laid out in a row along +X by its footprint.
cursor = 0.0
for part in connectors:
    obb = obb_of(part.modelgeometry)
    long_size = max(obb.xsize, obb.ysize, obb.zsize)
    part.transformation = orient_to_xy(part.modelgeometry, cursor + 0.5 * long_size) * part.transformation
    cursor += long_size + GAP

compas.json_dump(fab, data_dir / "quarter_connectors_fab_model.json")

flat = viewer.scene.add_group("connectors_flat")
for part in connectors:
    draw_connector(part, flat)

viewer.show()
