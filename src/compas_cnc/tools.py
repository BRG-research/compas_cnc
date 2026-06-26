import math

from compas.datastructures import Mesh
from compas.geometry import Point
from compas.geometry import Translation


def point_at(path, t):
    """Point at arc-length fraction ``t`` (0..1) along a polyline ``path``."""
    pts = [Point(*p) for p in path]
    if len(pts) < 2:
        return pts[0]
    seglen = [pts[i].distance_to_point(pts[i + 1]) for i in range(len(pts) - 1)]
    total = sum(seglen)
    if total == 0:
        return pts[0]
    target = max(0.0, min(1.0, t)) * total
    walked = 0.0
    for (a, b), d in zip(zip(pts, pts[1:]), seglen):
        if d and walked + d >= target:
            return a + (b - a) * ((target - walked) / d)
        walked += d
    return pts[-1]


def _revolve(profile, sides):
    """Mesh of the ``(r, z)`` profile revolved about the z-axis (``r == 0`` => axis point)."""
    mesh = Mesh()
    rings = []
    for r, z in profile:
        if r <= 1e-9:
            rings.append([mesh.add_vertex(x=0.0, y=0.0, z=z)])
        else:
            rings.append([mesh.add_vertex(x=r * math.cos(2 * math.pi * k / sides), y=r * math.sin(2 * math.pi * k / sides), z=z) for k in range(sides)])
    for lower, upper in zip(rings, rings[1:]):
        if len(lower) == 1:  # axis point -> ring (apex cone / bottom cap)
            for k in range(sides):
                mesh.add_face([lower[0], upper[k], upper[(k + 1) % sides]])
        elif len(upper) == 1:  # ring -> axis point (top cap)
            for k in range(sides):
                mesh.add_face([lower[k], lower[(k + 1) % sides], upper[0]])
        else:  # ring -> ring (cylinder / cone wall)
            for k in range(sides):
                mesh.add_face([lower[k], lower[(k + 1) % sides], upper[(k + 1) % sides], upper[k]])
    return mesh


class Tool:
    """A milling tool drawn as a solid of revolution: a cylinder of ``diameter`` and
    ``height``, with an optional sharp conical tip of ``cone_height`` (the cone is
    cut into the bottom of the cylinder, so the total length stays ``height``).

    The tool TIP sits at the local origin and the body rises +Z, so dropping the
    tip onto a tool-path point seats the cutter on the cut.
    """

    def __init__(self, diameter, height, cone_height=0.0, sides=24, name="tool"):
        self.diameter = float(diameter)
        self.height = float(height)
        self.cone_height = float(cone_height)
        self.sides = int(sides)
        self.name = name

    @property
    def radius(self):
        """Tool radius -- half the ``diameter``. Setting it scales ``diameter`` to match."""
        return self.diameter / 2.0

    @radius.setter
    def radius(self, value):
        self.diameter = float(value) * 2.0

    def solid(self, tip=(0.0, 0.0, 0.0)):
        """Tool :class:`compas.datastructures.Mesh` with its tip at ``tip``, body rising +Z."""
        r = self.radius
        if self.cone_height > 0:
            profile = [(0.0, 0.0), (r, self.cone_height), (r, self.height), (0.0, self.height)]
        else:
            profile = [(0.0, 0.0), (r, 0.0), (r, self.height), (0.0, self.height)]
        return _revolve(profile, self.sides).transformed(Translation.from_vector(list(tip)))

    def at(self, path, t=0.0):
        """Tool mesh placed at arc-length parameter ``t`` (0..1) along ``path``."""
        return self.solid(point_at(path, t))

    def __repr__(self):
        cone = f", cone_height={self.cone_height}" if self.cone_height else ""
        return f"Tool({self.name}, diameter={self.diameter}, height={self.height}{cone})"


def add_tool_slider(viewer, tool, path, title="t", **style):
    """Add ``tool`` to a LIVE ``viewer`` with a side-dock slider that scrubs it
    along ``path`` (0..100 %). Returns the tool scene object. Call before ``show``."""
    from compas.geometry import Translation
    from compas_viewer.components import Slider

    obj = viewer.scene.add(tool.solid(), name=tool.name, **style)

    def _move(_component, value):
        obj.transformation = Translation.from_vector(list(point_at(path, value / 100.0)))
        obj.update()
        viewer.renderer.update()

    obj.transformation = Translation.from_vector(list(point_at(path, 0.0)))  # seat at the start
    slider = Slider(title=title, value=0, min_val=0, max_val=100, step=0.1, action=_move)
    viewer.ui.sidedock.add(slider)
    viewer.ui.sidedock.show = True
    return obj


FLAT_3MM = Tool(3.0, 30.0, name="flat_3mm")
FLAT_3_175MM = Tool(3.175, 19.0, name="flat_3.175mm")
VBIT_3_175MM = Tool(3.175, 19.0, cone_height=3.175, name="vbit_3.175mm")
