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


def add_toolpath_slider(viewer, entries, path_objs=None, path_colors=None, red=(0.90, 0.10, 0.10)):
    """Two side-dock sliders on a LIVE ``viewer``: 'toolpath' selects a path by id
    (0..N-1) and 'position' scrubs 0..100 % along it.

    ``entries`` is a list of ``(tool, path)`` pairs. The cutter shown swaps to the
    selected path's ``tool`` -- so a 6mm path shows the 6mm cutter, a 3mm path the 3mm
    one -- and is dropped onto the scrub position. If ``path_objs`` (the live polyline
    scene objects, in the same order as ``entries``) and ``path_colors`` (their base
    RGB tuples) are given, the selected path's polyline is recoloured ``red`` and the
    rest restored -- a moving red highlight of the active tool-path. Call before
    ``viewer.show()``; a recorder/watch viewer has no ``ui`` so guard with
    ``if hasattr(viewer, "ui")``.
    """
    from compas.colors import Color
    from compas.geometry import Translation
    from compas_viewer.components import Slider

    red = Color(*red)

    # One scene solid per DISTINCT tool; only the selected path's tool is shown.
    solids = {}
    for tool, _path in entries:
        if id(tool) not in solids:
            obj = viewer.scene.add(tool.solid(), name=tool.name)
            obj.set_visible(False)
            solids[id(tool)] = obj

    state = {"idx": 0, "t": 0.0, "hl": -1}

    def refresh():
        idx = max(0, min(len(entries) - 1, int(round(state["idx"]))))
        # Recolour only the two paths whose colour changes: revert the previously
        # reddened one, redden the new one. Touching two objects instead of all N is
        # what keeps scrubbing responsive (rebuilding every buffer each tick crawls),
        # and moving 'position' -- which never changes idx -- skips this block entirely.
        if path_objs and idx != state["hl"]:
            prev = state["hl"]
            if 0 <= prev < len(path_objs):
                path_objs[prev].linecolor = Color(*path_colors[prev])
                path_objs[prev].update(update_data=True)
            path_objs[idx].linecolor = red
            path_objs[idx].update(update_data=True)
            state["hl"] = idx
        # Show only the selected path's tool, moved to its scrub position.
        tool, path = entries[idx]
        for key, obj in solids.items():
            active = key == id(tool)
            obj.set_visible(active)
            if active:
                obj.transformation = Translation.from_vector(list(point_at(path, state["t"])))
            obj.update()
        viewer.renderer.update()

    if len(entries) > 1:
        viewer.ui.sidedock.add(Slider(title="toolpath", value=0, min_val=0, max_val=len(entries) - 1, step=1, action=lambda _c, v: (state.update(idx=v), refresh())))
    viewer.ui.sidedock.add(Slider(title="position", value=0, min_val=0, max_val=100, step=0.1, action=lambda _c, v: (state.update(t=v / 100.0), refresh())))
    viewer.ui.sidedock.show = True
    refresh()


FLAT_3MM = Tool(3.0, 30.0, name="flat_3mm")
FLAT_3_175MM = Tool(3.175, 19.0, name="flat_3.175mm")
VBIT_3_175MM = Tool(3.175, 19.0, cone_height=3.175, name="vbit_3.175mm")
