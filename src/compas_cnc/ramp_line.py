"""Linear-ramp (plunge) milling tool-path for a narrow slot cut along one line."""

import math

from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import Vector

__all__ = ["toolpath_ramp_line"]

MIN_CLEARANCE = 10.0  # a retract must clear the tool-path by at least this (units)
RAMP_ANGLE_DEFAULT = math.radians(15.0)  # gentle default ramp when none is given


class toolpath_ramp_line:
    """Ramp (linear-plunge) tool-path that cuts a narrow slot the tool cannot
    clear sideways, by descending gradually back and forth along ONE line.

    A slot only as wide as the tool itself leaves no room for a zigzag clearing
    pass, so the tool follows the slot's CENTRELINE and ramps down: it traverses
    the line losing a little Z, reverses and traverses back losing a little more,
    sawtoothing down until it reaches the floor. An optional flat finishing pass
    then cleans the bottom of the slot at full depth.

    Sequence of the path:

    1. **Plunge** -- a vertical line from ``safe_z`` down onto the start end of
       the top centreline.
    2. **Ramp** -- back-and-forth sweeps along the line, each descending by
       ``step`` (so the ramp angle is ``atan(step / line_length)``), until the
       floor is reached.
    3. **Floor** -- one flat finishing pass along the line at full depth
       (optional, ``bottom_pass``).
    4. **Retract** -- a vertical line back up to ``safe_z``.

    Parameters
    ----------
    line : :class:`compas.geometry.Line`
        The slot centreline at the MOUTH (top) of the cut. Its length is the
        traverse distance of every ramp pass.
    descent : :class:`compas.geometry.Vector`
        Vector from the mouth straight to the floor; its length is the cut depth
        and its direction is the descent direction (typically straight down).
    step : float, optional
        Target vertical descent per ramp pass. The actual step is ``<= step``
        (the pass count is rounded up). Takes precedence over ``ramp_angle``.
    ramp_angle : float, optional
        Ramp angle in radians, used when ``step`` is not given: the per-pass
        descent becomes ``line_length * tan(ramp_angle)``. Defaults to
        :data:`RAMP_ANGLE_DEFAULT` when neither is supplied.
    bottom_pass : bool, optional
        Add one flat finishing pass along the line at full depth. Defaults to
        ``True``.
    safe_z : float, optional
        World-Z to plunge from / retract to. Clamped up to at least
        :data:`MIN_CLEARANCE` above the mouth. Defaults to
        ``mouth_z + MIN_CLEARANCE``.

    Attributes
    ----------
    path : :class:`compas.geometry.Polyline`
        The full tool-centre path: plunge, ramp, floor pass, retract.
    ramp : :class:`compas.geometry.Polyline`
        The descending sawtooth (and floor pass), without the lead-in/out.
    passes : int
        Number of ramp traverses.
    step : float
        Actual vertical descent per pass.
    ramp_angle : float
        Actual ramp angle (radians).
    depth : float
        Total descent (``= descent.length``).
    safe_z : float
    """

    def __init__(self, line, descent, step=None, ramp_angle=None, bottom_pass=True, safe_z=None):
        self.line = line
        self.descent = Vector(*descent)
        self._step_request = None if step is None else float(step)
        self._ramp_angle_request = None if ramp_angle is None else float(ramp_angle)
        self.bottom_pass = bottom_pass
        self._safe_z_request = safe_z
        self._build()

    # ------------------------------------------------------------------ #
    # Constructor from a thin box cutter
    # ------------------------------------------------------------------ #

    @classmethod
    def from_box(cls, mesh, end_inset=0.0, step=None, ramp_angle=None, bottom_pass=True, safe_z=None):
        """Build the ramp from a thin box-shaped cutter solid.

        The box's three edge directions are recovered from a corner. The edge
        most aligned with world Z is the DEPTH (descent) axis; of the other two
        the longer is the cut LINE and the shorter is the slot WIDTH (ignored --
        it is what makes the slot too narrow to clear sideways). The centreline
        runs along the cut axis, centred in width, at the top of the depth axis.

        Parameters
        ----------
        mesh : :class:`compas.datastructures.Mesh`
            A box cutter (8 vertices, every corner of degree 3).
        end_inset : float, optional
            Pull the centreline ends in from the short end faces by this much
            (e.g. the tool radius, so the tool stays within the slot). Defaults
            to ``0.0``.

        Returns
        -------
        :class:`toolpath_ramp_line` | None
            ``None`` if ``mesh`` is not a clean box.
        """
        verts = list(mesh.vertices())
        if len(verts) != 8 or any(len(list(mesh.vertex_neighbors(v))) != 3 for v in verts):
            return None
        coords = [Point(*mesh.vertex_coordinates(v)) for v in verts]
        centroid = Point(
            sum(p.x for p in coords) / 8.0,
            sum(p.y for p in coords) / 8.0,
            sum(p.z for p in coords) / 8.0,
        )
        p0 = coords[0]
        edges = [Point(*mesh.vertex_coordinates(n)) - p0 for n in mesh.vertex_neighbors(verts[0])]

        # DEPTH = the edge most aligned with world Z; of the rest the longer is
        # the cut LINE, the shorter the slot width.
        depth_edge = max(edges, key=lambda e: abs(e.unitized().z))
        rest = sorted((e for e in edges if e is not depth_edge), key=lambda e: e.length)
        cut_edge = rest[-1]

        depth_len = depth_edge.length
        up = depth_edge.unitized()
        if up.z < 0:
            up = up * -1.0  # point toward the mouth (higher Z)
        descent = up * (-depth_len)  # mouth -> floor

        cut_dir = cut_edge.unitized()
        half = max(0.0, cut_edge.length * 0.5 - end_inset)
        top_center = centroid + up * (depth_len * 0.5)
        line = Line(top_center - cut_dir * half, top_center + cut_dir * half)
        return cls(line, descent, step=step, ramp_angle=ramp_angle, bottom_pass=bottom_pass, safe_z=safe_z)

    # ------------------------------------------------------------------ #

    def _build(self):
        a = Point(*self.line.start)
        b = Point(*self.line.end)
        length = self.line.length
        if length < 1e-9:
            raise ValueError("ramp line has zero length.")
        self.depth = self.descent.length
        if self.depth < 1e-9:
            raise ValueError("descent has zero length.")

        # Vertical descent per pass: explicit `step`, else from `ramp_angle`,
        # else the default ramp angle. Clamp the pass count up so step <= request.
        if self._step_request is not None:
            target = self._step_request
        else:
            angle = RAMP_ANGLE_DEFAULT if self._ramp_angle_request is None else self._ramp_angle_request
            target = length * math.tan(angle)
        target = max(target, 1e-6)
        n = max(1, math.ceil(self.depth / target))
        self.passes = n
        self.step = self.depth / n
        self.ramp_angle = math.atan2(self.step, length)

        # Sawtooth down: alternate ends, descending by descent/n each traverse.
        ends = [a, b]
        points = [ends[i % 2] + self.descent * (i / n) for i in range(n + 1)]

        # One flat finishing pass along the floor to the opposite end.
        if self.bottom_pass:
            points.append(ends[(n + 1) % 2] + self.descent)

        self.ramp = Polyline(points)

        # Vertical plunge-in / retract-out, kept >= MIN_CLEARANCE above the mouth.
        mouth_z = max(p[2] for p in points)
        floor = mouth_z + MIN_CLEARANCE
        self.safe_z = floor if self._safe_z_request is None else max(self._safe_z_request, floor)
        lead_in = Point(a[0], a[1], self.safe_z)
        last = points[-1]
        lead_out = Point(last[0], last[1], self.safe_z)
        self.path = Polyline([lead_in] + points + [lead_out])

    def __repr__(self):
        return (
            f"toolpath_ramp_line(passes={self.passes}, step={self.step:.3f}, "
            f"ramp_angle={math.degrees(self.ramp_angle):.1f}deg, depth={self.depth:.3f}, safe_z={self.safe_z})"
        )
