"""Helical drilling / boring milling tool-path."""

import math

from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import Vector

__all__ = ["toolpath_helix_drill"]

MIN_CLEARANCE = 10.0  # a retract must clear the tool-path by at least this (units)


class toolpath_helix_drill:
    """Helical drilling / boring tool-path that spirals down a cylindrical hole.

    A flat end mill of diameter ``tool_diameter`` cannot plunge straight into a
    hole wider than itself and clear it, so it descends on a HELIX instead
    (helical interpolation). The tool CENTRE travels on a helix of radius
    ``hole_radius - tool_radius`` around the axis -- so the tool's outer edge just
    reaches the hole wall -- descending by ``pitch`` per revolution. An optional
    finishing circle at the bottom cleans the floor.

    The axis need not be vertical: the helix is built in the plane perpendicular
    to ``axis`` and descends along it, so it works for the horizontal bolt-holes
    in the column example as well as for vertical drilling.

    If the tool is as wide as (or wider than) the hole a helix is impossible, so
    the path degenerates to a straight plunge down the axis (an ordinary drill)
    and :attr:`is_drill` is set.

    Parameters
    ----------
    axis : :class:`compas.geometry.Line`
        Hole axis. The endpoint with the larger Z is taken as the TOP (the mouth,
        where the tool enters); the other is the bottom. Its length is the
        drilling depth and its direction is the descent direction.
    hole_radius : float
        Radius of the hole to bore.
    tool_diameter : float
        Diameter of the cutting tool.
    pitch : float, optional
        Axial descent per revolution. Defaults to ``tool_diameter`` (one tool
        width per turn). Must be > 0.
    segments : int, optional
        Straight samples per revolution (helix smoothness). Defaults to ``64``.
    bottom_pass : bool, optional
        Add one full finishing circle at the bottom of the hole. Defaults to
        ``True``.
    length : float, optional
        Override the total drill length. The bottom is anchored and the TOP is
        extended outward along the axis so the path spans exactly ``length`` --
        use this to start the tool above the stock when the hole itself is short.
        Applied after ``floor``. Defaults to ``None`` (use the axis as given).
    floor : float, optional
        Lowest world-Z the tool may reach. If the bottom is below it, the bottom
        is raised along the axis until it sits at ``floor`` (the drill "stops
        there"). Skipped for a horizontal axis (no Z component to clamp). Defaults
        to ``None`` (no limit).
    safe_z : float, optional
        World-Z to retract to at the end. Once the tool finishes at the bottom, a
        final point is appended straight above where it ended (the last point's X
        and Y, at this Z) so the tool lifts clear. Clamped up to at least
        :data:`MIN_CLEARANCE` above the whole path. Defaults to ``None`` (no
        retract).

    Attributes
    ----------
    path : :class:`compas.geometry.Polyline`
        The full tool-centre path (helix, the optional bottom circle, then the
        optional Z-safety retract).
    safe_z : float | None
        The effective retract height, or ``None`` when no retract was requested.
    drill_axis : :class:`compas.geometry.Line`
        The effective axis actually drilled (top -> bottom) after ``length`` and
        ``floor`` are applied.
    depth : float
        Length of :attr:`drill_axis`.
    helix_radius : float
        Radius of the helical tool-centre path (``0`` for a straight plunge).
    turns : float
        Number of revolutions in the descent.
    is_drill : bool
        ``True`` when the path degenerated to a straight plunge (tool >= hole).
    """

    def __init__(self, axis, hole_radius, tool_diameter, pitch=None, segments=64, bottom_pass=True, length=None, floor=None, safe_z=None):
        self.axis = axis
        self.hole_radius = float(hole_radius)
        self.tool_diameter = float(tool_diameter)
        self.tool_radius = self.tool_diameter / 2.0
        self.pitch = self.tool_diameter if pitch is None else float(pitch)
        if self.pitch <= 0:
            raise ValueError("pitch must be positive.")
        self.segments = int(segments)
        self.bottom_pass = bottom_pass
        self.length = None if length is None else float(length)
        self.floor = None if floor is None else float(floor)
        self._safe_z_request = None if safe_z is None else float(safe_z)
        self._build()

    def _build(self):
        # Orient the axis so `top` is the higher-Z end (the mouth) and `bottom`
        # the deeper end; `up` is the unit vector from bottom toward top.
        top = Point(*self.axis.start)
        bottom = Point(*self.axis.end)
        if bottom[2] > top[2]:
            top, bottom = bottom, top
        up = top - bottom
        if up.length < 1e-9:
            raise ValueError("axis has zero length.")
        up.unitize()

        # The drill must not go below `floor`: slide the bottom up along the axis
        # until it sits at that Z. A horizontal axis has no Z to clamp, so skip it.
        if self.floor is not None and bottom[2] < self.floor and abs(up[2]) > 1e-9:
            bottom = bottom + up * ((self.floor - bottom[2]) / up[2])

        # Override the total length, extending the TOP away from the anchored bottom.
        if self.length is not None:
            top = bottom + up * self.length

        self.drill_axis = Line(top, bottom)
        direction = bottom - top
        depth = direction.length
        if depth < 1e-9:
            raise ValueError("drill axis collapsed to zero length.")
        self.depth = depth
        direction = direction * (1.0 / depth)  # unit top -> bottom

        self.helix_radius = max(0.0, self.hole_radius - self.tool_radius)
        self.is_drill = self.helix_radius < 1e-9

        if self.is_drill:
            # Tool as wide as the hole: a helix is impossible, just plunge.
            self.turns = 0.0
            points = [top, bottom]
        else:
            # An orthonormal frame (u, v) spanning the plane perpendicular to the axis.
            reference = Vector(0, 0, 1)
            if abs(direction.dot(reference)) > 0.9:
                reference = Vector(1, 0, 0)
            u = direction.cross(reference)
            u.unitize()
            v = direction.cross(u)
            v.unitize()

            self.turns = depth / self.pitch
            count = max(1, round(self.segments * self.turns))

            points = []
            for i in range(count + 1):
                t = i / count  # 0..1 along the axis
                angle = 2.0 * math.pi * self.turns * t
                center = top + direction * (depth * t)
                offset = u * (self.helix_radius * math.cos(angle)) + v * (self.helix_radius * math.sin(angle))
                points.append(center + offset)

            if self.bottom_pass:
                # One full circle at the bottom of the hole to clean the floor.
                base = 2.0 * math.pi * self.turns
                ring = max(1, self.segments)
                for i in range(1, ring + 1):
                    angle = base + 2.0 * math.pi * (i / ring)
                    offset = u * (self.helix_radius * math.cos(angle)) + v * (self.helix_radius * math.sin(angle))
                    points.append(bottom + offset)

        # Z-safety retract: lift straight up from the LAST point (its X, Y) to
        # safe_z, clamped to clear the whole path.
        if self._safe_z_request is not None:
            clearance = max(p[2] for p in points) + MIN_CLEARANCE
            self.safe_z = max(self._safe_z_request, clearance)
            last = points[-1]
            points.append(Point(last[0], last[1], self.safe_z))
        else:
            self.safe_z = None

        self.path = Polyline(points)

    def __repr__(self):
        if self.is_drill:
            return f"toolpath_helix_drill(drill plunge, depth={self.depth:.3f})"
        return f"toolpath_helix_drill(helix_radius={self.helix_radius:.3f}, turns={self.turns:.2f}, depth={self.depth:.3f})"
