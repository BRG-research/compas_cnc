import math

from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import Vector

__all__ = ["toolpath_2d_drill"]

MIN_CLEARANCE = 10.0  # a retract must clear the tool-path by at least this (units)
SEGMENTS = 64  # straight samples per revolution (helix smoothness)
RAMP_ANGLE_DEFAULT = math.radians(15.0)  # gentle default helix descent angle


class toolpath_2d_drill:
    """Helical drilling / boring tool-path that spirals down a cylindrical hole.

    A flat end mill of diameter ``tool_diameter`` cannot plunge straight into a
    hole wider than itself and clear it, so it descends on a HELIX instead
    (helical interpolation). The tool CENTRE travels on a helix of radius
    ``hole_radius - tool_radius`` around the axis -- so the tool's outer edge just
    reaches the hole wall -- spiralling down at ``ramp_angle``. An optional
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
    ramp_angle : float | None, optional
        Helix descent angle in RADIANS -- how steeply the tool spirals down. The
        tool drops ``2*pi*helix_radius * tan(ramp_angle)`` per turn, which sets the
        turn count. Gentle (a few degrees) for hard material, steeper for soft.
        Defaults to :data:`RAMP_ANGLE_DEFAULT` (15 deg). Must be in (0, 90) degrees.
    bottom_pass : bool, optional
        Add one full finishing circle at the bottom of the hole. Defaults to
        ``True``.
    floor : float | None, optional
        Lowest world-Z the tool may reach. If the bottom is below it, the bottom
        is raised along the axis until it sits at ``floor`` (the drill "stops
        there"). Skipped for a horizontal axis (no Z component to clamp). Defaults
        to ``None`` (no limit).
    safe_z : float | None, optional
        World-Z for the approach and retract. The path rapids down to the entry at
        this Z, descends, retracts straight up at the end (same XY, Z to safe_z)
        and traverses home. Clamped up to at least :data:`MIN_CLEARANCE` above the
        whole descent. Defaults to that clearance height -- z-safety is ALWAYS
        part of the path.
    direction : {None, "climb", "conventional"}, optional
        Milling direction for hard materials. Assumes a CW (M3) tool. The default
        helix already CLIMBS the bore, so ``None`` and ``"climb"`` are identical;
        ``"conventional"`` reverses the helix winding. No-op for a straight plunge
        (tool >= hole). On an M4 spindle, pick the opposite label. Defaults to ``None``.

    Attributes
    ----------
    path : :class:`compas.geometry.Polyline`
        The full tool-centre path: safe-Z approach, the helix (and optional floor
        circle), the retract, then the safe-Z move home. Starts and ends at the
        same safe-Z home position.
    helix : :class:`compas.geometry.Polyline`
        Just the raw descent (helix + optional floor circle), without z-safety.
    safe_z : float
        The effective approach/retract height (always set).
    drill_axis : :class:`compas.geometry.Line`
        The effective axis actually drilled (top -> bottom) after ``floor`` is applied.
    depth : float
        Length of :attr:`drill_axis`.
    helix_radius : float
        Radius of the helical tool-centre path (``0`` for a straight plunge).
    turns : int
        Number of WHOLE revolutions in the descent (rounded UP so the actual ramp
        angle is <= the requested one, and so the helix ends under the entry, same
        XY, letting the tool retract straight up).
    ramp_angle : float
        The ACTUAL helix angle (radians) after rounding to whole turns -- close to
        the requested angle; ``0`` for a straight plunge.
    is_drill : bool
        ``True`` when the path degenerated to a straight plunge (tool >= hole).
    """

    def __init__(self, axis, hole_radius, tool_diameter, ramp_angle=None, bottom_pass=True, floor=None, safe_z=None, direction=None):
        self.axis = axis
        self.hole_radius = float(hole_radius)
        self.tool_diameter = float(tool_diameter)
        self.tool_radius = self.tool_diameter / 2.0
        self._ramp_angle_request = RAMP_ANGLE_DEFAULT if ramp_angle is None else float(ramp_angle)
        if not (0.0 < self._ramp_angle_request < math.pi / 2.0):
            raise ValueError("ramp_angle must be between 0 and 90 degrees (in radians).")
        self.bottom_pass = bottom_pass
        self.floor = None if floor is None else float(floor)
        self._safe_z_request = None if safe_z is None else float(safe_z)
        if direction not in (None, "climb", "conventional"):
            raise ValueError("direction must be None, 'climb', or 'conventional'.")
        self.direction = direction
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
            self.ramp_angle = 0.0
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

            # Milling direction on the bore wall (a pocket-like wall: cleared region is
            # the hole interior, inside the tool-centre circle). With `u x v == direction`
            # the +sin winding is CW seen from the mouth -> CLIMB for a CW (M3) tool, so
            # `None`/`"climb"` keep the default and only `"conventional"` flips the sin.
            plus_is_cw_from_mouth = u.cross(v).dot(direction) > 0
            want_cw = self.direction != "conventional"
            sin_sign = 1.0 if (want_cw == plus_is_cw_from_mouth) else -1.0

            # The ramp angle sets the descent: per turn the tool drops
            # circumference * tan(ramp_angle). Turns are rounded UP to a WHOLE
            # number so (a) the actual angle is <= the requested one (gentler, never
            # steeper) and (b) the helix ends directly under the entry (same XY) and
            # the floor circle returns there too -- the tool then retracts STRAIGHT
            # UP from the entry. The ACTUAL angle is recomputed from the turn count.
            circumference = 2.0 * math.pi * self.helix_radius
            target_step = circumference * math.tan(self._ramp_angle_request)
            self.turns = max(1, math.ceil(depth / target_step))
            self.ramp_angle = math.atan2(depth / self.turns, circumference)
            count = max(1, round(SEGMENTS * self.turns))

            points = []
            for i in range(count + 1):
                t = i / count  # 0..1 along the axis
                angle = 2.0 * math.pi * self.turns * t
                center = top + direction * (depth * t)
                offset = u * (self.helix_radius * math.cos(angle)) + v * (self.helix_radius * sin_sign * math.sin(angle))
                points.append(center + offset)

            if self.bottom_pass:
                # One full circle at the bottom of the hole to clean the floor.
                base = 2.0 * math.pi * self.turns
                ring = max(1, SEGMENTS)
                for i in range(1, ring + 1):
                    angle = base + 2.0 * math.pi * (i / ring)
                    offset = u * (self.helix_radius * math.cos(angle)) + v * (self.helix_radius * sin_sign * math.sin(angle))
                    points.append(bottom + offset)

        self.helix = Polyline(points)  # the raw descent (+ floor circle), no z-safety

        # Z-safety is ALWAYS part of the path: rapid down to the entry at safe_z,
        # descend the hole, retract straight up at the end (same XY, Z to safe_z),
        # then traverse home at safe_z so the path ENDS where it began. safe_z
        # defaults to MIN_CLEARANCE above the whole descent.
        clearance = max(p[2] for p in points) + MIN_CLEARANCE
        self.safe_z = clearance if self._safe_z_request is None else max(self._safe_z_request, clearance)
        first, last = points[0], points[-1]
        lead_in = Point(first[0], first[1], self.safe_z)
        retract = Point(last[0], last[1], self.safe_z)
        home = Point(first[0], first[1], self.safe_z)
        self.path = Polyline([lead_in] + points + [retract, home])

    def __repr__(self):
        if self.is_drill:
            return f"toolpath_2d_drill(drill plunge, depth={self.depth:.3f})"
        return f"toolpath_2d_drill(helix_radius={self.helix_radius:.3f}, turns={self.turns}, ramp_angle={math.degrees(self.ramp_angle):.1f}deg, depth={self.depth:.3f})"
