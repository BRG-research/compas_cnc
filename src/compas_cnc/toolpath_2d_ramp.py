import math

from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import Vector

from compas_cnc._milling import oriented
from compas_cnc._milling import signed_area_xy
from compas_cnc._milling import want_ccw

__all__ = ["toolpath_2d_ramp"]

MIN_CLEARANCE = 10.0  # a retract must clear the tool-path by at least this (units)
RAMP_ANGLE_DEFAULT = math.radians(15.0)  # gentle default ramp when none is given


def _longest_circular_run(mask):
    """Indices of the longest contiguous run of ``True`` in a CIRCULAR ``mask``.

    The list wraps, so a run that straddles the start/end of the list is still
    returned as one contiguous block (in ring order).
    """
    n = len(mask)
    if all(mask):
        return list(range(n))
    if not any(mask):
        return []
    start = mask.index(False)  # begin scanning just past a gap so runs stay whole
    runs, cur = [], []
    for k in range(n):
        i = (start + 1 + k) % n
        if mask[i]:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return max(runs, key=len)


class toolpath_2d_ramp:
    """Ramp (linear-plunge) tool-path that descends gradually along an OPEN
    polyline, sweeping it back and forth.

    The path may be a single straight line -- a narrow slot the tool cannot
    clear sideways, so it follows the centreline and saws straight down -- or a
    multi-segment path, e.g. the end-cap arc sliced out of a part's silhouette
    outline, so the tool can ramp down AROUND the end of a beam to cut it off
    instead of plunging straight. Each back-and-forth traverse of the path
    descends by ``step`` (Z interpolated by arc length, so the slope is even),
    down to the floor, then an optional flat finishing pass cleans the bottom.

    Sequence of the path: plunge from ``safe_z`` to the start of the path, the
    descending sweeps, an optional flat floor pass, then a retract to ``safe_z``.

    Parameters
    ----------
    path : :class:`compas.geometry.Polyline` | :class:`compas.geometry.Line`
        The open path to ramp along, positioned at the MOUTH (top) of the cut.
        A :class:`~compas.geometry.Line` or 2-point polyline gives a straight slot.
    descent : :class:`compas.geometry.Vector`
        Vector from the mouth straight to the floor; its length is the cut depth
        and its direction is the descent direction (typically straight down).
    step : float, optional
        Target vertical descent per pass; actual is ``<= step`` (pass count
        rounded up). Takes precedence over ``ramp_angle``.
    ramp_angle : float, optional
        Ramp angle in radians, used when ``step`` is not given: per-pass descent
        is ``path_length * tan(ramp_angle)``. Defaults to
        :data:`RAMP_ANGLE_DEFAULT`.
    bottom_pass : bool, optional
        Add one flat finishing pass along the path at full depth. Defaults to
        ``True``.
    safe_z : float, optional
        World-Z to plunge from / retract to. Clamped up to at least
        :data:`MIN_CLEARANCE` above the mouth. Defaults to
        ``mouth_z + MIN_CLEARANCE``.
    offset : float, optional
        Offset the path in its plane BEFORE ramping. For an OPEN path this is a
        single-sided parallel offset (sign picks the side, via
        :func:`compas.geometry.offset_polyline`) -- use it to space parallel ramp
        lanes off one centreline. For a CLOSED path it is a uniform inset of the
        whole perimeter (positive insets, via
        :func:`compas.geometry.offset_polygon`), so a tool-radius band wraps every
        side and the incline is preserved. Defaults to ``0.0`` (no offset).
    notch : float, optional
        Tool radius for inside-corner NOTCHES (dogbone corner relief). At every
        concave vertex the cut overcuts along the corner BISECTOR -- the average
        direction of the two edges, away from both neighbours into the solid -- so
        a round tool of this radius clears the corner. Penetration follows NGon's
        ``Ears.cs``: ``R/sin(phi/2) - R`` for a corner of angle ``phi``. The notch
        is an in-and-out excursion cut on every descending pass. Defaults to
        ``0.0`` (off).
    notch_flip : bool, optional
        Which corner HANDEDNESS gets notched. By default the convex corners of a
        closed pocket (turn matching the path winding) -- the ones a round tool
        leaves material in; flip for the other handedness (an island's corners, or
        the other side of an open path). Defaults to ``False``.
    direction : {None, "climb", "conventional"}, optional
        Milling direction for hard materials (assumes a CW / M3 tool; on M4 pick the
        opposite label). Only applies to a CLOSED ramp loop: it winds every lap the
        same way to suit. An OPEN ramp is a plunge, so ``direction`` is a no-op there.
        Defaults to ``None``.
    pocket : bool, optional
        For a closed loop with ``direction`` set: ``True`` if the loop is a pocket /
        outer wall (cleared region inside), ``False`` if it profiles an island / part
        outline (kept material inside). Sets which winding is climb. Defaults to ``True``.
    tabs : iterable, optional
        Hold-down TAB markers, each an ``(x, y[, z])`` point where the cut must NOT go
        through -- an uncut bridge is left so the part stays fixed to the stock (like a
        standard CNC tab). Only XY is used: each marker is snapped to the nearest point
        on the ramp centre-path, and a marker farther than ``tab_width`` from the path
        (belonging to another contour) is ignored -- so every marker may be passed to
        every ramp. Defaults to ``None`` (no tabs).
    tab_height : float, optional
        Bridge height: how far ABOVE the descent FLOOR (the deepest cut point) the tool
        is held over each tab, i.e. the thickness of uncut stock left. Defaults to 0.5.
    tab_width : float, optional
        Flat span of each tab along the path; the tab region is a disk of radius
        ``tab_width / 2`` about the snapped marker. Defaults to 3.0.

    Attributes
    ----------
    path : :class:`compas.geometry.Polyline`
        The full tool-centre path: plunge, ramp, floor pass, retract.
    ramp : :class:`compas.geometry.Polyline`
        The descending sweeps (and floor pass), without the lead-in/out.
    notches : list[tuple[:class:`compas.geometry.Point`, :class:`compas.geometry.Point`]]
        ``(corner, tip)`` pairs at the mouth, one per notched inside corner.
    tabs : list[:class:`compas.geometry.Point`]
        Bridge-centre points (at the tab-top Z), one per applied hold-down tab.
    passes : int
    step : float
    ramp_angle : float
    depth : float
    safe_z : float
    offset : float
    """

    def __init__(self, path, descent, step=None, ramp_angle=None, bottom_pass=True, safe_z=None, offset=0.0, notch=0.0, notch_flip=False, direction=None, pocket=True, tabs=None, tab_height=0.5, tab_width=3.0):
        pts = [Point(*p) for p in path]
        if len(pts) < 2:
            raise ValueError("ramp path needs at least 2 points.")
        if offset:
            if len(pts) >= 4 and pts[0].distance_to_point(pts[-1]) < 1e-9:
                # CLOSED boundary: inset the whole perimeter uniformly IN ITS PLANE
                # (positive insets), so a tool-radius band wraps every side and the
                # incline is preserved.
                from compas.geometry import offset_polygon

                ring = [Point(*p) for p in offset_polygon(pts[:-1], float(offset))]
                pts = ring + [ring[0]]
            else:
                # OPEN path: single-sided planar parallel offset (sign picks the side).
                from compas.geometry import offset_polyline

                pts = [Point(*p) for p in offset_polyline(pts, float(offset))]
        self._pts = pts
        if direction not in (None, "climb", "conventional"):
            raise ValueError("direction must be None, 'climb', or 'conventional'.")
        self.direction = direction
        self.pocket = bool(pocket)
        # CLOSED ramp loop: rewind the lap so the cut runs climb/conventional (a 2D
        # peripheral cut). OPEN ramps are plunges, so `direction` is a no-op there.
        if direction is not None and len(self._pts) >= 4 and self._pts[0].distance_to_point(self._pts[-1]) < 1e-9:
            ring = oriented(self._pts[:-1], want_ccw(direction, self.pocket))
            self._pts = list(ring) + [ring[0]]
        self.offset = float(offset)
        self.descent = Vector(*descent)
        self._step_request = None if step is None else float(step)
        self._ramp_angle_request = None if ramp_angle is None else float(ramp_angle)
        self.bottom_pass = bottom_pass
        self._safe_z_request = safe_z
        self.notch = float(notch)
        self.notch_flip = bool(notch_flip)
        self.notches = []
        # HOLD-DOWN TABS: where the cut must stop short of full depth, leaving an
        # uncut bridge so the part stays fixed to the stock. `tabs` are marker XY
        # points (the ramp snaps each onto its own centre-path); `tab_height` is how
        # far ABOVE the descent floor the bridge top sits; `tab_width` is its span.
        self._tabs = [Point(*t) for t in tabs] if tabs else []
        self.tab_height = float(tab_height)
        self.tab_width = float(tab_width)
        self.tabs = []
        self._build()

    # ------------------------------------------------------------------ #
    # Constructor from a silhouette outline: slice the cap around one end
    # ------------------------------------------------------------------ #

    @classmethod
    def from_outline(
        cls,
        outline,
        depth,
        top_z=0.0,
        end="start",
        cap=None,
        axis=None,
        step=None,
        ramp_angle=None,
        bottom_pass=True,
        safe_z=None,
        offset=0.0,
        notch=0.0,
        notch_flip=False,
    ):
        """Slice the cap around ONE end out of a closed silhouette and ramp it down.

        The beam's long axis is taken from ``axis`` or, by default, the longer
        side of the outline's bounding box. Outline vertices within ``cap`` of
        the chosen end (along that axis) form a single contiguous arc -- the end
        cap -- which is lifted to ``top_z`` and ramped straight down by ``depth``.

        Parameters
        ----------
        outline : :class:`compas.geometry.Polyline`
            A closed silhouette outline (e.g. from :func:`compas_cnc.outline`).
        depth : float
            How far to descend (the beam height at the end).
        top_z : float, optional
            Z of the mouth -- the top of the beam where cutting starts. The
            outline is flat, so its own Z is replaced by this. Defaults to 0.0.
        end : {"start", "end"}, optional
            Which end along the axis: ``"start"`` is the low-coordinate end,
            ``"end"`` the high one. Defaults to ``"start"``.
        cap : float, optional
            How far in from the end (along the axis) to include. Small values
            give a near-straight line across the end; larger values wrap further
            down the sides. Defaults to the full cross-width of the outline.
        axis : tuple, optional
            ``(x, y)`` beam long-axis direction. Defaults to the outline's
            longer bounding-box side.

        Returns
        -------
        :class:`toolpath_2d_ramp`
        """
        pts = [Point(*p) for p in outline]
        if len(pts) > 1 and pts[0].distance_to_point(pts[-1]) < 1e-9:
            pts = pts[:-1]  # drop the duplicated closing point
        if len(pts) < 3:
            raise ValueError("outline needs at least 3 distinct points.")

        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        if axis is None:
            axis = (1.0, 0.0) if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else (0.0, 1.0)
        ax = Vector(axis[0], axis[1], 0.0).unitized()
        perp = Vector(-ax.y, ax.x, 0.0)
        along = [p.x * ax.x + p.y * ax.y for p in pts]
        across = [p.x * perp.x + p.y * perp.y for p in pts]
        if cap is None:
            cap = max(across) - min(across)

        if end == "start":
            thresh = min(along) + cap
            mask = [a <= thresh for a in along]
        elif end == "end":
            thresh = max(along) - cap
            mask = [a >= thresh for a in along]
        else:
            raise ValueError("end must be 'start' or 'end'.")

        run = _longest_circular_run(mask)
        if len(run) < 2:
            raise ValueError("no outline segment falls within `cap` of the chosen end.")
        segment = Polyline([[pts[i].x, pts[i].y, top_z] for i in run])
        descent = Vector(0.0, 0.0, -abs(depth))
        return cls(segment, descent, step=step, ramp_angle=ramp_angle, bottom_pass=bottom_pass, safe_z=safe_z, offset=offset, notch=notch, notch_flip=notch_flip)

    # ------------------------------------------------------------------ #
    # Constructor from a thin box cutter: the straight-slot case
    # ------------------------------------------------------------------ #

    @classmethod
    def from_box(cls, mesh, end_inset=0.0, step=None, ramp_angle=None, bottom_pass=True, safe_z=None, offset=0.0):
        """Build a straight-slot ramp from a thin box-shaped cutter solid.

        A slot only as wide as the tool cannot be cleared sideways, so the tool
        follows the slot's CENTRELINE and ramps down. The box's three edge
        directions are recovered from a corner. The edge most aligned with world
        Z is the DEPTH (descent) axis; of the other two the longer is the cut
        LINE and the shorter is the slot WIDTH (ignored -- it is what makes the
        slot too narrow to clear sideways). The centreline runs along the cut
        axis, centred in width, at the top of the depth axis.

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
        :class:`toolpath_2d_ramp` | None
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
        return cls(line, descent, step=step, ramp_angle=ramp_angle, bottom_pass=bottom_pass, safe_z=safe_z, offset=offset)

    # ------------------------------------------------------------------ #

    def _notch_vectors(self):
        """Map each inside-corner vertex index -> in-plane dogbone overcut vector.

        A corner is CONVEX or CONCAVE by the sign of its turn about the mouth
        normal versus the path winding (NOT a centroid guess): like NGon's
        ``Ears.cs``, only one handedness is notched. The default is the corners a
        round tool leaves material in -- the CONVEX corners of a closed pocket
        (turn matching the winding); ``notch_flip`` selects the other handedness
        (e.g. an island's corners, or the other side of an open path). The
        direction is the corner BISECTOR ``unit(V-N) + unit(V-P)`` (which points
        into the solid for the selected corner type) and the length is the
        ``Ears.cs`` penetration ``R/sin(phi/2) - R``. Populates :attr:`notches`
        (mouth-level ``(corner, tip)`` pairs); returns ``{}`` when ``notch`` is off.
        """
        self.notches = []
        R = self.notch
        pts = self._pts
        if not R or R <= 0 or len(pts) < 3:
            return {}
        closed = len(pts) >= 4 and pts[0].distance_to_point(pts[-1]) < 1e-9
        ring = pts[:-1] if closed else pts
        m = len(ring)
        if m < 3:
            return {}
        # Reference normal = out of the solid toward the mouth; turns are measured
        # about it. A closed ring has a winding (CCW/CW) that fixes which turn sign
        # is convex; an open path has none, so default to a left turn (+1).
        up = self.descent.unitized() * -1.0 if self.descent.length > 1e-9 else Vector(0, 0, 1)
        winding = (1.0 if signed_area_xy(ring) > 0 else -1.0) if closed else 1.0
        target = -winding if self.notch_flip else winding
        out = {}
        for idx in range(m):
            if not closed and (idx == 0 or idx == m - 1):
                continue  # an OPEN path's free ends are not corners
            V, P, N = ring[idx], ring[(idx - 1) % m], ring[(idx + 1) % m]
            turn = up.dot((V - P).cross(N - V))  # signed turn about the mouth normal
            if abs(turn) < 1e-9:
                continue  # collinear -- no corner
            if (1.0 if turn > 0 else -1.0) != target:
                continue  # wrong handedness (convex vs concave) -- skip
            e0, e1 = V - N, V - P  # edges pointing from each neighbour TOWARD the corner
            l0, l1 = e0.length, e1.length
            if l0 < 1e-9 or l1 < 1e-9:
                continue
            u0, u1 = e0 * (1.0 / l0), e1 * (1.0 / l1)
            b = u0 + u1  # corner bisector: the average direction of the two edges
            lb = b.length
            if lb < 1e-6:
                continue  # straight-through vertex -- no corner
            sphi = math.sin(math.acos(max(-1.0, min(1.0, u0.dot(u1)))))  # sin of the corner angle
            if sphi < 1e-6:
                continue
            pen = (R / sphi) * lb - R  # Ears.cs: r*|u0+u1| - R  ==  R/sin(phi/2) - R
            if pen <= 1e-9:
                continue
            vec = b * (pen / lb)  # along the bisector, length `pen`
            out[idx] = vec
            self.notches.append((Point(*V), Point(V[0] + vec[0], V[1] + vec[1], V[2] + vec[2])))
        return out

    def _apply_tabs(self, points):
        """Lift the cut over hold-down TABS so uncut bridges hold the part down.

        Each marker in :attr:`_tabs` is snapped to the nearest point ON the ramp
        centre-path (in XY) -- the markers come from the un-offset contour, so a raw
        disk about them could miss the offset path. A marker whose nearest-path
        distance exceeds ``tab_width`` (belongs to another contour) is dropped, so
        the caller may pass every marker to every ramp. Around each snapped centre a
        disk of radius ``tab_width / 2`` is the tab region. Each tab is an ISOLATED
        up-and-down: the tool walls straight UP to ``floor + tab_height`` where it
        ENTERS a disk, rides flat across the bridge, walls straight DOWN where it
        LEAVES, and returns to the cut floor -- so between two tabs the tool drops
        back down and keeps cutting, it does not stay lifted. Because EVERY pass is
        clamped (not just the deepest), no pass cuts below a bridge. Populates
        :attr:`tabs` with the bridge-centre points; returns the (longer) point list.
        """
        self.tabs = []
        if not self._tabs:
            return points
        r = 0.5 * self.tab_width
        snap = self.tab_width
        disks = []  # (cx, cy) bridge centres snapped onto the path
        for t in self._tabs:
            tx, ty = float(t[0]), float(t[1])
            best = None
            for i in range(len(points) - 1):
                ax, ay = points[i][0], points[i][1]
                dx, dy = points[i + 1][0] - ax, points[i + 1][1] - ay
                ll = dx * dx + dy * dy
                if ll < 1e-18:
                    px, py = ax, ay
                else:
                    u = max(0.0, min(1.0, ((tx - ax) * dx + (ty - ay) * dy) / ll))
                    px, py = ax + u * dx, ay + u * dy
                d = math.hypot(tx - px, ty - py)
                if best is None or d < best[0]:
                    best = (d, px, py)
            if best is not None and best[0] <= snap:
                disks.append((best[1], best[2]))
        if not disks:
            return points

        floor_z = min(p[2] for p in points)
        tab_top_z = floor_z + self.tab_height

        def inside(x, y):
            for cx, cy in disks:
                if math.hypot(x - cx, y - cy) <= r + 1e-9:
                    return True
            return False

        # Walk each segment, inserting a vertical wall UP where it enters a disk and
        # DOWN where it leaves, so every tab is a self-contained lift that returns to
        # the cut floor. A vertex inside a disk is raised (deeper Z only -- passes
        # already above the bridge are left flat).
        out = []
        for i in range(len(points) - 1):
            a, b = points[i], points[i + 1]
            ax, ay, az = a[0], a[1], a[2]
            dx, dy, dz = b[0] - ax, b[1] - ay, b[2] - az
            out.append(Point(ax, ay, max(az, tab_top_z) if inside(ax, ay) else az))
            aa = dx * dx + dy * dy
            if aa < 1e-18:
                continue
            crossings = []
            for cx, cy in disks:
                fx, fy = ax - cx, ay - cy
                bb = 2.0 * (fx * dx + fy * dy)
                cc = fx * fx + fy * fy - r * r
                disc = bb * bb - 4.0 * aa * cc
                if disc <= 0.0:
                    continue
                sq = math.sqrt(disc)
                for tt in ((-bb - sq) / (2.0 * aa), (-bb + sq) / (2.0 * aa)):
                    if 1e-9 < tt < 1.0 - 1e-9:
                        crossings.append(tt)
            for tt in sorted(crossings):
                xt, yt, zt = ax + tt * dx, ay + tt * dy, az + tt * dz
                eps = 1e-6
                before = inside(ax + (tt - eps) * dx, ay + (tt - eps) * dy)
                after = inside(ax + (tt + eps) * dx, ay + (tt + eps) * dy)
                if (not before) and after:  # entering a tab -> wall UP onto the bridge
                    out.append(Point(xt, yt, zt))
                    if zt < tab_top_z:
                        out.append(Point(xt, yt, tab_top_z))
                elif before and (not after):  # leaving a tab -> wall DOWN to the floor
                    if zt < tab_top_z:
                        out.append(Point(xt, yt, tab_top_z))
                    out.append(Point(xt, yt, zt))
                else:  # boundary between overlapping disks -- stay on the bridge
                    out.append(Point(xt, yt, max(zt, tab_top_z)))
        lp = points[-1]
        out.append(Point(lp[0], lp[1], max(lp[2], tab_top_z) if inside(lp[0], lp[1]) else lp[2]))
        self.tabs = [Point(cx, cy, tab_top_z) for cx, cy in disks]
        return out

    def _build(self):
        pts = self._pts
        seglen = [pts[i].distance_to_point(pts[i + 1]) for i in range(len(pts) - 1)]
        total = sum(seglen)
        if total < 1e-9:
            raise ValueError("ramp path has zero length.")
        cum = [0.0]
        for s in seglen:
            cum.append(cum[-1] + s)

        self.depth = self.descent.length
        if self.depth < 1e-9:
            raise ValueError("descent has zero length.")

        # Vertical descent per pass: explicit `step`, else from `ramp_angle`,
        # else the default ramp angle. Clamp the pass count up so step <= request.
        if self._step_request is not None:
            target = self._step_request
        else:
            angle = RAMP_ANGLE_DEFAULT if self._ramp_angle_request is None else self._ramp_angle_request
            target = total * math.tan(angle)
        target = max(target, 1e-6)
        n = max(1, math.ceil(self.depth / target))
        self.passes = n
        self.step = self.depth / n
        self.ramp_angle = math.atan2(self.step, total)

        # Inside-corner NOTCHES (dogbone relief), sized per NGon's Ears.cs. Cut as
        # an in-and-out excursion at every pass, so it carves a full-depth notch.
        notch = self._notch_vectors()
        closed = len(pts) >= 4 and pts[0].distance_to_point(pts[-1]) < 1e-9
        one_dir = self.direction is not None and closed  # every lap the same way (climb/conventional)
        mring = (len(pts) - 1) if closed else len(pts)  # map the duplicated seam back to ring 0

        def emit(out, base, vidx, do_notch):
            out.append(base)
            rv = vidx % mring if closed else vidx
            if do_notch and rv in notch:
                out.append(base + notch[rv])  # penetrate along the corner bisector
                out.append(base)  # come back to the path and carry on

        # Each traverse spans arc fraction 0..1; pass k descends from k/n to
        # (k+1)/n of the full descent, so the slope is even along the path.
        forward = [(cum[j] / total, pts[j], j) for j in range(len(pts))]
        backward = [((total - cum[j]) / total, pts[j], j) for j in range(len(pts) - 1, -1, -1)]

        points = []
        for k in range(n):
            seq = forward if (one_dir or k % 2 == 0) else backward
            for j, (arc, p, vidx) in enumerate(seq):
                if k > 0 and j == 0:
                    continue  # turn point already added by the previous pass
                # A closed ring revisits its seam at the sweep END, so notch there
                # too (j != 0); an open path's free ENDS are not corners (skip both).
                do_notch = (j != 0) if closed else (0 < j < len(seq) - 1)
                emit(points, p + self.descent * ((k + arc) / n), vidx, do_notch)

        # Floor finishing that ENDS back at the start, so the cut returns to its
        # origin (like a drill) and the tool retracts straight up -- no diagonal.
        # A descent of n passes ends at the far end when n is odd (one traverse
        # home) or at the start when n is even (clean out to the far end and back).
        if self.bottom_pass:
            if one_dir:
                seq = forward[1:]  # one more forward lap; a closed lap ends back at the seam
            elif n % 2 == 1:
                seq = backward[1:]
            else:
                seq = forward[1:] + backward[1:]
            for pos, (_arc, p, vidx) in enumerate(seq):
                emit(points, p + self.descent, vidx, pos < len(seq) - 1)

        # HOLD-DOWN TABS: lift the cut over each bridge so the part stays fixed.
        points = self._apply_tabs(points)
        self.ramp = Polyline(points)

        # Vertical plunge-in / retract-out, kept >= MIN_CLEARANCE above the mouth.
        mouth_z = max(p[2] for p in points)
        floor = mouth_z + MIN_CLEARANCE
        self.safe_z = floor if self._safe_z_request is None else max(self._safe_z_request, floor)
        # Plunge in at the start, retract straight up at the end (same XY, Z to
        # safe_z), then traverse home at safe_z so the path ENDS where it began.
        first, last = points[0], points[-1]
        lead_in = Point(first[0], first[1], self.safe_z)
        lead_out = Point(last[0], last[1], self.safe_z)
        home = Point(first[0], first[1], self.safe_z)
        self.path = Polyline([lead_in] + points + [lead_out, home])

    def __repr__(self):
        return f"toolpath_2d_ramp(passes={self.passes}, step={self.step:.3f}, ramp_angle={math.degrees(self.ramp_angle):.1f}deg, depth={self.depth:.3f}, safe_z={self.safe_z})"
