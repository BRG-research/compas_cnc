import math

from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import is_point_in_polygon_xy

from compas_cnc._milling import oriented
from compas_cnc._milling import want_ccw
from compas_cnc.clipper2 import hatch
from compas_cnc.clipper2 import offset_polyline

__all__ = ["toolpath_2d_hatch"]

MIN_CLEARANCE = 10.0  # a retract must clear the tool-path by at least this (units)


class toolpath_2d_hatch:
    """Raster (zig-zag) fill of a closed polygon -- optionally with holes -- by
    parallel hatch lines at a user angle, connected into ONE continuous path, and
    optionally repeated DOWN a user depth in equal layers joined by a gentle
    helical ramp around the contour.

    The cut region is ``boundary`` minus ``holes``, compensated for the tool: the
    boundary is shrunk inward and every hole is grown outward by ``radius`` so the
    tool CENTRE path keeps a full radius clear of every wall instead of riding the
    edge. The compensated region is filled with hatch lines ``spacing`` apart at
    ``angle``, then the lines are walked in BOUSTROPHEDON (snake) order: sorted
    across the hatch direction row by row, and each next line entered by its
    NEARER end so the tool reverses every row instead of jumping around at random.

    A link between two successive lines that would drag the cutter through a hole
    is replaced by a RETRACT over the island (lift to ``safe_z``, rapid across,
    plunge back), so no cutting move ever enters a hole.

    **Depth (layered roughing).** With ``depth > 0`` the fill is not a single flat
    pass but a stack of equal layers ``step`` deep, cut from ``z`` straight down to
    ``z - depth`` (the pass count is rounded up so the actual ``step`` is ``<=`` the
    request). Material is removed layer by layer. Between layers the tool does NOT
    lift and plunge: it follows the boundary contour ONCE around while descending
    the next ``step``, a HELICAL ramp at the shallowest angle the perimeter allows
    (``atan(step / perimeter)``) -- the same loop that finishes the wall. So the
    path ramps gently into the stock, clears a layer, ramps down the wall into the
    next, and repeats until ``z - depth`` is reached. With ``depth == 0`` (the
    default) it is the original single flat fill at ``z``.

    The path plunges/ramps once at the start, sweeps each layer, and traverses
    home at ``safe_z`` -- so it ends where it began.

    Parameters
    ----------
    boundary : :class:`compas.geometry.Polyline` | :class:`compas.geometry.Polygon` | sequence of points
        Closed outer boundary (XY plane; the fill is laid at ``z``).
    spacing : float
        Perpendicular distance between hatch lines -- the stepover. Must be > 0.
    angle : float, optional
        Hatch direction in radians (CCW from +X). Defaults to ``0.0`` (horizontal).
    holes : sequence, optional
        Closed rings the fill must avoid (islands). Defaults to ``None``.
    radius : float, optional
        Tool radius for cutter compensation: the boundary is inset and the holes
        are grown by this much before hatching. Defaults to ``0.0`` (the raw
        boundary/holes, tool centre rides the edge).
    z : float, optional
        World-Z of the TOP fill plane. Defaults to the mean Z of ``boundary`` --
        i.e. the toolpath lands on the geometry's own plane unless told otherwise.
        With ``depth`` set this is the top of the stock; cutting goes downward.
    safe_z : float, optional
        Plunge/retract height. Clamped up to at least :data:`MIN_CLEARANCE` above
        ``z``. Defaults to ``z + MIN_CLEARANCE``.
    contour : bool, optional
        After the raster fill, run a finishing pass once around each wall (the inset
        boundary and the grown holes) for a clean edge. Defaults to ``True``. With
        ``depth`` it runs once at the FINAL (bottom) level.
    direction : {None, "climb", "conventional"}, optional
        Milling direction for hard materials (assumes a CW / M3 tool; on M4 pick the
        opposite label). When set, the contour walls and the descent ramp are wound
        accordingly (boundary as a pocket, holes as islands) and the fill goes
        one-directional (every pass cut the same way, with a lift + rapid return
        between). ``None`` (default) doesn't care -- the faster mixed zig-zag, fine
        for soft materials.
    depth : float, optional
        Total material depth to remove below ``z``. ``0`` (default) is a single flat
        fill at ``z``; ``> 0`` clears in layers down to ``z - depth``.
    step : float, optional
        Maximum Z per layer when ``depth > 0``. The depth is split into
        ``ceil(depth / step)`` EQUAL layers so the actual step is ``<= step``. When
        omitted the whole ``depth`` is one layer. Ignored if ``depth == 0``.

    Attributes
    ----------
    path : :class:`compas.geometry.Polyline`
        The full tool-centre path: plunge/ramp, snake fill per layer (with retracts
        over islands and helical descents between layers), retract, home. Starts and
        ends at the same safe-Z position.
    fill : :class:`compas.geometry.Polyline`
        The working raster snake of the TOP cutting layer (without lead-in/out).
    lines : list[:class:`compas.geometry.Line`]
        The raw clipped hatch segments (over the compensated region), before
        connection.
    offset_boundary : list[:class:`compas.geometry.Polyline`]
        The boundary after inward tool-radius compensation (the actual fill edge).
    offset_holes : list[:class:`compas.geometry.Polyline`]
        The holes after outward tool-radius compensation (the kept-out islands).
    levels : list[float]
        The world-Z of every cutting layer, top to bottom (one entry when
        ``depth == 0``).
    layers : int
        Number of cutting layers (``len(levels)``).
    step : float
        Actual vertical descent per layer (``0`` when ``depth == 0``).
    ramp_angle : float
        Angle of the helical inter-layer descent in radians (``0`` when there is no
        ramp).
    radius, depth, safe_z : float
    """

    def __init__(self, boundary, spacing, angle=0.0, holes=None, radius=0.0, z=None, safe_z=None, contour=True, direction=None, depth=0.0, step=None):
        self.boundary = boundary
        self.holes = list(holes) if holes else []
        self.spacing = float(spacing)
        self.angle = float(angle)
        self.radius = float(radius)
        if z is None:  # default the fill plane to the geometry's own plane
            zs = [c[2] for c in (list(p) for p in boundary) if len(c) > 2]
            self.z = float(sum(zs) / len(zs)) if zs else 0.0
        else:
            self.z = float(z)
        self._safe_z_request = safe_z
        self.contour = bool(contour)
        if direction not in (None, "climb", "conventional"):
            raise ValueError("direction must be None, 'climb', or 'conventional'.")
        self.direction = direction
        # A milling direction needs a one-directional fill (every pass cut the same
        # way); with no direction we take the faster zig-zag (mixed senses).
        self.one_directional = direction is not None
        self.depth = float(depth)
        self._step_request = None if step is None else float(step)
        self._build()

    # ------------------------------------------------------------------ #

    def _build(self):
        # Cutter compensation: shrink the boundary and grow the holes by the tool
        # radius, so the tool CENTRE stays a radius clear of every wall (otherwise
        # the path rides the edge and the cutter gouges a radius past it).
        if self.radius > 0:
            ob = offset_polyline(self.boundary, -self.radius)
            oh = [g for h in self.holes for g in offset_polyline(h, self.radius)]
        else:
            ob, oh = [self.boundary], list(self.holes)
        # Clipper returns the offsets at z=0; lay every ring on the fill plane so the
        # compensated boundary/holes are coplanar with the toolpath (and the input).
        self.offset_boundary = [Polyline([Point(p[0], p[1], self.z) for p in r]) for r in ob]
        self.offset_holes = [Polyline([Point(p[0], p[1], self.z) for p in r]) for r in oh]

        # Hatch every (compensated) boundary contour against the grown holes.
        self.lines = []
        for ring in self.offset_boundary:
            self.lines += hatch(ring, self.spacing, self.angle, holes=self.offset_holes or None)

        ordered = self._order()

        # Z layers: split `depth` below `z` into equal steps <= the requested `step`.
        if self.depth > 1e-9:
            request = self._step_request if (self._step_request and self._step_request > 1e-9) else self.depth
            n = max(1, math.ceil(self.depth / request))
            self.step = self.depth / n
            self.levels = [self.z - self.step * i for i in range(1, n + 1)]
        else:
            self.step = 0.0
            self.levels = [self.z]
        self.layers = len(self.levels)
        self.ramp_angle = 0.0

        floor = self.z + MIN_CLEARANCE
        self.safe_z = floor if self._safe_z_request is None else max(self._safe_z_request, floor)

        if not ordered:
            self.fill = Polyline([])
            self.path = Polyline([])
            return

        fill_z = self.levels[0] if self.depth > 1e-9 else self.z
        self.fill = Polyline([Point(p[0], p[1], fill_z) for ab in ordered for p in ab])

        if self.depth > 1e-9:
            self._assemble_layers(ordered)
        else:
            self._assemble_single(ordered)

    # ------------------------------------------------------------------ #
    # Raster ordering (boustrophedon snake) -- Z-independent
    # ------------------------------------------------------------------ #

    def _order(self):
        """Order the clipped hatch segments into a snake: a list of ``(a, b)`` point
        pairs, each the entry/exit of one cutting pass, in traversal order."""
        # Unit vectors: `u` along the hatch lines, `s` across them (the scan/row
        # direction). Order the segments row by row across `s`, then connect each
        # next segment by its nearer end -- a boustrophedon snake.
        ux, uy = math.cos(self.angle), math.sin(self.angle)
        sx, sy = -uy, ux

        def along(p):
            return p[0] * ux + p[1] * uy

        def across(p):
            return p[0] * sx + p[1] * sy

        segs = []
        for line in self.lines:
            a = Point(line.start[0], line.start[1], self.z)
            b = Point(line.end[0], line.end[1], self.z)
            if along(a) > along(b):
                a, b = b, a  # a is always the "low" end along the hatch direction
            segs.append((a, b))
        # Bucket the segments into ROWS (hatch lines a spacing apart across `s`),
        # then walk the rows in order, entering each from the end nearest the tool, so
        # the tool snakes straight to the adjacent line. A row's segments stay in
        # travel order, so a split row (cut by a hole) is crossed once in passing --
        # no full-width jump back and forth.
        segs.sort(key=lambda ab: across(ab[0]))
        rows = []
        for seg in segs:
            if rows and abs(across(seg[0]) - across(rows[-1][-1][0])) < self.spacing * 0.5:
                rows[-1].append(seg)
            else:
                rows.append([seg])

        ordered = []  # cut segments oriented for traversal
        if self.one_directional:
            # Every pass cuts the SAME way (low -> high along the hatch line), so the
            # cut direction -- and thus climb/conventional -- is consistent. Returns
            # between passes are retracts (added during assembly).
            for row in rows:
                row.sort(key=lambda ab: along(ab[0]))
                ordered.extend(row)
        else:
            # Boustrophedon snake: enter each row from the end nearest the tool.
            tail = None
            for row in rows:
                row.sort(key=lambda ab: along(ab[0]))  # left -> right along the hatch line
                if tail is not None and tail.distance_to_point(row[-1][1]) < tail.distance_to_point(row[0][0]):
                    row = [(b, a) for a, b in reversed(row)]  # nearer the far end -> traverse back
                for a, b in row:
                    ordered.append((a, b))
                    tail = b
        return ordered

    # ------------------------------------------------------------------ #
    # Path pieces shared by single- and multi-layer assembly
    # ------------------------------------------------------------------ #

    def _layer_cut(self, ordered, level):
        """The raster snake at world-Z ``level``: each pass cut at ``level``, with a
        retract over ``safe_z`` whenever a link would cross an island (or always, when
        one-directional). Starts at the first cut point, ends at the last -- no
        lead-in or trailing retract."""
        pts = []
        for i, (a, b) in enumerate(ordered):
            pts.append(Point(a[0], a[1], level))
            pts.append(Point(b[0], b[1], level))
            if i + 1 < len(ordered):
                nxt = ordered[i + 1][0]
                if self.one_directional or self._link_leaves_region(b, nxt):
                    pts.append(Point(b[0], b[1], self.safe_z))      # retract over the wall/island
                    pts.append(Point(nxt[0], nxt[1], self.safe_z))  # rapid across (next pass plunges)
        return pts

    def _contour_points(self, level):
        """Finishing pass once around each wall (inset boundary + grown holes) at
        ``level``, each reached by a safe-Z rapid and retracted after."""
        pts = []
        # boundary rings are pocket walls (cleared inside); hole rings are islands.
        walls = [(r, True) for r in self.offset_boundary] + [(r, False) for r in self.offset_holes]
        for ring, cleared_inside in walls:
            loop = [Point(p[0], p[1], level) for p in ring]
            ccw = want_ccw(self.direction, cleared_inside)
            if ccw is not None:  # wind the wall for the requested milling direction
                loop = oriented(loop, ccw)
            if loop[0].distance_to_point(loop[-1]) > 1e-9:
                loop.append(loop[0])  # close the loop
            pts.append(Point(loop[0][0], loop[0][1], self.safe_z))    # rapid to the loop start
            pts += loop                                               # cut around the wall
            pts.append(Point(loop[-1][0], loop[-1][1], self.safe_z))  # retract
        return pts

    # ------------------------------------------------------------------ #
    # Single flat layer (depth == 0) -- the original behaviour
    # ------------------------------------------------------------------ #

    def _assemble_single(self, ordered):
        start = ordered[0][0]
        last = ordered[-1][1]
        points = [Point(start[0], start[1], self.safe_z)]      # lead-in (plunge)
        points += self._layer_cut(ordered, self.z)
        points.append(Point(last[0], last[1], self.safe_z))    # retract after the fill
        if self.contour:
            points += self._contour_points(self.z)
        points.append(Point(start[0], start[1], self.safe_z))  # home
        self.path = Polyline(points)

    # ------------------------------------------------------------------ #
    # Layered roughing (depth > 0) with a helical inter-layer descent
    # ------------------------------------------------------------------ #

    def _assemble_layers(self, ordered):
        start = ordered[0][0]
        last = ordered[-1][1]
        ring = self._descent_ring()
        if ring is None:  # no boundary loop to ramp around -> plain plunge per layer
            self._assemble_layers_plunge(ordered)
            return

        i0 = min(range(len(ring)), key=lambda i: ring[i].distance_to_point(start))  # ramp entry near the fill start
        entry = ring[i0]
        self.ramp_angle = math.atan2(self.step, self._ring_perimeter(ring)) if self.step else 0.0

        points = [Point(entry[0], entry[1], self.safe_z)]  # lead-in over the ramp entry
        prev_top = self.z
        first = True
        for level in self.levels:
            if not first:
                # we finished the previous layer at `last` @ prev_top -- reposition to
                # the ramp entry (in-plane across the cleared floor, or retract if the
                # straight hop would leave the region).
                if self._link_leaves_region(last, entry):
                    points.append(Point(last[0], last[1], self.safe_z))
                    points.append(Point(entry[0], entry[1], self.safe_z))
            first = False
            # Helical ramp once around the contour, descending prev_top -> level: a
            # gentle wall-cutting entry into the next layer (the shallowest single-loop
            # angle the perimeter allows). The first point sits at entry @ prev_top, so
            # it plunges/links onto the previous floor before biting downward.
            points += self._spiral(ring, i0, prev_top, level)
            # Hop from the ramp end (entry @ level) to the fill start, retracting only
            # if that hop would leave the region.
            if entry.distance_to_point(start) > 1e-9:
                if self._link_leaves_region(entry, start):
                    points.append(Point(entry[0], entry[1], self.safe_z))
                    points.append(Point(start[0], start[1], self.safe_z))
                points.append(Point(start[0], start[1], level))
            # Raster-clear this layer.
            points += self._layer_cut(ordered, level)
            prev_top = level

        points.append(Point(last[0], last[1], self.safe_z))  # retract after the last layer
        if self.contour:
            points += self._contour_points(self.levels[-1])  # clean the wall at full depth
        points.append(Point(entry[0], entry[1], self.safe_z))  # home (== lead-in)
        self.path = Polyline(points)

    def _assemble_layers_plunge(self, ordered):
        """Fallback layering when there is no boundary loop to ramp around: each layer
        is reached by a vertical plunge instead of a helical descent."""
        start = ordered[0][0]
        last = ordered[-1][1]
        points = [Point(start[0], start[1], self.safe_z)]
        first = True
        for level in self.levels:
            if not first:
                points.append(Point(last[0], last[1], self.safe_z))    # retract
                points.append(Point(start[0], start[1], self.safe_z))  # rapid back to the start
            first = False
            points += self._layer_cut(ordered, level)  # plunge to start @ level, clear
        points.append(Point(last[0], last[1], self.safe_z))
        if self.contour:
            points += self._contour_points(self.levels[-1])
        points.append(Point(start[0], start[1], self.safe_z))
        self.path = Polyline(points)

    # ------------------------------------------------------------------ #
    # Geometry helpers for the descent ramp
    # ------------------------------------------------------------------ #

    def _descent_ring(self):
        """The boundary loop the helix ramps around: the longest inset-boundary ring,
        wound for the milling direction. ``None`` if there is no usable loop."""
        rings = []
        for r in self.offset_boundary:
            pts = [Point(p[0], p[1], p[2]) for p in r]
            if len(pts) > 1 and pts[0].distance_to_point(pts[-1]) < 1e-9:
                pts = pts[:-1]  # drop the duplicated closing vertex
            if len(pts) >= 3:
                rings.append(pts)
        if not rings:
            return None
        ring = max(rings, key=self._ring_perimeter)  # the main perimeter
        ccw = want_ccw(self.direction, True)  # boundary = pocket wall
        if ccw is not None:
            ring = oriented(ring, ccw)
        return ring

    @staticmethod
    def _ring_perimeter(ring):
        m = len(ring)
        return sum(ring[i].distance_to_point(ring[(i + 1) % m]) for i in range(m))

    def _spiral(self, ring, i0, z_from, z_to):
        """One helical loop around ``ring`` starting at vertex ``i0``, descending Z
        linearly from ``z_from`` to ``z_to`` by arc length (so the ramp angle is
        constant). First point sits at ``z_from``, the closing point back at ``i0`` at
        ``z_to``."""
        m = len(ring)
        verts = [ring[(i0 + k) % m] for k in range(m)] + [ring[i0]]  # full loop, back to the entry
        seglen = [verts[k].distance_to_point(verts[k + 1]) for k in range(len(verts) - 1)]
        total = sum(seglen) or 1.0
        pts, walked = [], 0.0
        for k, v in enumerate(verts):
            z = z_from + (z_to - z_from) * (walked / total)
            pts.append(Point(v[0], v[1], z))
            if k < len(seglen):
                walked += seglen[k]
        return pts

    # Ignore sub-micron straddling of a wall: the clipped hatch ends and the offset
    # boundary can disagree by a floating-point hair, so a link running ALONG a wall
    # must not read as "outside". A real void crossing is mm-deep, far above this.
    _REGION_TOL = 1e-2  # mm

    def _link_leaves_region(self, p, q):
        """Whether the straight in-plane link ``p -> q`` would leave the cut region --
        exit the inset boundary (a concavity, or the gap between disjoint hatch
        regions) OR enter a kept-out island -- i.e. drag the cutter across the
        geometry. Such a link must be replaced by a safe-Z retract; a link that only
        runs ALONG a wall stays in (the excursion is within :data:`_REGION_TOL`)."""
        boundary = [[[v[0], v[1]] for v in ring] for ring in self.offset_boundary]
        holes = [[[v[0], v[1]] for v in hole] for hole in self.offset_holes]
        if not boundary and not holes:
            return False
        dist = math.hypot(q[0] - p[0], q[1] - p[1])
        n = min(400, max(16, int(dist / 0.5)))  # sample finely enough to catch a thin neck
        for k in range(1, n):
            t = k / n
            pt = [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]
            if boundary and not any(is_point_in_polygon_xy(pt, ring) for ring in boundary):
                if min(self._dist_to_ring(pt, ring) for ring in boundary) > self._REGION_TOL:
                    return True  # left the inset boundary -- would cut across a concavity / void
            for ring in holes:
                if is_point_in_polygon_xy(pt, ring) and self._dist_to_ring(pt, ring) > self._REGION_TOL:
                    return True  # entered an island
        return False

    @staticmethod
    def _dist_to_ring(pt, ring):
        """Shortest distance from ``pt`` (x, y) to the closed polyline ``ring`` (its edges)."""
        x, y = pt[0], pt[1]
        m = len(ring)
        best = float("inf")
        for i in range(m):
            ax, ay = ring[i][0], ring[i][1]
            bx, by = ring[(i + 1) % m][0], ring[(i + 1) % m][1]
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            s = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg2))
            best = min(best, math.hypot(x - (ax + s * dx), y - (ay + s * dy)))
        return best

    def __repr__(self):
        return (
            f"toolpath_2d_hatch(lines={len(self.lines)}, spacing={self.spacing}, "
            f"angle={math.degrees(self.angle):.1f}deg, layers={self.layers}, depth={self.depth})"
        )
