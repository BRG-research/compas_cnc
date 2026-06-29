import math

from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import Translation

from compas_cnc._milling import signed_area_xy
from compas_cnc._milling import want_ccw

__all__ = ["toolpath_2d_surfacing"]

MIN_CLEARANCE = 10.0  # lead-in/out must clear the tool-path by at least this (units)


class toolpath_2d_surfacing:
    """Zigzag (boustrophedon) milling tool-path that clears the rectangle bounded
    by two parallel edges, then runs a finishing contour around the full boundary.

    Sequence of the path:

    1. **Plunge** -- a vertical line from ``safe_z`` straight down onto the start
       corner.
    2. **Zigzag** -- back-and-forth sweeps clearing the inside. The tool-CENTRE
       path is inset from both edges by ``radius`` so a tool of that radius stays
       WITHIN the rectangle, and the spacing between passes is forced
       ``<= stepover`` (``ceil`` on the pass count) so passes overlap.
    3. **Contour** -- trace the FULL inset perimeter (all four edges) and return to
       the start corner, leaving a clean wall all the way round. The connecting
       move from the zigzag end retraces one edge, so it goes round a little more
       than once -- which is fine for a finishing pass.
    4. **Retract** -- a vertical line back up to ``safe_z`` at the start corner.

    Parameters
    ----------
    line0, line1 : :class:`compas.geometry.Line`
        The two (roughly) parallel edges to fill between. Passes sweep across the
        gap and step along their length, so the choice of edges sets the zigzag
        orientation.
    radius : float
        Tool radius. Used as the boundary inset and, by default, the max spacing.
    safe_z : float, optional
        Plunge/retract height. Clamped up to at least :data:`MIN_CLEARANCE` above
        the tool-path plane. Defaults to ``toolpath_z + MIN_CLEARANCE``.
    stepover : float, optional
        Distance between passes. Defaults to ``radius``; actual spacing is always
        ``<= stepover`` (the pass count is rounded up).
    direction : {None, "climb", "conventional"}, optional
        Milling direction for hard materials (assumes a CW / M3 tool; on M4 pick the
        opposite label). When set, the finishing contour is wound to match (the
        rectangle is a pocket wall) and the zig-zag goes one-directional (every pass
        cut the same way, lift + rapid between). ``None`` (default) is the faster
        bidirectional zig-zag, fine for soft materials.

    Attributes
    ----------
    path : :class:`compas.geometry.Polyline`
        The full tool-centre path: plunge, zigzag, contour, retract.
    zigzag, contour : :class:`compas.geometry.Polyline`
        The cutting zigzag and the finishing perimeter loop, separately.
    passes : int
    spacing : float
    safe_z : float
    """

    def __init__(self, line0, line1, radius, safe_z=None, stepover=None, incline=False, direction=None):
        self.line0 = line0
        self.line1 = line1
        self.radius = radius
        self.stepover = radius if stepover is None else stepover
        if incline:
            self.stepover *= 0.5  # tilted face: halve the spacing so tool passes overlap >= half (scallop control)
        self._safe_z_request = safe_z
        self.incline = incline
        if direction not in (None, "climb", "conventional"):
            raise ValueError("direction must be None, 'climb', or 'conventional'.")
        self.direction = direction
        # A milling direction needs a one-directional zig-zag (every pass cut the same
        # way); with no direction we take the faster bidirectional zig-zag.
        self.one_directional = direction is not None
        self._build()

    # ------------------------------------------------------------------ #
    # Constructors from real geometry
    # ------------------------------------------------------------------ #

    @classmethod
    def from_quad(cls, points, radius, safe_z=None, stepover=None, flip=False, incline=False, direction=None, start=None):
        """Build a tool-path that fills a 4-corner face.

        ``points`` are the face's 4 corners in order -- as an open list/polyline of
        4 points, OR a CLOSED polyline of 5 (the repeated closing corner is dropped),
        so the same outline you draw can be passed straight in. Returns ``None`` if
        it is not 4 corners.

        ``flip`` -- the zigzag direction (boolean): ``False`` (default) sweeps along
        the LONGER side (fewer, longer passes); ``True`` sweeps the PERPENDICULAR
        way (the other subdivision direction).

        ``incline`` -- if the face is tilted, shift the whole tool-path by ``radius``
        up-slope along the subdivision direction, so the FLAT tool's edge rides the
        surface instead of its centre digging into the material (see ``_build``).

        ``start`` -- where the sweep BEGINS (it becomes ``line0.start``): ``None``
        (default) starts at the HIGHEST corner, so a tilted face is cut from the top
        downhill; an integer ``0-3`` selects that corner of the face outline by index
        (the order ``points`` are given / the mesh face's corner loop); or pass a
        point ``(x, y, z)`` to start at the corner nearest it. On a flat face ``None``
        keeps the ``flip`` start. Re-orients the rails only; the zigzag is unchanged.
        """
        pts = [Point(*p) for p in points]
        if len(pts) >= 2 and pts[0].distance_to_point(pts[-1]) < 1e-9:
            pts = pts[:-1]  # accept a CLOSED polyline too -- drop the duplicated closing corner
        if len(pts) != 4:
            return None  # from_quad needs exactly four corners
        v0, v1, v2, v3 = pts
        pair_a = (Line(v0, v1), Line(v3, v2))  # one pair of opposite edges
        pair_b = (Line(v1, v2), Line(v0, v3))  # the other pair
        len_a = pair_a[0].length + pair_a[1].length
        len_b = pair_b[0].length + pair_b[1].length
        auto = pair_b if len_a >= len_b else pair_a  # rails = shorter edges -> sweep long side
        other = pair_a if len_a >= len_b else pair_b
        line0, line1 = other if flip else auto
        # Pick the start corner (it becomes line0.start): the HIGHEST corner by
        # default, or the corner nearest `start`. Swapping the rails / flipping both
        # keeps line0.start and line1.start edge-connected, so the raster and contour
        # walk stay valid. On a flat face the default leaves the `flip` start as is.
        ends = [line0.start, line0.end, line1.start, line1.end]
        if start is None:
            which = max(range(4), key=lambda i: ends[i][2])  # highest corner by Z
        else:
            target = pts[start % 4] if isinstance(start, int) else Point(*start)  # outline id, or a point
            which = min(range(4), key=lambda i: ends[i].distance_to_point(target))
        if which >= 2:  # the chosen corner is on line1 -> make that rail line0
            line0, line1 = line1, line0
            which -= 2
        if which == 1:  # the chosen corner is the rail END -> flip both rails so it is the START
            line0 = Line(line0.end, line0.start)
            line1 = Line(line1.end, line1.start)
        # A tool of `radius` insets `radius` off every edge, so the face must be
        # wider than the tool DIAMETER on both axes -- else the inset collapses or
        # crosses over (garbage). Too small for this tool => no tool-path.
        gap = (line1.start - line0.start).length
        if min(line0.length, line1.length) <= 2 * radius or gap <= 2 * radius:
            return None
        return cls(line0, line1, radius, safe_z=safe_z, stepover=stepover, incline=incline, direction=direction)

    @classmethod
    def from_mesh_face(cls, mesh, face_id, radius, safe_z=None, stepover=None, flip=False, incline=False, direction=None, start=None):
        """Build a tool-path from the face ``face_id`` you choose on a mesh.

        ``flip``/``incline``/``start`` -- see :meth:`from_quad`. Returns ``None`` if
        that face is not a quad.
        """
        coords = mesh.face_coordinates(face_id)
        if len(coords) != 4:
            return None
        return cls.from_quad(coords, radius, safe_z=safe_z, stepover=stepover, flip=flip, incline=incline, direction=direction, start=start)

    @classmethod
    def from_plate(cls, mesh, radius, safe_z=None, stepover=None, top=False, flip=False, incline=False, direction=None, start=None):
        """Build a tool-path from a plate-like cutter's large face.

        A plate's two LARGEST faces are its top and bottom (both quads); the thin
        side faces are ignored. Coplanar triangles (from a triangulated cutter) are
        first merged back into quads, so this is robust to triangulation -- no need
        to clean the cutter mesh first. Uses the LOWER face by default, or the upper
        one with ``top=True``. ``flip`` (boolean) sets the zigzag direction and
        ``incline`` shifts the path for tilted faces (see :meth:`from_quad`). Returns
        ``None`` if no quad face is found.
        """
        try:  # recover a clean quad box if the cutter was stored triangulated
            from compas_tf.solid_difference_modifier import merge_coplanar_faces

            mesh = merge_coplanar_faces(mesh.copy())
        except Exception:
            pass
        quads = [fk for fk in mesh.faces() if len(mesh.face_vertices(fk)) == 4]
        if not quads:
            return None
        big = sorted(quads, key=mesh.face_area, reverse=True)[:2]  # top + bottom
        big.sort(key=lambda fk: sum(p[2] for p in mesh.face_coordinates(fk)))  # by Z
        face_id = big[1] if top else big[0]
        return cls.from_quad(mesh.face_coordinates(face_id), radius, safe_z=safe_z, stepover=stepover, flip=flip, incline=incline, direction=direction, start=start)

    # ------------------------------------------------------------------ #

    def _build(self):
        r = self.radius
        # Inclined cuts EXTEND to the plate edges (no inset): on a 3-axis machine the
        # offset compensated centre sits above/over the edge so the rim still cuts the
        # full face. Flat cuts keep the normal inset (the tool stays inside).
        inset_r = 0.0 if self.incline else r
        across = self.line1.start - self.line0.start  # vector line0 -> line1
        across = across * (1.0 / across.length)
        along = self.line0.vector * (1.0 / self.line0.length)

        def inset(line, sign):
            start = line.start + across * (inset_r * sign) + along * inset_r
            end = line.end + across * (inset_r * sign) - along * inset_r
            return Line(start, end)

        a = inset(self.line0, +1)  # move inward toward line1
        b = inset(self.line1, -1)  # move inward toward line0

        n = max(1, math.ceil(a.length / self.stepover))  # ceil -> spacing <= stepover
        zpts = []
        for i in range(n + 1):
            t = i / n
            pa = a.start + a.vector * t
            pb = b.start + b.vector * t
            if self.one_directional:
                zpts.extend([pa, pb])  # every pass cuts pa -> pb (consistent direction)
            else:
                zpts.extend([pa, pb] if i % 2 == 0 else [pb, pa])  # boustrophedon snake

        self.zigzag = Polyline(zpts)
        self.passes = n + 1
        self.spacing = a.length / n

        # Finishing contour: walk the inset perimeter along EDGES ONLY (never
        # diagonally) from where the zigzag ended, around the full boundary, back to
        # the start corner. Consecutive corners in `cycle` share an edge, so there is
        # never a diagonal cross; the walk goes round a little more than once -- fine
        # for a finishing pass. With a milling `direction` set it runs CW or CCW to
        # match (the rectangle is a pocket wall -> cleared_inside=True); else legacy.
        bl, tl = a.start, a.end  # rail a (line0 edge): bottom/top corner
        br, tr = b.start, b.end  # rail b (line1 edge): bottom/top corner
        cycle = [bl, br, tr, tl]  # consecutive corners are edge-connected (no diagonal)
        start_i = min(range(4), key=lambda i: cycle[i].distance_to_point(zpts[-1]))  # nearest the zigzag end
        ccw = want_ccw(self.direction, cleared_inside=True)
        step = +1 if ccw is None else (+1 if (signed_area_xy(cycle) > 0) == ccw else -1)
        contour = []
        k, edges = start_i, 0
        while True:
            k = (k + step) % 4
            contour.append(cycle[k])
            edges += 1
            if cycle[k] is bl and edges >= 4:  # back at start, full boundary covered
                break
        self.contour = Polyline([zpts[-1]] + contour)

        # Vertical plunge-in / retract-out at the start corner (initial position),
        # kept >= MIN_CLEARANCE above the tool-path plane.
        toolpath_z = max(p[2] for p in zpts)
        floor = toolpath_z + MIN_CLEARANCE
        self.safe_z = floor if self._safe_z_request is None else max(self._safe_z_request, floor)
        lead_in = Point(bl[0], bl[1], self.safe_z)
        lead_out = Point(bl[0], bl[1], self.safe_z)
        if self.one_directional:
            # Each pass: plunge -> cut pa->pb -> retract; rapid across to the next pass.
            body = [lead_in]
            for i in range(0, len(zpts), 2):
                pa, pb = zpts[i], zpts[i + 1]
                body += [Point(pa[0], pa[1], pa[2]), Point(pb[0], pb[1], pb[2])]
                if i + 2 < len(zpts):
                    nxt = zpts[i + 2]
                    body.append(Point(pb[0], pb[1], self.safe_z))  # retract
                    body.append(Point(nxt[0], nxt[1], self.safe_z))  # rapid to the next pass
            self.path = Polyline(body + list(contour) + [lead_out])
        else:
            self.path = Polyline([lead_in] + zpts + list(contour) + [lead_out])

        # Flat tool on a TILTED face: the tool-path is the tool CENTRE, but a flat
        # bit only touches an inclined plane at its UP-SLOPE rim -- so if the centre
        # rode the surface, the disk would sink into the material on the up-slope
        # side. Shift the whole path DOWN-SLOPE by the tool `radius` (in XY, keeping
        # Z), so the up-slope rim lands on each target point and the rest of the disk
        # stays above the surface. The face's full slope is used, so a face tilted on
        # both axes is handled. No shift on a horizontal face.
        if self.incline:
            normal = along.cross(across)
            if normal[2] < 0:
                normal = normal * -1.0  # point up, out of the kept material
            hx, hy = normal[0], normal[1]  # horizontal part of the up-normal = DOWN-slope dir
            hlen = (hx * hx + hy * hy) ** 0.5  # 0 if the face is horizontal -> no shift
            if hlen > 1e-9:
                shift = Translation.from_vector([hx / hlen * r, hy / hlen * r, 0.0])  # down-slope by the tool radius
                self.zigzag = self.zigzag.transformed(shift)
                self.contour = self.contour.transformed(shift)
                self.path = self.path.transformed(shift)

    def __repr__(self):
        return f"toolpath_2d_surfacing(passes={self.passes}, spacing={self.spacing:.3f}, radius={self.radius}, safe_z={self.safe_z})"
