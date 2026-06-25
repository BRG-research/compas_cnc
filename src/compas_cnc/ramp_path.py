"""Linear-ramp (plunge) milling tool-path descended along an open polyline."""

import math

from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import Vector

__all__ = ["toolpath_ramp_path"]

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


class toolpath_ramp_path:
    """Ramp (linear-plunge) tool-path that descends gradually along an OPEN
    polyline, sweeping it back and forth.

    This generalises :class:`toolpath_ramp_line` from a straight line to a
    multi-segment path -- e.g. the end-cap arc sliced out of a part's silhouette
    outline -- so the tool can ramp down AROUND the end of a beam to cut it off,
    instead of plunging straight. Each back-and-forth traverse of the path
    descends by ``step`` (Z interpolated by arc length, so the slope is even),
    down to the floor, then an optional flat finishing pass cleans the bottom.

    Sequence of the path: plunge from ``safe_z`` to the start of the path, the
    descending sweeps, an optional flat floor pass, then a retract to ``safe_z``.

    Parameters
    ----------
    path : :class:`compas.geometry.Polyline`
        The open path to ramp along, positioned at the MOUTH (top) of the cut.
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

    Attributes
    ----------
    path : :class:`compas.geometry.Polyline`
        The full tool-centre path: plunge, ramp, floor pass, retract.
    ramp : :class:`compas.geometry.Polyline`
        The descending sweeps (and floor pass), without the lead-in/out.
    passes : int
    step : float
    ramp_angle : float
    depth : float
    safe_z : float
    """

    def __init__(self, path, descent, step=None, ramp_angle=None, bottom_pass=True, safe_z=None):
        self._pts = [Point(*p) for p in path]
        if len(self._pts) < 2:
            raise ValueError("ramp path needs at least 2 points.")
        self.descent = Vector(*descent)
        self._step_request = None if step is None else float(step)
        self._ramp_angle_request = None if ramp_angle is None else float(ramp_angle)
        self.bottom_pass = bottom_pass
        self._safe_z_request = safe_z
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
        :class:`toolpath_ramp_path`
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
        return cls(segment, descent, step=step, ramp_angle=ramp_angle, bottom_pass=bottom_pass, safe_z=safe_z)

    # ------------------------------------------------------------------ #

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

        # Each traverse spans arc fraction 0..1; pass k descends from k/n to
        # (k+1)/n of the full descent, so the slope is even along the path.
        forward = [(cum[j] / total, pts[j]) for j in range(len(pts))]
        backward = [((total - cum[j]) / total, pts[j]) for j in range(len(pts) - 1, -1, -1)]

        points = []
        for k in range(n):
            seq = forward if k % 2 == 0 else backward
            for j, (arc, p) in enumerate(seq):
                if k > 0 and j == 0:
                    continue  # turn point already added by the previous pass
                points.append(p + self.descent * ((k + arc) / n))

        # One flat finishing pass back across the path at full depth.
        if self.bottom_pass:
            seq = backward if n % 2 == 1 else forward
            for j, (_arc, p) in enumerate(seq):
                if j == 0:
                    continue
                points.append(p + self.descent)

        self.ramp = Polyline(points)

        # Vertical plunge-in / retract-out, kept >= MIN_CLEARANCE above the mouth.
        mouth_z = max(p[2] for p in points)
        floor = mouth_z + MIN_CLEARANCE
        self.safe_z = floor if self._safe_z_request is None else max(self._safe_z_request, floor)
        first, last = points[0], points[-1]
        lead_in = Point(first[0], first[1], self.safe_z)
        lead_out = Point(last[0], last[1], self.safe_z)
        self.path = Polyline([lead_in] + points + [lead_out])

    def __repr__(self):
        return (
            f"toolpath_ramp_path(passes={self.passes}, step={self.step:.3f}, "
            f"ramp_angle={math.degrees(self.ramp_angle):.1f}deg, depth={self.depth:.3f}, safe_z={self.safe_z})"
        )
