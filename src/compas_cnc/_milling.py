def signed_area_xy(points):
    """Twice-area shoelace SIGN on XY: ``> 0`` = CCW (seen from +Z), ``< 0`` = CW.

    Tolerates a duplicated closing point (its wrap term contributes zero area), so a
    closed or open ring both give the same sign.
    """
    pts = list(points)
    n = len(pts)
    if n < 3:
        return 0.0
    return 0.5 * sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))


def oriented(points, ccw):
    """Return ``points`` reversed iff their XY winding does not match ``ccw`` (True=CCW).

    A closed list (first point repeated at the end) stays closed after reversal.
    """
    pts = list(points)
    is_ccw = signed_area_xy(pts) > 0
    return pts if is_ccw == ccw else pts[::-1]


def want_ccw(direction, cleared_inside):
    """Target winding (True=CCW) for a milling ``direction``, or ``None`` to leave as-is.

    Assumes a right-hand CW (M3) tool, for which CLIMB keeps the uncut material on the
    LEFT of travel. ``direction`` is ``"climb"``, ``"conventional"``, or ``None``.
    ``cleared_inside`` is ``True`` for a pocket / outer-boundary wall (the cleared region
    is inside the loop) and ``False`` for an island / hole (kept material inside).

    * pocket  : climb -> CW,  conventional -> CCW
    * island  : climb -> CCW, conventional -> CW
    """
    if direction is None:
        return None
    return (direction == "climb") != cleared_inside
