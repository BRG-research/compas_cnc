"""Polyline offsetting and hatching, backed by Clipper2.

This is the friendly layer over the :mod:`compas_cnc._clipper2` extension. It
speaks COMPAS geometry (:class:`~compas.geometry.Polyline`,
:class:`~compas.geometry.Polygon`, :class:`~compas.geometry.Line`) and works in
the XY plane -- the Z coordinate of inputs is ignored and outputs are returned
at ``z = 0``.

Two operations are provided:

* :func:`offset_polyline` -- grow / shrink a polyline (closed or open).
* :func:`hatch` -- fill a closed polyline with a set of parallel lines clipped
  to its interior.
"""

import math

import numpy as np
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Polyline

from compas_cnc import _clipper2  # type: ignore

__all__ = ["offset_polyline", "outline", "hatch", "clip"]

# Clipper2 enum mappings (kept in sync with src/clipper2.cpp).
_JOIN_TYPES = {"square": 0, "bevel": 1, "round": 2, "miter": 3}
_END_TYPES = {"polygon": 0, "joined": 1, "butt": 2, "square": 3, "round": 4}
_FILL_RULES = {"even_odd": 0, "nonzero": 1, "positive": 2, "negative": 3}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _as_xy(points) -> np.ndarray:
    """Return an ``(N, 2)`` float array of the XY coordinates of ``points``.

    Accepts a COMPAS ``Polyline``/``Polygon``, any iterable of points, or an
    array-like. The Z component, if present, is dropped.
    """
    array = np.asarray([list(point) for point in points], dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("Expected a sequence of points with at least 2 coordinates each.")
    return array[:, :2]


def _is_closed(points: np.ndarray, tol: float = 1e-9) -> bool:
    """Whether the first and last point coincide (within ``tol``)."""
    return len(points) > 1 and bool(np.allclose(points[0], points[-1], atol=tol))


def _open_ring(points: np.ndarray) -> np.ndarray:
    """Drop a duplicated closing point so Clipper2 sees a clean polygon ring."""
    if _is_closed(points):
        return points[:-1]
    return points


def _ensure_ccw(ring: np.ndarray) -> np.ndarray:
    """Return ``ring`` oriented counter-clockwise (positive signed area).

    Offsetting a closed polygon is sign-sensitive, so we normalise the winding
    up front. With a CCW ring, ``distance > 0`` grows the polygon outward and
    ``distance < 0`` shrinks it inward.
    """
    x = ring[:, 0]
    y = ring[:, 1]
    # shoelace formula
    area2 = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    if area2 < 0:
        return ring[::-1]
    return ring


def _to_paths(rings) -> list:
    """Convert ``(N, 2)`` arrays into the list-of-[x, y] paths the C++ layer wants."""
    return [[[float(x), float(y)] for x, y in ring] for ring in rings]


def _chain(paths, tol=1e-6) -> list:
    """Join open paths that share XY endpoints into continuous polylines.

    Clipper2 returns each clipped line piece on its own; this stitches pieces back
    together at coincident endpoints, so a clipped outline comes out as one (or a
    few) continuous polylines instead of hundreds of fragments.
    """
    from collections import defaultdict

    segs = [list(p) for p in paths if len(p) >= 2]

    def key(pt):
        return (round(pt[0] / tol), round(pt[1] / tol))

    ends = defaultdict(list)  # endpoint key -> [(segment index, 0=start | 1=end)]
    for i, s in enumerate(segs):
        ends[key(s[0])].append((i, 0))
        ends[key(s[-1])].append((i, 1))

    used = [False] * len(segs)
    chains = []
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        chain = list(segs[i])
        while True:  # grow at the tail
            nxt = next(((j, w) for j, w in ends[key(chain[-1])] if not used[j]), None)
            if nxt is None:
                break
            j, w = nxt
            used[j] = True
            chain += segs[j][1:] if w == 0 else segs[j][-2::-1]
        while True:  # grow at the head
            prv = next(((j, w) for j, w in ends[key(chain[0])] if not used[j]), None)
            if prv is None:
                break
            j, w = prv
            used[j] = True
            chain[:0] = segs[j][:-1] if w == 1 else segs[j][:0:-1]
        chains.append(chain)
    return chains


# --------------------------------------------------------------------------
# offset
# --------------------------------------------------------------------------


def offset_polyline(
    polyline,
    distance,
    closed=None,
    join_type="round",
    end_type=None,
    miter_limit=2.0,
    arc_tolerance=0.0,
    precision=4,
) -> list[Polyline]:
    """Offset a polyline by ``distance``.

    Parameters
    ----------
    polyline : :class:`compas.geometry.Polyline` | :class:`compas.geometry.Polygon` | sequence of points
        The polyline to offset (XY plane; Z is ignored).
    distance : float
        Offset distance. For a closed polyline a positive distance grows it
        outward and a negative distance shrinks it inward (the input winding is
        normalised so this is always true). For an open polyline the offset is
        applied to both sides.
    closed : bool, optional
        Whether to treat the input as a closed polygon. By default this is
        inferred: the input is closed if its first and last point coincide.
    join_type : {"round", "miter", "bevel", "square"}, optional
        How corners are handled. Defaults to ``"round"``.
    end_type : str, optional
        How path ends are handled. Defaults to ``"polygon"`` for closed inputs
        and ``"round"`` for open ones. One of ``"polygon"``, ``"joined"``,
        ``"butt"``, ``"square"``, ``"round"``.
    miter_limit : float, optional
        Miter limit, used only with ``join_type="miter"``. Defaults to ``2.0``.
    arc_tolerance : float, optional
        Maximum deviation when approximating round joins/ends. ``0`` lets
        Clipper2 pick a sensible value. Defaults to ``0.0``.
    precision : int, optional
        Decimal places Clipper2 keeps internally (0..8). Defaults to ``4``.

    Returns
    -------
    list[:class:`compas.geometry.Polyline`]
        The offset polyline(s). Offsetting can split or merge contours, so a
        list is always returned. Closed results are returned closed (first
        point repeated at the end).

    Examples
    --------
    >>> from compas.geometry import Polyline
    >>> square = Polyline([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [0, 0, 0]])
    >>> result = offset_polyline(square, 2.0)            # grow outward
    >>> len(result)
    1
    """
    points = _as_xy(polyline)
    if closed is None:
        closed = _is_closed(points)

    if closed:
        ring = _ensure_ccw(_open_ring(points))
        et = "polygon" if end_type is None else end_type
        paths_in = _to_paths([ring])
    else:
        et = "round" if end_type is None else end_type
        paths_in = _to_paths([points])

    if join_type not in _JOIN_TYPES:
        raise ValueError(f"Unknown join_type {join_type!r}; expected one of {sorted(_JOIN_TYPES)}.")
    if et not in _END_TYPES:
        raise ValueError(f"Unknown end_type {et!r}; expected one of {sorted(_END_TYPES)}.")

    result = _clipper2.offset_paths(
        paths_in,
        float(distance),
        _JOIN_TYPES[join_type],
        _END_TYPES[et],
        float(miter_limit),
        float(arc_tolerance),
        int(precision),
    )

    polylines = []
    for path in result:
        coords = [[x, y, 0.0] for x, y in path]
        if not coords:
            continue
        if et == "polygon":
            coords.append(coords[0])  # close the ring
        polylines.append(Polyline(coords))
    return polylines


# --------------------------------------------------------------------------
# outline
# --------------------------------------------------------------------------


def outline(
    mesh,
    distance,
    join_type="round",
    miter_limit=2.0,
    arc_tolerance=0.0,
    precision=4,
    z=0.0,
) -> list[Polyline]:
    """Compute the 2D outline (XY silhouette) of a mesh, grown by ``distance``.

    Every face of ``mesh`` is projected onto the XY plane (its Z is dropped) and
    the union of those projections is taken -- that union is the mesh's
    silhouette (its shadow on the table). The silhouette is then offset outward
    by ``distance``: pass the milling tool *radius* so a tool of that radius can
    run around the OUTSIDE of the part, its cutting edge just grazing the
    silhouette.

    Both steps happen in a single Clipper2 ``InflatePaths`` call. Offsetting
    (Minkowski sum with a disc) distributes over union, so offsetting every face
    by ``distance`` and unioning the results is exactly the silhouette grown by
    ``distance`` -- no separate boolean-union pass is needed. Faces that project
    to (near-)zero area -- vertical walls seen edge-on -- are skipped: the
    horizontal faces already cover the whole silhouette, so they add nothing.

    Parameters
    ----------
    mesh : :class:`compas.datastructures.Mesh`
        The mesh to outline. Each face's Z is ignored (the part is flattened
        onto the XY plane).
    distance : float
        Outward offset, i.e. the milling tool radius. Must be > 0: the projected
        faces only merge into a single clean outline when they are grown
        outward.
    join_type : {"round", "miter", "bevel", "square"}, optional
        How corners are rounded by the offset. Defaults to ``"round"``.
    miter_limit : float, optional
        Miter limit, used only with ``join_type="miter"``. Defaults to ``2.0``.
    arc_tolerance : float, optional
        Maximum deviation when approximating round joins. ``0`` lets Clipper2
        pick a sensible value. Defaults to ``0.0``.
    precision : int, optional
        Decimal places Clipper2 keeps internally (0..8). Defaults to ``4``.
    z : float, optional
        Z height at which the resulting outline is placed. Defaults to ``0.0``.

    Returns
    -------
    list[:class:`compas.geometry.Polyline`]
        The offset outline contour(s), each a closed polyline. The largest is
        the outer perimeter; a mesh with disconnected parts or with through
        holes wider than ``2 * distance`` yields more than one polyline.

    Examples
    --------
    >>> from compas.datastructures import Mesh
    >>> box = Mesh.from_vertices_and_faces(
    ...     [[0, 0, 0], [10, 0, 0], [10, 8, 0], [0, 8, 0],
    ...      [0, 0, 5], [10, 0, 5], [10, 8, 5], [0, 8, 5]],
    ...     [[0, 1, 2, 3], [4, 5, 6, 7]],
    ... )
    >>> contours = outline(box, 3.0)            # 10x8 footprint grown by 3
    >>> len(contours)
    1
    """
    if distance <= 0:
        raise ValueError("distance must be positive (the milling tool radius); faces only union when grown outward.")
    if join_type not in _JOIN_TYPES:
        raise ValueError(f"Unknown join_type {join_type!r}; expected one of {sorted(_JOIN_TYPES)}.")

    rings = []
    for face in mesh.faces():
        ring = _open_ring(_as_xy(mesh.face_coordinates(face)))
        if len(ring) < 3:
            continue
        x = ring[:, 0]
        y = ring[:, 1]
        area2 = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))  # shoelace
        if abs(area2) < 1e-12:
            continue  # vertical wall projecting to a line -- adds nothing to the silhouette
        rings.append(ring[::-1] if area2 < 0 else ring)  # normalise to CCW (positive fill)

    if not rings:
        return []

    result = _clipper2.offset_paths(
        _to_paths(rings),
        float(distance),
        _JOIN_TYPES[join_type],
        _END_TYPES["polygon"],
        float(miter_limit),
        float(arc_tolerance),
        int(precision),
    )

    polylines = []
    for path in result:
        if not path:
            continue
        coords = [[x, y, z] for x, y in path]
        coords.append(coords[0])  # close the ring
        polylines.append(Polyline(coords))
    return polylines


# --------------------------------------------------------------------------
# hatch
# --------------------------------------------------------------------------


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    """Rotate ``(N, 2)`` ``points`` about the origin by ``angle`` (radians, CCW)."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return points @ rotation.T


def hatch(
    boundary,
    spacing,
    angle=0.0,
    fill_rule="nonzero",
    precision=4,
    holes=None,
) -> list[Line]:
    """Fill a closed polyline with parallel lines clipped to its interior.

    A family of parallel lines at ``angle`` is laid out so it just covers the
    boundary's bounding box *measured in the hatch direction* (the minimal box
    aligned to the lines, so none are wasted), then each line is clipped to the
    inside of the boundary using Clipper2.

    Parameters
    ----------
    boundary : :class:`compas.geometry.Polyline` | :class:`compas.geometry.Polygon` | sequence of points
        A closed polyline / polygon (XY plane; Z is ignored). It does not need
        to be convex -- a single hatch line may yield several clipped segments.
    spacing : float
        Perpendicular distance between successive hatch lines. Must be > 0.
    angle : float, optional
        Direction of the hatch lines in radians, measured CCW from the X axis.
        Defaults to ``0.0`` (horizontal lines).
    fill_rule : {"nonzero", "even_odd", "positive", "negative"}, optional
        How the boundary interior is determined. ``"even_odd"`` treats inner
        rings as holes. Defaults to ``"nonzero"``.
    precision : int, optional
        Decimal places Clipper2 keeps internally (0..8). Defaults to ``4``.
    holes : sequence, optional
        Closed rings to subtract from ``boundary`` (islands the fill must avoid).
        When given, ``fill_rule`` is forced to ``"even_odd"``. Defaults to ``None``.

    Returns
    -------
    list[:class:`compas.geometry.Line`]
        The clipped hatch segments lying inside the boundary, at ``z = 0``.

    Examples
    --------
    >>> from compas.geometry import Polyline
    >>> square = Polyline([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [0, 0, 0]])
    >>> lines = hatch(square, spacing=2.0)
    >>> all(isinstance(line, Line) for line in lines)
    True
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive.")
    if fill_rule not in _FILL_RULES:
        raise ValueError(f"Unknown fill_rule {fill_rule!r}; expected one of {sorted(_FILL_RULES)}.")

    ring = _open_ring(_as_xy(boundary))
    if len(ring) < 3:
        raise ValueError("A hatch boundary needs at least 3 distinct points.")

    centroid = ring.mean(axis=0)
    # Work in a frame where the hatch lines are horizontal: rotate the boundary
    # by -angle, lay out horizontal scan lines over its axis-aligned box, then
    # rotate everything back.
    local = _rotate(ring - centroid, -angle)
    xmin, ymin = local.min(axis=0)
    xmax, ymax = local.max(axis=0)

    pad = max(xmax - xmin, 1.0) * 1e-6  # nudge endpoints just past the box in x

    lines_local = []
    y = ymin + spacing
    while y < ymax - 1e-12:
        lines_local.append([[xmin - pad, y], [xmax + pad, y]])
        y += spacing

    if not lines_local:
        return []

    # Rotate the scan lines back into world coordinates.
    lines_world = []
    for start, end in lines_local:
        pts = _rotate(np.array([start, end]), angle) + centroid
        lines_world.append([[float(pts[0, 0]), float(pts[0, 1])], [float(pts[1, 0]), float(pts[1, 1])]])

    rings = [ring]
    if holes:
        rings += [_open_ring(_as_xy(hole)) for hole in holes]
        fill_rule = "even_odd"  # inner rings are subtracted as holes
    boundary_path = _to_paths(rings)

    clipped = _clipper2.clip_lines(lines_world, boundary_path, _FILL_RULES[fill_rule], int(precision))

    segments = []
    for path in clipped:
        if len(path) < 2:
            continue
        start = Point(path[0][0], path[0][1], 0.0)
        end = Point(path[-1][0], path[-1][1], 0.0)
        segments.append(Line(start, end))
    return segments


def clip(polyline, boundary, z=0.0, fill_rule="nonzero", precision=4) -> list[Polyline]:
    """Boolean-intersect a polyline with a closed boundary region.

    Returns the parts of ``polyline`` that lie INSIDE ``boundary`` -- the open
    polyline(s) you get by intersecting the path with the region, via Clipper2.

    Parameters
    ----------
    polyline : :class:`compas.geometry.Polyline` | :class:`compas.geometry.Polygon` | sequence of points
        The path to clip (XY plane; Z is ignored).
    boundary : :class:`compas.geometry.Polyline` | :class:`compas.geometry.Polygon` | sequence of points
        A closed region to intersect against (XY plane; Z is ignored).
    z : float, optional
        Z assigned to every returned point. Defaults to ``0.0``.
    fill_rule : {"nonzero", "even_odd", "positive", "negative"}, optional
        How the boundary interior is determined. Defaults to ``"nonzero"``.
    precision : int, optional
        Decimal places Clipper2 keeps internally (0..8). Defaults to ``4``.

    Returns
    -------
    list[:class:`compas.geometry.Polyline`]
        The clipped open polylines lying inside the boundary, at ``z``.
    """
    if fill_rule not in _FILL_RULES:
        raise ValueError(f"Unknown fill_rule {fill_rule!r}; expected one of {sorted(_FILL_RULES)}.")
    pts = _as_xy(polyline)
    lines = [[[float(pts[i][0]), float(pts[i][1])], [float(pts[i + 1][0]), float(pts[i + 1][1])]] for i in range(len(pts) - 1)]
    if not lines:
        return []
    ring = _open_ring(_as_xy(boundary))
    if len(ring) < 3:
        raise ValueError("A clip boundary needs at least 3 distinct points.")
    boundary_path = _to_paths([ring])
    clipped = _clipper2.clip_lines(lines, boundary_path, _FILL_RULES[fill_rule], int(precision))
    chains = _chain([p for p in clipped if len(p) >= 2])  # stitch fragments into continuous polylines
    return [Polyline([Point(pt[0], pt[1], z) for pt in chain]) for chain in chains]
