from compas.geometry import Point
from compas.geometry import Polyline


def toolpath_merge(*toolpaths, home=True):
    """Merge tool-paths (or polylines) into ONE :class:`compas.geometry.Polyline`, in order.

    Every tool-path begins and ends at its safe-Z height, so joining them
    end-to-start gives a clean rapid at safe Z between cuts -- no diagonal dive
    through the stock. Accepts tool-path objects (anything with a ``.path``) or
    raw polylines. Consecutive coincident points are dropped. With ``home`` (the
    default) the merged path traverses back at safe Z to the global start, so the
    whole job ENDS where it began.
    """
    points = []
    for tp in toolpaths:
        path = getattr(tp, "path", tp)
        for p in path:
            point = Point(*p)
            if points and points[-1].distance_to_point(point) < 1e-9:
                continue
            points.append(point)
    if home and len(points) > 1 and points[0].distance_to_point(points[-1]) > 1e-9:
        points.append(points[0].copy())
    return Polyline(points)
