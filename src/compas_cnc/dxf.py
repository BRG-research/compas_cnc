"""Load 2D geometry from DXF files into COMPAS objects.

``ezdxf`` reads **DXF only**, not DWG. To use a ``.dwg`` file, export / convert
it to ASCII DXF first (e.g. "Save As -> AutoCAD DXF", or the ODA File Converter).
"""

import ezdxf

from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Polyline

__all__ = ["load_dxf"]


def load_dxf(path, sag=0.5):
    """Read 2D entities from a DXF file as COMPAS geometry.

    Parameters
    ----------
    path : str | os.PathLike
        Path to a DXF file. ``ezdxf`` reads DXF, not DWG -- convert a ``.dwg``
        to ASCII DXF first.
    sag : float, optional
        Maximum chord error (in the drawing's units) used to flatten curved
        entities (ARC, CIRCLE, ELLIPSE, SPLINE, bulged LWPOLYLINE) into
        polylines. Smaller values give finer curves.

    Returns
    -------
    list[:class:`compas.geometry.Line` | :class:`compas.geometry.Polyline`]
        One geometry per supported modelspace entity (LINE -> Line, everything
        else -> Polyline). Unsupported entities are skipped.

    Examples
    --------
    >>> from compas_cnc.dxf import load_dxf
    >>> curves = load_dxf("data/cnc_table.dxf")  # doctest: +SKIP
    """
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    geometry = []
    for entity in msp:
        kind = entity.dxftype()
        try:
            if kind == "LINE":
                geometry.append(Line(Point(*entity.dxf.start), Point(*entity.dxf.end)))
            elif hasattr(entity, "flattening"):  # LWPOLYLINE, ARC, CIRCLE, ELLIPSE, SPLINE
                points = [Point(*p) for p in entity.flattening(sag)]
                if len(points) >= 2:
                    geometry.append(Polyline(points))
            elif kind == "POLYLINE":
                points = [Point(*v.dxf.location) for v in entity.vertices]
                if len(points) >= 2:
                    geometry.append(Polyline(points))
        except Exception:
            continue
    return geometry
