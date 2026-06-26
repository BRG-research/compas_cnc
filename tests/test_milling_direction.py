"""Tests for climb/conventional milling-direction control across the toolpaths."""

import math

import pytest
from compas.geometry import Line
from compas.geometry import Polyline
from compas.geometry import Vector

from compas_cnc import toolpath_2d_drill
from compas_cnc import toolpath_2d_hatch
from compas_cnc import toolpath_2d_ramp
from compas_cnc import toolpath_2d_surfacing
from compas_cnc._milling import oriented
from compas_cnc._milling import signed_area_xy
from compas_cnc._milling import want_ccw

CW = [[0, 0], [0, 10], [10, 10], [10, 0]]  # clockwise square (negative area)
CCW = [[0, 0], [10, 0], [10, 10], [0, 10]]  # counter-clockwise (positive area)


def test_signed_area_sign_and_closing_point():
    assert signed_area_xy(CCW) > 0
    assert signed_area_xy(CW) < 0
    assert signed_area_xy(CCW + [CCW[0]]) > 0  # tolerates a duplicated closing point
    assert signed_area_xy([[0, 0], [1, 1]]) == 0.0  # fewer than 3 points -> 0


def test_oriented_reverses_only_when_needed():
    assert oriented(CCW, ccw=True) == CCW  # already CCW -> unchanged
    assert oriented(CCW, ccw=False) == CCW[::-1]  # flip to CW
    assert signed_area_xy(oriented(CW, ccw=True)) > 0  # CW -> CCW


def test_want_ccw_truth_table():
    assert want_ccw(None, True) is None and want_ccw(None, False) is None
    assert want_ccw("climb", True) is False  # pocket climb -> CW
    assert want_ccw("climb", False) is True  # island climb -> CCW
    assert want_ccw("conventional", True) is True  # pocket conventional -> CCW
    assert want_ccw("conventional", False) is False  # island conventional -> CW


# --------------------------------------------------------------------------- #
# drill
# --------------------------------------------------------------------------- #

DRILL = dict(hole_radius=10.0, tool_diameter=3.175)


def test_drill_climb_equals_default_conventional_flips():
    axis = Line([40, 25, 20], [40, 25, 0])
    none = toolpath_2d_drill(axis, **DRILL)
    climb = toolpath_2d_drill(axis, direction="climb", **DRILL)
    conv = toolpath_2d_drill(axis, direction="conventional", **DRILL)
    # default helix already climbs the bore -> None and "climb" are identical
    assert all(p.distance_to_point(q) < 1e-9 for p, q in zip(none.helix, climb.helix))
    # conventional reverses the winding (signed area sign flips)
    assert (signed_area_xy([list(p) for p in climb.helix]) > 0) != (signed_area_xy([list(p) for p in conv.helix]) > 0)
    # every variant still retracts straight up (start == end) and keeps z-safety
    for tp in (none, climb, conv):
        assert list(tp.path[0]) == pytest.approx(list(tp.path[-1]))


def test_drill_bad_direction_raises():
    with pytest.raises(ValueError):
        toolpath_2d_drill(Line([0, 0, 10], [0, 0, 0]), hole_radius=5.0, tool_diameter=3.0, direction="down")


# --------------------------------------------------------------------------- #
# hatch
# --------------------------------------------------------------------------- #

BOUNDARY = Polyline([[0, 0, 0], [60, 0, 0], [60, 40, 0], [0, 40, 0], [0, 0, 0]])
HOLE = Polyline([[25, 18, 0], [35, 18, 0], [35, 28, 0], [25, 28, 0], [25, 18, 0]])


def _along_senses(fill, ux=1.0, uy=0.0):
    pts = list(fill)
    return {1 if (b[0] * ux + b[1] * uy) > (a[0] * ux + a[1] * uy) else -1 for a, b in zip(pts[0::2], pts[1::2])}


def test_hatch_direction_makes_fill_one_directional():
    none = toolpath_2d_hatch(BOUNDARY, spacing=3.0, holes=[HOLE], radius=1.5)
    climb = toolpath_2d_hatch(BOUNDARY, spacing=3.0, holes=[HOLE], radius=1.5, direction="climb")
    assert none.one_directional is False and climb.one_directional is True
    assert len(_along_senses(climb.fill)) == 1  # every pass cut the same way
    assert list(climb.path[0]) == pytest.approx(list(climb.path[-1]))  # still returns home


# --------------------------------------------------------------------------- #
# surfacing
# --------------------------------------------------------------------------- #

QUAD = [[0, 0, 0], [80, 0, 0], [80, 50, 0], [0, 50, 0]]


def test_surfacing_contour_winding_matches_direction():
    climb = toolpath_2d_surfacing.from_quad(QUAD, radius=1.5, stepover=3.0, direction="climb")
    conv = toolpath_2d_surfacing.from_quad(QUAD, radius=1.5, stepover=3.0, direction="conventional")
    # the rectangle is a pocket wall: climb -> CW (negative), conventional -> CCW (positive)
    assert signed_area_xy([list(p) for p in climb.contour]) < 0
    assert signed_area_xy([list(p) for p in conv.contour]) > 0
    for tp in (climb, conv):
        assert list(tp.path[0]) == pytest.approx(list(tp.path[-1]))  # ends at the start corner


def test_surfacing_none_unchanged_and_one_directional_passes():
    none = toolpath_2d_surfacing.from_quad(QUAD, radius=1.5, stepover=3.0)
    climb = toolpath_2d_surfacing.from_quad(QUAD, radius=1.5, stepover=3.0, direction="climb")
    assert none.one_directional is False and climb.one_directional is True
    senses = {1 if b[0] > a[0] else -1 for a, b in zip(list(climb.zigzag)[0::2], list(climb.zigzag)[1::2])}
    assert len(senses) == 1  # every zig-zag pass the same way


# --------------------------------------------------------------------------- #
# ramp
# --------------------------------------------------------------------------- #


def test_ramp_open_path_direction_is_noop():
    line = Line([10, 25, 0], [70, 25, 0])
    none = toolpath_2d_ramp(line, Vector(0, 0, -15), step=2.0)
    climb = toolpath_2d_ramp(line, Vector(0, 0, -15), step=2.0, direction="climb")
    assert all(p.distance_to_point(q) < 1e-9 for p, q in zip(none.path, climb.path))


def test_ramp_closed_loop_winding_and_home():
    ring = Polyline([[0, 0, 5], [40, 0, 5], [40, 30, 5], [0, 30, 5], [0, 0, 5]])
    climb = toolpath_2d_ramp(ring, Vector(0, 0, -10), step=3.0, direction="climb")
    conv = toolpath_2d_ramp(ring, Vector(0, 0, -10), step=3.0, direction="conventional")
    # pocket loop (default pocket=True): climb -> CW (negative), conventional -> CCW
    assert signed_area_xy([list(p) for p in climb._pts[:-1]]) < 0
    assert signed_area_xy([list(p) for p in conv._pts[:-1]]) > 0
    for tp in (climb, conv):
        assert list(tp.path[0]) == pytest.approx(list(tp.path[-1]))  # returns home
