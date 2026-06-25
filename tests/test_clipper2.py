"""Tests for the Clipper2 wrapper (offset_polyline and hatch).

Skipped when the compiled extension has not been built.
"""

import math

import pytest
from compas.geometry import Line
from compas.geometry import Polyline

clipper2 = pytest.importorskip("compas_cnc.clipper2")

offset_polyline = clipper2.offset_polyline
hatch = clipper2.hatch


def _bbox(polylines):
    pts = [pt for pl in polylines for pt in pl]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


@pytest.fixture
def square():
    return Polyline([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [0, 0, 0]])


# ---------------------------------------------------------------- offset


def test_offset_outward_grows(square):
    result = offset_polyline(square, 2.0)
    assert len(result) == 1
    xmin, ymin, xmax, ymax = _bbox(result)
    assert xmin == pytest.approx(-2.0, abs=1e-3)
    assert ymin == pytest.approx(-2.0, abs=1e-3)
    assert xmax == pytest.approx(12.0, abs=1e-3)
    assert ymax == pytest.approx(12.0, abs=1e-3)


def test_offset_inward_shrinks(square):
    result = offset_polyline(square, -2.0)
    assert len(result) == 1
    xmin, ymin, xmax, ymax = _bbox(result)
    assert xmin == pytest.approx(2.0, abs=1e-3)
    assert ymin == pytest.approx(2.0, abs=1e-3)
    assert xmax == pytest.approx(8.0, abs=1e-3)
    assert ymax == pytest.approx(8.0, abs=1e-3)


def test_offset_returns_closed_ring(square):
    (result,) = offset_polyline(square, 1.0)
    assert result[0] == result[-1]


def test_offset_orientation_independent(square):
    """Reversing the input winding must not flip the in/out convention."""
    reversed_square = Polyline(square.points[::-1])
    forward = offset_polyline(square, 2.0)
    backward = offset_polyline(reversed_square, 2.0)
    assert _bbox(forward) == pytest.approx(_bbox(backward), abs=1e-6)


# ---------------------------------------------------------------- hatch


def test_hatch_horizontal_spans_width(square):
    lines = hatch(square, spacing=2.0)
    # 10-wide box, lines at y = 2,4,6,8 -> 4 lines
    assert len(lines) == 4
    for line in lines:
        assert isinstance(line, Line)
        assert line.length == pytest.approx(10.0, abs=1e-3)
        assert min(line.start[0], line.end[0]) == pytest.approx(0.0, abs=1e-3)
        assert max(line.start[0], line.end[0]) == pytest.approx(10.0, abs=1e-3)


def test_hatch_spacing_controls_count(square):
    coarse = hatch(square, spacing=5.0)
    fine = hatch(square, spacing=1.0)
    assert len(fine) > len(coarse)


def test_hatch_angle_rotates(square):
    lines = hatch(square, spacing=2.0, angle=math.radians(45))
    assert len(lines) > 0
    # diagonal of a 10x10 square is ~14.14; clipped diagonals must not exceed it
    assert max(line.length for line in lines) <= 14.15


def test_hatch_concave_splits_into_segments():
    # A 'U' shape: a horizontal line through the prongs is cut into two pieces.
    u_shape = Polyline(
        [
            [0, 0, 0],
            [10, 0, 0],
            [10, 10, 0],
            [7, 10, 0],
            [7, 3, 0],
            [3, 3, 0],
            [3, 10, 0],
            [0, 10, 0],
            [0, 0, 0],
        ]
    )
    lines = hatch(u_shape, spacing=1.0)
    # scan lines above y=3 must yield two segments (left and right prong)
    upper = [line for line in lines if (line.start[1] + line.end[1]) / 2 > 3.5]
    assert len(upper) >= 2 * 5  # at least two segments for several scan lines


def test_hatch_zero_spacing_raises(square):
    with pytest.raises(ValueError):
        hatch(square, spacing=0.0)
