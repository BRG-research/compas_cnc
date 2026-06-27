"""Tests for the helical drilling tool-path."""

import math

import pytest
from compas.geometry import Line

from compas_cnc.toolpath_2d_drill import toolpath_2d_drill


def _axis_distance(point, line):
    """Perpendicular distance from ``point`` to the infinite line of ``line``."""
    start = line.start
    direction = line.end - line.start
    direction.unitize()
    vec = point - start
    along = vec.dot(direction)
    foot = start + direction * along
    return (point - foot).length


def test_helix_radius_and_turns_from_ramp_angle():
    axis = Line([0, 0, 10], [0, 0, 0])  # 10 deep, downward
    angle = math.radians(10.0)
    drill = toolpath_2d_drill(axis, hole_radius=5.0, tool_diameter=3.0, ramp_angle=angle)
    assert not drill.is_drill
    assert drill.helix_radius == pytest.approx(5.0 - 1.5)
    # turns = ceil(depth / (circumference * tan(angle))) -- rounded up, gentler
    circumference = 2.0 * math.pi * drill.helix_radius
    assert drill.turns == max(1, math.ceil(10.0 / (circumference * math.tan(angle))))
    assert drill.ramp_angle <= angle + 1e-9  # actual angle never steeper than requested
    # the ACTUAL ramp angle is recomputed from the whole-turn count
    assert drill.ramp_angle == pytest.approx(math.atan2(10.0 / drill.turns, circumference))


def test_helix_points_ride_the_offset_radius():
    axis = Line([0, 0, 10], [0, 0, 0])
    drill = toolpath_2d_drill(axis, hole_radius=5.0, tool_diameter=3.0, bottom_pass=False)
    for point in drill.helix:
        assert _axis_distance(point, axis) == pytest.approx(drill.helix_radius, abs=1e-6)


def test_helix_descends_monotonically():
    axis = Line([0, 0, 10], [0, 0, 0])
    drill = toolpath_2d_drill(axis, hole_radius=5.0, tool_diameter=3.0, bottom_pass=False)
    zs = [p[2] for p in drill.helix]
    assert zs[0] == pytest.approx(10.0)
    assert zs[-1] == pytest.approx(0.0)
    assert all(b <= a + 1e-9 for a, b in zip(zs, zs[1:]))  # non-increasing


def test_helix_on_horizontal_axis():
    # Mirrors the column bolt-holes: axis along -Y, away from world Z.
    axis = Line([0, 0, 0], [0, -15, 0])
    drill = toolpath_2d_drill(axis, hole_radius=2.5, tool_diameter=3.0)
    assert not drill.is_drill
    assert drill.helix_radius == pytest.approx(1.0)
    # every point stays on the offset radius around the (horizontal) axis
    for point in drill.helix:
        assert _axis_distance(point, axis) == pytest.approx(1.0, abs=1e-6)


def test_tool_wider_than_hole_degenerates_to_plunge():
    axis = Line([0, 0, 10], [0, 0, 0])
    drill = toolpath_2d_drill(axis, hole_radius=1.0, tool_diameter=3.0)
    assert drill.is_drill
    assert len(drill.helix) == 2
    assert list(drill.helix[0]) == pytest.approx([0, 0, 10])
    assert list(drill.helix[-1]) == pytest.approx([0, 0, 0])


def test_bottom_pass_adds_a_closing_circle():
    axis = Line([0, 0, 10], [0, 0, 0])
    with_pass = toolpath_2d_drill(axis, hole_radius=5.0, tool_diameter=3.0, bottom_pass=True)
    without = toolpath_2d_drill(axis, hole_radius=5.0, tool_diameter=3.0, bottom_pass=False)
    assert len(with_pass.helix) > len(without.helix)
    # the extra points all sit at the bottom of the hole
    extra = list(with_pass.helix)[len(without.helix) :]
    assert all(p[2] == pytest.approx(0.0, abs=1e-6) for p in extra)


def test_invalid_ramp_angle_raises():
    axis = Line([0, 0, 10], [0, 0, 0])
    with pytest.raises(ValueError):
        toolpath_2d_drill(axis, hole_radius=5.0, tool_diameter=3.0, ramp_angle=0.0)
    with pytest.raises(ValueError):
        toolpath_2d_drill(axis, hole_radius=5.0, tool_diameter=3.0, ramp_angle=math.radians(90.0))


def test_floor_clamps_the_bottom():
    # Hole reaches z=-2.5; floor at -1 must raise the bottom to exactly -1.
    axis = Line([0, 0, 12.5], [0, 0, -2.5])
    drill = toolpath_2d_drill(axis, hole_radius=5.0, tool_diameter=3.0, floor=-1.0)
    assert drill.drill_axis.end[2] == pytest.approx(-1.0)
    assert min(p[2] for p in drill.path) >= -1.0 - 1e-9
    assert drill.drill_axis.start[2] == pytest.approx(12.5)  # top untouched by floor alone


def test_floor_above_hole_is_noop():
    axis = Line([0, 0, 12.5], [0, 0, 5.0])  # bottom already above the floor
    drill = toolpath_2d_drill(axis, hole_radius=5.0, tool_diameter=3.0, floor=-1.0)
    assert drill.drill_axis.end[2] == pytest.approx(5.0)


def test_safe_z_retract_sits_above_last_cut_point():
    axis = Line([0, 0, 10], [0, 0, 0])
    drill = toolpath_2d_drill(axis, hole_radius=5.0, tool_diameter=3.0, bottom_pass=False, safe_z=60.0)
    last_cut = drill.helix[-1]
    retract = drill.path[-2]  # path tail is [..., retract, home]
    assert drill.safe_z == pytest.approx(60.0)
    assert retract[0] == pytest.approx(last_cut[0])  # same X
    assert retract[1] == pytest.approx(last_cut[1])  # same Y
    assert retract[2] == pytest.approx(60.0)  # lifted straight up to safe_z


def test_safe_z_clamped_above_the_path():
    # safe_z below the top of the path is clamped up to clear it.
    axis = Line([0, 0, 30], [0, 0, 0])
    drill = toolpath_2d_drill(axis, hole_radius=5.0, tool_diameter=3.0, safe_z=5.0)
    assert drill.safe_z >= 30.0  # at least the top, plus clearance
    assert drill.path[-1][2] == pytest.approx(drill.safe_z)


def test_z_safety_is_always_present():
    axis = Line([0, 0, 10], [0, 0, 0])
    drill = toolpath_2d_drill(axis, hole_radius=5.0, tool_diameter=3.0)
    assert drill.safe_z == pytest.approx(10.0 + 10.0)  # top + MIN_CLEARANCE, set by default
    assert drill.path[0][2] == pytest.approx(drill.safe_z)  # approaches at safe_z
    assert drill.path[-1][2] == pytest.approx(drill.safe_z)  # retracts to safe_z
    assert list(drill.path[0]) == pytest.approx(list(drill.path[-1]))  # ends back home
