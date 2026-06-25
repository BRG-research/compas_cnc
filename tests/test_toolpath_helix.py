"""Tests for the helical drilling tool-path."""

import pytest
from compas.geometry import Line

from compas_cnc.helix_drill import toolpath_helix_drill


def _axis_distance(point, line):
    """Perpendicular distance from ``point`` to the infinite line of ``line``."""
    start = line.start
    direction = line.end - line.start
    direction.unitize()
    vec = point - start
    along = vec.dot(direction)
    foot = start + direction * along
    return (point - foot).length


def test_helix_radius_and_turns():
    axis = Line([0, 0, 10], [0, 0, 0])  # 10 deep, downward
    drill = toolpath_helix_drill(axis, hole_radius=5.0, tool_diameter=3.0, pitch=2.0)
    assert not drill.is_drill
    assert drill.helix_radius == pytest.approx(5.0 - 1.5)
    assert drill.turns == pytest.approx(10.0 / 2.0)


def test_helix_points_ride_the_offset_radius():
    axis = Line([0, 0, 10], [0, 0, 0])
    drill = toolpath_helix_drill(axis, hole_radius=5.0, tool_diameter=3.0, bottom_pass=False)
    for point in drill.path:
        assert _axis_distance(point, axis) == pytest.approx(drill.helix_radius, abs=1e-6)


def test_helix_descends_monotonically():
    axis = Line([0, 0, 10], [0, 0, 0])
    drill = toolpath_helix_drill(axis, hole_radius=5.0, tool_diameter=3.0, bottom_pass=False)
    zs = [p[2] for p in drill.path]
    assert zs[0] == pytest.approx(10.0)
    assert zs[-1] == pytest.approx(0.0)
    assert all(b <= a + 1e-9 for a, b in zip(zs, zs[1:]))  # non-increasing


def test_helix_on_horizontal_axis():
    # Mirrors the column bolt-holes: axis along -Y, away from world Z.
    axis = Line([0, 0, 0], [0, -15, 0])
    drill = toolpath_helix_drill(axis, hole_radius=2.5, tool_diameter=3.0)
    assert not drill.is_drill
    assert drill.helix_radius == pytest.approx(1.0)
    # every point stays on the offset radius around the (horizontal) axis
    for point in drill.path:
        assert _axis_distance(point, axis) == pytest.approx(1.0, abs=1e-6)


def test_tool_wider_than_hole_degenerates_to_plunge():
    axis = Line([0, 0, 10], [0, 0, 0])
    drill = toolpath_helix_drill(axis, hole_radius=1.0, tool_diameter=3.0)
    assert drill.is_drill
    assert len(drill.path) == 2
    assert list(drill.path[0]) == pytest.approx([0, 0, 10])
    assert list(drill.path[-1]) == pytest.approx([0, 0, 0])


def test_bottom_pass_adds_a_closing_circle():
    axis = Line([0, 0, 10], [0, 0, 0])
    with_pass = toolpath_helix_drill(axis, hole_radius=5.0, tool_diameter=3.0, bottom_pass=True)
    without = toolpath_helix_drill(axis, hole_radius=5.0, tool_diameter=3.0, bottom_pass=False)
    assert len(with_pass.path) > len(without.path)
    # the extra points all sit at the bottom of the hole
    extra = list(with_pass.path)[len(without.path):]
    assert all(p[2] == pytest.approx(0.0, abs=1e-6) for p in extra)


def test_non_positive_pitch_raises():
    axis = Line([0, 0, 10], [0, 0, 0])
    with pytest.raises(ValueError):
        toolpath_helix_drill(axis, hole_radius=5.0, tool_diameter=3.0, pitch=0.0)


def test_length_override_extends_the_top():
    # 15-deep hole, override to 30: bottom stays at z=0, top extends up to z=30.
    axis = Line([0, 0, 15], [0, 0, 0])
    drill = toolpath_helix_drill(axis, hole_radius=5.0, tool_diameter=3.0, length=30.0)
    assert drill.depth == pytest.approx(30.0)
    assert drill.drill_axis.start[2] == pytest.approx(30.0)  # top extended
    assert drill.drill_axis.end[2] == pytest.approx(0.0)  # bottom anchored


def test_length_override_orientation_independent():
    # Input the axis "upside down" (start below end): top is still the +Z end.
    axis = Line([0, 0, 0], [0, 0, 15])
    drill = toolpath_helix_drill(axis, hole_radius=5.0, tool_diameter=3.0, length=30.0)
    assert drill.drill_axis.start[2] == pytest.approx(30.0)
    assert drill.drill_axis.end[2] == pytest.approx(0.0)


def test_floor_clamps_the_bottom():
    # Hole reaches z=-2.5; floor at -1 must raise the bottom to exactly -1.
    axis = Line([0, 0, 12.5], [0, 0, -2.5])
    drill = toolpath_helix_drill(axis, hole_radius=5.0, tool_diameter=3.0, floor=-1.0)
    assert drill.drill_axis.end[2] == pytest.approx(-1.0)
    assert min(p[2] for p in drill.path) >= -1.0 - 1e-9
    assert drill.drill_axis.start[2] == pytest.approx(12.5)  # top untouched by floor alone


def test_length_and_floor_together():
    # Bottom clamped to -1, then top extended so the total length is 30 -> top at 29.
    axis = Line([0, 0, 12.5], [0, 0, -2.5])
    drill = toolpath_helix_drill(axis, hole_radius=5.0, tool_diameter=3.0, length=30.0, floor=-1.0)
    assert drill.drill_axis.end[2] == pytest.approx(-1.0)
    assert drill.drill_axis.start[2] == pytest.approx(29.0)
    assert drill.depth == pytest.approx(30.0)
    assert min(p[2] for p in drill.path) >= -1.0 - 1e-9


def test_floor_above_hole_is_noop():
    axis = Line([0, 0, 12.5], [0, 0, 5.0])  # bottom already above the floor
    drill = toolpath_helix_drill(axis, hole_radius=5.0, tool_diameter=3.0, floor=-1.0)
    assert drill.drill_axis.end[2] == pytest.approx(5.0)


def test_safe_z_appends_retract_at_last_point_xy():
    axis = Line([0, 0, 10], [0, 0, 0])
    drill_plain = toolpath_helix_drill(axis, hole_radius=5.0, tool_diameter=3.0, bottom_pass=False)
    drill_safe = toolpath_helix_drill(axis, hole_radius=5.0, tool_diameter=3.0, bottom_pass=False, safe_z=60.0)
    # one extra point versus the no-retract path
    assert len(drill_safe.path) == len(drill_plain.path) + 1
    last_cut = drill_plain.path[-1]
    retract = drill_safe.path[-1]
    assert drill_safe.safe_z == pytest.approx(60.0)
    assert retract[0] == pytest.approx(last_cut[0])  # same X
    assert retract[1] == pytest.approx(last_cut[1])  # same Y
    assert retract[2] == pytest.approx(60.0)  # lifted to safe_z


def test_safe_z_clamped_above_the_path():
    # safe_z below the top of the path is clamped up to clear it.
    axis = Line([0, 0, 30], [0, 0, 0])
    drill = toolpath_helix_drill(axis, hole_radius=5.0, tool_diameter=3.0, safe_z=5.0)
    assert drill.safe_z >= 30.0  # at least the top, plus clearance
    assert drill.path[-1][2] == pytest.approx(drill.safe_z)


def test_no_safe_z_means_no_retract():
    axis = Line([0, 0, 10], [0, 0, 0])
    drill = toolpath_helix_drill(axis, hole_radius=5.0, tool_diameter=3.0)
    assert drill.safe_z is None
