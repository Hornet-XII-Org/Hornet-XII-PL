"""Tests for the yaw aligner maths, using SYNTHETIC orientations only."""

import math

import pytest

from hornet_localization.yaw_aligner_node import YawAligner


def quaternion_from_rpy(roll, pitch, yaw):
    """Build an x, y, z, w quaternion from roll, pitch and yaw in radians."""
    sin_roll, cos_roll = math.sin(roll / 2), math.cos(roll / 2)
    sin_pitch, cos_pitch = math.sin(pitch / 2), math.cos(pitch / 2)
    sin_yaw, cos_yaw = math.sin(yaw / 2), math.cos(yaw / 2)
    return (
        sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw,
        cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw,
        cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw,
        cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw,
    )


def rpy_from_quaternion(quaternion):
    """Return roll, pitch and yaw in radians."""
    x, y, z, w = quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def angle_difference(first, second):
    """Return first - second, wrapped into the range -pi to pi."""
    return math.remainder(first - second, 2 * math.pi)


@pytest.mark.parametrize('start_yaw_degrees', [0, 45, 90, -90, 180])
def test_first_output_has_yaw_zero(start_yaw_degrees):
    first = quaternion_from_rpy(0.1, -0.2, math.radians(start_yaw_degrees))

    _, _, yaw = rpy_from_quaternion(YawAligner().align(first))

    assert angle_difference(yaw, 0.0) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    'start_degrees, later_degrees, expected_degrees',
    [
        (90, 203, 113),
        (-90, -135, -45),
        (170, -170, 20),
    ],
    ids=['turn_left', 'turn_right', 'across_plus_minus_180'],
)
def test_later_relative_yaw_is_preserved(
    start_degrees, later_degrees, expected_degrees
):
    aligner = YawAligner()
    aligner.align(quaternion_from_rpy(0.0, 0.0, math.radians(start_degrees)))

    later = aligner.align(
        quaternion_from_rpy(0.0, 0.0, math.radians(later_degrees)))
    _, _, yaw = rpy_from_quaternion(later)

    expected = math.radians(expected_degrees)
    assert angle_difference(yaw, expected) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    'bad_first',
    [
        (float('nan'), 0.0, 0.0, 1.0),
        (0.0, float('inf'), 0.0, 1.0),
        (0.0, 0.0, 0.0, 0.0),
    ],
    ids=['nan', 'inf', 'zero'],
)
def test_bad_first_message_does_not_set_the_starting_yaw(bad_first):
    aligner = YawAligner()

    assert aligner.align(bad_first) is bad_first

    # The first good message, at yaw 90, must become the reference.
    first_good = aligner.align(quaternion_from_rpy(0.0, 0.0, math.radians(90)))
    later = aligner.align(quaternion_from_rpy(0.0, 0.0, math.radians(120)))

    _, _, first_good_yaw = rpy_from_quaternion(first_good)
    _, _, later_yaw = rpy_from_quaternion(later)
    assert first_good_yaw == pytest.approx(0.0, abs=1e-9)
    assert later_yaw == pytest.approx(math.radians(30), abs=1e-9)


def test_roll_and_pitch_are_preserved():
    aligner = YawAligner()

    first = aligner.align(quaternion_from_rpy(0.3, -0.2, 1.2))
    later = aligner.align(quaternion_from_rpy(-0.4, 0.1, 2.0))

    first_roll, first_pitch, _ = rpy_from_quaternion(first)
    later_roll, later_pitch, _ = rpy_from_quaternion(later)
    assert first_roll == pytest.approx(0.3, abs=1e-9)
    assert first_pitch == pytest.approx(-0.2, abs=1e-9)
    assert later_roll == pytest.approx(-0.4, abs=1e-9)
    assert later_pitch == pytest.approx(0.1, abs=1e-9)
