import math

import pytest

from hornet_localization.imu_localization_node import (
    normalized_quaternion_xyzw,
    rotate_map_vector_into_base,
)


def test_normalizes_quaternion():
    assert normalized_quaternion_xyzw((0.0, 0.0, 0.0, 2.0)) == (
        0.0,
        0.0,
        0.0,
        1.0,
    )


def test_rejects_zero_quaternion():
    assert normalized_quaternion_xyzw((0.0, 0.0, 0.0, 0.0)) is None


def test_inverse_rotates_map_velocity_into_base_axes():
    half_sqrt_two = math.sqrt(0.5)
    orientation_map_from_base = (
        0.0,
        0.0,
        half_sqrt_two,
        half_sqrt_two,
    )

    velocity_base = rotate_map_vector_into_base(
        (0.0, 1.0, 0.0),
        orientation_map_from_base,
    )

    assert velocity_base == pytest.approx((1.0, 0.0, 0.0))
