"""Tests for the inertial estimator, using SYNTHETIC IMU data only."""

import math

import pytest

from hornet_localization.inertial_estimator import InertialEstimator


GRAVITY = 9.81
SAMPLE_RATE_HZ = 100

NAN = float('nan')
INF = float('inf')

LEVEL = (0.0, 0.0, 0.0, 1.0)
YAW_90_DEGREES = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
ZERO_QUATERNION = (0.0, 0.0, 0.0, 0.0)

STILL_AND_LEVEL = (0.0, 0.0, GRAVITY)
FORWARD_1_MPS2 = (1.0, 0.0, GRAVITY)
FORWARD_5_MPS2 = (5.0, 0.0, GRAVITY)

ZERO = (0.0, 0.0, 0.0)


def assert_near(actual, expected):
    assert actual == pytest.approx(expected, abs=1e-9)


def run_constant(acceleration, orientation, duration_sec):
    """Feed one constant reading at 100 Hz and return the last result."""
    estimator = InertialEstimator()
    result = None
    for step in range(round(duration_sec * SAMPLE_RATE_HZ) + 1):
        result = estimator.update(
            step / SAMPLE_RATE_HZ, acceleration, orientation)
    return result


def moving_estimator():
    """Return an estimator that has already moved for 0.1 s."""
    estimator = InertialEstimator()
    estimator.update(0.0, FORWARD_1_MPS2, LEVEL)
    estimator.update(0.1, FORWARD_1_MPS2, LEVEL)
    return estimator


def assert_state_unchanged(estimator):
    """Check the estimator acts as if the rejected sample never arrived.

    The next good sample must give exactly the same answer as an estimator
    that never saw the rejected sample.
    """
    untouched = moving_estimator()
    next_sample = (0.2, FORWARD_1_MPS2, LEVEL)
    assert estimator.update(*next_sample) == untouched.update(*next_sample)


def test_still_and_level_for_10_seconds_stays_at_zero():
    position, velocity = run_constant(STILL_AND_LEVEL, LEVEL, 10.0)

    assert_near(position, ZERO)
    assert_near(velocity, ZERO)


def test_still_with_30_degree_roll_stays_at_zero():
    # Roll 30 degrees about X.  The map "up" axis, seen from the tilted
    # robot, is (0, sin 30, cos 30), so gravity's push spreads over Y and Z.
    half_roll = math.radians(30.0) / 2.0
    rolled = (math.sin(half_roll), 0.0, 0.0, math.cos(half_roll))
    tilted_reading = (
        0.0,
        GRAVITY * math.sin(math.radians(30.0)),
        GRAVITY * math.cos(math.radians(30.0)),
    )

    position, velocity = run_constant(tilted_reading, rolled, 10.0)

    assert_near(position, ZERO)
    assert_near(velocity, ZERO)


def test_constant_forward_acceleration_for_2_seconds():
    position, velocity = run_constant(FORWARD_1_MPS2, LEVEL, 2.0)

    assert_near(velocity, (2.0, 0.0, 0.0))
    assert_near(position, (2.0, 0.0, 0.0))


def test_forward_with_yaw_90_degrees_moves_along_map_y():
    position, velocity = run_constant(FORWARD_1_MPS2, YAW_90_DEGREES, 2.0)

    assert_near(velocity, (0.0, 2.0, 0.0))
    assert_near(position, (0.0, 2.0, 0.0))


def test_first_sample_returns_zeros():
    estimator = InertialEstimator()

    assert estimator.update(123.4, FORWARD_5_MPS2, LEVEL) == (ZERO, ZERO)


@pytest.mark.parametrize(
    'timestamp',
    [0.1, 0.05],
    ids=['repeated', 'backward'],
)
def test_bad_timestamp_returns_none_and_keeps_state(timestamp):
    estimator = moving_estimator()

    assert estimator.update(timestamp, FORWARD_5_MPS2, LEVEL) is None
    assert_state_unchanged(estimator)


@pytest.mark.parametrize(
    'timestamp, acceleration, orientation',
    [
        (NAN, FORWARD_5_MPS2, LEVEL),
        (0.15, (NAN, 0.0, GRAVITY), LEVEL),
        (0.15, FORWARD_5_MPS2, (NAN, 0.0, 0.0, 1.0)),
        (INF, FORWARD_5_MPS2, LEVEL),
        (0.15, (0.0, INF, GRAVITY), LEVEL),
        (0.15, FORWARD_5_MPS2, (0.0, 0.0, INF, 1.0)),
        (0.15, FORWARD_5_MPS2, ZERO_QUATERNION),
    ],
    ids=[
        'nan_timestamp',
        'nan_acceleration',
        'nan_quaternion',
        'inf_timestamp',
        'inf_acceleration',
        'inf_quaternion',
        'zero_quaternion',
    ],
)
def test_bad_input_returns_none_and_keeps_state(
    timestamp, acceleration, orientation
):
    estimator = moving_estimator()

    assert estimator.update(timestamp, acceleration, orientation) is None
    assert_state_unchanged(estimator)


def test_gap_returns_none_and_does_not_integrate_across_it():
    # Before the gap: velocity 0.1 m/s, position 0.005 m.
    estimator = moving_estimator()

    assert estimator.update(10.0, (3.0, 0.0, GRAVITY), LEVEL) is None
    position, velocity = estimator.update(10.1, FORWARD_1_MPS2, LEVEL)

    # Only the 0.1 s after the gap counts, with average acceleration
    # (3 + 1) / 2 = 2.  The 9.9 s gap adds nothing.
    assert_near(velocity, (0.3, 0.0, 0.0))
    assert_near(position, (0.025, 0.0, 0.0))


def test_reset_returns_to_start_state():
    estimator = moving_estimator()

    estimator.reset()

    # The time 0.0 is earlier than the last sample, so this would be
    # rejected if reset() had not cleared the old time.
    assert estimator.update(0.0, FORWARD_1_MPS2, LEVEL) == (ZERO, ZERO)

    fresh = InertialEstimator()
    fresh.update(0.0, FORWARD_1_MPS2, LEVEL)
    next_sample = (0.1, FORWARD_1_MPS2, LEVEL)
    assert estimator.update(*next_sample) == fresh.update(*next_sample)
