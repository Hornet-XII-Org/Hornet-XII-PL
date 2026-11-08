"""Interface between the ROS node and the inertial estimator.

Dao Quang Linh owns the estimator implementation.  Keep the public update
signature stable so the ROS integration can be developed in parallel.
"""

import math
from typing import Optional, Tuple


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]

# A quaternion shorter than this has no usable direction.
_MIN_QUATERNION_LENGTH = 1e-6


def _rotate_vector(quaternion_xyzw: Quaternion, vector: Vector3) -> Vector3:
    """Rotate ``vector`` by a unit quaternion in x, y, z, w order."""
    x, y, z, w = quaternion_xyzw
    vx, vy, vz = vector

    # v' = v + w * t + (q_xyz cross t), where t = 2 * (q_xyz cross v).
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


class InertialEstimator:
    """Estimate map-frame position and velocity from timestamped IMU data."""

    def __init__(
        self,
        gravity_mps2: float = 9.81,
        max_dt_sec: float = 0.5,
    ) -> None:
        self._gravity_mps2 = gravity_mps2
        self._max_dt_sec = max_dt_sec
        self.reset()

    def reset(self) -> None:
        """Reset the estimator to the agreed startup origin."""
        self._position_map: Vector3 = (0.0, 0.0, 0.0)
        self._velocity_map: Vector3 = (0.0, 0.0, 0.0)
        self._last_timestamp_sec: Optional[float] = None
        self._previous_acceleration_map: Optional[Vector3] = None

    def update(
        self,
        timestamp_sec: float,
        acceleration_body_mps2: Vector3,
        orientation_map_from_base_xyzw: Quaternion,
    ) -> Optional[Tuple[Vector3, Vector3]]:
        """Return ``(position_map_m, velocity_map_mps)``, or ``None``.

        ``acceleration_body_mps2`` is the raw accelerometer reading in
        ``base_link`` axes.  It includes gravity: a level, still robot reads
        ``(0, 0, +g)``.  The supplied quaternion describes the robot
        orientation relative to the map frame.  The returned velocity remains
        in map axes; the ROS node must rotate it into ``base_link`` before
        filling Odometry.twist.

        The first sample after construction or ``reset()`` returns zero
        position and zero velocity.

        Returns ``None`` when no valid estimate can be produced:

        - an input is NaN or infinite, or the quaternion length is near zero
          (state is not changed);
        - the timestamp is repeated or goes backward (state is not changed);
        - the time since the last sample is longer than ``max_dt_sec``.  The
          estimator restarts its timing from this sample and does not
          integrate across the gap.  Position and velocity keep their old
          values.
        """
        values = (
            timestamp_sec,
            *acceleration_body_mps2,
            *orientation_map_from_base_xyzw,
        )
        if not all(math.isfinite(value) for value in values):
            return None

        quaternion_length = math.sqrt(
            sum(part * part for part in orientation_map_from_base_xyzw))
        if quaternion_length < _MIN_QUATERNION_LENGTH:
            return None
        orientation = tuple(
            part / quaternion_length
            for part in orientation_map_from_base_xyzw
        )

        acceleration_map = self._acceleration_in_map(
            acceleration_body_mps2, orientation)

        if self._last_timestamp_sec is None:
            self._last_timestamp_sec = timestamp_sec
            self._previous_acceleration_map = acceleration_map
            return self._position_map, self._velocity_map

        dt = timestamp_sec - self._last_timestamp_sec

        if dt <= 0.0:
            return None

        if dt > self._max_dt_sec:
            self._last_timestamp_sec = timestamp_sec
            self._previous_acceleration_map = acceleration_map
            return None

        # Trapezoid rule: use the average of the old and new values.
        new_velocity = tuple(
            velocity + 0.5 * (previous + current) * dt
            for velocity, previous, current in zip(
                self._velocity_map,
                self._previous_acceleration_map,
                acceleration_map,
            )
        )
        new_position = tuple(
            position + 0.5 * (old_velocity + velocity) * dt
            for position, old_velocity, velocity in zip(
                self._position_map, self._velocity_map, new_velocity)
        )

        self._position_map = new_position
        self._velocity_map = new_velocity
        self._last_timestamp_sec = timestamp_sec
        self._previous_acceleration_map = acceleration_map

        return self._position_map, self._velocity_map

    def _acceleration_in_map(
        self,
        acceleration_body_mps2: Vector3,
        orientation_map_from_base_xyzw: Quaternion,
    ) -> Vector3:
        """Rotate a body reading into map axes and remove gravity."""
        x, y, z = _rotate_vector(
            orientation_map_from_base_xyzw, acceleration_body_mps2)
        return (x, y, z - self._gravity_mps2)
