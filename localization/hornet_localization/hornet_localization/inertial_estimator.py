"""Interface between the ROS node and the inertial estimator.

Dao Quang Linh owns the estimator implementation.  Keep the public update
signature stable so the ROS integration can be developed in parallel.
"""

from typing import Tuple


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]


class InertialEstimator:
    """Estimate map-frame position and velocity from timestamped IMU data."""

    def reset(self) -> None:
        """Reset the estimator to the agreed startup origin."""

    def update(
        self,
        timestamp_sec: float,
        acceleration_body_mps2: Vector3,
        orientation_map_from_base_xyzw: Quaternion,
    ) -> Tuple[Vector3, Vector3]:
        """Return ``(position_map_m, velocity_map_mps)``.

        ``acceleration_body_mps2`` is measured in the sensor/robot frame.
        The supplied quaternion describes the robot orientation relative to
        the map frame.  The returned velocity remains in map axes; the ROS
        node must rotate it into ``base_link`` before filling Odometry.twist.
        """
        raise NotImplementedError('Estimator implementation belongs here')
