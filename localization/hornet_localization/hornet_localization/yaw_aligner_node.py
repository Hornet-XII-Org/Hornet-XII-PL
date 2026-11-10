"""Remove the random starting yaw from the filtered IMU orientation.

Without a compass, Madgwick's starting yaw depends on tiny sensor tilt.  This
node saves the yaw of the first message and subtracts it from every message,
so the robot always starts facing yaw 0.  Roll and pitch do not change.
"""

import math
from typing import Optional, Tuple

from hornet_localization.inertial_estimator import _MIN_QUATERNION_LENGTH


Quaternion = Tuple[float, float, float, float]


def _is_usable(quaternion_xyzw: Quaternion) -> bool:
    """Return whether a quaternion is finite and not near zero."""
    return (
        all(math.isfinite(part) for part in quaternion_xyzw)
        and math.hypot(*quaternion_xyzw) >= _MIN_QUATERNION_LENGTH
    )


def yaw_from_quaternion(quaternion_xyzw: Quaternion) -> float:
    """Return the yaw, in radians, of a quaternion in x, y, z, w order."""
    x, y, z, w = quaternion_xyzw
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rotate_about_z(angle_rad: float, quaternion_xyzw: Quaternion) -> Quaternion:
    """Return ``q_z(angle_rad) * quaternion_xyzw`` (left multiply)."""
    x, y, z, w = quaternion_xyzw
    sin_half = math.sin(angle_rad / 2.0)
    cos_half = math.cos(angle_rad / 2.0)
    return (
        cos_half * x - sin_half * y,
        cos_half * y + sin_half * x,
        cos_half * z + sin_half * w,
        cos_half * w - sin_half * z,
    )


class YawAligner:
    """Make every orientation relative to the yaw of the first one."""

    def __init__(self) -> None:
        self._first_yaw_rad: Optional[float] = None

    def align(self, quaternion_xyzw: Quaternion) -> Quaternion:
        """Return the orientation with the starting yaw removed.

        The starting yaw is saved from the first finite, non-zero quaternion.
        Until one arrives, the input is returned unchanged.
        """
        if self._first_yaw_rad is None:
            if not _is_usable(quaternion_xyzw):
                return quaternion_xyzw
            self._first_yaw_rad = yaw_from_quaternion(quaternion_xyzw)
        return rotate_about_z(-self._first_yaw_rad, quaternion_xyzw)


def main(args=None) -> None:
    # ROS is imported here, not at the top of the file, so that pytest can
    # test the maths above on a computer without ROS.
    import rclpy
    from sensor_msgs.msg import Imu

    rclpy.init(args=args)
    node = rclpy.create_node('yaw_aligner_node')
    aligner = YawAligner()
    publisher = node.create_publisher(Imu, '/localization/imu/aligned', 10)

    def imu_callback(message: Imu) -> None:
        # Only the orientation changes.  The header, gyro and acceleration
        # stay as they arrived.
        orientation = message.orientation
        orientation.x, orientation.y, orientation.z, orientation.w = (
            aligner.align((
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ))
        )
        publisher.publish(message)

    node.create_subscription(
        Imu, '/localization/imu/filtered', imu_callback, 10)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
