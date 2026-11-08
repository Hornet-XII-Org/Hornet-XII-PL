"""ROS interface for the initial Hornet XII localization estimate."""

import math

from geometry_msgs.msg import TransformStamped
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, MagneticField
from tf2_ros import TransformBroadcaster

from hornet_localization.inertial_estimator import InertialEstimator


def normalized_quaternion_xyzw(quaternion):
    """Return a unit quaternion, or ``None`` when the input is invalid."""
    if len(quaternion) != 4 or not all(math.isfinite(v) for v in quaternion):
        return None

    norm = math.sqrt(sum(v * v for v in quaternion))
    if norm < 1.0e-12:
        return None

    return tuple(v / norm for v in quaternion)


def rotate_map_vector_into_base(vector_map, orientation_map_from_base):
    """Rotate a map-axis vector into ``base_link`` axes.

    ``orientation_map_from_base`` uses ROS's quaternion order: x, y, z, w.
    """
    x, y, z, w = orientation_map_from_base
    vx, vy, vz = vector_map

    # This is R(q).T @ vector_map. R(q) rotates base axes into map axes,
    # therefore its transpose (the inverse rotation) gives base axes.
    return (
        (1.0 - 2.0 * (y * y + z * z)) * vx
        + 2.0 * (x * y + z * w) * vy
        + 2.0 * (x * z - y * w) * vz,
        2.0 * (x * y - z * w) * vx
        + (1.0 - 2.0 * (x * x + z * z)) * vy
        + 2.0 * (y * z + x * w) * vz,
        2.0 * (x * z + y * w) * vx
        + 2.0 * (y * z - x * w) * vy
        + (1.0 - 2.0 * (x * x + y * y)) * vz,
    )


class ImuLocalizationNode(Node):
    """Connect standard sensor topics to odometry and TF outputs."""

    def __init__(self) -> None:
        super().__init__('imu_localization_node')

        self.declare_parameter('imu_topic', '/sensors/imu')
        self.declare_parameter(
            'magnetic_field_topic', '/sensors/magnetic_field')
        self.declare_parameter(
            'odometry_topic', '/localization/odometry')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        imu_topic = self.get_parameter('imu_topic').value
        magnetic_field_topic = self.get_parameter(
            'magnetic_field_topic').value
        odometry_topic = self.get_parameter('odometry_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.imu_subscription = self.create_subscription(
            Imu,
            imu_topic,
            self.imu_callback,
            qos_profile_sensor_data,
        )
        self.magnetic_field_subscription = self.create_subscription(
            MagneticField,
            magnetic_field_topic,
            self.magnetic_field_callback,
            qos_profile_sensor_data,
        )
        self.odometry_publisher = self.create_publisher(
            Odometry, odometry_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.estimator = InertialEstimator()
        self.latest_magnetic_field = None
        self.estimator_pending_warning_sent = False

        self.get_logger().info(
            'Localization interface ready; estimator integration is pending')

    def imu_callback(self, message: Imu) -> None:
        """Update the estimate and publish matching Odometry and TF data."""
        if message.orientation_covariance[0] < 0.0:
            self.get_logger().warning(
                'Ignoring IMU message without an orientation estimate',
                throttle_duration_sec=5.0,
            )
            return

        orientation = normalized_quaternion_xyzw((
            message.orientation.x,
            message.orientation.y,
            message.orientation.z,
            message.orientation.w,
        ))
        if orientation is None:
            self.get_logger().warning(
                'Ignoring IMU message with an invalid orientation quaternion',
                throttle_duration_sec=5.0,
            )
            return

        timestamp_sec = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1.0e-9
        )
        acceleration_body = (
            message.linear_acceleration.x,
            message.linear_acceleration.y,
            message.linear_acceleration.z,
        )

        try:
            position_map, velocity_map = self.estimator.update(
                timestamp_sec,
                acceleration_body,
                orientation,
            )
        except NotImplementedError:
            if not self.estimator_pending_warning_sent:
                self.get_logger().warning(
                    'Waiting for the inertial estimator implementation')
                self.estimator_pending_warning_sent = True
            return

        if not self._valid_vector(position_map) or not self._valid_vector(
                velocity_map):
            self.get_logger().warning(
                'Estimator returned a non-finite position or velocity',
                throttle_duration_sec=5.0,
            )
            return

        velocity_base = rotate_map_vector_into_base(velocity_map, orientation)
        self._publish_odometry_and_tf(
            message,
            position_map,
            velocity_base,
            orientation,
        )

    def magnetic_field_callback(self, message: MagneticField) -> None:
        """Keep the newest magnetic-field reading for filter integration."""
        self.latest_magnetic_field = message

    @staticmethod
    def _valid_vector(vector) -> bool:
        """Return whether a value is a finite three-component vector."""
        try:
            return len(vector) == 3 and all(math.isfinite(v) for v in vector)
        except (TypeError, ValueError):
            return False

    def _publish_odometry_and_tf(
        self,
        imu_message,
        position_map,
        velocity_base,
        orientation,
    ) -> None:
        """Publish one estimate through both standard ROS representations."""
        odometry = Odometry()
        odometry.header.stamp = imu_message.header.stamp
        odometry.header.frame_id = self.odom_frame
        odometry.child_frame_id = self.base_frame

        odometry.pose.pose.position.x = position_map[0]
        odometry.pose.pose.position.y = position_map[1]
        odometry.pose.pose.position.z = position_map[2]
        odometry.pose.pose.orientation.x = orientation[0]
        odometry.pose.pose.orientation.y = orientation[1]
        odometry.pose.pose.orientation.z = orientation[2]
        odometry.pose.pose.orientation.w = orientation[3]

        odometry.twist.twist.linear.x = velocity_base[0]
        odometry.twist.twist.linear.y = velocity_base[1]
        odometry.twist.twist.linear.z = velocity_base[2]

        # This initial interface assumes the IMU axes are aligned with
        # base_link. Replace this direct copy with a measured mounting
        # transform if the hardware uses a different orientation.
        odometry.twist.twist.angular = imu_message.angular_velocity
        self.odometry_publisher.publish(odometry)

        transform = TransformStamped()
        transform.header.stamp = imu_message.header.stamp
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = position_map[0]
        transform.transform.translation.y = position_map[1]
        transform.transform.translation.z = position_map[2]
        transform.transform.rotation.x = orientation[0]
        transform.transform.rotation.y = orientation[1]
        transform.transform.rotation.z = orientation[2]
        transform.transform.rotation.w = orientation[3]
        self.tf_broadcaster.sendTransform(transform)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImuLocalizationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
