"""ROS interface for the initial Hornet XII localization estimate."""

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField
from tf2_ros import TransformBroadcaster

from hornet_localization.inertial_estimator import InertialEstimator


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

        self.imu_subscription = self.create_subscription(
            Imu, imu_topic, self.imu_callback, 10)
        self.magnetic_field_subscription = self.create_subscription(
            MagneticField,
            magnetic_field_topic,
            self.magnetic_field_callback,
            10,
        )
        self.odometry_publisher = self.create_publisher(
            Odometry, odometry_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.estimator = InertialEstimator()
        self.latest_magnetic_field = None

        self.get_logger().info(
            'Localization interface ready; estimator integration is pending')

    def imu_callback(self, message: Imu) -> None:
        """Receive IMU data; estimator/output wiring is implemented next."""
        del message

    def magnetic_field_callback(self, message: MagneticField) -> None:
        """Keep the newest magnetic-field reading for filter integration."""
        self.latest_magnetic_field = message


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
