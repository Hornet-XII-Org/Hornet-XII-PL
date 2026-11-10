"""Convert the simulator IMU into the /sensors/imu interface.

SIMULATOR ONLY.  The simulated IMU is mounted rolled upside down
(``<pose>0 0 0 3.142 0 0</pose>`` in the robot's model.sdf), so a level robot
reads about (0, 0, -9.81).  Rolling a vector 180 degrees about X gives
(x, -y, -z), which puts the reading into base_link axes.
"""

import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


def flip_to_base_link(vector: Vector3) -> None:
    """Apply (x, -y, -z) to ``vector`` in place."""
    vector.y = -vector.y
    vector.z = -vector.z


class SimImuAdapterNode(Node):
    """Republish the simulator IMU in base_link axes."""

    def __init__(self) -> None:
        super().__init__('sim_imu_adapter_node')

        self.publisher = self.create_publisher(Imu, '/sensors/imu', 10)
        # The simulator's QoS is not known yet.  A best-effort subscriber
        # can receive from both reliable and best-effort publishers.
        self.subscription = self.create_subscription(
            Imu,
            '/model/bluerov2/imu',
            self.imu_callback,
            qos_profile_sensor_data,
        )

    def imu_callback(self, message: Imu) -> None:
        """Fix the axes and mark the orientation as unavailable."""
        flip_to_base_link(message.linear_acceleration)
        flip_to_base_link(message.angular_velocity)
        message.header.frame_id = 'base_link'
        message.orientation_covariance[0] = -1.0
        self.publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimImuAdapterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
