"""Initial launch scaffold for the localization component."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Start the ROS interface node.

    The Madgwick filter and temporary map-to-odom transform will be added as
    the estimator configuration is integrated.
    """
    return LaunchDescription([
        Node(
            package='hornet_localization',
            executable='imu_localization_node',
            name='imu_localization_node',
            output='screen',
        ),
    ])
