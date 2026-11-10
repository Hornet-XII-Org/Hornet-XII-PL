"""Launch the IMU localization chain for Hornet XII."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Start the filter, yaw aligner, odometry node and map-to-odom TF.

    With ``use_sim:=true``, also start the simulator IMU adapter and use the
    simulator clock.
    """
    use_sim = LaunchConfiguration('use_sim')
    sim_time = {'use_sim_time': ParameterValue(use_sim, value_type=bool)}
    madgwick_config = PathJoinSubstitution([
        FindPackageShare('hornet_localization'),
        'config',
        'madgwick.yaml',
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim',
            default_value='false',
            description='Use the simulator IMU adapter and simulator clock.',
        ),
        Node(
            package='hornet_localization',
            executable='sim_imu_adapter_node',
            name='sim_imu_adapter_node',
            output='screen',
            parameters=[sim_time],
            condition=IfCondition(use_sim),
        ),
        Node(
            package='imu_filter_madgwick',
            executable='imu_filter_madgwick_node',
            # This name must match the top-level key in madgwick.yaml.
            name='imu_filter_madgwick',
            output='screen',
            parameters=[madgwick_config, sim_time],
            remappings=[
                ('imu/data_raw', '/sensors/imu'),
                ('imu/data', '/localization/imu/filtered'),
            ],
        ),
        Node(
            package='hornet_localization',
            executable='yaw_aligner_node',
            name='yaw_aligner_node',
            output='screen',
            parameters=[sim_time],
        ),
        Node(
            package='hornet_localization',
            executable='imu_localization_node',
            name='imu_localization_node',
            output='screen',
            parameters=[
                {'imu_topic': '/localization/imu/aligned'},
                sim_time,
            ],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_static_transform',
            output='screen',
            arguments=['--frame-id', 'map', '--child-frame-id', 'odom'],
            parameters=[sim_time],
        ),
    ])
