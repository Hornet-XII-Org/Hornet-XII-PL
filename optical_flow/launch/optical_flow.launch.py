from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory('optical_flow'))
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=str(share / 'config/optical_flow.yaml')),
        DeclareLaunchArgument('calibration_file', default_value=str(share / 'config/bottom_camera.yaml')),
        DeclareLaunchArgument('input_reliable', default_value='false',
                              description='Default false uses sensor-data image QoS'),
        DeclareLaunchArgument('play_video', default_value='false',
                              description='Only enable for an already rectified video'),
        DeclareLaunchArgument('assume_fixed_orientation', default_value='false',
                              description='TEST ONLY: assume level body and no rotation'),
        DeclareLaunchArgument('video_path', default_value='',
                              description='Empty uses VIDEO_PATH in of_video_player.cpp'),
        DeclareLaunchArgument('loop', default_value='false'),
        DeclareLaunchArgument('image_width', default_value='640'),
        DeclareLaunchArgument('opencv_threads', default_value='0'),
        Node(package='optical_flow', executable='optical_flow_node', name='optical_flow_node', output='screen',
             parameters=[LaunchConfiguration('params_file'), {
                 'assume_fixed_orientation': ParameterValue(
                     LaunchConfiguration('assume_fixed_orientation'), value_type=bool),
                 'calibration_file': ParameterValue(LaunchConfiguration('calibration_file'), value_type=str),
                 'input_reliable': ParameterValue(LaunchConfiguration('input_reliable'), value_type=bool),
                 'opencv_threads': ParameterValue(LaunchConfiguration('opencv_threads'), value_type=int),
             }]),
        Node(package='optical_flow', executable='of_video_player', name='of_video_player', output='screen',
             condition=IfCondition(LaunchConfiguration('play_video')),
             remappings=[('video_input', '/bottom/image_rect')],
             parameters=[{
                 'video_path': ParameterValue(LaunchConfiguration('video_path'), value_type=str),
                 'loop': ParameterValue(LaunchConfiguration('loop'), value_type=bool),
                 'opencv_threads': ParameterValue(LaunchConfiguration('opencv_threads'), value_type=int),
                 'image_width': ParameterValue(LaunchConfiguration('image_width'), value_type=int),
             }]),
    ])
