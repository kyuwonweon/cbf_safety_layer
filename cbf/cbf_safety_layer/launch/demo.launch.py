import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    """Generate ROS 2 launch description."""
    franka_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('franka_bringup'),
                         'launch',
                         'franka.launch.py')
        ),
        launch_arguments={
            'robot_ip': 'dont_care',
            'use_fake_hardware': 'true',
            'arm_id': 'fer',
            'use_rviz': 'true'
        }.items(),
    )

    spawn_velocity = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["velocity_group_controller", "--controller-manager", "/controller_manager", "--param-file", 
                   os.path.join(get_package_share_directory('cbf_safety_layer'), 'config', 'controllers.yaml')],
    )

    return LaunchDescription([
        franka_bringup,
        spawn_velocity,
    ])