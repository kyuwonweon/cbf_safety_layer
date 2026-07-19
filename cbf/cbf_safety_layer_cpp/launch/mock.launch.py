from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    franka_desc_pkg = get_package_share_directory('franka_description')
    urdf_path = os.path.join(franka_desc_pkg, 'robots', 'fer', 'fer.urdf')

    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    rviz_config = os.path.join(
        get_package_share_directory('cbf_safety_layer'), 'config', 'visualize_franka.rviz')

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}],
            remappings=[('/joint_states', '/safety/joint_states')]
        ),
        Node(
            package='cbf_safety_layer_cpp',
            executable='safety_node_cpp',
            output='screen',
            parameters=[{'self_robot_description': robot_desc, 'hardware_mode': False}]
        ),
        Node(
            package='cbf_safety_layer_cpp',
            executable='test.py',
            name='test_mover',
            output='screen',
            remappings=[('/joint_states', '/joint_states_source')]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['-d', rviz_config]
        ),
    ])
