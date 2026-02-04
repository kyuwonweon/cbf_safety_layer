from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    franka_desc_pkg = get_package_share_directory('franka_description')
    urdf_path = os.path.join(franka_desc_pkg, 'robots', 'fer', 'fer.urdf')
    
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

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
            parameters=[{'robot_description': robot_desc}]
        ),
        Node(
            package='cbf_safety_layer_cpp',
            executable='test.py',
            name='test_mover',
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            output='screen'
        ),
    ])
