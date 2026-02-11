import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    urdf_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots', 'fer', 'fer.urdf'
    )

    with open(urdf_file, 'r') as infp:
        robot_descrip = infp.read()

    left_r_descrip = robot_descrip.replace('"fer_', '"left_fer_')
    right_r_descrip = robot_descrip.replace('"fer_', '"right_fer_')

    # Robot 1
    # Position: x=0, y=0.5, z=0
    left_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='left',
        output='screen',
        parameters=[{
            'robot_description': left_r_descrip
        }]
    )

    left_joint_pub = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        namespace='left',
        output='screen'
    )

    left_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '1.0', '0', '0', '0', '0',
                   'base', 'left_fer_link0']
    )

    # Robot 2
    # Position: x=0, y=-0.5, z=0
    right_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='right',
        output='screen',
        parameters=[{
            'robot_description': right_r_descrip
        }]
    )

    right_joint_pub = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        namespace='right',
        output='screen'
    )

    right_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '-1.0', '0', '0', '0', '0',
                   'base', 'right_fer_link0']
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    return LaunchDescription([
        left_robot_state_publisher,
        left_joint_pub,
        left_tf,
        right_robot_state_publisher,
        right_joint_pub,
        right_tf,
        rviz_node
    ])
