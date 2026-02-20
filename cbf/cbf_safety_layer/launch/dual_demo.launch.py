import os
import re
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction


def generate_launch_description():
    urdf_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots', 'fer', 'fer.urdf'
    )

    with open(urdf_file, 'r') as infp:
        robot_descrip = infp.read()

    def apply_namespace(desc, ns_prefix):
        # rename all fer_ references
        pattern = r'=\s*(["\'])fer_'
        result = re.sub(pattern, f'=\\1{ns_prefix}fer_', desc)
        # remove the base_joint that connects base->link0 at origin
        result = re.sub(
            r'\s*<joint\s+name=["\'][^"\']*base_joint["\'][^>]*>.*?</joint>',
            '',
            result,
            flags=re.DOTALL
        )
        # remove the <link name="base"/> stub that base_joint was attached
        result = re.sub(
            r'\s*<link\s+name=["\']base["\'][^/]*/?>(\s*</link>)?',
            '',
            result,
            flags=re.DOTALL
        )
        return result

    # Apply the replacements
    robot1_r_descrip = apply_namespace(robot_descrip, "robot1_")
    robot2_r_descrip = apply_namespace(robot_descrip, "robot2_")

    ready_pose_map = {
        'fer_joint1': 0.0,
        'fer_joint2': -0.785,
        'fer_joint3': 0.0,
        'fer_joint4': -2.356,
        'fer_joint5': 0.0,
        'fer_joint6': 1.571,
        'fer_joint7': 0.785,
        'fer_finger_joint1': 0.0,
        'fer_finger_joint2': 0.0,
    }
    
    def get_ready_pose(prefix):
        return {f'{prefix}{k}': v for k, v in ready_pose_map.items()}

    # Robot1
    robot1_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='robot1',
        output='screen',
        parameters=[{
            'robot_description': robot1_r_descrip
        }]
    )

    robot1_joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        namespace='robot1',
        parameters=[{
            'robot_description': robot1_r_descrip,
            'zeros': get_ready_pose('robot1_')
        }],
        remappings=[('joint_states', 'desired_joint_states')]
    )

    robot1_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='robot1_base_broadcaster',
        output='screen',
        arguments=[
            '--x', '0', '--y', '1.0', '--z', '0',
            '--yaw', '0', '--pitch', '0', '--roll', '0',
            '--frame-id', 'base',
            '--child-frame-id', 'robot1_fer_link0'
        ]
    )

    robot1_safety_node = Node(
        package='cbf_safety_layer_cpp',
        executable='safety_node_cpp',
        namespace='robot1',
        output='screen',
        parameters=[{
            'self_robot_description': robot1_r_descrip,
            'other_robot_description': robot2_r_descrip,
            'self_frame_prefix': 'robot1_fer_',
            'other_frame_prefix': 'robot2_fer_',
            'reference_frame': 'base',
            'base_offset_x': 0.0,
            'base_offset_y': 1.0,
            'base_offset_z': 0.0,
            'use_fallback_urdf': False 
        }],
        remappings=[
            ('/joint_states_source', 'desired_joint_states'),
            ('/joint_states_source_other', '/robot2/joint_states'),
            ('/safety_marker', '/robot1/safety_marker'),
            ('/safety/joint_states', 'joint_states')
        ]
    )

    # Robot2
    robot2_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='robot2',
        output='screen',
        parameters=[{
            'robot_description': robot2_r_descrip
        }]
    )

    robot2_joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        namespace='robot2',
        parameters=[{
            'robot_description': robot2_r_descrip,
            'zeros': get_ready_pose('robot2_')
        }],
        remappings=[('joint_states', 'desired_joint_states')]
    )

    robot2_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='robot2_base_broadcaster',
        output='screen',
        arguments=[
            '--x', '0', '--y', '-1.0', '--z', '0',
            '--yaw', '0', '--pitch', '0', '--roll', '0',
            '--frame-id', 'base',
            '--child-frame-id', 'robot2_fer_link0'
        ]
    )

    robot2_safety_node = Node(
        package='cbf_safety_layer_cpp',
        executable='safety_node_cpp',
        namespace='robot2',
        output='screen',
        parameters=[{
            'self_robot_description': robot2_r_descrip,
            'other_robot_description': robot1_r_descrip,
            'self_frame_prefix': 'robot2_fer_',
            'other_frame_prefix': 'robot1_fer_',
            'reference_frame': 'base',
            'base_offset_x': 0.0,
            'base_offset_y': -1.0,
            'base_offset_z': 0.0,
            'use_fallback_urdf': False
        }],
        remappings=[
            ('/joint_states_source', 'desired_joint_states'),
            ('/joint_states_source_other', '/robot1/joint_states'),
            ('/safety_marker', '/robot2/safety_marker'),
            ('/safety/joint_states', 'joint_states')
        ]
    )

    world_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_base_broadcaster',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--yaw', '0', '--pitch', '0', '--roll', '0',
            '--frame-id', 'world', '--child-frame-id', 'base'
        ]
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    return LaunchDescription([
        world_to_base,
        robot1_tf,
        robot2_tf,

        TimerAction(
            period=1.0,
            actions=[
                robot1_robot_state_publisher,
                robot1_joint_state_publisher,
                robot1_safety_node,
                robot2_robot_state_publisher,
                robot2_joint_state_publisher,
                robot2_safety_node,
                ]
            ),
        TimerAction(period=1.5, actions=[rviz_node]),
        ])