import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction, RegisterEventHandler # Added RegisterEventHandler
from launch.event_handlers import OnProcessExit # Added OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetRemap

def generate_launch_description():
    franka_sim = GroupAction([
        SetRemap(src='/joint_states', dst='/joint_states_source'),
        SetRemap(src='/tf', dst='/tf_garbage'),
        SetRemap(src='/tf_static', dst='/tf_static_garbage'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('franka_bringup'),
                             'launch',
                             'franka.launch.py')
            ),
            launch_arguments={
                'robot_ip': '192.168.51.20',
                'use_fake_hardware': 'false',
                'robot_type': 'fer',
                'arm_id': 'fer',
                'load_gripper': 'false',
                'use_rviz': 'false',
            }.items(),
        )
    ])

    spawn_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=['franka_robot_state_broadcaster', '--controller-manager', '/controller_manager'],
    )

    spawn_velocity = Node(
        package="controller_manager",
        executable="spawner",
        arguments=['velocity_group_controller', '--controller-manager',
                   '/controller_manager', '--param-file',
                   os.path.join(get_package_share_directory('cbf_safety_layer'),'config', 'controllers.yaml')],
    )

    # Fixed variable name here: target_action=spawn_broadcaster, on_exit=[spawn_velocity]
    delayed_velocity_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_broadcaster,
            on_exit=[spawn_velocity],
        )
    )

    safety_node = Node(
        package='cbf_safety_layer_cpp',
        executable='safety_node_one',
        output='screen',
        remappings=[
            ('/safety/joint_states', '/joint_states'),
        ]
    )

    urdf_file = os.path.join(get_package_share_directory('franka_description'), 'robots', 'fer', 'fer.urdf')
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='safety_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}],
        remappings=[
            ('/robot_description', '/robot_description_viz'),
        ]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(get_package_share_directory('franka_description'), 'rviz', 'visualize_franka.rviz')],
    )

    joy_node = Node(package='joy', executable='joy_node')
    teleop_node = Node(package='cbf_safety_layer', executable='teleop_node')
    # teleop_node = Node(
    #     package='cbf_safety_layer', 
    #     executable='teleop_node',
    #     remappings=[
    #         ('/safety/input_joint_states', '/velocity_group_controller/commands'),
    #     ]
    # )
    return LaunchDescription([
        franka_sim,
        spawn_broadcaster,
        delayed_velocity_spawner,
        safety_node,
        robot_state_publisher,
        rviz_node,
        joy_node,
        teleop_node
    ])