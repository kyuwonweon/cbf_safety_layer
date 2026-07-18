import os
import re
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, GroupAction,
                             RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetRemap


def generate_launch_description():

    # -----------------------------------------------------------------------
    # URDF preparation
    # -----------------------------------------------------------------------
    urdf_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots', 'fer', 'fer.urdf'
    )
    with open(urdf_file, 'r') as f:
        robot_descrip = f.read()

    def apply_namespace(desc, ns_prefix):
        result = re.sub(r'=\s*(["\'])fer_', f'=\\1{ns_prefix}fer_', desc)
        result = re.sub(
            r'\s*<joint\s+name=["\'][^"\']*base_joint["\'][^>]*>.*?</joint>',
            '', result, flags=re.DOTALL)
        result = re.sub(
            r'\s*<link\s+name=["\']base["\'][^/]*/?>(\s*</link>)?',
            '', result, flags=re.DOTALL)
        return result

    robot1_r_descrip = apply_namespace(robot_descrip, "robot1_")
    robot2_r_descrip = apply_namespace(robot_descrip, "robot2_")

    # -----------------------------------------------------------------------
    # Static TF — must come first, no delay
    # -----------------------------------------------------------------------
    world_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_base_broadcaster',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--yaw', '0', '--pitch', '0', '--roll', '0',
                   '--frame-id', 'world', '--child-frame-id', 'base']
    )

    robot1_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='robot1_base_broadcaster',
        arguments=['--x', '0', '--y', '0.4', '--z', '0',
                   '--yaw', '-1.5708', '--pitch', '0', '--roll', '0',
                   '--frame-id', 'base', '--child-frame-id', 'robot1_fer_link0']
    )

    robot2_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='robot2_base_broadcaster',
        arguments=['--x', '0', '--y', '-0.4', '--z', '0',
                   '--yaw', '1.5708', '--pitch', '0', '--roll', '0',
                   '--frame-id', 'base', '--child-frame-id', 'robot2_fer_link0']
    )

    # -----------------------------------------------------------------------
    # Robot 1 hardware bringup
    # -----------------------------------------------------------------------
    robot1_franka = GroupAction([
        SetRemap(src='/joint_states',  dst='/robot1/franka_joint_states'),
        SetRemap(src='/tf',            dst='/tf_garbage_r1'),
        SetRemap(src='/tf_static',     dst='/tf_static_garbage_r1'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('franka_bringup'),
                             'launch', 'franka.launch.py')
            ),
            launch_arguments={
                'robot_ip':           '192.168.51.20',
                'use_fake_hardware':  'false',
                'robot_type':         'fer',
                'arm_id':             'fer',
                'load_gripper':       'false',
                'use_rviz':           'false',
                'namespace':          'robot1',
            }.items(),
        )
    ])

    robot2_franka = GroupAction([
        SetRemap(src='/joint_states',  dst='/robot2/franka_joint_states'),
        SetRemap(src='/tf',            dst='/tf_garbage_r2'),
        SetRemap(src='/tf_static',     dst='/tf_static_garbage_r2'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('franka_bringup'),
                             'launch', 'franka.launch.py')
            ),
            launch_arguments={
                'robot_ip':           '192.168.51.19',
                'use_fake_hardware':  'false',
                'robot_type':         'fer',
                'arm_id':             'fer',
                'load_gripper':       'false',
                'use_rviz':           'false',
                'namespace':          'robot2',
            }.items(),
        )
    ])

    # -----------------------------------------------------------------------
    # Controller spawning — broadcaster first, then velocity controller
    # -----------------------------------------------------------------------
    robot1_spawn_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        name='robot1_spawn_broadcaster',
        arguments=['franka_robot_state_broadcaster',
                   '--controller-manager', '/robot1/controller_manager'],
    )

    robot1_spawn_velocity = Node(
        package='controller_manager',
        executable='spawner',
        name='robot1_spawn_velocity',
        arguments=['velocity_group_controller',
                   '--controller-manager', '/robot1/controller_manager',
                   '--param-file',
                   os.path.join(get_package_share_directory('cbf_safety_layer'),
                                'config', 'controllers.yaml')],
    )

    robot1_delayed_velocity = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=robot1_spawn_broadcaster,
            on_exit=[robot1_spawn_velocity],
        )
    )

    robot2_spawn_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        name='robot2_spawn_broadcaster',
        arguments=['franka_robot_state_broadcaster',
                   '--controller-manager', '/robot2/controller_manager'],
    )

    robot2_spawn_velocity = Node(
        package='controller_manager',
        executable='spawner',
        name='robot2_spawn_velocity',
        arguments=['velocity_group_controller',
                   '--controller-manager', '/robot2/controller_manager',
                   '--param-file',
                   os.path.join(get_package_share_directory('cbf_safety_layer'),
                                'config', 'controllers.yaml')],
    )

    robot2_delayed_velocity = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=robot2_spawn_broadcaster,
            on_exit=[robot2_spawn_velocity],
        )
    )

    # -----------------------------------------------------------------------
    # Robot State Publishers — for RViz TF chain only, not for control
    # -----------------------------------------------------------------------
    robot1_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot1_state_publisher',
        namespace='robot1',
        output='screen',
        parameters=[{'robot_description': robot1_r_descrip}],
    )

    robot2_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot2_state_publisher',
        namespace='robot2',
        output='screen',
        parameters=[{'robot_description': robot2_r_descrip}],
    )

    # -----------------------------------------------------------------------
    # CBF Safety Nodes
    # /joint_states_source       → hardware franka_robot_state_broadcaster topic
    # /joint_states_source_other → other robot's hardware broadcaster topic
    # -----------------------------------------------------------------------
    robot1_safety_node = Node(
        package='cbf_safety_layer_cpp',
        executable='safety_node_cpp',
        namespace='robot1',
        output='screen',
        parameters=[{
            'self_robot_description':  robot1_r_descrip,
            'other_robot_description': robot2_r_descrip,
            'self_frame_prefix':       'robot1_fer_',
            'other_frame_prefix':      'robot2_fer_',
            'reference_frame':         'base',
            'base_offset_x':           0.0,
            'base_offset_y':           0.4,
            'base_offset_z':           0.0,
            'other_base_offset_x':     0.0,
            'other_base_offset_y':    -0.4,
            'other_base_offset_z':     0.0,
            'base_yaw': -1.5708,
            'other_base_yaw': 1.5708,
            'use_fallback_urdf':        False,
            'hardware_mode':            True,
        }],
        remappings=[
            ('/joint_states_source',
             '/robot1/franka_robot_state_broadcaster/robot_state'),
            ('/joint_states_source_other',
             '/robot2/franka_robot_state_broadcaster/robot_state'),
            ('/safety_marker',           '/robot1/safety_marker'),
            ('safety/joint_states',      'joint_states'),
            ('/manual_vel',              '/robot1/manual_vel'),
            ('/teleop_vel',              '/robot1/teleop_vel'),
            ('/velocity_group_controller/commands',
             '/robot1/velocity_group_controller/commands'),
        ]
    )

    robot2_safety_node = Node(
        package='cbf_safety_layer_cpp',
        executable='safety_node_cpp',
        namespace='robot2',
        output='screen',
        parameters=[{
            'self_robot_description':  robot2_r_descrip,
            'other_robot_description': robot1_r_descrip,
            'self_frame_prefix':       'robot2_fer_',
            'other_frame_prefix':      'robot1_fer_',
            'reference_frame':         'base',
            'base_offset_x':           0.0,
            'base_offset_y':          -0.4,
            'base_offset_z':           0.0,
            'other_base_offset_x':     0.0,
            'other_base_offset_y':     0.4,
            'other_base_offset_z':     0.0,
            'base_yaw': 1.5708,
            'other_base_yaw': -1.5708,
            'use_fallback_urdf':        False,
            'hardware_mode':            True,
        }],
        remappings=[
            ('/joint_states_source',
             '/robot2/franka_robot_state_broadcaster/robot_state'),
            ('/joint_states_source_other',
             '/robot1/franka_robot_state_broadcaster/robot_state'),
            ('/safety_marker',           '/robot2/safety_marker'),
            ('safety/joint_states',      'joint_states'),
            ('/manual_vel',              '/robot2/manual_vel'),
            ('/teleop_vel',              '/robot2/teleop_vel'),
            ('/velocity_group_controller/commands',
             '/robot2/velocity_group_controller/commands'),
        ]
    )

    # -----------------------------------------------------------------------
    # Teleop, RViz, joystick
    # -----------------------------------------------------------------------
    robot1_teleop_node = Node(
        package='cbf_safety_layer',
        executable='teleop_node',
        name='teleop_robot1',
        parameters=[{'robot_id': 1}],
        remappings=[('safety/input_joint_states', '/robot1/teleop_vel')]
    )

    robot2_teleop_node = Node(
        package='cbf_safety_layer',
        executable='teleop_node',
        name='teleop_robot2',
        parameters=[{'robot_id': 2}],
        remappings=[('safety/input_joint_states', '/robot2/teleop_vel')]
    )

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[{'deadzone': 0.05, 'autorepeat_rate': 20.0}]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
    )

    # -----------------------------------------------------------------------
    # Launch order
    # -----------------------------------------------------------------------
    return LaunchDescription([
        world_to_base,
        robot1_tf,
        robot2_tf,

        robot1_franka,
        robot2_franka,
        robot1_spawn_broadcaster,
        robot1_delayed_velocity,
        robot2_spawn_broadcaster,
        robot2_delayed_velocity,

        TimerAction(period=1.0, actions=[
            robot1_robot_state_publisher,
            robot2_robot_state_publisher,
            robot1_safety_node,
            robot2_safety_node,
            robot1_teleop_node,
            robot2_teleop_node,
            joy_node,
        ]),

        TimerAction(period=2.0, actions=[rviz_node]),
    ])