from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _patched_robot_description(context):
    urdf_file = Path(LaunchConfiguration('urdf_file').perform(context)).expanduser()
    text = urdf_file.read_text()
    mesh_dir = urdf_file.parent / 'meshes'
    text = text.replace('./meshes/', f'file://{mesh_dir}/')

    package_share = FindPackageShare('elf3_arm_ik_baselink')
    config_file = LaunchConfiguration('config_file')
    urdf_dir = LaunchConfiguration('urdf_dir')
    rviz_config = LaunchConfiguration('rviz_config')

    return [
        Node(
            package='elf3_arm_ik_baselink',
            executable='baselink_arm_ik_node',
            name='baselink_arm_ik_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                config_file,
                {'urdf_dir': urdf_dir},
            ],
        ),
        Node(
            package='elf3_arm_ik_baselink',
            executable='arm_command_joint_state_node',
            name='arm_command_joint_state_node',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': text}],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
        ),
    ]


def generate_launch_description():
    package_share = FindPackageShare('elf3_arm_ik_baselink')
    default_config = PathJoinSubstitution(
        [package_share, 'config', 'baselink_arm_ik.yaml']
    )
    default_rviz_config = PathJoinSubstitution(
        [package_share, 'config', 'elf3_arm_ik.rviz']
    )
    default_urdf_dir = PathJoinSubstitution([package_share, 'data'])
    default_urdf_file = PathJoinSubstitution(
        [
            EnvironmentVariable('HOME'),
            'bxi_arm_v2',
            'src',
            'bxi_rl_controller_ros2_example',
            'resources',
            'elf3_dof29',
            'urdf',
            'elf3.urdf',
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'config_file',
                default_value=default_config,
                description='YAML config for the base_link arm IK node.',
            ),
            DeclareLaunchArgument(
                'rviz_config',
                default_value=default_rviz_config,
                description='RViz config file.',
            ),
            DeclareLaunchArgument(
                'urdf_dir',
                default_value=default_urdf_dir,
                description='Directory containing elf3_arm_l.urdf and elf3_arm_r.urdf.',
            ),
            DeclareLaunchArgument(
                'urdf_file',
                default_value=default_urdf_file,
                description='Full ELF3 URDF for RViz robot model.',
            ),
            OpaqueFunction(function=_patched_robot_description),
        ]
    )
