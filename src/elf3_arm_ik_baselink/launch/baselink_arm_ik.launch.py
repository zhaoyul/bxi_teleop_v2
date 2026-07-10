from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare('elf3_arm_ik_baselink')
    default_config = PathJoinSubstitution(
        [package_share, 'config', 'baselink_arm_ik.yaml']
    )
    default_urdf_dir = PathJoinSubstitution([package_share, 'data'])

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'config_file',
                default_value=default_config,
                description='YAML config for the base_link arm IK node.',
            ),
            DeclareLaunchArgument(
                'urdf_dir',
                default_value=default_urdf_dir,
                description='Directory containing elf3_arm_l.urdf and elf3_arm_r.urdf.',
            ),
            Node(
                package='elf3_arm_ik_baselink',
                executable='baselink_arm_ik_node',
                name='baselink_arm_ik_node',
                output='screen',
                emulate_tty=True,
                parameters=[
                    LaunchConfiguration('config_file'),
                    {'urdf_dir': LaunchConfiguration('urdf_dir')},
                ],
            ),
        ]
    )
