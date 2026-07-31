from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    package_share = FindPackageShare('elf3_arm_ik_baselink')
    base_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [package_share, 'launch', 'baselink_arm_ik_rviz.launch.py']
            )
        ),
        launch_arguments={'home_duration_sec': '0.8'}.items(),
    )
    reachability_demo = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='elf3_arm_ik_baselink',
                executable='demo_cross_midline_reachability',
                name='demo_cross_midline_reachability',
                output='screen',
                emulate_tty=True,
                arguments=['--speed-scale', '5.0'],
            )
        ],
    )
    return LaunchDescription([base_demo, reachability_demo])
