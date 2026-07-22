from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'elf3_arm_ik_baselink'
repo_data_dir = os.path.join('..', '..', 'data')

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            'share/' + package_name + '/config',
            glob(os.path.join('config', '*.yaml')),
        ),
        (
            'share/' + package_name + '/config',
            glob(os.path.join('config', '*.rviz')),
        ),
        (
            'share/' + package_name + '/config/models',
            glob(os.path.join('config', 'models', '*.yaml')),
        ),
        (
            'share/' + package_name + '/launch',
            glob(os.path.join('launch', '*.launch.py')),
        ),
        (
            'share/' + package_name + '/data',
            [
                os.path.join(repo_data_dir, 'elf3_arm_l.urdf'),
                os.path.join(repo_data_dir, 'elf3_arm_r.urdf'),
            ],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='dev@todo.todo',
    description='Base-link target pose to ELF3 dual arm IK command bridge.',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
            'console_scripts': [
            (
                'baselink_arm_ik_node = '
                'elf3_arm_ik_baselink.baselink_arm_ik_node:main'
            ),
            (
                'arm_command_joint_state_node = '
                'elf3_arm_ik_baselink.arm_command_joint_state_node:main'
            ),
            (
                'demo_reset_scenarios = '
                'elf3_arm_ik_baselink.demo_reset_scenarios:main'
            ),
        ],
    },
)
