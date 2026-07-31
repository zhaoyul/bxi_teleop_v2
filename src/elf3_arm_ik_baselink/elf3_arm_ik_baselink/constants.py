"""Shared ELF3 arm names and neutral poses."""

LEFT_JOINT_NAMES = [
    'l_shoulder_y_joint',
    'l_shoulder_x_joint',
    'l_shoulder_z_joint',
    'l_elbow_y_joint',
    'l_wrist_x_joint',
    'l_wrist_y_joint',
    'l_wrist_z_joint',
]

RIGHT_JOINT_NAMES = [
    'r_shoulder_y_joint',
    'r_shoulder_x_joint',
    'r_shoulder_z_joint',
    'r_elbow_y_joint',
    'r_wrist_x_joint',
    'r_wrist_y_joint',
    'r_wrist_z_joint',
]

JOINT_NAMES = LEFT_JOINT_NAMES + RIGHT_JOINT_NAMES

LEFT_HOME = [0.5, 0.3, -0.1, -0.2, 0.0, 0.0, 0.0]
RIGHT_HOME = [0.5, -0.3, 0.1, -0.2, 0.0, 0.0, 0.0]

RIGHT_IK_REFERENCE_SEEDS = [
    [-0.6516, 0.1924, 0.5021, -0.1126, 0.2656, 0.3008, -0.1597],
    [-1.0020, 0.3140, 1.2880, 0.4780, 0.6430, -0.2830, -0.7760],
    [-1.1010, 0.2770, 0.9710, 0.2000, 0.6490, 0.1270, -0.7010],
]

SIDE_LEFT = 'left'
SIDE_RIGHT = 'right'
