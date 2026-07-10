"""Publish full ELF3 joint states from arm IK commands for RViz."""

from __future__ import annotations

from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
import sensor_msgs.msg

from .constants import JOINT_NAMES


FULL_JOINT_NAMES = [
    'waist_y_joint',
    'waist_x_joint',
    'waist_z_joint',
    'l_hip_y_joint',
    'l_hip_x_joint',
    'l_hip_z_joint',
    'l_knee_y_joint',
    'l_ankle_y_joint',
    'l_ankle_x_joint',
    'r_hip_y_joint',
    'r_hip_x_joint',
    'r_hip_z_joint',
    'r_knee_y_joint',
    'r_ankle_y_joint',
    'r_ankle_x_joint',
] + JOINT_NAMES

NOMINAL_POSITIONS = np.array(
    [
        0.0,
        0.0,
        0.0,
        -0.4,
        0.0,
        0.0,
        0.8,
        -0.4,
        0.0,
        -0.4,
        0.0,
        0.0,
        0.8,
        -0.4,
        0.0,
        0.5,
        0.3,
        -0.1,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.5,
        -0.3,
        0.1,
        -0.2,
        0.0,
        0.0,
        0.0,
    ],
    dtype=float,
)


class ArmCommandJointStateNode(Node):
    """Bridge the 14-joint arm command topic into a full RViz joint state."""

    def __init__(self) -> None:
        super().__init__('arm_command_joint_state_node')
        self.declare_parameter('arm_command_topic', 'pico_control_joint_commands')
        self.declare_parameter('joint_state_topic', 'joint_states')
        self.declare_parameter('publish_period_sec', 1.0 / 30.0)

        self.arm_command_topic = str(self.get_parameter('arm_command_topic').value)
        self.joint_state_topic = str(self.get_parameter('joint_state_topic').value)
        self.publish_period_sec = float(
            self.get_parameter('publish_period_sec').value
        )
        if self.publish_period_sec <= 0.0:
            raise ValueError('publish_period_sec must be positive')

        self.positions = NOMINAL_POSITIONS.copy()
        qos = QoSProfile(
            depth=1,
            durability=qos_profile_sensor_data.durability,
            reliability=qos_profile_sensor_data.reliability,
        )
        self.create_subscription(
            sensor_msgs.msg.JointState,
            self.arm_command_topic,
            self._arm_command_callback,
            qos,
        )
        self.joint_state_pub = self.create_publisher(
            sensor_msgs.msg.JointState,
            self.joint_state_topic,
            qos,
        )
        self.timer = self.create_timer(self.publish_period_sec, self._publish)

    def _arm_command_callback(self, msg: sensor_msgs.msg.JointState) -> None:
        if len(msg.position) < len(JOINT_NAMES):
            self.get_logger().warn(
                f'ignoring incomplete arm command: {len(msg.position)} values'
            )
            return
        self.positions[-14:] = np.asarray(msg.position[:14], dtype=float)

    def _publish(self) -> None:
        msg = sensor_msgs.msg.JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = FULL_JOINT_NAMES
        msg.position = self.positions.astype(float).tolist()
        self.joint_state_pub.publish(msg)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = ArmCommandJointStateNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
