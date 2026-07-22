#!/usr/bin/env python3
"""Publish a known reachable target and verify the ROS2 IK status response."""

from __future__ import annotations

import json
from pathlib import Path
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from elf3_arm_ik_baselink.ik_solver import Elf3ArmIkSolver


class IkSmokeTestNode(Node):
    def __init__(self) -> None:
        super().__init__('arm_ik_smoke_test')
        self.status = None
        self.publisher = self.create_publisher(
            PoseStamped,
            'arm_ik/left_target',
            10,
        )
        self.create_subscription(String, 'arm_ik/status', self._on_status, 10)
        self.target = self._build_target()

    def _build_target(self) -> PoseStamped:
        share = Path(get_package_share_directory('elf3_arm_ik_baselink'))
        solver = Elf3ArmIkSolver(str(share / 'data'))
        joints = np.array([0.5, 0.3, -0.1, -0.2, 0.0, 0.0, 0.0])
        transform = solver.left_chain.forward_kinematics(np.r_[0.0, joints])
        tcp_offset = np.array([0.08, 0.0, 0.0])
        tcp_position = (
            solver.shoulder_origins['left']
            + transform[:3, 3]
            + transform[:3, :3] @ tcp_offset
        )

        msg = PoseStamped()
        msg.header.frame_id = 'base_link'
        msg.pose.position.x = float(tcp_position[0])
        msg.pose.position.y = float(tcp_position[1])
        msg.pose.position.z = float(tcp_position[2])
        quaternion = rotation_matrix_to_quaternion_xyzw(transform[:3, :3])
        msg.pose.orientation.x = float(quaternion[0])
        msg.pose.orientation.y = float(quaternion[1])
        msg.pose.orientation.z = float(quaternion[2])
        msg.pose.orientation.w = float(quaternion[3])
        return msg

    def _on_status(self, msg: String) -> None:
        status = json.loads(msg.data)
        if status.get('side') == 'left':
            self.status = status


def rotation_matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a normalized XYZW quaternion."""
    rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        values = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        next_axis = (axis + 1) % 3
        last_axis = (axis + 2) % 3
        scale = 2.0 * np.sqrt(
            1.0 + rotation[axis, axis]
            - rotation[next_axis, next_axis]
            - rotation[last_axis, last_axis]
        )
        values = np.zeros(4)
        values[axis] = 0.25 * scale
        values[3] = (rotation[last_axis, next_axis] - rotation[next_axis, last_axis]) / scale
        values[next_axis] = (rotation[next_axis, axis] + rotation[axis, next_axis]) / scale
        values[last_axis] = (rotation[last_axis, axis] + rotation[axis, last_axis]) / scale
    return values / np.linalg.norm(values)


def main() -> int:
    rclpy.init()
    node = IkSmokeTestNode()
    deadline = time.monotonic() + 10.0
    try:
        while rclpy.ok() and time.monotonic() < deadline and node.status is None:
            node.target.header.stamp = node.get_clock().now().to_msg()
            node.publisher.publish(node.target)
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if node.status is None:
        print('FAILED: no arm_ik/status response within 10 seconds')
        return 1
    print(json.dumps(node.status, indent=2, sort_keys=True))
    if not node.status.get('success'):
        print('FAILED: reachable target was rejected')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
