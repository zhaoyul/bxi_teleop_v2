#!/usr/bin/env python3
"""Exercise the formal SolveArmIK service with success and failure cases."""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from elf3_arm_ik_interfaces.srv import SolveArmIK
import numpy as np
import rclpy
from rclpy.node import Node

from elf3_arm_ik_baselink.constants import LEFT_JOINT_NAMES
from elf3_arm_ik_baselink.ik_solver import Elf3ArmIkSolver


HOME = np.array([0.5, 0.3, -0.1, -0.2, 0.0, 0.0, 0.0])
TCP_OFFSET = np.array([0.08, 0.0, 0.0])


class SolveServiceTest(Node):
    def __init__(self) -> None:
        super().__init__('solve_arm_ik_service_test')
        self.client = self.create_client(SolveArmIK, 'arm_ik/solve')

    def call(self, request: SolveArmIK.Request) -> SolveArmIK.Response:
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.result() is None:
            raise RuntimeError('SolveArmIK service timed out')
        return future.result()


def reachable_request() -> SolveArmIK.Request:
    share = Path(get_package_share_directory('elf3_arm_ik_baselink'))
    solver = Elf3ArmIkSolver(str(share / 'data'))
    transform = solver.left_chain.forward_kinematics(np.r_[0.0, HOME])
    tcp_position = (
        solver.shoulder_origins['left']
        + transform[:3, 3]
        + transform[:3, :3] @ TCP_OFFSET
    )
    quaternion = rotation_matrix_to_quaternion_xyzw(transform[:3, :3])

    request = SolveArmIK.Request()
    request.robot_model = 'elf3'
    request.arm_side = 'left'
    request.current_joint_state.name = list(LEFT_JOINT_NAMES)
    request.current_joint_state.position = HOME.tolist()
    request.target_pose.header.frame_id = 'base_link'
    request.target_pose.pose.position.x = float(tcp_position[0])
    request.target_pose.pose.position.y = float(tcp_position[1])
    request.target_pose.pose.position.z = float(tcp_position[2])
    request.target_pose.pose.orientation.x = float(quaternion[0])
    request.target_pose.pose.orientation.y = float(quaternion[1])
    request.target_pose.pose.orientation.z = float(quaternion[2])
    request.target_pose.pose.orientation.w = float(quaternion[3])
    request.use_tcp_offset = True
    request.tcp_offset.x = float(TCP_OFFSET[0])
    request.tcp_offset.y = float(TCP_OFFSET[1])
    request.tcp_offset.z = float(TCP_OFFSET[2])
    return request


def rotation_matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
    values = np.empty(4)
    values[3] = np.sqrt(max(0.0, 1.0 + np.trace(rotation))) / 2.0
    denominator = max(4.0 * values[3], 1e-12)
    values[0] = (rotation[2, 1] - rotation[1, 2]) / denominator
    values[1] = (rotation[0, 2] - rotation[2, 0]) / denominator
    values[2] = (rotation[1, 0] - rotation[0, 1]) / denominator
    return values / np.linalg.norm(values)


def main() -> int:
    rclpy.init()
    node = SolveServiceTest()
    failures = []
    try:
        if not node.client.wait_for_service(timeout_sec=10.0):
            print('FAILED: arm_ik/solve service is unavailable')
            return 1

        success = node.call(reachable_request())
        print(
            f'reachable: success={success.success} status={success.status} '
            f'position={success.position_error_m * 1000.0:.3f}mm '
            f'solve={success.solve_time_ms:.3f}ms'
        )
        if not success.success or len(success.joint_solution.position) != 7:
            failures.append('reachable target')

        unsupported_request = reachable_request()
        unsupported_request.robot_model = 'unknown_robot'
        unsupported = node.call(unsupported_request)
        print(f'unsupported: success={unsupported.success} status={unsupported.status}')
        if unsupported.success or unsupported.status != 'unsupported_model':
            failures.append('unsupported model')

        collision_request = reachable_request()
        collision_request.require_collision_check = True
        collision = node.call(collision_request)
        print(f'collision-required: success={collision.success} status={collision.status}')
        if collision.success or collision.status != 'collision_check_unavailable':
            failures.append('collision unavailable')
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if failures:
        print('FAILED: ' + ', '.join(failures))
        return 1
    print('PASS: SolveArmIK service contract')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
