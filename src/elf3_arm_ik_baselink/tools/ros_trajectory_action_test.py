#!/usr/bin/env python3
"""Verify trajectory planning, constraints, execution cancel, and safe return."""

from __future__ import annotations

from pathlib import Path
import time

from ament_index_python.packages import get_package_share_directory
from elf3_arm_ik_interfaces.action import PlanArmTrajectory
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from elf3_arm_ik_baselink.constants import LEFT_JOINT_NAMES
from elf3_arm_ik_baselink.ik_solver import Elf3ArmIkSolver


HOME = np.array([0.5, 0.3, -0.1, -0.2, 0.0, 0.0, 0.0])
TARGET = np.array([0.4, 1.2, 0.4, 0.8, -0.4, 0.4, -0.2])
TCP_OFFSET = np.array([0.08, 0.0, 0.0])


class TrajectoryActionTest(Node):
    def __init__(self) -> None:
        super().__init__('trajectory_action_test')
        self.client = ActionClient(
            self,
            PlanArmTrajectory,
            'arm_ik/plan_trajectory',
        )

    def send(self, goal: PlanArmTrajectory.Goal):
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError('trajectory goal was rejected')
        return handle

    def result(self, handle):
        future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        wrapped = future.result()
        if wrapped is None:
            raise RuntimeError('trajectory result timed out')
        return wrapped.result


def build_goal(execute: bool) -> PlanArmTrajectory.Goal:
    share = Path(get_package_share_directory('elf3_arm_ik_baselink'))
    solver = Elf3ArmIkSolver(str(share / 'data'))
    transform = solver.left_chain.forward_kinematics(np.r_[0.0, TARGET])
    tcp_position = (
        solver.shoulder_origins['left']
        + transform[:3, 3]
        + transform[:3, :3] @ TCP_OFFSET
    )
    quaternion = rotation_matrix_to_quaternion_xyzw(transform[:3, :3])

    goal = PlanArmTrajectory.Goal()
    goal.robot_model = 'elf3'
    goal.arm_side = 'left'
    goal.current_joint_state.name = list(LEFT_JOINT_NAMES)
    goal.current_joint_state.position = HOME.tolist()
    goal.target_pose.header.frame_id = 'base_link'
    goal.target_pose.pose.position.x = float(tcp_position[0])
    goal.target_pose.pose.position.y = float(tcp_position[1])
    goal.target_pose.pose.position.z = float(tcp_position[2])
    goal.target_pose.pose.orientation.x = float(quaternion[0])
    goal.target_pose.pose.orientation.y = float(quaternion[1])
    goal.target_pose.pose.orientation.z = float(quaternion[2])
    goal.target_pose.pose.orientation.w = float(quaternion[3])
    goal.use_tcp_offset = True
    goal.tcp_offset.x = float(TCP_OFFSET[0])
    goal.max_velocity_rad_s = 0.4
    goal.max_acceleration_rad_s2 = 0.8
    goal.control_period_sec = 0.02
    goal.minimum_duration_sec = 3.0
    goal.execute = execute
    goal.return_to_safe_on_cancel = True
    goal.safe_return_mode = 'home'
    return goal


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
    node = TrajectoryActionTest()
    failures = []
    try:
        if not node.client.wait_for_server(timeout_sec=10.0):
            print('FAILED: arm_ik/plan_trajectory action is unavailable')
            return 1

        planned = node.result(node.send(build_goal(execute=False)))
        points = planned.trajectory.points
        max_velocity = max(
            abs(value)
            for point in points
            for value in point.velocities
        )
        max_acceleration = max(
            abs(value)
            for point in points
            for value in point.accelerations
        )
        print(
            f'plan: success={planned.success} status={planned.status} '
            f'points={len(points)} velocity={max_velocity:.3f} '
            f'acceleration={max_acceleration:.3f}'
        )
        if not planned.success or not points:
            failures.append('plan result')
        if max_velocity > 0.4 + 1e-9 or max_acceleration > 0.8 + 1e-9:
            failures.append('trajectory limits')

        executing_handle = node.send(build_goal(execute=True))
        time.sleep(0.3)
        cancel_future = executing_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(node, cancel_future, timeout_sec=5.0)
        if not cancel_future.result().goals_canceling:
            failures.append('cancel acceptance')
        cancelled = node.result(executing_handle)
        print(
            f'cancel: success={cancelled.success} status={cancelled.status} '
            f'message={cancelled.message}'
        )
        if cancelled.success or cancelled.status != 'cancelled_safe_return_started':
            failures.append('safe return after cancel')
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if failures:
        print('FAILED: ' + ', '.join(failures))
        return 1
    print('PASS: PlanArmTrajectory action contract')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
