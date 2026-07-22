"""ROS2 node that converts base_link TCP targets into ELF3 arm commands."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Optional

from ament_index_python.packages import get_package_share_directory
import geometry_msgs.msg
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
import sensor_msgs.msg
import std_msgs.msg
import std_srvs.srv

from .constants import JOINT_NAMES, LEFT_HOME, RIGHT_HOME, SIDE_LEFT, SIDE_RIGHT
from .ik_solver import Elf3ArmIkSolver, IkResult
from .math_utils import exponential_smooth, quaternion_xyzw_to_matrix, rate_limit


@dataclass
class TargetPose:
    position: np.ndarray
    orientation: np.ndarray
    stamp_monotonic: float


@dataclass
class JointTrajectory:
    start_left: np.ndarray
    start_right: np.ndarray
    goal_left: np.ndarray
    goal_right: np.ndarray
    start_time: float
    duration: float
    name: str


class BaseLinkArmIkNode(Node):
    """Bridge base_link PoseStamped targets into the existing teleop arm API."""

    def __init__(self) -> None:
        super().__init__('baselink_arm_ik_node')
        self._declare_parameters()
        self._load_parameters()

        self.solver = Elf3ArmIkSolver(
            self.urdf_dir,
            max_position_error_m=self.max_position_error_m,
            max_orientation_error_rad=np.deg2rad(self.max_orientation_error_deg),
            max_jacobian_condition=self.max_jacobian_condition,
            min_joint_limit_margin_rad=self.min_joint_limit_margin_rad,
            left_shoulder_origin=self.left_shoulder_origin,
            right_shoulder_origin=self.right_shoulder_origin,
        )
        self._lock = threading.Lock()
        self.targets: dict[str, Optional[TargetPose]] = {
            SIDE_LEFT: None,
            SIDE_RIGHT: None,
        }
        self.current_joints: dict[str, Optional[np.ndarray]] = {
            SIDE_LEFT: None,
            SIDE_RIGHT: None,
        }
        self.last_solutions = {
            SIDE_LEFT: np.asarray(LEFT_HOME, dtype=float),
            SIDE_RIGHT: np.asarray(RIGHT_HOME, dtype=float),
        }
        self.active_trajectory: Optional[JointTrajectory] = None
        self.last_command_time = time.monotonic()

        qos = QoSProfile(
            depth=1,
            durability=qos_profile_sensor_data.durability,
            reliability=qos_profile_sensor_data.reliability,
        )
        self.create_subscription(
            geometry_msgs.msg.PoseStamped,
            self.left_target_topic,
            lambda msg: self._target_callback(SIDE_LEFT, msg),
            qos,
        )
        self.create_subscription(
            geometry_msgs.msg.PoseStamped,
            self.right_target_topic,
            lambda msg: self._target_callback(SIDE_RIGHT, msg),
            qos,
        )
        self.create_subscription(
            sensor_msgs.msg.JointState,
            self.joint_state_topic,
            self._joint_state_callback,
            qos,
        )

        self.joint_pub = self.create_publisher(
            sensor_msgs.msg.JointState,
            self.joint_command_topic,
            qos,
        )
        self.status_pub = self.create_publisher(
            std_msgs.msg.String,
            self.ik_status_topic,
            10,
        )
        self.left_enable_pub = self.create_publisher(
            std_msgs.msg.Float32,
            self.left_enable_topic,
            qos,
        )
        self.right_enable_pub = self.create_publisher(
            std_msgs.msg.Float32,
            self.right_enable_topic,
            qos,
        )
        self.create_service(
            std_srvs.srv.Trigger,
            self.go_home_service,
            self._go_home_callback,
        )
        self.create_service(
            std_srvs.srv.Trigger,
            self.go_handshake_service,
            self._go_handshake_callback,
        )
        self.create_service(
            std_srvs.srv.Trigger,
            self.go_demo_pose_a_service,
            self._go_demo_pose_a_callback,
        )
        self.create_service(
            std_srvs.srv.Trigger,
            self.go_demo_pose_b_service,
            self._go_demo_pose_b_callback,
        )
        self.timer = self.create_timer(self.publish_period_sec, self._on_timer)

    def _declare_parameters(self) -> None:
        default_urdf_dir = str(
            Path(get_package_share_directory('elf3_arm_ik_baselink')) / 'data'
        )
        self.declare_parameter('urdf_dir', default_urdf_dir)
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('left_target_topic', 'arm_ik/left_target')
        self.declare_parameter('right_target_topic', 'arm_ik/right_target')
        self.declare_parameter('joint_state_topic', 'pico_control_joint_states')
        self.declare_parameter('joint_command_topic', 'pico_control_joint_commands')
        self.declare_parameter('left_enable_topic', 'pico/left_grip')
        self.declare_parameter('right_enable_topic', 'pico/right_grip')
        self.declare_parameter('publish_period_sec', 0.01)
        self.declare_parameter('target_timeout_sec', 0.5)
        self.declare_parameter('smoothing_alpha', 0.35)
        self.declare_parameter('max_joint_step_rad', 0.08)
        self.declare_parameter('solve_orientation', True)
        self.declare_parameter('tcp_offset', [0.08, 0.0, 0.0])
        self.declare_parameter('left_shoulder_origin', [0.0, 0.178, 0.087])
        self.declare_parameter('right_shoulder_origin', [0.0, -0.178, 0.087])
        self.declare_parameter('workspace_min', [-0.25, -0.85, -0.25])
        self.declare_parameter('workspace_max', [0.75, 0.85, 0.95])
        self.declare_parameter('min_lr_tcp_distance', 0.12)
        self.declare_parameter('max_position_error_m', 0.002)
        self.declare_parameter('max_orientation_error_deg', 2.0)
        self.declare_parameter('max_jacobian_condition', 1000.0)
        self.declare_parameter('min_joint_limit_margin_rad', 0.005)
        self.declare_parameter('ik_status_topic', 'arm_ik/status')
        self.declare_parameter('publish_enable_grip', True)
        self.declare_parameter('go_home_service', 'arm_ik/go_home')
        self.declare_parameter(
            'go_handshake_service',
            'arm_ik/go_handshake_ready',
        )
        self.declare_parameter('home_duration_sec', 1.5)
        self.declare_parameter('handshake_duration_sec', 1.2)
        self.declare_parameter('demo_pose_duration_sec', 2.4)
        self.declare_parameter('handshake_tcp_position', [0.35, -0.28, 0.55])
        self.declare_parameter(
            'handshake_right_joints',
            [0.2, -0.65, -0.35, 0.95, 0.0, 0.2, 0.0],
        )
        self.declare_parameter('go_demo_pose_a_service', 'arm_ik/go_demo_pose_a')
        self.declare_parameter('go_demo_pose_b_service', 'arm_ik/go_demo_pose_b')
        self.declare_parameter(
            'demo_pose_a_left_joints',
            [0.4, 2.45, 1.0, 1.45, -1.2, 0.8, -0.6],
        )
        self.declare_parameter(
            'demo_pose_a_right_joints',
            [-0.9, -0.55, -1.4, -0.2, 1.2, -0.6, 0.6],
        )
        self.declare_parameter(
            'demo_pose_b_left_joints',
            [-0.9, 0.45, -1.4, -0.2, -1.2, 0.6, -0.6],
        )
        self.declare_parameter(
            'demo_pose_b_right_joints',
            [0.45, -2.65, -0.95, 1.55, 1.55, 0.95, 0.78],
        )

    def _load_parameters(self) -> None:
        self.urdf_dir = str(self.get_parameter('urdf_dir').value)
        self.target_frame = str(self.get_parameter('target_frame').value)
        self.left_target_topic = str(self.get_parameter('left_target_topic').value)
        self.right_target_topic = str(self.get_parameter('right_target_topic').value)
        self.joint_state_topic = str(self.get_parameter('joint_state_topic').value)
        self.joint_command_topic = str(self.get_parameter('joint_command_topic').value)
        self.left_enable_topic = str(self.get_parameter('left_enable_topic').value)
        self.right_enable_topic = str(self.get_parameter('right_enable_topic').value)
        self.publish_period_sec = self._positive_float('publish_period_sec')
        self.target_timeout_sec = self._positive_float('target_timeout_sec')
        self.smoothing_alpha = self._bounded_float('smoothing_alpha', 0.0, 1.0)
        self.max_joint_step_rad = self._positive_float('max_joint_step_rad')
        self.solve_orientation = bool(self.get_parameter('solve_orientation').value)
        self.tcp_offset = self._float_array_parameter('tcp_offset', 3)
        self.left_shoulder_origin = self._float_array_parameter(
            'left_shoulder_origin',
            3,
        )
        self.right_shoulder_origin = self._float_array_parameter(
            'right_shoulder_origin',
            3,
        )
        self.workspace_min = self._float_array_parameter('workspace_min', 3)
        self.workspace_max = self._float_array_parameter('workspace_max', 3)
        self.min_lr_tcp_distance = self._positive_float('min_lr_tcp_distance')
        self.max_position_error_m = self._positive_float('max_position_error_m')
        self.max_orientation_error_deg = self._positive_float(
            'max_orientation_error_deg'
        )
        self.max_jacobian_condition = self._positive_float(
            'max_jacobian_condition'
        )
        self.min_joint_limit_margin_rad = self._positive_float(
            'min_joint_limit_margin_rad'
        )
        self.ik_status_topic = str(self.get_parameter('ik_status_topic').value)
        self.publish_enable_grip = bool(
            self.get_parameter('publish_enable_grip').value
        )
        self.go_home_service = str(self.get_parameter('go_home_service').value)
        self.go_handshake_service = str(
            self.get_parameter('go_handshake_service').value
        )
        self.home_duration_sec = self._positive_float('home_duration_sec')
        self.handshake_duration_sec = self._positive_float(
            'handshake_duration_sec'
        )
        self.demo_pose_duration_sec = self._positive_float(
            'demo_pose_duration_sec'
        )
        self.handshake_tcp_position = self._float_array_parameter(
            'handshake_tcp_position',
            3,
        )
        self.handshake_right_joints = self._float_array_parameter(
            'handshake_right_joints',
            7,
        )
        self.go_demo_pose_a_service = str(
            self.get_parameter('go_demo_pose_a_service').value
        )
        self.go_demo_pose_b_service = str(
            self.get_parameter('go_demo_pose_b_service').value
        )
        self.demo_pose_a_left_joints = self._float_array_parameter(
            'demo_pose_a_left_joints',
            7,
        )
        self.demo_pose_a_right_joints = self._float_array_parameter(
            'demo_pose_a_right_joints',
            7,
        )
        self.demo_pose_b_left_joints = self._float_array_parameter(
            'demo_pose_b_left_joints',
            7,
        )
        self.demo_pose_b_right_joints = self._float_array_parameter(
            'demo_pose_b_right_joints',
            7,
        )

    def _positive_float(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if value <= 0.0:
            raise ValueError(f'{name} must be positive')
        return value

    def _bounded_float(self, name: str, lower: float, upper: float) -> float:
        value = float(self.get_parameter(name).value)
        if value < lower or value > upper:
            raise ValueError(f'{name} must be in [{lower}, {upper}]')
        return value

    def _float_array_parameter(self, name: str, size: int) -> np.ndarray:
        value = np.asarray(self.get_parameter(name).value, dtype=float).reshape(-1)
        if len(value) != size:
            raise ValueError(f'{name} must have {size} values')
        return value

    def _target_callback(
        self,
        side: str,
        msg: geometry_msgs.msg.PoseStamped,
    ) -> None:
        frame_id = msg.header.frame_id.strip()
        if frame_id and frame_id != self.target_frame:
            self.get_logger().warn(
                f'ignoring {side} target in frame {frame_id}; '
                f'expected {self.target_frame}'
            )
            return

        position = np.asarray(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=float,
        )
        if not self._is_in_workspace(position):
            self.get_logger().warn(
                f'ignoring {side} target outside workspace: {position.tolist()}'
            )
            return

        orientation = quaternion_xyzw_to_matrix(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        )

        with self._lock:
            self.targets[side] = TargetPose(
                position=position,
                orientation=orientation,
                stamp_monotonic=time.monotonic(),
            )

    def _joint_state_callback(self, msg: sensor_msgs.msg.JointState) -> None:
        if len(msg.position) < len(JOINT_NAMES):
            self.get_logger().warn(
                f'ignoring incomplete arm joint state: {len(msg.position)} values'
            )
            return
        with self._lock:
            self.current_joints[SIDE_LEFT] = np.asarray(msg.position[0:7], dtype=float)
            self.current_joints[SIDE_RIGHT] = np.asarray(msg.position[7:14], dtype=float)

    def _on_timer(self) -> None:
        now = time.monotonic()
        with self._lock:
            targets = dict(self.targets)
            seeds = dict(self.current_joints)
            previous = {
                SIDE_LEFT: self.last_solutions[SIDE_LEFT].copy(),
                SIDE_RIGHT: self.last_solutions[SIDE_RIGHT].copy(),
            }
            trajectory = self.active_trajectory

        if trajectory is not None:
            solved = self._sample_trajectory(trajectory, now)
            with self._lock:
                self.last_solutions = {
                    SIDE_LEFT: solved[SIDE_LEFT].copy(),
                    SIDE_RIGHT: solved[SIDE_RIGHT].copy(),
                }
                if now - trajectory.start_time >= trajectory.duration:
                    self.active_trajectory = None
                    self.get_logger().info(
                        f'{trajectory.name} trajectory completed'
                    )
                self.last_command_time = now

            self._publish_joint_command(solved[SIDE_LEFT], solved[SIDE_RIGHT])
            if self.publish_enable_grip:
                self._publish_arm_enable()
            return

        solved = previous
        for side in (SIDE_LEFT, SIDE_RIGHT):
            target = targets[side]
            if target is None or now - target.stamp_monotonic > self.target_timeout_sec:
                continue

            orientation = target.orientation if self.solve_orientation else None
            result = self.solver.solve(
                side=side,
                tcp_position=target.position,
                tcp_orientation=orientation,
                seed_joints=seeds[side] if seeds[side] is not None else previous[side],
                tcp_offset=self.tcp_offset,
            )
            if not result.success:
                self.get_logger().warn(
                    f'{side} IK rejected [{result.status}]: {result.message}'
                )
                self._publish_ik_status(side, result)
                continue

            limited = rate_limit(previous[side], result.joints, self.max_joint_step_rad)
            solved[side] = exponential_smooth(
                previous[side],
                limited,
                self.smoothing_alpha,
            )
            self._publish_ik_status(side, result)

        if not self._passes_dual_arm_guard(targets, now):
            solved = previous

        with self._lock:
            self.last_solutions = {
                SIDE_LEFT: solved[SIDE_LEFT].copy(),
                SIDE_RIGHT: solved[SIDE_RIGHT].copy(),
            }
            self.last_command_time = now

        self._publish_joint_command(solved[SIDE_LEFT], solved[SIDE_RIGHT])
        if self.publish_enable_grip:
            self._publish_arm_enable()

    def _go_home_callback(
        self,
        request: std_srvs.srv.Trigger.Request,
        response: std_srvs.srv.Trigger.Response,
    ) -> std_srvs.srv.Trigger.Response:
        del request
        self._start_joint_trajectory(
            name='home',
            goal_left=np.asarray(LEFT_HOME, dtype=float),
            goal_right=np.asarray(RIGHT_HOME, dtype=float),
            duration=self.home_duration_sec,
        )
        response.success = True
        response.message = 'started home trajectory'
        return response

    def _go_handshake_callback(
        self,
        request: std_srvs.srv.Trigger.Request,
        response: std_srvs.srv.Trigger.Response,
    ) -> std_srvs.srv.Trigger.Response:
        del request
        self._start_joint_trajectory(
            name='handshake_ready',
            goal_left=np.asarray(LEFT_HOME, dtype=float),
            goal_right=self.handshake_right_joints,
            duration=self.handshake_duration_sec,
        )
        response.success = True
        response.message = 'started handshake ready trajectory'
        return response

    def _go_demo_pose_a_callback(
        self,
        request: std_srvs.srv.Trigger.Request,
        response: std_srvs.srv.Trigger.Response,
    ) -> std_srvs.srv.Trigger.Response:
        del request
        self._start_joint_trajectory(
            name='demo_pose_a',
            goal_left=self.demo_pose_a_left_joints,
            goal_right=self.demo_pose_a_right_joints,
            duration=self.demo_pose_duration_sec,
        )
        response.success = True
        response.message = 'started demo pose A trajectory'
        return response

    def _go_demo_pose_b_callback(
        self,
        request: std_srvs.srv.Trigger.Request,
        response: std_srvs.srv.Trigger.Response,
    ) -> std_srvs.srv.Trigger.Response:
        del request
        self._start_joint_trajectory(
            name='demo_pose_b',
            goal_left=self.demo_pose_b_left_joints,
            goal_right=self.demo_pose_b_right_joints,
            duration=self.demo_pose_duration_sec,
        )
        response.success = True
        response.message = 'started demo pose B trajectory'
        return response

    def _start_joint_trajectory(
        self,
        name: str,
        goal_left: np.ndarray,
        goal_right: np.ndarray,
        duration: float,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            start_left = (
                self.current_joints[SIDE_LEFT].copy()
                if self.current_joints[SIDE_LEFT] is not None
                else self.last_solutions[SIDE_LEFT].copy()
            )
            start_right = (
                self.current_joints[SIDE_RIGHT].copy()
                if self.current_joints[SIDE_RIGHT] is not None
                else self.last_solutions[SIDE_RIGHT].copy()
            )
            self.active_trajectory = JointTrajectory(
                start_left=start_left,
                start_right=start_right,
                goal_left=np.asarray(goal_left, dtype=float).reshape(7),
                goal_right=np.asarray(goal_right, dtype=float).reshape(7),
                start_time=now,
                duration=float(duration),
                name=name,
            )
        self.get_logger().info(f'started {name} trajectory')

    @staticmethod
    def _sample_trajectory(
        trajectory: JointTrajectory,
        now: float,
    ) -> dict[str, np.ndarray]:
        progress = (now - trajectory.start_time) / trajectory.duration
        t = float(np.clip(progress, 0.0, 1.0))
        blend = t * t * (3.0 - 2.0 * t)
        return {
            SIDE_LEFT: trajectory.start_left
            + (trajectory.goal_left - trajectory.start_left) * blend,
            SIDE_RIGHT: trajectory.start_right
            + (trajectory.goal_right - trajectory.start_right) * blend,
        }

    def _is_in_workspace(self, position: np.ndarray) -> bool:
        return bool(
            np.all(position >= self.workspace_min)
            and np.all(position <= self.workspace_max)
        )

    def _passes_dual_arm_guard(
        self,
        targets: dict[str, Optional[TargetPose]],
        now: float,
    ) -> bool:
        left = targets[SIDE_LEFT]
        right = targets[SIDE_RIGHT]
        if left is None or right is None:
            return True
        if now - left.stamp_monotonic > self.target_timeout_sec:
            return True
        if now - right.stamp_monotonic > self.target_timeout_sec:
            return True

        distance = float(np.linalg.norm(left.position - right.position))
        if distance < self.min_lr_tcp_distance:
            self.get_logger().warn(
                'holding previous arm command: target TCPs too close '
                f'({distance:.3f} m)'
            )
            return False
        return True

    def _publish_joint_command(
        self,
        left_joints: np.ndarray,
        right_joints: np.ndarray,
    ) -> None:
        msg = sensor_msgs.msg.JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = list(left_joints.astype(float)) + list(
            right_joints.astype(float)
        )
        self.joint_pub.publish(msg)

    def _publish_arm_enable(self) -> None:
        enable = std_msgs.msg.Float32()
        enable.data = 1.0
        self.left_enable_pub.publish(enable)
        self.right_enable_pub.publish(enable)

    def _publish_ik_status(self, side: str, result: IkResult) -> None:
        def finite_or_none(value: float) -> Optional[float]:
            return float(value) if np.isfinite(value) else None

        msg = std_msgs.msg.String()
        msg.data = json.dumps(
            {
                'side': side,
                'success': result.success,
                'status': result.status,
                'position_error_m': finite_or_none(result.position_error_m),
                'orientation_error_deg': finite_or_none(
                    np.rad2deg(result.orientation_error_rad),
                ),
                'solve_time_ms': finite_or_none(result.solve_time_ms),
                'validation_time_ms': finite_or_none(result.validation_time_ms),
                'jacobian_condition': finite_or_none(result.jacobian_condition),
                'min_joint_limit_margin_rad': finite_or_none(
                    result.min_joint_limit_margin_rad,
                ),
                'message': result.message,
            },
            allow_nan=False,
            separators=(',', ':'),
        )
        self.status_pub.publish(msg)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = BaseLinkArmIkNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
