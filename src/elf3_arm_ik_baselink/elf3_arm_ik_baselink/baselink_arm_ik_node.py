"""ROS2 node that converts base_link TCP targets into ELF3 arm commands."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Optional

from ament_index_python.packages import get_package_share_directory
from elf3_arm_ik_interfaces.action import PlanArmTrajectory
from elf3_arm_ik_interfaces.srv import SolveArmIK
import geometry_msgs.msg
import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.task import Future
import sensor_msgs.msg
import std_msgs.msg
import std_srvs.srv
import trajectory_msgs.msg

from .constants import (
    JOINT_NAMES,
    LEFT_HOME,
    LEFT_JOINT_NAMES,
    RIGHT_HOME,
    RIGHT_IK_REFERENCE_SEEDS,
    RIGHT_JOINT_NAMES,
    SIDE_LEFT,
    SIDE_RIGHT,
)
from .ik_solver import Elf3ArmIkSolver, IkResult
from .math_utils import exponential_smooth, quaternion_xyzw_to_matrix, rate_limit
from .trajectory_planner import PlannedTrajectory, plan_smoothstep_trajectory


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
        self._solver_lock = threading.Lock()
        self.targets: dict[str, Optional[TargetPose]] = {
            SIDE_LEFT: None,
            SIDE_RIGHT: None,
        }
        self.current_joints: dict[str, Optional[np.ndarray]] = {
            SIDE_LEFT: None,
            SIDE_RIGHT: None,
        }
        self.last_joint_state_time = 0.0
        self.last_solutions = {
            SIDE_LEFT: np.asarray(LEFT_HOME, dtype=float),
            SIDE_RIGHT: np.asarray(RIGHT_HOME, dtype=float),
        }
        self._action_active = False
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
        self.create_service(
            SolveArmIK,
            self.solve_ik_service,
            self._solve_ik_callback,
        )
        self._trajectory_action = ActionServer(
            self,
            PlanArmTrajectory,
            self.plan_trajectory_action,
            execute_callback=self._execute_trajectory_action,
            goal_callback=self._trajectory_goal_callback,
            cancel_callback=self._trajectory_cancel_callback,
            callback_group=ReentrantCallbackGroup(),
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
        self.declare_parameter('joint_state_timeout_sec', 0.2)
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
        self.declare_parameter('solve_ik_service', 'arm_ik/solve')
        self.declare_parameter('plan_trajectory_action', 'arm_ik/plan_trajectory')
        self.declare_parameter('trajectory_max_velocity_rad_s', 0.6)
        self.declare_parameter('trajectory_max_acceleration_rad_s2', 1.2)
        self.declare_parameter('trajectory_control_period_sec', 0.02)
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
        self.joint_state_timeout_sec = self._positive_float(
            'joint_state_timeout_sec'
        )
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
        self.solve_ik_service = str(self.get_parameter('solve_ik_service').value)
        self.plan_trajectory_action = str(
            self.get_parameter('plan_trajectory_action').value
        )
        self.trajectory_max_velocity_rad_s = self._positive_float(
            'trajectory_max_velocity_rad_s'
        )
        self.trajectory_max_acceleration_rad_s2 = self._positive_float(
            'trajectory_max_acceleration_rad_s2'
        )
        self.trajectory_control_period_sec = self._positive_float(
            'trajectory_control_period_sec'
        )
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
            self.last_joint_state_time = time.monotonic()

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
            with self._solver_lock:
                result = self.solver.solve(
                    side=side,
                    tcp_position=target.position,
                    tcp_orientation=orientation,
                    seed_joints=(
                        seeds[side] if seeds[side] is not None else previous[side]
                    ),
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

    def _solve_ik_callback(
        self,
        request: SolveArmIK.Request,
        response: SolveArmIK.Response,
    ) -> SolveArmIK.Response:
        model = request.robot_model.strip()
        if model and model != 'elf3':
            return self._reject_solve_response(
                response,
                SolveArmIK.Response.STATUS_UNSUPPORTED_MODEL,
                'unsupported_model',
                f'robot model {model!r} is not configured',
            )

        side = request.arm_side.strip().lower()
        if side not in (SIDE_LEFT, SIDE_RIGHT):
            return self._reject_solve_response(
                response,
                SolveArmIK.Response.STATUS_INVALID_REQUEST,
                'invalid_arm_side',
                "arm_side must be 'left' or 'right'",
            )

        frame_id = request.target_pose.header.frame_id.strip()
        if frame_id and frame_id != self.target_frame:
            return self._reject_solve_response(
                response,
                SolveArmIK.Response.STATUS_INVALID_FRAME,
                'invalid_frame',
                f'expected frame {self.target_frame!r}, received {frame_id!r}',
            )

        position = np.asarray(
            [
                request.target_pose.pose.position.x,
                request.target_pose.pose.position.y,
                request.target_pose.pose.position.z,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(position)) or not self._is_in_workspace(position):
            return self._reject_solve_response(
                response,
                SolveArmIK.Response.STATUS_OUTSIDE_WORKSPACE,
                'outside_workspace',
                f'target position {position.tolist()} is outside configured workspace',
            )

        if request.require_collision_check:
            return self._reject_solve_response(
                response,
                SolveArmIK.Response.STATUS_COLLISION_CHECK_UNAVAILABLE,
                'collision_check_unavailable',
                'collision checking is not available in this build',
            )

        try:
            seed = self._seed_from_joint_state(side, request.current_joint_state)
        except ValueError as exc:
            return self._reject_solve_response(
                response,
                SolveArmIK.Response.STATUS_INVALID_REQUEST,
                'invalid_joint_state',
                str(exc),
            )

        orientation = quaternion_xyzw_to_matrix(
            [
                request.target_pose.pose.orientation.x,
                request.target_pose.pose.orientation.y,
                request.target_pose.pose.orientation.z,
                request.target_pose.pose.orientation.w,
            ]
        )
        tcp_offset = (
            np.asarray(
                [
                    request.tcp_offset.x,
                    request.tcp_offset.y,
                    request.tcp_offset.z,
                ],
                dtype=float,
            )
            if request.use_tcp_offset
            else self.tcp_offset
        )
        fallback_seeds = [
            self.last_solutions[side],
            np.asarray(LEFT_HOME if side == SIDE_LEFT else RIGHT_HOME, dtype=float),
        ]
        if side == SIDE_RIGHT:
            fallback_seeds.extend(
                np.asarray(item, dtype=float) for item in RIGHT_IK_REFERENCE_SEEDS
            )
        with self._solver_lock:
            result = self.solver.solve_with_seeds(
                side=side,
                tcp_position=position,
                tcp_orientation=orientation,
                primary_seed=seed,
                fallback_seeds=fallback_seeds,
                tcp_offset=tcp_offset,
                max_position_error_m=request.max_position_error_m,
                max_orientation_error_rad=request.max_orientation_error_rad,
                max_jacobian_condition=request.max_jacobian_condition,
                min_joint_limit_margin_rad=request.min_joint_limit_margin_rad,
            )
        self._publish_ik_status(side, result)

        response.success = result.success
        response.status_code = (
            SolveArmIK.Response.STATUS_SUCCESS
            if result.success
            else SolveArmIK.Response.STATUS_IK_REJECTED
        )
        response.status = result.status
        response.message = result.message or 'IK solution validated'
        response.position_error_m = result.position_error_m
        response.orientation_error_rad = result.orientation_error_rad
        response.solve_time_ms = result.solve_time_ms
        response.validation_time_ms = result.validation_time_ms
        response.jacobian_condition = result.jacobian_condition
        response.min_joint_limit_margin_rad = result.min_joint_limit_margin_rad
        response.collision_checked = False
        response.collision = False
        if result.success:
            response.joint_solution.header.stamp = self.get_clock().now().to_msg()
            response.joint_solution.name = self._joint_names(side)
            response.joint_solution.position = result.joints.astype(float).tolist()
        return response

    def _seed_from_joint_state(
        self,
        side: str,
        joint_state: sensor_msgs.msg.JointState,
    ) -> np.ndarray:
        expected_names = self._joint_names(side)
        positions = np.asarray(joint_state.position, dtype=float)
        if len(positions) == 0:
            with self._lock:
                current = self.current_joints[side]
                return (
                    current.copy()
                    if current is not None
                    else self.last_solutions[side].copy()
                )
        if not np.all(np.isfinite(positions)):
            raise ValueError('current_joint_state contains non-finite positions')

        names = list(joint_state.name)
        if names:
            if len(names) != len(positions):
                raise ValueError('joint state name and position lengths differ')
            by_name = dict(zip(names, positions))
            missing = [name for name in expected_names if name not in by_name]
            if missing:
                raise ValueError(f'missing arm joints: {missing}')
            return np.asarray([by_name[name] for name in expected_names], dtype=float)
        if len(positions) == 7:
            return positions.copy()
        if len(positions) == 14:
            return positions[0:7].copy() if side == SIDE_LEFT else positions[7:14].copy()
        raise ValueError('unnamed joint state must contain 7 or 14 positions')

    @staticmethod
    def _joint_names(side: str) -> list[str]:
        return list(LEFT_JOINT_NAMES if side == SIDE_LEFT else RIGHT_JOINT_NAMES)

    @staticmethod
    def _reject_solve_response(
        response: SolveArmIK.Response,
        status_code: int,
        status: str,
        message: str,
    ) -> SolveArmIK.Response:
        response.success = False
        response.status_code = status_code
        response.status = status
        response.message = message
        response.collision_checked = False
        response.collision = False
        return response

    def _trajectory_goal_callback(
        self,
        goal: PlanArmTrajectory.Goal,
    ) -> GoalResponse:
        model = goal.robot_model.strip()
        side = goal.arm_side.strip().lower()
        safe_mode = goal.safe_return_mode.strip() or 'home'
        if model and model != 'elf3':
            return GoalResponse.REJECT
        if side not in (SIDE_LEFT, SIDE_RIGHT):
            return GoalResponse.REJECT
        if goal.minimum_duration_sec < 0.0:
            return GoalResponse.REJECT
        if goal.return_to_safe_on_cancel and safe_mode not in (
            'home',
            'handshake_ready',
        ):
            return GoalResponse.REJECT
        with self._lock:
            if self._action_active:
                return GoalResponse.REJECT
            self._action_active = True
        return GoalResponse.ACCEPT

    @staticmethod
    def _trajectory_cancel_callback(goal_handle) -> CancelResponse:
        del goal_handle
        return CancelResponse.ACCEPT

    async def _execute_trajectory_action(self, goal_handle):
        goal = goal_handle.request
        result = PlanArmTrajectory.Result()
        planning_started = time.perf_counter()
        try:
            solve_request = SolveArmIK.Request()
            solve_request.robot_model = goal.robot_model
            solve_request.arm_side = goal.arm_side
            solve_request.current_joint_state = goal.current_joint_state
            solve_request.target_pose = goal.target_pose
            solve_request.use_tcp_offset = goal.use_tcp_offset
            solve_request.tcp_offset = goal.tcp_offset
            solve_request.require_collision_check = goal.require_collision_check
            solve_response = self._solve_ik_callback(
                solve_request,
                SolveArmIK.Response(),
            )
            self._copy_solve_result_to_action(solve_response, result)
            if not solve_response.success:
                result.planning_time_ms = (
                    time.perf_counter() - planning_started
                ) * 1000.0
                goal_handle.abort()
                return result

            side = goal.arm_side.strip().lower()
            start_joints = self._seed_from_joint_state(
                side,
                goal.current_joint_state,
            )
            goal_joints = np.asarray(
                solve_response.joint_solution.position,
                dtype=float,
            )
            max_velocity = self._goal_value_or_default(
                goal.max_velocity_rad_s,
                self.trajectory_max_velocity_rad_s,
            )
            max_acceleration = self._goal_value_or_default(
                goal.max_acceleration_rad_s2,
                self.trajectory_max_acceleration_rad_s2,
            )
            control_period = self._goal_value_or_default(
                goal.control_period_sec,
                self.trajectory_control_period_sec,
            )
            planned = plan_smoothstep_trajectory(
                start=start_joints,
                goal=goal_joints,
                max_velocity_rad_s=max_velocity,
                max_acceleration_rad_s2=max_acceleration,
                control_period_sec=control_period,
                minimum_duration_sec=goal.minimum_duration_sec,
            )
            result.trajectory = self._to_trajectory_message(
                side,
                planned,
            )
            result.planning_time_ms = (
                time.perf_counter() - planning_started
            ) * 1000.0

            if not goal.execute:
                result.success = True
                result.status = 'planned'
                result.message = 'trajectory generated without execution'
                goal_handle.succeed()
                return result

            active = self._start_action_trajectory(
                side=side,
                start_joints=start_joints,
                goal_joints=goal_joints,
                duration=planned.duration_sec,
            )
            execution_started = time.monotonic()
            while rclpy.ok():
                now = time.monotonic()
                elapsed = now - execution_started
                commands = self._sample_trajectory(active, now)
                if goal_handle.is_cancel_requested:
                    stopped = self._stop_action_trajectory(active, commands)
                    safe_mode = goal.safe_return_mode.strip() or 'home'
                    if goal.return_to_safe_on_cancel:
                        self._start_safe_return(stopped, safe_mode)
                        result.status = 'cancelled_safe_return_started'
                        result.message = f'cancelled; started {safe_mode} return'
                    else:
                        result.status = 'cancelled'
                        result.message = 'trajectory execution cancelled'
                    result.success = False
                    goal_handle.canceled()
                    return result

                feedback = PlanArmTrajectory.Feedback()
                feedback.progress = float(
                    np.clip(elapsed / planned.duration_sec, 0.0, 1.0)
                )
                feedback.phase = 'executing'
                feedback.elapsed_ms = elapsed * 1000.0
                feedback.current_command.header.stamp = (
                    self.get_clock().now().to_msg()
                )
                feedback.current_command.name = self._joint_names(side)
                feedback.current_command.position = commands[side].tolist()
                goal_handle.publish_feedback(feedback)

                if elapsed >= planned.duration_sec:
                    break
                await self._async_wait(min(max(control_period, 0.005), 0.1))

            result.success = True
            result.status = 'completed'
            result.message = 'trajectory execution completed'
            goal_handle.succeed()
            return result
        except (ValueError, RuntimeError) as exc:
            result.success = False
            result.status = 'planning_error'
            result.message = str(exc)
            self.get_logger().error(f'trajectory action failed: {exc}')
            result.planning_time_ms = (
                time.perf_counter() - planning_started
            ) * 1000.0
            goal_handle.abort()
            return result
        finally:
            with self._lock:
                self._action_active = False

    @staticmethod
    def _goal_value_or_default(value: float, default: float) -> float:
        return float(value) if value > 0.0 else float(default)

    async def _async_wait(self, duration_sec: float) -> None:
        future = Future()

        def wake() -> None:
            if not future.done():
                future.set_result(None)

        timer = self.create_timer(float(duration_sec), wake)
        try:
            await future
        finally:
            self.destroy_timer(timer)

    @staticmethod
    def _copy_solve_result_to_action(
        solve: SolveArmIK.Response,
        result: PlanArmTrajectory.Result,
    ) -> None:
        result.success = solve.success
        result.status = solve.status
        result.message = solve.message
        result.position_error_m = solve.position_error_m
        result.orientation_error_rad = solve.orientation_error_rad
        result.solve_time_ms = solve.solve_time_ms
        result.collision_checked = solve.collision_checked
        result.collision = solve.collision
        result.collision_pairs = list(solve.collision_pairs)

    def _to_trajectory_message(
        self,
        side: str,
        planned: PlannedTrajectory,
    ) -> trajectory_msgs.msg.JointTrajectory:
        message = trajectory_msgs.msg.JointTrajectory()
        message.header.stamp = self.get_clock().now().to_msg()
        message.joint_names = self._joint_names(side)
        for index, time_sec in enumerate(planned.times_sec):
            point = trajectory_msgs.msg.JointTrajectoryPoint()
            point.positions = planned.positions[index].tolist()
            point.velocities = planned.velocities[index].tolist()
            point.accelerations = planned.accelerations[index].tolist()
            point.time_from_start = Duration(seconds=float(time_sec)).to_msg()
            message.points.append(point)
        return message

    def _start_action_trajectory(
        self,
        side: str,
        start_joints: np.ndarray,
        goal_joints: np.ndarray,
        duration: float,
    ) -> JointTrajectory:
        with self._lock:
            other_side = SIDE_RIGHT if side == SIDE_LEFT else SIDE_LEFT
            other = (
                self.current_joints[other_side].copy()
                if self.current_joints[other_side] is not None
                else self.last_solutions[other_side].copy()
            )
            active = JointTrajectory(
                start_left=start_joints.copy() if side == SIDE_LEFT else other,
                start_right=start_joints.copy() if side == SIDE_RIGHT else other,
                goal_left=goal_joints.copy() if side == SIDE_LEFT else other,
                goal_right=goal_joints.copy() if side == SIDE_RIGHT else other,
                start_time=time.monotonic(),
                duration=float(duration),
                name=f'{side}_action',
            )
            self.active_trajectory = active
        return active

    def _stop_action_trajectory(
        self,
        active: JointTrajectory,
        commands: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        with self._lock:
            use_measured = (
                time.monotonic() - self.last_joint_state_time
                <= self.joint_state_timeout_sec
            )
            stopped = {
                side: (
                    self.current_joints[side].copy()
                    if use_measured and self.current_joints[side] is not None
                    else commands[side].copy()
                )
                for side in (SIDE_LEFT, SIDE_RIGHT)
            }
            if self.active_trajectory is active:
                self.active_trajectory = None
            self.last_solutions = {
                SIDE_LEFT: stopped[SIDE_LEFT].copy(),
                SIDE_RIGHT: stopped[SIDE_RIGHT].copy(),
            }
        return stopped

    def _start_safe_return(
        self,
        commands: dict[str, np.ndarray],
        mode: str,
    ) -> None:
        goal_left = np.asarray(LEFT_HOME, dtype=float)
        goal_right = (
            self.handshake_right_joints.copy()
            if mode == 'handshake_ready'
            else np.asarray(RIGHT_HOME, dtype=float)
        )
        start = np.r_[commands[SIDE_LEFT], commands[SIDE_RIGHT]]
        goal = np.r_[goal_left, goal_right]
        planned = plan_smoothstep_trajectory(
            start=start,
            goal=goal,
            max_velocity_rad_s=self.trajectory_max_velocity_rad_s,
            max_acceleration_rad_s2=self.trajectory_max_acceleration_rad_s2,
            control_period_sec=self.trajectory_control_period_sec,
        )
        with self._lock:
            self.active_trajectory = JointTrajectory(
                start_left=commands[SIDE_LEFT].copy(),
                start_right=commands[SIDE_RIGHT].copy(),
                goal_left=goal_left,
                goal_right=goal_right,
                start_time=time.monotonic(),
                duration=planned.duration_sec,
                name=f'safe_return_{mode}',
            )

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
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
