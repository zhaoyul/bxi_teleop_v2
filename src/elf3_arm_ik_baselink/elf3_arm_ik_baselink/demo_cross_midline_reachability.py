"""RViz demo of right-arm handshake reach across the body centerline."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Optional

from ament_index_python.packages import get_package_share_directory
from elf3_arm_ik_interfaces.action import PlanArmTrajectory
import geometry_msgs.msg
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
import std_srvs.srv
import visualization_msgs.msg

from .constants import RIGHT_JOINT_NAMES
from .ik_solver import Elf3ArmIkSolver
from .math_utils import rotation_matrix_to_quaternion_xyzw
from .reachability import (
    HANDSHAKE_ORIENTATION,
    RIGHT_HOME,
    TCP_OFFSET,
    ReachabilitySample,
    cross_midline_targets,
    scan_right_arm_reachability,
    select_boundary_demo_samples,
)


class CrossMidlineReachabilityDemo(Node):
    """Show the right arm reaching several fixed-orientation body-left points."""

    def __init__(self) -> None:
        super().__init__('right_arm_cross_midline_reachability_demo')
        share = Path(get_package_share_directory('elf3_arm_ik_baselink'))
        self.solver = Elf3ArmIkSolver(str(share / 'data'))
        self.marker_pub = self.create_publisher(
            visualization_msgs.msg.MarkerArray,
            'arm_ik/reachability_markers',
            10,
        )
        self.status_pub = self.create_publisher(
            visualization_msgs.msg.Marker,
            'arm_ik/demo_text',
            10,
        )
        self.go_home = self.create_client(std_srvs.srv.Trigger, 'arm_ik/go_home')
        self.action = ActionClient(
            self,
            PlanArmTrajectory,
            'arm_ik/plan_trajectory',
        )
        self.samples: list[ReachabilitySample] = []
        self.demo_samples: list[ReachabilitySample] = []
        self.active_sample: Optional[ReachabilitySample] = None
        self.current_right = RIGHT_HOME.copy()
        self.active_label = ''

    def prepare(self) -> None:
        self.publish_status('SCANNING RIGHT-ARM CROSS-MIDLINE REGION')
        self.samples = scan_right_arm_reachability(
            self.solver,
            cross_midline_targets(),
        )
        self.demo_samples = select_boundary_demo_samples(self.samples)
        reachable = sum(sample.reachable for sample in self.samples)
        self.publish_markers()
        self.get_logger().info(
            f'reachability scan: {reachable}/{len(self.samples)} points reachable; '
            f'{len(self.demo_samples)} selected for motion'
        )

    def run(self, once: bool) -> None:
        self.prepare()
        if not self.go_home.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('arm_ik/go_home service is unavailable')
        if not self.action.wait_for_server(timeout_sec=10.0):
            raise RuntimeError('arm_ik/plan_trajectory action is unavailable')

        loop = 1
        while rclpy.ok():
            try:
                self.run_loop(loop)
            except RuntimeError as exc:
                self.get_logger().error(f'loop {loop} failed: {exc}')
                self.active_sample = None
                self.publish_status(
                    f'L{loop}|RECOVERING-TO-HOME'
                )
                self.publish_markers()
                if once:
                    raise
                try:
                    self.call_home()
                except RuntimeError as recovery_error:
                    self.get_logger().error(
                        f'loop {loop} recovery failed: {recovery_error}'
                    )
                self.hold_status(f'L{loop}|RECOVERING-TO-HOME', 5.0)
            if once:
                return
            loop += 1

    def run_loop(self, loop: int) -> None:
        self.call_home()
        self.hold_status(
            f'L{loop}|RESET-BOTH-ARMS-HOME',
            4.5,
        )
        self.current_right = RIGHT_HOME.copy()

        total = len(self.demo_samples)
        for index, sample in enumerate(self.demo_samples, start=1):
            self.active_sample = sample
            position = sample.position
            self.active_label = f'L{loop} P{index}/{total}'
            self.publish_markers()
            result = self.execute_target(sample)
            if not result.success:
                raise RuntimeError(
                    f'target {position.tolist()} failed: {result.status} '
                    f'{result.message}'
                )
            self.current_right = np.asarray(
                result.trajectory.points[-1].positions,
                dtype=float,
            )
            self.get_logger().info(
                f'target {index}/{total} reached: '
                f'y={position[1]:+.2f} z={position[2]:+.2f} '
                f'IK={result.solve_time_ms:.2f}ms '
                f'error={result.position_error_m * 1000.0:.2f}mm'
            )
            self.hold_status(
                f'{self.active_label}|'
                f'IK={result.solve_time_ms:.1f}ms|'
                f'E={result.position_error_m * 1000.0:.1f}mm',
                1.4,
            )

        self.active_sample = None
        self.publish_markers()
        self.hold_status(
            f'L{loop}|CROSS-MIDLINE-DEMO-COMPLETE',
            1.0,
        )

    def call_home(self) -> None:
        future = self.go_home.call_async(std_srvs.srv.Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        response = future.result()
        if response is None or not response.success:
            message = response.message if response is not None else 'timeout'
            raise RuntimeError(f'go_home failed: {message}')

    def execute_target(
        self,
        sample: ReachabilitySample,
    ) -> PlanArmTrajectory.Result:
        quaternion = rotation_matrix_to_quaternion_xyzw(HANDSHAKE_ORIENTATION)
        goal = PlanArmTrajectory.Goal()
        goal.robot_model = 'elf3'
        goal.arm_side = 'right'
        goal.current_joint_state.name = list(RIGHT_JOINT_NAMES)
        goal.current_joint_state.position = self.current_right.tolist()
        goal.target_pose.header.frame_id = 'base_link'
        goal.target_pose.pose.position.x = float(sample.position[0])
        goal.target_pose.pose.position.y = float(sample.position[1])
        goal.target_pose.pose.position.z = float(sample.position[2])
        goal.target_pose.pose.orientation.x = float(quaternion[0])
        goal.target_pose.pose.orientation.y = float(quaternion[1])
        goal.target_pose.pose.orientation.z = float(quaternion[2])
        goal.target_pose.pose.orientation.w = float(quaternion[3])
        goal.use_tcp_offset = True
        goal.tcp_offset.x = float(TCP_OFFSET[0])
        goal.tcp_offset.y = float(TCP_OFFSET[1])
        goal.tcp_offset.z = float(TCP_OFFSET[2])
        goal.max_velocity_rad_s = 0.45
        goal.max_acceleration_rad_s2 = 0.90
        goal.control_period_sec = 0.02
        goal.minimum_duration_sec = 1.6
        goal.require_collision_check = False
        goal.execute = True
        goal.return_to_safe_on_cancel = True
        goal.safe_return_mode = 'home'

        send_future = self.action.send_goal_async(
            goal,
            feedback_callback=self.on_feedback,
        )
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError('trajectory action goal was rejected')
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
        wrapped = result_future.result()
        if wrapped is None:
            cancel_future = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
            raise RuntimeError('trajectory action result timed out')
        return wrapped.result

    def on_feedback(self, message) -> None:
        feedback = message.feedback
        self.publish_status(
            f'{self.active_label}|MOVE={feedback.progress * 100.0:.0f}%'
        )
        self.publish_markers()

    def hold_status(self, text: str, seconds: float) -> None:
        end_time = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end_time:
            self.publish_status(text)
            self.publish_markers()
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.08)

    def publish_status(self, text: str) -> None:
        marker = self._marker(0, visualization_msgs.msg.Marker.TEXT_VIEW_FACING)
        marker.ns = 'demo_status'
        marker.pose.position.x = 0.90
        marker.pose.position.y = 0.0
        marker.pose.position.z = -0.36
        marker.scale.z = 0.045
        marker.color.r = 0.0
        marker.color.g = 0.95
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.lifetime.sec = 1
        marker.text = text
        self.status_pub.publish(marker)

    def publish_markers(self) -> None:
        if not self.samples:
            return
        marker_array = visualization_msgs.msg.MarkerArray()
        marker_array.markers.extend(self._region_markers())
        for index, sample in enumerate(self.samples):
            marker_array.markers.extend(self._sample_markers(index, sample))
        marker_array.markers.extend(self._boundary_markers())
        self.marker_pub.publish(marker_array)

    def _region_markers(self) -> list[visualization_msgs.msg.Marker]:
        plane = self._marker(1000, visualization_msgs.msg.Marker.CUBE)
        plane.ns = 'reachability_region'
        plane.pose.position.x = 0.34
        plane.pose.position.y = 0.12
        plane.pose.position.z = 0.08
        plane.scale.x = 0.012
        plane.scale.y = 0.25
        plane.scale.z = 0.22
        plane.color.r = 0.1
        plane.color.g = 0.5
        plane.color.b = 1.0
        plane.color.a = 0.10

        centerline = self._marker(1001, visualization_msgs.msg.Marker.LINE_STRIP)
        centerline.ns = 'body_centerline'
        centerline.scale.x = 0.009
        centerline.color.r = 0.0
        centerline.color.g = 0.9
        centerline.color.b = 1.0
        centerline.color.a = 1.0
        centerline.points = [
            self._point(0.34, 0.0, -0.02),
            self._point(0.34, 0.0, 0.19),
        ]

        title = self._text_marker(
            1002,
            y=0.34,
            z=0.21,
            text='RIGHT-ARM->ROBOT-LEFT',
            color=(1.0, 1.0, 1.0),
        )
        reachable_label = self._text_marker(
            1003,
            y=0.34,
            z=0.17,
            text='GREEN=REACHABLE',
            color=(0.1, 1.0, 0.2),
        )
        rejected_label = self._text_marker(
            1004,
            y=0.34,
            z=0.13,
            text='RED=REJECTED',
            color=(1.0, 0.12, 0.05),
        )

        centerline_label = self._marker(
            1005,
            visualization_msgs.msg.Marker.TEXT_VIEW_FACING,
        )
        centerline_label.ns = 'body_centerline'
        centerline_label.pose.position.x = 0.34
        centerline_label.pose.position.y = 0.0
        centerline_label.pose.position.z = 0.205
        centerline_label.scale.z = 0.024
        centerline_label.color.r = 0.0
        centerline_label.color.g = 0.9
        centerline_label.color.b = 1.0
        centerline_label.color.a = 1.0
        centerline_label.text = 'BODY-CENTER:y=0'
        return [
            plane,
            centerline,
            title,
            reachable_label,
            rejected_label,
            centerline_label,
        ]

    def _sample_markers(
        self,
        index: int,
        sample: ReachabilitySample,
    ) -> list[visualization_msgs.msg.Marker]:
        active = self.active_sample is sample
        sphere = self._marker(index, visualization_msgs.msg.Marker.SPHERE)
        sphere.ns = 'reachability_points'
        sphere.pose.position = self._point(*sample.position)
        scale = 0.052 if active else 0.036
        sphere.scale.x = scale
        sphere.scale.y = scale
        sphere.scale.z = scale
        if active:
            sphere.color.r = 1.0
            sphere.color.g = 0.85
            sphere.color.b = 0.0
        elif sample.reachable:
            sphere.color.r = 0.1
            sphere.color.g = 1.0
            sphere.color.b = 0.2
        else:
            sphere.color.r = 1.0
            sphere.color.g = 0.12
            sphere.color.b = 0.05
        sphere.color.a = 1.0

        return [sphere]

    def _boundary_markers(self) -> list[visualization_msgs.msg.Marker]:
        boundary = self._marker(1100, visualization_msgs.msg.Marker.LINE_STRIP)
        boundary.ns = 'reachability_boundary'
        boundary.scale.x = 0.012
        boundary.color.r = 1.0
        boundary.color.g = 0.55
        boundary.color.b = 0.0
        boundary.color.a = 1.0
        for z in sorted({round(float(sample.position[2]), 6) for sample in self.samples}):
            row = [
                sample
                for sample in self.samples
                if sample.reachable and abs(float(sample.position[2]) - z) < 1e-6
            ]
            if row:
                farthest = max(row, key=lambda sample: float(sample.position[1]))
                boundary.points.append(self._point(*farthest.position))

        path = self._marker(1101, visualization_msgs.msg.Marker.LINE_STRIP)
        path.ns = 'demo_target_path'
        path.scale.x = 0.006
        path.color.r = 0.2
        path.color.g = 0.6
        path.color.b = 1.0
        path.color.a = 0.8
        path.points = [self._point(*sample.position) for sample in self.demo_samples]

        markers = [boundary, path]
        if self.active_sample is not None:
            arrow = self._marker(1102, visualization_msgs.msg.Marker.ARROW)
            arrow.ns = 'active_handshake_orientation'
            arrow.scale.x = 0.012
            arrow.scale.y = 0.025
            arrow.scale.z = 0.025
            arrow.color.r = 1.0
            arrow.color.g = 0.85
            arrow.color.b = 0.0
            arrow.color.a = 1.0
            start = self.active_sample.position
            end = start + HANDSHAKE_ORIENTATION[:, 0] * 0.10
            arrow.points = [self._point(*start), self._point(*end)]
            markers.append(arrow)
        return markers

    def _marker(self, marker_id: int, marker_type: int):
        marker = visualization_msgs.msg.Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'torso_link'
        marker.id = marker_id
        marker.type = marker_type
        marker.action = visualization_msgs.msg.Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _text_marker(
        self,
        marker_id: int,
        y: float,
        z: float,
        text: str,
        color: tuple[float, float, float],
    ) -> visualization_msgs.msg.Marker:
        marker = self._marker(
            marker_id,
            visualization_msgs.msg.Marker.TEXT_VIEW_FACING,
        )
        marker.ns = 'reachability_legend'
        marker.pose.position.x = 0.34
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.scale.z = 0.024
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0
        marker.text = text
        return marker

    @staticmethod
    def _point(x: float, y: float, z: float) -> geometry_msgs.msg.Point:
        point = geometry_msgs.msg.Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)
        return point


def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run one reachability sequence and exit.',
    )
    parsed_args, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = CrossMidlineReachabilityDemo()
    try:
        node.run(once=parsed_args.once)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
