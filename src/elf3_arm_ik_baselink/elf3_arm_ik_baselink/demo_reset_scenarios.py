"""Demo sequence for arbitrary-pose reset and handshake-ready reset."""

from __future__ import annotations

import argparse
import time
from typing import Optional

import rclpy
from rclpy.node import Node
import std_srvs.srv
import visualization_msgs.msg


class ResetScenarioDemo(Node):
    """Drive the RViz demo through customer-facing reset scenarios."""

    def __init__(self) -> None:
        super().__init__('arm_ik_reset_scenario_demo')
        self.status_pub = self.create_publisher(
            visualization_msgs.msg.Marker,
            'arm_ik/demo_text',
            10,
        )
        self.go_home = self.create_client(
            std_srvs.srv.Trigger,
            'arm_ik/go_home',
        )
        self.go_handshake = self.create_client(
            std_srvs.srv.Trigger,
            'arm_ik/go_handshake_ready',
        )
        self.go_demo_pose_a = self.create_client(
            std_srvs.srv.Trigger,
            'arm_ik/go_demo_pose_a',
        )
        self.go_demo_pose_b = self.create_client(
            std_srvs.srv.Trigger,
            'arm_ik/go_demo_pose_b',
        )

    def run(self) -> None:
        self.get_logger().info('waiting for reset services')
        self.publish_status_for('Waiting-Services', 0.6)
        self.go_home.wait_for_service(timeout_sec=10.0)
        self.go_handshake.wait_for_service(timeout_sec=10.0)
        self.go_demo_pose_a.wait_for_service(timeout_sec=10.0)
        self.go_demo_pose_b.wait_for_service(timeout_sec=10.0)
        self.publish_status_for('Demo-Ready', 0.6)

        loop_count = 1
        while rclpy.ok():
            prefix = f'L{loop_count}'
            self.run_loop(prefix)
            loop_count += 1
            self.publish_status_for('Next-Loop', 0.8)

    def run_loop(self, prefix: str) -> None:
        step = f'{prefix} 1/4:Pose-A'
        self.get_logger().info(step)
        elapsed_ms = self.call_trigger(self.go_demo_pose_a)
        self.publish_status_for(f'{step} calc={elapsed_ms:.1f}ms', 3.0)

        step = f'{prefix} 2/4:Reset-Home'
        self.get_logger().info(step)
        elapsed_ms = self.call_trigger(self.go_home)
        self.publish_status_for(f'{step} calc={elapsed_ms:.1f}ms', 4.8)

        step = f'{prefix} 3/4:Pose-B'
        self.get_logger().info(step)
        elapsed_ms = self.call_trigger(self.go_demo_pose_b)
        self.publish_status_for(f'{step} calc={elapsed_ms:.1f}ms', 3.0)

        step = f'{prefix} 4/4:Reset-Hand'
        self.get_logger().info(step)
        elapsed_ms = self.call_trigger(self.go_handshake)
        self.publish_status_for(f'{step} calc={elapsed_ms:.1f}ms', 4.8)
        self.get_logger().info(f'{prefix} demo loop finished')

    def publish_status_for(self, text: str, seconds: float) -> None:
        end_time = time.monotonic() + seconds
        while time.monotonic() < end_time:
            self.publish_status(text)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.1)

    def publish_status(self, text: str) -> None:
        marker = visualization_msgs.msg.Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'torso_link'
        marker.ns = 'demo_status'
        marker.id = 0
        marker.type = visualization_msgs.msg.Marker.TEXT_VIEW_FACING
        marker.action = visualization_msgs.msg.Marker.ADD
        marker.pose.position.x = 0.65
        marker.pose.position.y = 0.0
        marker.pose.position.z = -0.45
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.075
        marker.color.r = 0.0
        marker.color.g = 0.95
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.lifetime.sec = 1
        marker.text = text
        self.status_pub.publish(marker)

    def call_trigger(self, client) -> float:
        started = time.perf_counter()
        future = client.call_async(std_srvs.srv.Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = future.result()
        if result is None:
            raise RuntimeError('service call timed out')
        if not result.success:
            raise RuntimeError(result.message)
        self.get_logger().info(f'{result.message}; calc={elapsed_ms:.1f}ms')
        return elapsed_ms


def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run one loop and exit. Default is to repeat forever.',
    )
    parsed_args, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    node = ResetScenarioDemo()
    try:
        if parsed_args.once:
            node.go_home.wait_for_service(timeout_sec=10.0)
            node.go_handshake.wait_for_service(timeout_sec=10.0)
            node.go_demo_pose_a.wait_for_service(timeout_sec=10.0)
            node.go_demo_pose_b.wait_for_service(timeout_sec=10.0)
            node.run_loop('L1')
        else:
            node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
