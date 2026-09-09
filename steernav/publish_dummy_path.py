#!/usr/bin/env python3
import numpy as np
from PIL import Image as PILImage
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, Float32MultiArray, Empty
from nav_msgs.msg import Path, Odometry

IMAGE_TOPIC = "/camera/camera/color/image_raw/compressed"
ODOM_TOPIC = "/a200_0648/platform/odom/filtered"
WAYPOINT_TOPIC = "/path"
OVERLAY_TOPIC = "/overlay"

class TestPathPublisher(Node):

    def __init__(self):
        super().__init__("test_path_publisher")

        self._started_sent = False
        self.pub_started = self.create_publisher(Empty, "/started", 10)
        self.robot_velocity_base = np.zeros(3, dtype=np.float64)
        self.robot_angular_velocity_base = np.zeros(3, dtype=np.float64)

        # ROS 2 Topics
        self.odom_sub = self.create_subscription(
            Odometry, ODOM_TOPIC, self.odom_callback_obs,
            qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                   history=QoSHistoryPolicy.KEEP_LAST,
                                   depth=10))
        self.path_pub = self.create_publisher(Path, WAYPOINT_TOPIC, 10)
        # self.trajectory_visual_pub = self.create_publisher(
        #     Image, OVERLAY_TOPIC, qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
        #                                                  history=QoSHistoryPolicy.KEEP_LAST,
        #                                                  depth=10))

        # self.timer = self.create_timer(0.1, self.publish_path)
        self.publish_path()
        self.get_logger().info(f"Publishing dummy path on {WAYPOINT_TOPIC}")

        # Publish /started once, when we actually start inferencing
        if not self._started_sent:
            self._started_sent = True
            self._have_cur_img = False
            self._have_cur_pose = False
            self.pub_started.publish(Empty())
            self.get_logger().info("Published /started (once).")

    def odom_callback_obs(self, msg: Odometry):
        # self.get_logger().info("Reached Odom callback!")
        self.robot_velocity_base[:] = [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ]

        self.robot_angular_velocity_base[:] = [
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ]

    def _to_path_msg(self, path_xy: np.ndarray) -> Path:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"  # semantic: "start frame"

        for x, y in path_xy:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        return msg

    def generate_waypoints(self, end=(2, 2), n=20):
        start = np.array([0.0, 0.0])
        end = np.array(end, dtype=float)

        t = np.linspace(0, 1, n)
        s = 3 * t ** 2 - 2 * t ** 3  # smoothstep

        waypoints = start + s[:, None] * (end - start)
        return waypoints

    def publish_path(self):
        path_msg = Path()
        now = self.get_clock().now().to_msg()
        path_msg.header.stamp = now
        path_msg.header.frame_id = "base_link"
        end_point = np.array([2.0, 2.0])
        dummy_waypoints = self.generate_waypoints(end_point)
        print(f"end point: {end_point}")
        print(f"dummy waypoints: {dummy_waypoints}")
        self.path_pub.publish(self._to_path_msg(dummy_waypoints))


def main(args=None):
    rclpy.init(args=args)

    node = TestPathPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()