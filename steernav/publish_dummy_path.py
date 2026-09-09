#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
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
# WAYPOINT_TOPIC = "/path"
WAYPOINT_TOPIC = "/husky/policy_path"  # this lets dummy path to be steered
OVERLAY_TOPIC = "/overlay"

class TestPathPublisher(Node):

    def __init__(self, end_pt):
        super().__init__("test_path_publisher")

        self._started_sent = False
        self.pub_started = self.create_publisher(Empty, "/started", 10)
        self.robot_velocity_base = np.zeros(3, dtype=np.float64)
        self.robot_angular_velocity_base = np.zeros(3, dtype=np.float64)
        self.end_point = end_pt
        # ROS 2 Topics
        # self.odom_sub = self.create_subscription(
        #     Odometry, ODOM_TOPIC, self.odom_callback_obs,
        #     qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
        #                            history=QoSHistoryPolicy.KEEP_LAST,
        #                            depth=10))
        self.path_pub = self.create_publisher(Path, WAYPOINT_TOPIC, 10)
        # self.trajectory_visual_pub = self.create_publisher(
        #     Image, OVERLAY_TOPIC, qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
        #                                                  history=QoSHistoryPolicy.KEEP_LAST,
        #                                                  depth=10))

        self.timer = self.create_timer(0.1, self.publish_path)
        # self.publish_path()
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

    def publish_path(self):
        path_msg = Path()
        now = self.get_clock().now().to_msg()
        path_msg.header.stamp = now
        path_msg.header.frame_id = "base_link"
        dummy_waypoints, _ = generate_waypoints(self.end_point)
        print(f"end point: {self.end_point}")
        print(f"dummy waypoints: {dummy_waypoints}")
        self.path_pub.publish(self._to_path_msg(dummy_waypoints))

def generate_waypoints(end=(2, 2), n=20, ):
    # NOTE: we assume robot faces FORWARD. else adjust "forward_distance" logic
    start = np.array([0.0, 0.0])
    end = np.array(end, dtype=float)

    # Initial robot heading: north (+y)
    heading = np.array([0.0, 1.0])
    # Start and end points
    p0, p3 = start, end
    forward_distance = max(1, (end[1] - start[1])// 2)
    turn_distance = (end[0] - start[0]) // 2
    # Enforces initial tangent pointing north
    p1 = p0 + forward_distance * heading
    # Controls how the path bends into the endpoint
    p2 = p3 - np.array([turn_distance, 0.0])
    t = np.linspace(0.0, 1.0, n)[:, None]
    waypoints = (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2
            + t ** 3 * p3
    )
    # you just need waypoints, rest of the pts are for plotting
    details = (p0, p1, p2, p3, n)
    return waypoints, details

def plot_dummy_path(end_pt):
    # Generate path data
    waypoints, details = generate_waypoints(end_pt)
    p0, p1, p2, p3, n = details
    # Plot setup
    plt.figure(figsize=(7, 6))

    # Plot Bézier curve & waypoints
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'b-', label='Trajectory Path', zorder=2)
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'bo', markersize=4, label=f"Waypoints (n={n})", zorder=3)

    # Plot control polygon & points
    control_pts = np.array([p0, p1, p2, p3])
    plt.plot(control_pts[:, 0], control_pts[:, 1], 'r--', alpha=0.5, label='Control Polygon', zorder=1)
    plt.scatter(control_pts[:, 0], control_pts[:, 1], color='red', s=50, zorder=4)

    # Annotate control points
    labels = ['$p_0$ (Start)', '$p_1$', '$p_2$', '$p_3$ (End)']
    for i, txt in enumerate(labels):
        plt.annotate(f"{txt}\n{tuple(control_pts[i])}", (control_pts[i, 0], control_pts[i, 1]),
                     textcoords="offset points", xytext=(10, -5), ha='left')

    plt.title('Generated Waypoint Trajectory')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')
    plt.legend()
    plt.show()

def main(args=None):
    end_point = (0, 2)
    plot_dummy_path(end_point)

    rclpy.init(args=args)
    node = TestPathPublisher(end_point)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()