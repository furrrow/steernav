#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


WAYPOINT_TOPIC = "/ghost/policy_path"


class TestPathPublisher(Node):

    def __init__(self):
        super().__init__("test_path_publisher")
        self.path_pub = self.create_publisher(Path, WAYPOINT_TOPIC, 10)

        # Publish every 0.1 seconds (10 Hz)
        self.timer = self.create_timer(0.1, self.publish_path)
        self.get_logger().info(f"Publishing test path on {WAYPOINT_TOPIC}")

    def publish_path(self):
        path_msg = Path()
        now = self.get_clock().now().to_msg()
        path_msg.header.stamp = now
        path_msg.header.frame_id = "base_link"

        dummy_waypoints = [
            (0.6, -0.0),
            (1.2, -0.1),
            (1.8, -0.1),
            (2.3, -0.2),
            (2.8, -0.2),
            (3.4, -0.3),
            (4.0, -0.3),
            (4.6, -0.4),
        ]

        # Add each waypoint to the Path
        for x, y in dummy_waypoints:
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = "base_link"

            # Position
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            # Identity orientation
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0

            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)


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