#!/usr/bin/env python3
"""
overlay node to replace "visualize_path" in ros_inference
a work in progress.... NOT READY!!

"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from typing import Optional
import cv2
import numpy as np
import rclpy
import tf2_ros
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from cv_bridge import CvBridge
from nav_msgs.msg import Path
from sensor_msgs.msg import Image, CompressedImage

from custom_utils.io_utils import load_calibration, overlay_path


class VisualOverlayNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("visual_overlay_node")

        self.args = args
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.latest_image: Optional[np.ndarray] = None
        self.latest_path: Optional[np.ndarray] = None
        self.image_stamp = 0
        self.last_rendered_policy_stamp_ns = -1

        # image and path queues that contain:
        # (timestamp, image)
        # (timestamp, path_xy)
        # (timestamp, path_xy)
        self.image_queue = deque(maxlen=150)
        self.policy_path_queue = deque(maxlen=300)
        self.steered_path_queue = deque(maxlen=300)

        # CONSTANTS
        parent_dir = "/home/jim/Projects/steernav"
        # parent_dir = "/home/gamma-nav/Documents/Projects/git_repos/steernav"
        # parent_dir = "/workspace/steernav"
        DEPLOY_CONFIG_PATH = f"{parent_dir}/steernav/config/robot.yaml"
        MODEL_CONFIG_PATH = "config/models.yaml"
        CAMERA_MATRIX_DIR = f"{parent_dir}/steernav/cam_matrix.json"
        with open(DEPLOY_CONFIG_PATH, "r") as f:
            deploy_config = yaml.safe_load(f)
        self.rate = deploy_config["frame_rate"]
        self.waypoint_idx = deploy_config['waypoint_idx']
        robot_config = deploy_config[args.robot]
        print(f"using robot config for: {args.robot}")
        self.max_v = robot_config["max_v"]
        self.max_w = robot_config["max_w"]
        args.robot_radius = robot_config["robot_radius"]
        self.original_img_size = (deploy_config["img_w"], deploy_config["img_h"])  # (1280, 720)
        self.shrink_img_size = (deploy_config["shrink_w"], deploy_config["shrink_h"])  # (640, 480)
        self.robot_velocity_base = np.zeros(3, dtype=np.float64)
        self.robot_angular_velocity_base = np.zeros(3, dtype=np.float64)
        self.dt = 1 / self.rate
        self.reached_goal = False
        self.path_frame_id = "base_link"
        self._started_sent = False
        self.show_time_performance = True

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self,
        )
        self.inference_count = 0
        self.inference_start_time = time.perf_counter()
        self.plotting_steered_path = False

        # ROS Topics
        OVERLAY_TOPIC = deploy_config['overlay_topic'] # no robot, just /path_overlay
        IMAGE_TOPIC = robot_config['image_topic']
        ODOM_TOPIC = robot_config['odom_topic']
        self.compressed_img_topic = True if "compressed" in IMAGE_TOPIC else False
        print(f"IMAGE_TOPIC: {IMAGE_TOPIC} compressed_img_topic: {self.compressed_img_topic}")
        POLICY_PATH_TOPIC = robot_config['policy_path_topic']
        STEERED_PATH_TOPIC = robot_config['steered_path_topic']

        self.compressed = "compressed" in IMAGE_TOPIC.lower()

        # load_calibration() in the steering node returns:
        # cam_matrix, dist_coeffs, T_base_from_cam
        self.cam_matrix, self.dist_coeffs, self.T_base_from_cam = load_calibration(CAMERA_MATRIX_DIR)
        self.T_cam_from_base = np.linalg.inv(self.T_base_from_cam)

        best_effort_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ROS 2 Topics
        # subscribers:
        msg_type = CompressedImage if self.compressed_img_topic else Image
        self.image_sub = self.create_subscription(
            msg_type, IMAGE_TOPIC, self.image_callback,
            qos_profile=reliable_qos)
        # self.odom_sub = self.create_subscription(
        #     Odometry, ODOM_TOPIC, self.odom_callback_obs,
        #     qos_profile=reliable_qos)
        self.policy_path_sub = self.create_subscription(
            Path, POLICY_PATH_TOPIC, self.policy_path_callback,
            qos_profile=reliable_qos)
        self.steered_path_sub = self.create_subscription(
            Path, STEERED_PATH_TOPIC, self.steered_path_callback,
            qos_profile=reliable_qos)

        self.overlay_pub = self.create_publisher(
            Image, OVERLAY_TOPIC, qos_profile=reliable_qos)
        self.timer = self.create_timer(1.0 / args.rate, self.render_callback)

        self.get_logger().info(
            f"Visual overlay node started\n"
            f"  image:   {IMAGE_TOPIC}\n"
            f"  overlay: {OVERLAY_TOPIC}\n"
            f"  rate:    {args.rate:.1f} Hz\n"
            f"  compressed image: {self.compressed}"
        )

    @staticmethod
    def stamp_to_ns(stamp) -> int:
        """Convert ROS builtin_interfaces/msg/Time to integer nanoseconds."""
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def image_callback(self, msg):
        try:
            if self.compressed:
                bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            else:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            timestamp_ns = self.stamp_to_ns(msg.header.stamp)
            self.image_queue.append((timestamp_ns, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), msg.header.frame_id))

        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")

    def policy_path_callback(self, msg: Path):
        if not msg.poses:
            return
        # self.get_logger().info("Reached policy_path_callback!")
        path_timestamp_ns = self.stamp_to_ns(msg.header.stamp)
        path_xy = [(pose_stamped.pose.position.x, pose_stamped.pose.position.y) for pose_stamped in msg.poses]
        self.policy_path_queue.append((path_timestamp_ns, path_xy))

    def steered_path_callback(self, msg: Path):
        if not msg.poses:
            return
        # self.get_logger().info("Reached steered_path_callback!")
        path_timestamp_ns = self.stamp_to_ns(msg.header.stamp)
        path_xy = [(pose_stamped.pose.position.x, pose_stamped.pose.position.y) for pose_stamped in msg.poses]
        self.steered_path_queue.append((path_timestamp_ns, path_xy))

    def get_matching_image_and_paths(self, max_dt_sec: float = 5.0):
        # max_dt_sec: max time difference allowed
        # returns everything as None if no match found
        max_dt_ns = int(max_dt_sec * 1e9)
        adjusted_path = None
        # policy_path as anchor:
        target_ns, policy_path = self.policy_path_queue[-1]

        # check steered_path_queue
        min_steer_idx = np.argmin(np.abs([stamp_ns - target_ns for (stamp_ns, path) in self.steered_path_queue]))
        steered_stamp_ns, steered_path = self.steered_path_queue[min_steer_idx]
        steered_dt_ns = abs(steered_stamp_ns - target_ns)
        print(f"steered_dt s:{steered_dt_ns/1e9:.3f} min_steer_idx {min_steer_idx}")
        # check of closest unadjusted path time is too old/new
        if steered_dt_ns < max_dt_ns:
            adjusted_path = steered_path
            print(f"closest steered path idx:{min_steer_idx}")
        else:
            print(f"steered_dt s:{steered_dt_ns/1e9:.3f}")
            print(f"max_dt s:{max_dt_ns/1e9:.3f}")
            print(f"no path match, returning all none")
            return None, None, None, None
        # check image_queue
        min_img_idx = np.argmin(
            np.abs([stamp_ns - target_ns for (stamp_ns, path, fr_id) in self.image_queue]))
        image_stamp_ns, image, fr_id = self.image_queue[min_img_idx]
        image_dt_ns = abs(image_stamp_ns - target_ns)
        print(f"image_dt s:{image_dt_ns / 1e9:.3f} min_img_idx {min_img_idx}")
        # check of closest img time is too old/new
        if image_dt_ns > max_dt_ns:
            print(f"no img match, returning all none")
            return None, None, None, None

        return image, policy_path, adjusted_path, target_ns

    def render_callback(self):
        # Copy references/snapshot while holding the lock very briefly.
        with self.lock:
            if len(self.image_queue) == 0:
                return
            if len(self.policy_path_queue) == 0:
                return
            if len(self.steered_path_queue) == 0:
                return
            image, policy_path, adjusted_path, policy_stamp_ns = self.get_matching_image_and_paths()
            # Do not redraw the same camera frame if last drawn frame is within n seconds (to nanosec)
            if image is None:
                return
            if policy_stamp_ns <= self.last_rendered_policy_stamp_ns:
                return

        try:
            if adjusted_path is not None:
                path = np.stack((policy_path, adjusted_path), axis=0)
            else:
                path = policy_path
            resized = cv2.resize(image, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)
            overlay_bgr = overlay_path(trajectories=np.array(path),
                                       img=resized,
                                       cam_matrix=self.cam_matrix,
                                       T_cam_from_base=self.T_cam_from_base,)

            msg = self.bridge.cv2_to_imgmsg(
                overlay_bgr,
                encoding="rgb8",
            )
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.path_frame_id

            self.overlay_pub.publish(msg)
            self.last_rendered_policy_stamp_ns = policy_stamp_ns

        except Exception as exc:
            self.get_logger().error(f"Overlay rendering failed: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Independent ROS 2 camera/path visual overlay node."
    )
    parser.add_argument("--rate",type=float,default=10.0,help="Maximum visualization rate in Hz.")
    parser.add_argument("-r", "--robot", type=str, help="Robot Name", default="husky")

    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = VisualOverlayNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
