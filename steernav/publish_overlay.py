#!/usr/bin/env python3
"""
overlay node to replace "visualize_path" in ros_inference
a work in progress.... NOT READY!!

"""

from __future__ import annotations

import argparse
import threading
import time
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

from custom_utils.io_utils import load_calibration


class VisualOverlayNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("visual_overlay_node")

        self.args = args
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.latest_image: Optional[np.ndarray] = None
        self.latest_path: Optional[np.ndarray] = None
        self.image_seq = 0
        self.last_rendered_image_seq = -1

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
        self.detection_queue = []
        self.detection_queue_len = 20
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

        # ROS Topics
        IMAGE_TOPIC = robot_config['image_topic']
        ODOM_TOPIC = robot_config['odom_topic']
        self.compressed_img_topic = True if "compressed" in IMAGE_TOPIC else False
        print(f"IMAGE_TOPIC: {IMAGE_TOPIC} compressed_img_topic: {self.compressed_img_topic}")
        PATH_TOPIC = "/path"
        STEERED_WAYPOINT_TOPIC = robot_config['steered_waypoint_topic']
        SAMPLED_ACTIONS_TOPIC = robot_config['sampled_actions_topic']
        REACHED_GOAL_TOPIC = robot_config['reached_goal_topic']
        OVERLAY_TOPIC = robot_config['overlay_topic']

        self.compressed = "compressed" in IMAGE_TOPIC.lower()

        # load_calibration() in the steering node returns:
        # cam_matrix, dist_coeffs, T_base_from_cam
        self.cam_matrix, self.dist_coeffs, self.T_base_from_cam = load_calibration(CAMERA_MATRIX_DIR)
        self.T_cam_from_base = np.linalg.inv(self.T_base_from_cam)

        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        path_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        msg_type = CompressedImage if self.compressed else Image

        self.image_sub = self.create_subscription(
            msg_type,
            IMAGE_TOPIC,
            self.image_callback,
            image_qos,
        )

        self.path_sub = self.create_subscription(
            Path,
            PATH_TOPIC,
            self.path_callback,
            path_qos,
        )

        self.overlay_pub = self.create_publisher(
            Image,
            OVERLAY_TOPIC,
            QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )

        self.timer = self.create_timer(1.0 / args.rate, self.render_callback)

        self.get_logger().info(
            f"Visual overlay node started\n"
            f"  image:   {args.image_topic}\n"
            f"  path:    {args.path_topic}\n"
            f"  overlay: {args.overlay_topic}\n"
            f"  rate:    {args.rate:.1f} Hz\n"
            f"  compressed image: {self.compressed}"
        )

    def image_callback(self, msg):
        try:
            if self.compressed:
                bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            else:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            with self.lock:
                self.latest_image = bgr
                self.image_seq += 1

        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")

    def path_callback(self, msg: Path):
        if not msg.poses:
            return

        path_xyz = np.asarray(
            [
                [
                    pose.pose.position.x,
                    pose.pose.position.y,
                    pose.pose.position.z,
                ]
                for pose in msg.poses
            ],
            dtype=np.float64,
        )

        with self.lock:
            self.latest_path = path_xyz

    def project_base_points_to_image(self, points_base: np.ndarray):
        """
        Project Nx3 points expressed in base_link into image pixels.

        Assumes T_cam_from_base maps homogeneous points:
            p_cam = T_cam_from_base @ p_base
        and that the calibrated camera frame follows the normal optical-camera
        convention expected by the intrinsic matrix.
        """
        if len(points_base) == 0:
            return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=bool)

        ones = np.ones((points_base.shape[0], 1), dtype=np.float64)
        points_h = np.hstack((points_base, ones))

        points_cam = (self.T_cam_from_base @ points_h.T).T[:, :3]

        # Camera optical coordinates: Z must be positive/in front of camera.
        valid = np.isfinite(points_cam).all(axis=1) & (
            points_cam[:, 2] > self.args.min_depth
        )

        pixels = np.full((points_base.shape[0], 2), -1, dtype=np.int32)

        if np.any(valid):
            cam = points_cam[valid]

            normalized = np.column_stack(
                (
                    cam[:, 0] / cam[:, 2],
                    cam[:, 1] / cam[:, 2],
                    np.ones(cam.shape[0]),
                )
            )

            uvw = (self.cam_matrix @ normalized.T).T
            uv = np.rint(uvw[:, :2]).astype(np.int32)
            pixels[valid] = uv

        return pixels, valid

    def draw_path(self, image: np.ndarray, path_base: np.ndarray) -> np.ndarray:
        overlay = image.copy()
        height, width = overlay.shape[:2]

        pixels, valid = self.project_base_points_to_image(path_base)

        in_image = (
            valid
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )

        # Draw only contiguous segments whose endpoints are visible.
        for i in range(len(pixels) - 1):
            if in_image[i] and in_image[i + 1]:
                p0 = tuple(pixels[i])
                p1 = tuple(pixels[i + 1])
                cv2.line(
                    overlay,
                    p0,
                    p1,
                    (0, 255, 0),
                    self.args.line_thickness,
                    cv2.LINE_AA,
                )

        # Draw waypoint samples.
        for i, pixel in enumerate(pixels):
            if not in_image[i]:
                continue

            radius = self.args.point_radius
            color = (0, 255, 255)

            # Make the first visible point visually distinct.
            if i == 0:
                radius += 2
                color = (0, 0, 255)

            cv2.circle(
                overlay,
                tuple(pixel),
                radius,
                color,
                -1,
                cv2.LINE_AA,
            )

        if self.args.show_status:
            visible_count = int(np.count_nonzero(in_image))
            text = f"path points visible: {visible_count}/{len(path_base)}"
            cv2.putText(
                overlay,
                text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return overlay

    def render_callback(self):
        # Copy references/snapshot while holding the lock very briefly.
        with self.lock:
            if self.latest_image is None or self.latest_path is None:
                return

            # Do not redraw the same camera frame repeatedly.
            if self.image_seq == self.last_rendered_image_seq:
                return

            image = self.latest_image.copy()
            path = self.latest_path.copy()
            image_seq = self.image_seq

        try:
            overlay_bgr = self.draw_path(image, path)

            msg = self.bridge.cv2_to_imgmsg(
                overlay_bgr,
                encoding="bgr8",
            )
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.args.camera_frame

            self.overlay_pub.publish(msg)
            self.last_rendered_image_seq = image_seq

        except Exception as exc:
            self.get_logger().error(f"Overlay rendering failed: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Independent ROS 2 camera/path visual overlay node."
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Maximum visualization rate in Hz.",
    )

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
