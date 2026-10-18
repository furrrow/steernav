from __future__ import annotations

from collections import deque
import json
import matplotlib
matplotlib.use("Agg")

import time
import argparse
from argparse import Namespace
from pathlib import Path as FilePath

import cv2
from cv_bridge import CvBridge
import numpy as np
import torch
import yaml
from transformers import AutoProcessor, AutoModelForCausalLM

from PIL import Image as PILImage
from torch import Tensor
import threading
from queue import Queue, Full, Empty
# ROS2
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, Float32MultiArray, Empty, String
from nav_msgs.msg import Path, Odometry
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import tf2_ros

from custom_utils.esdf_utils import visualize_esdf, visualize_path_debug, visualize_camera_esdf
from custom_utils.io_utils import load_calibration, filter_unwanted_results

from moge.model.v2 import MoGeModel
import supervision as sv
from steer_dummy_path import update_trajectories, transform_point
from custom_utils.pointcloud_utils import update_points


def resample_path(path, num_waypoints: int = 8, eps: float = 1e-8) -> np.ndarray:
    """Resample a path to equally spaced XY waypoints along its arc length."""
    if not isinstance(num_waypoints, (int, np.integer)) or num_waypoints <= 0:
        raise ValueError("num_waypoints must be a positive integer")

    path_xy = np.asarray(path, dtype=np.float64)
    if path_xy.ndim == 1:
        if path_xy.shape[0] < 2:
            return np.zeros((num_waypoints, 2), dtype=np.float64)
        path_xy = path_xy[:2][None, :]
    elif path_xy.ndim != 2 or path_xy.shape[1] < 2:
        raise ValueError("path must have shape (N, 2) or (N, >=2)")
    else:
        path_xy = path_xy[:, :2]

    path_xy = path_xy[np.isfinite(path_xy).all(axis=1)]
    if len(path_xy) == 0:
        return np.zeros((num_waypoints, 2), dtype=np.float64)

    if len(path_xy) > 1:
        keep = np.r_[True, np.linalg.norm(np.diff(path_xy, axis=0), axis=1) > eps]
        path_xy = path_xy[keep]
    if len(path_xy) == 1:
        return np.repeat(path_xy, num_waypoints, axis=0)

    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    cumulative_distance = np.r_[0.0, np.cumsum(segment_lengths)]
    sample_distances = np.linspace(0.0, cumulative_distance[-1], num_waypoints)
    return np.column_stack([
        np.interp(sample_distances, cumulative_distance, path_xy[:, axis])
        for axis in range(2)
    ])


def pick_waypoint(
    opt_path_xy,
    target_distance: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """Pick a waypoint at a fixed lookahead distance along a path.

    The path is expressed in the robot base frame, so the robot is treated as
    the origin. If the path is shorter than target_distance, return the last
    reachable point.
    """
    path_xy = np.asarray(opt_path_xy, dtype=np.float64)
    if path_xy.size == 0:
        return np.zeros(2, dtype=np.float64)

    if path_xy.ndim == 1:
        if path_xy.shape[0] < 2:
            return np.zeros(2, dtype=np.float64)
        path_xy = path_xy[:2][None, :]
    elif path_xy.ndim != 2 or path_xy.shape[1] < 2:
        raise ValueError("opt_path_xy must have shape (N, 2) or (N, >=2)")
    else:
        path_xy = path_xy[:, :2]

    path_xy = path_xy[np.isfinite(path_xy).all(axis=1)]
    if len(path_xy) == 0:
        return np.zeros(2, dtype=np.float64)

    points = np.vstack((np.zeros(2, dtype=np.float64), path_xy))
    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)

    remaining = float(target_distance)
    if remaining <= eps:
        return np.zeros(2, dtype=np.float64)

    last_reachable = points[0]
    for start, vector, length in zip(points[:-1], segment_vectors, segment_lengths):
        if length <= eps:
            continue
        last_reachable = start + vector
        if remaining <= length:
            return start + (remaining / length) * vector
        remaining -= length

    return last_reachable


def transform_robot_twist_to_camera(linear_velocity_base, angular_velocity_base,
                                    T_base_from_cam):
    """Return camera-origin linear and angular velocity in camera axes."""
    linear_velocity_base = np.asarray(linear_velocity_base, dtype=np.float64)
    angular_velocity_base = np.asarray(angular_velocity_base, dtype=np.float64)
    T_base_from_cam = np.asarray(T_base_from_cam, dtype=np.float64)
    if linear_velocity_base.shape != (3,) or angular_velocity_base.shape != (3,):
        raise ValueError("linear and angular velocity must each have shape (3,)")
    if T_base_from_cam.shape != (4, 4):
        raise ValueError("T_base_from_cam must have shape (4, 4)")

    rotation_cam_from_base = T_base_from_cam[:3, :3].T
    camera_offset_base = T_base_from_cam[:3, 3]
    camera_linear_velocity_base = (
        linear_velocity_base
        + np.cross(angular_velocity_base, camera_offset_base)
    )
    return (
        rotation_cam_from_base @ camera_linear_velocity_base,
        rotation_cam_from_base @ angular_velocity_base,
    )


def quaternion_to_rotation_matrix(quaternion_xyzw):
    """Convert an XYZW quaternion to a 3-by-3 rotation matrix."""
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if quaternion.shape != (4,) or not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion must be a finite, nonzero XYZW vector")
    x, y, z, w = quaternion / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def interpolate_pose_samples(pose_samples, timestamp, max_endpoint_gap=0.1):
    """Interpolate a pose, allowing a nearby endpoint when one side is absent."""
    if not pose_samples:
        return None
    times = np.asarray([sample[0] for sample in pose_samples], dtype=np.float64)
    timestamp = float(timestamp)
    right = int(np.searchsorted(times, timestamp, side="left"))
    if right < len(times) and abs(times[right] - timestamp) <= 1e-9:
        position, quaternion = pose_samples[right][1:]
    else:
        if right == 0 or right == len(times):
            endpoint = 0 if right == 0 else len(times) - 1
            if abs(timestamp - times[endpoint]) > float(max_endpoint_gap):
                return None
            position, quaternion = pose_samples[endpoint][1:]
            q0 = np.asarray(quaternion, dtype=np.float64).copy()
            q1 = q0.copy()
            fraction = 0.0
        else:
            left = right - 1
            interval = times[right] - times[left]
            if interval <= 1e-9:
                return None
            fraction = (timestamp - times[left]) / interval
            position = (
                (1.0 - fraction) * np.asarray(pose_samples[left][1], dtype=np.float64)
                + fraction * np.asarray(pose_samples[right][1], dtype=np.float64)
            )
            q0 = np.asarray(pose_samples[left][2], dtype=np.float64).copy()
            q1 = np.asarray(pose_samples[right][2], dtype=np.float64).copy()
        q0 /= np.linalg.norm(q0)
        q1 /= np.linalg.norm(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            quaternion = q0 + fraction * (q1 - q0)
            quaternion /= np.linalg.norm(quaternion)
        else:
            angle = np.arccos(np.clip(dot, -1.0, 1.0))
            quaternion = (
                np.sin((1.0 - fraction) * angle) * q0
                + np.sin(fraction * angle) * q1
            ) / np.sin(angle)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation_matrix(quaternion)
    transform[:3, 3] = position
    return transform


def interpolate_vector_samples(samples, timestamp, max_endpoint_gap=0.1):
    """Linearly interpolate timestamped vectors with bounded endpoint reuse."""
    if not samples:
        return None
    times = np.asarray([sample[0] for sample in samples], dtype=np.float64)
    timestamp = float(timestamp)
    right = int(np.searchsorted(times, timestamp, side="left"))
    if right < len(times) and abs(times[right] - timestamp) <= 1e-9:
        return np.asarray(samples[right][1], dtype=np.float64).copy()
    if right == 0 or right == len(times):
        endpoint = 0 if right == 0 else len(times) - 1
        if abs(timestamp - times[endpoint]) > float(max_endpoint_gap):
            return None
        return np.asarray(samples[endpoint][1], dtype=np.float64).copy()
    left = right - 1
    interval = times[right] - times[left]
    if interval <= 1e-9:
        return None
    fraction = (timestamp - times[left]) / interval
    return (
        (1.0 - fraction) * np.asarray(samples[left][1], dtype=np.float64)
        + fraction * np.asarray(samples[right][1], dtype=np.float64)
    )


def yaw_from_rotation(rotation):
    """Return planar yaw from a 3-by-3 rotation matrix."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


class SteeringNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('Steering_Node')

        self.buffer_size = None

        # CONSTANTS
        parent_dir = FilePath(__file__).resolve().parent.parent
        # parent_dir = "/home/jim/Projects/steernav"
        # parent_dir = "/home/gamma-nav/Documents/Projects/git_repos/steernav"
        # parent_dir = "/workspace/steernav"
        DEPLOY_CONFIG_PATH = f"{parent_dir}/steernav/config/robot.yaml"
        MODEL_CONFIG_PATH = f"{parent_dir}/steernav/config/models.yaml"
        with open(DEPLOY_CONFIG_PATH, "r") as f:
            deploy_config = yaml.safe_load(f)
        self.rate = deploy_config["frame_rate"]
        robot_config = deploy_config[args.robot]
        calibration_path = FilePath(
            robot_config.get(
                "camera_calibration",
                deploy_config.get("cam_matrix", "cam_matrix.json"),
            )
        )
        if not calibration_path.is_absolute():
            calibration_path = parent_dir / "steernav" / calibration_path
        print(f"using robot config for: {args.robot}")
        self.max_v = robot_config["max_v"]
        self.max_w = robot_config["max_w"]
        self.lookahead_distance = float(robot_config["lookahead_distance"])
        if not np.isfinite(self.lookahead_distance) or self.lookahead_distance < 0:
            raise ValueError("lookahead_distance must be finite and >= 0")
        args.robot_radius=robot_config["robot_radius"]
        self.original_img_size = (deploy_config["img_w"], deploy_config["img_h"])  # (1280, 720)
        self.shrink_img_size = (deploy_config["shrink_w"], deploy_config["shrink_h"])  # (640, 480)
        self.detection_queue = []
        self.detection_queue_len = 20
        self.detection_timestamp_queue = deque(maxlen=self.detection_queue_len)
        self.detection_receipt_ros_queue = deque(maxlen=self.detection_queue_len)
        self.detection_receipt_steady_queue = deque(maxlen=self.detection_queue_len)
        self.detection_base_pose_queue = deque(maxlen=self.detection_queue_len)
        self.detection_camera_pose_queue = deque(maxlen=self.detection_queue_len)
        self.odom_pose_buffer = deque(maxlen=2000)
        self.odom_linear_velocity_buffer = deque(maxlen=2000)
        self.odom_angular_velocity_buffer = deque(maxlen=2000)
        self.odom_receipt_time_buffer = deque(maxlen=2000)
        self.odom_frame_id = ""
        self.odom_child_frame_id = ""
        self.latest_odom_timestamp = None
        self.motion_track_debug_history = {}
        self.motion_track_debug_active_ids = set()
        # Estimate track motion from the central half of each detection. Large
        # grounding boxes often include background whose median depth can jump
        # between surfaces even when the box itself is stable.
        self.velocity_box_inset_fraction = 0.25
        self.robot_velocity_base = np.zeros(3, dtype=np.float64)
        self.robot_angular_velocity_base = np.zeros(3, dtype=np.float64)
        self.dt = 1 / self.rate
        self.reached_goal = False
        self.require_episode = args.robot == "sim_nova"
        self.episode_id = None
        self.path_frame_id = "base_link"
        self._started_sent = False
        self.show_time_performance = False
        self.debug_motion = args.debug_motion
        self.no_velocity = args.no_velocity
        self.visualize = True
        self.save_video = False
        self.video_writer = None
        self.video_path = "esdf_debug.mp4"
        self.video_fps = 5.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self,
        )

        # ROS Topics
        IMAGE_TOPIC = robot_config['image_topic']
        ODOM_TOPIC = robot_config['odom_topic']
        self.compressed_img_topic = True if "compressed" in IMAGE_TOPIC else False
        print(f"IMAGE_TOPIC: {IMAGE_TOPIC} compressed_img_topic: {self.compressed_img_topic}")
        POLICY_PATH_TOPIC = robot_config['policy_path_topic']
        STEERED_WAYPOINT_TOPIC = robot_config['steered_waypoint_topic']
        SAMPLED_ACTIONS_TOPIC = robot_config['sampled_actions_topic']
        OVERLAY_TOPIC = robot_config['overlay_topic']

        # load model parameters
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_params = yaml.safe_load(f)

        self.buffer_size = model_params["buffer_size"]
        self.cam_matrix, self.dist_coeffs, self.T_base_from_cam = load_calibration(
            str(calibration_path)
        )
        self.T_cam_from_base = np.linalg.inv(self.T_base_from_cam)
        self.image_queue = deque(maxlen=self.buffer_size+1)
        self.image_timestamp_queue = deque(maxlen=self.buffer_size+1)
        self.image_receipt_ros_queue = deque(maxlen=self.buffer_size+1)
        self.image_receipt_steady_queue = deque(maxlen=self.buffer_size+1)
        self.paths_queue = deque(maxlen=self.buffer_size+1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        depth_model_name = model_params['depth_model_name']
        vision_model_name = model_params['vision_model_name']
        self.depth_model = MoGeModel.from_pretrained(depth_model_name).to(self.device).eval()
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.obj_detect_model = None
        self.processor = None
        if not self.no_velocity:
            self.obj_detect_model = AutoModelForCausalLM.from_pretrained(
                vision_model_name,
                torch_dtype=self.torch_dtype,
                attn_implementation="eager",
                trust_remote_code=True,
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                vision_model_name, trust_remote_code=True
            )
        self.task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        self.text_prompt = "people"
        self.prompt = self.task_prompt + self.text_prompt
        self.tracker = sv.ByteTrack()

        # ROS 2 Topics
        # subscribers:
        msg_type = CompressedImage if self.compressed_img_topic else Image
        self.image_sub = self.create_subscription(
            msg_type, IMAGE_TOPIC, self.img_callback_obs,
            qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                   history=QoSHistoryPolicy.KEEP_LAST,
                                   depth=10))
        self.odom_sub = self.create_subscription(
            Odometry, ODOM_TOPIC, self.odom_callback_obs,
            qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                   history=QoSHistoryPolicy.KEEP_LAST,
                                   depth=10))
        self.policy_path_sub = self.create_subscription(
            Path, POLICY_PATH_TOPIC, self.path_callback_obs,
            qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                   history=QoSHistoryPolicy.KEEP_LAST,
                                   depth=10))
        self.episode_sub = None
        if self.require_episode:
            episode_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.episode_sub = self.create_subscription(
                String,
                args.episode_topic,
                self.episode_callback,
                qos_profile=episode_qos,
            )
        # publishers:
        self.steered_waypoint_pub = self.create_publisher(
            Float32MultiArray, STEERED_WAYPOINT_TOPIC,
            qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                   history=QoSHistoryPolicy.KEEP_LAST,
                                   depth=10))
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC,
            qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                   history=QoSHistoryPolicy.KEEP_LAST,
                                   depth=10))
        self.trajectory_visual_pub = self.create_publisher(
            Image, OVERLAY_TOPIC, qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                                         history=QoSHistoryPolicy.KEEP_LAST,
                                                         depth=10))
        # self.goal_pub = self.create_publisher(Bool, REACHED_GOAL_TOPIC, 1)
        self.pub_started = self.create_publisher(Empty, "/started", 10)
        self.timer = self.create_timer(
            1.0 / self.rate,
            lambda: self.run_steering_loop(args),
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

        # self.imsave_timer = self.create_timer(1, lambda: self.save_images_and_actions())

        if self.require_episode:
            print(f"Waiting for simulation episodes on {args.episode_topic}...")
        else:
            print("Waiting for image observations...")

        self.model_params = model_params

        self.closest_node = 0
        self.br = CvBridge()
        # Publish /started once, when we actually start inferencing
        if not self._started_sent:
            self._started_sent = True
            self._have_cur_img = False
            self._have_cur_pose = False
            self.pub_started.publish(Empty())
            self.get_logger().info("Published /started (once).")

    def img_callback_obs(self, msg: Image):
        receipt_ros = self.get_clock().now().nanoseconds * 1e-9
        receipt_steady = time.monotonic()
        if self.compressed_img_topic:
            self.obs_img = self.br.compressed_imgmsg_to_cv2(msg)
        else:
            self.obs_img = self.br.imgmsg_to_cv2(msg)
        # Keep the timestamp paired with the image. Inference latency is variable,
        # so wall-clock processing intervals are not valid observation intervals.
        image_timestamp = (
            float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) * 1e-9
        )
        if image_timestamp <= 0.0:
            image_timestamp = self.get_clock().now().nanoseconds * 1e-9
        self.obs_img = cv2.cvtColor(self.obs_img, cv2.COLOR_BGR2RGB)
        self.obs_img = PILImage.fromarray(self.obs_img)
        if self.obs_img.size != self.shrink_img_size:
            self.obs_img = self.obs_img.resize(self.shrink_img_size)

        if self.buffer_size is not None:
            self.image_queue.append(self.obs_img)
            self.image_timestamp_queue.append(image_timestamp)
            self.image_receipt_ros_queue.append(receipt_ros)
            self.image_receipt_steady_queue.append(receipt_steady)

    def odom_callback_obs(self, msg: Odometry):
        receipt_ros = self.get_clock().now().nanoseconds * 1e-9
        receipt_steady = time.monotonic()
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
        timestamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if timestamp <= 0.0:
            timestamp = self.get_clock().now().nanoseconds * 1e-9
        pose = msg.pose.pose
        sample = (
            timestamp,
            np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64),
            np.array([
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ], dtype=np.float64),
        )
        self.odom_pose_buffer.append(sample)
        self.odom_linear_velocity_buffer.append(
            (timestamp, self.robot_velocity_base.copy())
        )
        self.odom_angular_velocity_buffer.append(
            (timestamp, self.robot_angular_velocity_base.copy())
        )
        self.odom_receipt_time_buffer.append(
            (timestamp, np.array([receipt_ros, receipt_steady], dtype=np.float64))
        )
        if len(self.odom_pose_buffer) > 1 and timestamp < self.odom_pose_buffer[-2][0]:
            self.odom_pose_buffer = deque(
                sorted(self.odom_pose_buffer, key=lambda item: item[0]), maxlen=2000
            )
            self.odom_linear_velocity_buffer = deque(
                sorted(self.odom_linear_velocity_buffer, key=lambda item: item[0]),
                maxlen=2000,
            )
            self.odom_angular_velocity_buffer = deque(
                sorted(self.odom_angular_velocity_buffer, key=lambda item: item[0]),
                maxlen=2000,
            )
            self.odom_receipt_time_buffer = deque(
                sorted(self.odom_receipt_time_buffer, key=lambda item: item[0]),
                maxlen=2000,
            )
        frames_changed = (
            msg.header.frame_id != self.odom_frame_id
            or msg.child_frame_id != self.odom_child_frame_id
        )
        self.odom_frame_id = msg.header.frame_id
        self.odom_child_frame_id = msg.child_frame_id
        self.latest_odom_timestamp = timestamp
        if self.debug_motion and frames_changed:
            self.get_logger().info(
                "[motion-debug] odom frames "
                f"pose_frame={self.odom_frame_id!r} "
                f"twist_frame={self.odom_child_frame_id!r}"
            )

    def get_base_pose_world(self, timestamp):
        try:
            return interpolate_pose_samples(self.odom_pose_buffer, timestamp)
        except ValueError as exc:
            if self.debug_motion:
                self.get_logger().warn(f"[motion-debug] invalid odometry pose: {exc}")
            return None

    def episode_callback(self, msg: String) -> None:
        """Clear all steering state before accepting data for a new sim episode."""
        try:
            episode = json.loads(msg.data)
            episode_id = episode["episode_id"]
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError("episode_id must be a nonempty string")
            if episode.get("state") != "started":
                return
            if episode.get("robot_type") != "nova_carter":
                raise ValueError("expected robot_type nova_carter")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().error(f"Invalid episode message: {exc}")
            return

        if episode_id == self.episode_id:
            return

        self.episode_id = episode_id
        self.image_queue.clear()
        self.image_timestamp_queue.clear()
        self.image_receipt_ros_queue.clear()
        self.image_receipt_steady_queue.clear()
        self.paths_queue.clear()
        self.detection_queue.clear()
        self.detection_timestamp_queue.clear()
        self.detection_receipt_ros_queue.clear()
        self.detection_receipt_steady_queue.clear()
        self.detection_base_pose_queue.clear()
        self.detection_camera_pose_queue.clear()
        self.odom_pose_buffer.clear()
        self.odom_linear_velocity_buffer.clear()
        self.odom_angular_velocity_buffer.clear()
        self.odom_receipt_time_buffer.clear()
        self.odom_frame_id = ""
        self.odom_child_frame_id = ""
        self.latest_odom_timestamp = None
        self.motion_track_debug_history.clear()
        self.motion_track_debug_active_ids.clear()
        self.tracker = sv.ByteTrack()
        self.robot_velocity_base.fill(0.0)
        self.robot_angular_velocity_base.fill(0.0)

        stop = Float32MultiArray()
        stop.data = [0.0, 0.0]
        self.steered_waypoint_pub.publish(stop)
        self.get_logger().info(f"Reset steering state for episode {episode_id}")

    def get_robot_velocity_camera(self):
        # Odometry twist is at the base origin. Include the camera lever arm so
        # this is the velocity of the camera origin, then express it in optical
        # camera axes (x right, y down, z forward).
        linear_velocity_camera, _ = transform_robot_twist_to_camera(
            self.robot_velocity_base,
            self.robot_angular_velocity_base,
            self.T_base_from_cam,
        )
        return linear_velocity_camera

    def get_robot_angular_velocity_camera(self):
        _, angular_velocity_camera = transform_robot_twist_to_camera(
            self.robot_velocity_base,
            self.robot_angular_velocity_base,
            self.T_base_from_cam,
        )
        return angular_velocity_camera

    def path_callback_obs(self, msg: Path):
        waypoint_stamp = msg.header.stamp
        waypoints = []
        for pose_stamped in msg.poses:
            pose = pose_stamped.pose

            x = pose.position.x
            y = pose.position.y

            qz = pose.orientation.z
            qw = pose.orientation.w

            # quaternion -> yaw
            yaw = 2.0 * np.arctan2(qz, qw)

            # yaw -> heading vector
            hx = np.cos(yaw)
            hy = np.sin(yaw)

            waypoints.append((x, y, hx, hy))

        if self.buffer_size is not None:
            self.paths_queue.append(np.array(waypoints))

    def destroy_node(self):
        self.timer.cancel()
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.get_logger().info(f"Saved video to {self.video_path}")
        return super().destroy_node()

    def _infer_depth(self, obs_image):
        input_image = torch.from_numpy(obs_image).to(self.device).permute(2, 0, 1).float().div_(255.0)
        depth_model_output = self.depth_model.infer(input_image) #First big blocker
        moge_points = depth_model_output['points'].cpu().numpy()
        estimated_cam_matrix = depth_model_output['intrinsics'].cpu().numpy()
        points_input = moge_points.astype(np.float32, copy=True)
        points_input[~depth_model_output["mask"].cpu().numpy().astype(bool)] = np.nan
        depth = depth_model_output['depth'].cpu().numpy()
        return points_input, estimated_cam_matrix, depth

    def _infer_bboxes(self, obs_image):
        obj_detect_inputs = (self.processor(text=self.prompt, images=obs_image, return_tensors="pt")
                             .to(self.device, self.torch_dtype))
        generated_ids = self.obj_detect_model.generate(
            input_ids=obj_detect_inputs["input_ids"],
            pixel_values=obj_detect_inputs["pixel_values"],
            max_new_tokens=4096,
            num_beams=3,
            do_sample=False
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        obj_detect_result = self.processor.post_process_generation(generated_text, task=self.task_prompt,
                                                              image_size=(obs_image.shape[1], obs_image.shape[0]))
        bbox_result = obj_detect_result[self.task_prompt] #Second big blocker
        bbox_result = filter_unwanted_results(bbox_result, obs_image.shape[1], obs_image.shape[0])
        bbox_only = [bbox for bbox, label in zip(bbox_result['bboxes'], bbox_result['labels'])]
        return bbox_result, bbox_only

    def run_steering_loop(self, args):
        chosen_waypoint = np.zeros(2)
        if self.require_episode and self.episode_id is None:
            return
        if (len(self.image_queue) > self.buffer_size) and (len(self.paths_queue) > 0):
            t0 = time.perf_counter()
            last_path = self.paths_queue[-1]
            if len(last_path) > 0:
                obs_image = np.array(self.image_queue[-1])
                image_timestamp = self.image_timestamp_queue[-1]
                image_receipt_ros = self.image_receipt_ros_queue[-1]
                image_receipt_steady = self.image_receipt_steady_queue[-1]
                vla_path = resample_path(last_path, num_waypoints=8)

                if self.no_velocity:
                    points_input, estimated_cam_matrix, depth = self._infer_depth(obs_image)
                    bbox_result = {"bboxes": [], "labels": []}
                    bbox_only = []
                else:
                    points_input, estimated_cam_matrix, depth = self._infer_depth(obs_image)
                    if self.show_time_performance:
                        t1 = time.perf_counter()
                        self.get_logger().info(f" === > depth_model inference took {(t1 - t0) * 1000:.1f} ms")
                    bbox_result, bbox_only = self._infer_bboxes(obs_image)
                    if self.show_time_performance:
                        self.get_logger().info(f"obj_detect_result took {(time.perf_counter() - t1) * 1000:.1f} ms")
                if self.show_time_performance:
                    t2 = time.perf_counter()
                    self.get_logger().info(f"=== >  depth + bbox inference took {(t2 - t0) * 1000:.1f} ms")

                if len(bbox_only) > 0:
                    dummy_confidence = np.ones(len(bbox_only)) * 0.7
                    sv_detection = sv.Detections(xyxy=np.array(bbox_only), confidence=dummy_confidence)
                    detections = self.tracker.update_with_detections(sv_detection)
                    pos_dict = {}
                    for box, id in zip(detections.xyxy, detections.tracker_id):
                        x1, y1, x2, y2 = box
                        box_width = x2 - x1
                        box_height = y2 - y1

                        inset = self.velocity_box_inset_fraction
                        x1 += inset * box_width
                        x2 -= inset * box_width
                        y1 += inset * box_height
                        y2 -= inset * box_height

                        box_3d_pts = points_input[int(y1):int(y2), int(x1):int(x2)]  # (120, 40, 3)
                        if box_3d_pts.size == 0:
                            continue
                        pts_flat = box_3d_pts.reshape(-1, 3)  # (N, 3)
                        valid_mask = ~np.isnan(pts_flat).any(axis=1) & (pts_flat != 0).any(axis=1)  # (N,)
                        valid_pts = pts_flat[valid_mask]  # (N, 3)
                        if len(valid_pts) == 0:
                            continue
                        median_3d = np.median(valid_pts, axis=0)
                        pos_dict[id] = median_3d
                    # coopting the data field since it is unused.
                    detections.data = pos_dict
                    if self.debug_motion:
                        current_track_ids = set()
                        for box, track_id in zip(detections.xyxy, detections.tracker_id):
                            track_id = int(track_id)
                            current_track_ids.add(track_id)
                            position = pos_dict.get(track_id)
                            previous = self.motion_track_debug_history.get(track_id)
                            if position is None:
                                self.get_logger().info(
                                    f"[track-debug] stamp={image_timestamp:.6f} "
                                    f"track={track_id} box={np.asarray(box).tolist()} "
                                    "position_cam=None"
                                )
                                continue
                            if previous is None:
                                change_text = "new_track=True"
                            else:
                                previous_stamp, previous_position = previous
                                track_dt = image_timestamp - previous_stamp
                                position_delta = position - previous_position
                                raw_speed = (
                                    np.linalg.norm(position_delta) / track_dt
                                    if track_dt > 1e-6 else float("nan")
                                )
                                change_text = (
                                    f"dt={track_dt:.3f} "
                                    f"position_delta_cam={position_delta.tolist()} "
                                    f"raw_speed={raw_speed:.3f}"
                                )
                            self.get_logger().info(
                                f"[track-debug] stamp={image_timestamp:.6f} "
                                f"track={track_id} box={np.asarray(box).tolist()} "
                                f"position_cam={position.tolist()} {change_text}"
                            )
                            self.motion_track_debug_history[track_id] = (
                                image_timestamp,
                                position.copy(),
                            )
                        missing_track_ids = (
                            self.motion_track_debug_active_ids - current_track_ids
                        )
                        if missing_track_ids:
                            self.get_logger().info(
                                f"[track-debug] stamp={image_timestamp:.6f} "
                                f"missing_tracks={sorted(missing_track_ids)}"
                            )
                        self.motion_track_debug_active_ids = current_track_ids
                    # pred_color = plot_bbox(frame_rgb, bbox_result, detections.tracker_id, show_plot=False, return_img=True)
                else:
                    # pred_color = frame_rgb
                    detections = sv.Detections(xyxy=np.array([[0, 0, 0, 0]]), confidence=np.array([0.7]),
                                               tracker_id=np.array([0]), data={})
                    if self.debug_motion and self.motion_track_debug_active_ids:
                        self.get_logger().info(
                            f"[track-debug] stamp={image_timestamp:.6f} "
                            f"missing_tracks={sorted(self.motion_track_debug_active_ids)}"
                        )
                        self.motion_track_debug_active_ids.clear()
                # detection queue
                self.detection_queue.append(detections)
                self.detection_timestamp_queue.append(image_timestamp)
                self.detection_receipt_ros_queue.append(image_receipt_ros)
                self.detection_receipt_steady_queue.append(image_receipt_steady)
                base_pose_world = self.get_base_pose_world(image_timestamp)
                camera_pose_world = (
                    None if base_pose_world is None
                    else base_pose_world @ self.T_base_from_cam
                )
                self.detection_base_pose_queue.append(base_pose_world)
                self.detection_camera_pose_queue.append(camera_pose_world)
                if len(self.detection_queue) > self.detection_queue_len:
                    self.detection_queue.pop(0)
                if self.show_time_performance:
                    t3 = time.perf_counter()
                    self.get_logger().info(f"tracker update took {(t3 - t2) * 1000:.1f} ms")
                robot_velocity_camera = self.get_robot_velocity_camera()
                robot_angular_velocity_camera = self.get_robot_angular_velocity_camera()
                if self.debug_motion:
                    odom_age = (
                        float("nan") if self.latest_odom_timestamp is None
                        else image_timestamp - self.latest_odom_timestamp
                    )
                    self.get_logger().info(
                        "[motion-debug] timing "
                        f"image_stamp={image_timestamp:.6f} "
                        f"latest_odom_stamp={self.latest_odom_timestamp} "
                        f"image_minus_latest_odom={odom_age:.3f} "
                        f"processing_lag_wall="
                        f"{time.monotonic() - image_receipt_steady:.3f} "
                        f"pose_aligned={camera_pose_world is not None}"
                    )
                    if len(self.detection_timestamp_queue) >= 2:
                        t0, t1 = list(self.detection_timestamp_queue)[-2:]
                        image_ros0, image_ros1 = list(
                            self.detection_receipt_ros_queue
                        )[-2:]
                        image_steady0, image_steady1 = list(
                            self.detection_receipt_steady_queue
                        )[-2:]
                        base0, base1 = list(self.detection_base_pose_queue)[-2:]
                        _, cam1 = list(self.detection_camera_pose_queue)[-2:]
                        twist0 = interpolate_vector_samples(
                            self.odom_linear_velocity_buffer, t0
                        )
                        twist1 = interpolate_vector_samples(
                            self.odom_linear_velocity_buffer, t1
                        )
                        angular0 = interpolate_vector_samples(
                            self.odom_angular_velocity_buffer, t0
                        )
                        angular1 = interpolate_vector_samples(
                            self.odom_angular_velocity_buffer, t1
                        )
                        odom_receipt0 = interpolate_vector_samples(
                            self.odom_receipt_time_buffer, t0
                        )
                        odom_receipt1 = interpolate_vector_samples(
                            self.odom_receipt_time_buffer, t1
                        )
                        if base0 is not None and base1 is not None and t1 > t0:
                            dt_pose = t1 - t0
                            image_receipt_ros_dt = image_ros1 - image_ros0
                            image_receipt_steady_dt = image_steady1 - image_steady0
                            odom_receipt_ros_dt = (
                                None if odom_receipt0 is None or odom_receipt1 is None
                                else odom_receipt1[0] - odom_receipt0[0]
                            )
                            odom_receipt_steady_dt = (
                                None if odom_receipt0 is None or odom_receipt1 is None
                                else odom_receipt1[1] - odom_receipt0[1]
                            )
                            image_header_over_steady = (
                                None if image_receipt_steady_dt <= 1e-6
                                else dt_pose / image_receipt_steady_dt
                            )
                            odom_header_over_steady = (
                                None if odom_receipt_steady_dt is None
                                or odom_receipt_steady_dt <= 1e-6
                                else dt_pose / odom_receipt_steady_dt
                            )
                            base0_from_base1 = np.linalg.inv(base0) @ base1
                            cam1_from_cam0 = np.linalg.inv(cam1) @ (
                                base0 @ self.T_base_from_cam
                            )
                            pose_velocity_base = base0_from_base1[:3, 3] / dt_pose
                            relative_rotation = (np.linalg.inv(base0) @ base1)[:3, :3]
                            yaw_delta = yaw_from_rotation(relative_rotation)
                            yaw_rate = yaw_delta / dt_pose
                            mean_twist = (
                                None if twist0 is None or twist1 is None
                                else 0.5 * (twist0 + twist1)
                            )
                            mean_angular = (
                                None if angular0 is None or angular1 is None
                                else 0.5 * (angular0 + angular1)
                            )
                            pose_speed = np.linalg.norm(pose_velocity_base[:2])
                            twist_speed = (
                                None if mean_twist is None else
                                np.linalg.norm(mean_twist[:2])
                            )
                            linear_scale = (
                                None if twist_speed is None or pose_speed < 1e-6
                                else twist_speed / pose_speed
                            )
                            angular_scale = (
                                None if mean_angular is None or abs(yaw_rate) < 1e-6
                                else abs(mean_angular[2] / yaw_rate)
                            )
                            self.get_logger().info(
                                "[motion-debug] pose interval "
                                f"t0={t0:.6f} t1={t1:.6f} dt={dt_pose:.3f} "
                                f"image_receipt_ros_dt={image_receipt_ros_dt:.3f} "
                                f"image_receipt_steady_dt={image_receipt_steady_dt:.3f} "
                                f"image_header_over_steady={image_header_over_steady} "
                                f"odom_receipt_ros_dt={odom_receipt_ros_dt} "
                                f"odom_receipt_steady_dt={odom_receipt_steady_dt} "
                                f"odom_header_over_steady={odom_header_over_steady} "
                                f"translation_base={base0_from_base1[:3, 3].tolist()} "
                                f"pose_velocity_base={pose_velocity_base.tolist()} "
                                f"yaw_delta={yaw_delta:.6f} yaw_rate={yaw_rate:.6f} "
                                f"mean_twist_base={None if mean_twist is None else mean_twist.tolist()} "
                                f"mean_angular_base={None if mean_angular is None else mean_angular.tolist()} "
                                f"linear_twist_over_pose={linear_scale} "
                                f"angular_twist_over_pose={angular_scale} "
                                f"earlier_origin_in_later_cam={cam1_from_cam0[:3, 3].tolist()}"
                            )
                if self.no_velocity:
                    trajectory_points = points_input
                    point_movement_bev = []
                else:
                    # Update points according to detected obstacle velocities.
                    trajectory_points, point_movement_cam = update_points(
                        points_input,
                        self.detection_queue,
                        robot_velocity_camera=robot_velocity_camera,
                        robot_angular_velocity_camera=robot_angular_velocity_camera,
                        detection_timestamps=self.detection_timestamp_queue,
                        camera_poses_world=self.detection_camera_pose_queue,
                        last_n_records=2,
                        velocity_window=5,
                        min_dynamic_observations=3,
                        max_dynamic_speed=3.0,
                        time_look_ahead=1.0,
                        debug_motion=self.debug_motion,
                    )
                    point_movement_bev = [
                        (
                            transform_point(self.T_base_from_cam, prev_point),
                            transform_point(self.T_base_from_cam, after_point),
                        )
                        for prev_point, after_point in point_movement_cam
                    ]

                esdf_result, init_path_xy, opt_path_xy = update_trajectories(
                    args, trajectory_points, estimated_cam_matrix, vla_path,
                    n_iter=20, time_session=False)
                if self.show_time_performance:
                    t4 = time.perf_counter()
                    self.get_logger().info(f"update_trajectories took {(t4 - t3) * 1000:.1f} ms")

                chosen_waypoint = pick_waypoint(
                    opt_path_xy,
                    target_distance=self.lookahead_distance,
                )
                if self.debug_motion:
                    self.get_logger().info(
                        "[motion-debug] selected waypoint "
                        f"waypoint={chosen_waypoint.flatten().tolist()} "
                        f"target_distance={self.lookahead_distance:.3f}"
                    )

                # visualization code
                if self.visualize:
                    # visualize only esdf, ~ 150ms
                    # esdf_surface = visualize_esdf(esdf_result=esdf_result,
                    #                               before_path=init_path_xy, after_path=opt_path_xy,
                    #                               point_movement_bev=point_movement_bev,
                    #                               args=args)
                    esdf_surface = visualize_camera_esdf(rgb=obs_image,
                                                        esdf_result=esdf_result, bbox_result=bbox_result,
                                                        cam_matrix=self.cam_matrix,
                                                        T_cam_from_base=self.T_cam_from_base,
                                                        before_path=init_path_xy, after_path=opt_path_xy,
                                                        point_movement_bev=point_movement_bev,
                                                        tracker_ids=detections.tracker_id,
                                                        selected_waypoint=chosen_waypoint,
                                                        args=args)
                    # visualize full quad chart, ~ 500ms
                    # esdf_surface = visualize_path_debug(depth=depth, rgb=obs_image,
                    #                                     esdf_result=esdf_result, bbox_result=bbox_result,
                    #                                     cam_matrix=self.cam_matrix,
                    #                                     T_cam_from_base=self.T_cam_from_base,
                    #                                     before_path=init_path_xy, after_path=opt_path_xy,
                    #                                     point_movement_bev=point_movement_bev,
                    #                                     args=args)
                    frame_rgb = np.asarray(esdf_surface, dtype=np.uint8)
                    out_msg = self.br.cv2_to_imgmsg(frame_rgb, encoding="rgb8")

                    # Save visualization to video
                    if self.save_video:
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        h, w = frame_bgr.shape[:2]

                        # Initialize writer on first frame
                        if self.video_writer is None:
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            self.video_writer = cv2.VideoWriter(
                                self.video_path,
                                fourcc,
                                self.video_fps,
                                (w, h)
                            )

                            if not self.video_writer.isOpened():
                                raise RuntimeError(f"Failed to open video: {self.video_path}")

                        self.video_writer.write(frame_bgr)
                    self.trajectory_visual_pub.publish(out_msg)
                    if self.show_time_performance:
                        t5 = time.perf_counter()
                        self.get_logger().info(f"visualize + publish path took {(t5 - t4) * 1000:.1f} ms")

        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.flatten().tolist()
        self.steered_waypoint_pub.publish(waypoint_msg)

        # reached_goal = self.closest_node == self.goal_node
        # goal_reached_msg = Bool()
        # goal_reached_msg.data = bool(reached_goal)
        # self.goal_pub.publish(goal_reached_msg)

        # if reached_goal:

def main(args: argparse.Namespace, ros_args=None):
    rclpy.init(args=ros_args)
    steering_node = SteeringNode(args)

    try:
        rclpy.spin(steering_node)
    except KeyboardInterrupt:
        pass
    finally:
        steering_node.destroy_node()
        import matplotlib.pyplot as plt
        plt.close("all")

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ros inference pipeline according to a depth-map ESDF."
    )
    parser.add_argument("-r", "--robot", type=str, help="Robot Name", default="husky")
    parser.add_argument(
        "--episode-topic",
        default="/isaacsim/episode",
        help="Episode metadata topic used by sim_nova to reset steering state.",
    )
    parser.add_argument(
        "--debug-motion",
        action="store_true",
        help="Log odometry and per-track ego-motion compensation details.",
    )
    parser.add_argument(
        "--no-velocity",
        action="store_true",
        help=(
            "Disable object detection and velocity compensation, and pass the "
            "original depth points directly to trajectory optimization."
        ),
    )
    parser.add_argument("--h-min", type=float, default=0.5, help="Minimum kept height in meters.")
    parser.add_argument("--h-max", type=float, default=1.5, help="Maximum kept height in meters.")
    parser.add_argument("--x-min", type=float, default=0.0, help="Minimum forward extent in meters.")
    parser.add_argument("--x-max", type=float, default=5.0, help="Maximum forward extent in meters.")
    parser.add_argument("--y-min", type=float, default=-5.0, help="Minimum lateral extent in meters.")
    parser.add_argument("--y-max", type=float, default=5.0, help="Maximum lateral extent in meters.")
    parser.add_argument("--resolution", type=float, default=0.10, help="Grid resolution in meters per cell.")
    parser.add_argument("--sensor-x", type=float, default=0.0, help="Sensor x location in map frame.")
    parser.add_argument("--sensor-y", type=float, default=0.0, help="Sensor y location in map frame.")
    parser.add_argument("--camera-height", type=float, default=0.45, help="AGL, in meters")
    parser.add_argument("--img_w", type=int, default=1280, help="resize img width to correctly overlay path")
    parser.add_argument("--img_h", type=int, default=720, help="resize img height to correctly overlay path")
    parser.add_argument("--esdf-height-scale", type=float, default=1.8)
    parser.add_argument("--esdf-height-clip-m", type=float, default=2.0)
    parser.add_argument("--esdf-projected-smooth-sigma", type=float, default=1.5)
    parser.add_argument("--esdf-projected-color-percentile", type=float, default=80.0)
    args, ros_args = parser.parse_known_args()
    if ros_args and ros_args[0] != "--ros-args":
        parser.error(f"Unrecognized arguments: {ros_args}")
    main(args, ros_args)
