from __future__ import annotations

from collections import deque
import json
import matplotlib
matplotlib.use("Agg")

import time
import argparse
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext
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
        self.robot_velocity_base = np.zeros(3, dtype=np.float64)
        self.robot_angular_velocity_base = np.zeros(3, dtype=np.float64)
        self.dt = 1 / self.rate
        self.reached_goal = False
        self.require_episode = args.robot == "sim_nova"
        self.episode_id = None
        self.path_frame_id = "base_link"
        self._started_sent = False
        self.show_time_performance = False
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
        self.inference_count = 0
        self.inference_start_time = time.perf_counter()

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
        self.paths_queue = deque(maxlen=self.buffer_size+1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        depth_model_name = model_params['depth_model_name']
        vision_model_name = model_params['vision_model_name']
        self.depth_model = MoGeModel.from_pretrained(depth_model_name).to(self.device).eval()
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.obj_detect_model = AutoModelForCausalLM.from_pretrained(vision_model_name,
                                                            torch_dtype=self.torch_dtype,
                                                            attn_implementation="eager",
                                                            trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(vision_model_name, trust_remote_code=True)
        self.task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        self.text_prompt = "people"
        self.prompt = self.task_prompt + self.text_prompt
        self.tracker = sv.ByteTrack()

        self.inference_executor = None
        self.depth_stream = None
        self.bbox_stream = None
        if args.parallel_inference and self.device.type == "cuda":
            self.get_logger().info("Using Parallel inference.")
            self.depth_stream = torch.cuda.Stream(device=self.device)
            self.bbox_stream = torch.cuda.Stream(device=self.device)
            # Model weights were loaded on the current stream; workers must wait for them.
            model_stream = torch.cuda.current_stream(self.device)
            self.depth_stream.wait_stream(model_stream)
            self.bbox_stream.wait_stream(model_stream)
            self.inference_executor = ThreadPoolExecutor(max_workers=2)
        elif args.parallel_inference:
            self.get_logger().warn("Parallel inference requires CUDA; using serial inference.")

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
        self.timer = self.create_timer(1.0 / self.rate, lambda: self.run_steering_loop(args))

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
        # self.get_logger().info("Reached Image callback!")
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
            # print(f"resizing image from {self.obs_img.size} to {self.shrink_img_size}")
            self.obs_img = self.obs_img.resize(self.shrink_img_size)

        if self.buffer_size is not None:
            self.image_queue.append(self.obs_img)
            self.image_timestamp_queue.append(image_timestamp)

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
        self.paths_queue.clear()
        self.detection_queue.clear()
        self.detection_timestamp_queue.clear()
        self.tracker = sv.ByteTrack()
        self.robot_velocity_base.fill(0.0)
        self.robot_angular_velocity_base.fill(0.0)
        self.inference_count = 0
        self.inference_start_time = time.perf_counter()

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
        if self.inference_executor is not None:
            self.inference_executor.shutdown(wait=True)
            self.inference_executor = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.get_logger().info(f"Saved video to {self.video_path}")
        return super().destroy_node()

    def _infer_depth(self, obs_image):
        context = torch.cuda.stream(self.depth_stream) if self.depth_stream is not None else nullcontext()
        with context:
            input_image = torch.from_numpy(obs_image).to(self.device).permute(2, 0, 1).float().div_(255.0)
            depth_model_output = self.depth_model.infer(input_image) #First big blocker
            if self.depth_stream is not None:
                self.depth_stream.synchronize()
            moge_points = depth_model_output['points'].cpu().numpy()
            estimated_cam_matrix = depth_model_output['intrinsics'].cpu().numpy()
            points_input = moge_points.astype(np.float32, copy=True)
            points_input[~depth_model_output["mask"].cpu().numpy().astype(bool)] = np.nan
            depth = depth_model_output['depth'].cpu().numpy()
        return points_input, estimated_cam_matrix, depth

    def _infer_bboxes(self, obs_image):
        context = torch.cuda.stream(self.bbox_stream) if self.bbox_stream is not None else nullcontext()
        with context:
            obj_detect_inputs = (self.processor(text=self.prompt, images=obs_image, return_tensors="pt")
                                 .to(self.device, self.torch_dtype))
            generated_ids = self.obj_detect_model.generate(
                input_ids=obj_detect_inputs["input_ids"],
                pixel_values=obj_detect_inputs["pixel_values"],
                max_new_tokens=4096,
                num_beams=3,
                do_sample=False
            )
            if self.bbox_stream is not None:
                self.bbox_stream.synchronize()
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
            # self.get_logger().info(f"last path: {last_path}")
            if len(last_path) > 0:
                obs_image = np.array(self.image_queue[-1])
                image_timestamp = self.image_timestamp_queue[-1]
                vla_path = resample_path(last_path, num_waypoints=8)

                if self.inference_executor is not None:
                    depth_future = self.inference_executor.submit(self._infer_depth, obs_image)
                    bbox_future = self.inference_executor.submit(self._infer_bboxes, obs_image)
                    # Finish both workers before advancing this frame, even if one failed.
                    wait((depth_future, bbox_future))
                    points_input, estimated_cam_matrix, depth = depth_future.result()
                    bbox_result, bbox_only = bbox_future.result()
                    torch.cuda.synchronize(self.device)
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
                        # update box size
                        box_width = x2 - x1
                        box_height = y2 - y1

                        x1 += 0.10 * box_width
                        x2 -= 0.10 * box_width
                        y1 += 0.10 * box_height
                        y2 -= 0.10 * box_height

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
                        # print(f"detect id {id} median loc: {median_3d}")
                    # coopting the data field since it is unused.
                    detections.data = pos_dict
                    # pred_color = plot_bbox(frame_rgb, bbox_result, detections.tracker_id, show_plot=False, return_img=True)
                else:
                    # pred_color = frame_rgb
                    detections = sv.Detections(xyxy=np.array([[0, 0, 0, 0]]), confidence=np.array([0.7]),
                                               tracker_id=np.array([0]), data={})
                # detection queue
                self.detection_queue.append(detections)
                self.detection_timestamp_queue.append(image_timestamp)
                if len(self.detection_queue) > self.detection_queue_len:
                    self.detection_queue.pop(0)
                if self.show_time_performance:
                    t3 = time.perf_counter()
                    self.get_logger().info(f"tracker update took {(t3 - t2) * 1000:.1f} ms")
                robot_velocity_camera = self.get_robot_velocity_camera()
                robot_angular_velocity_camera = self.get_robot_angular_velocity_camera()
                self.get_logger().info(f"robot_velocity_camera: {robot_velocity_camera}")
                # update points according to velocity obstacles...
                updated_points, point_movement_cam = update_points(points_input, self.detection_queue,
                                                                   robot_velocity_camera=robot_velocity_camera,
                                                                   robot_angular_velocity_camera=robot_angular_velocity_camera,
                                                                   detection_timestamps=self.detection_timestamp_queue,
                                                                   last_n_records=3,
                                                                   time_look_ahead=1.0)
                point_movement_bev = [
                    (
                        transform_point(self.T_base_from_cam, prev_point),
                        transform_point(self.T_base_from_cam, after_point),
                    )
                    for prev_point, after_point in point_movement_cam
                ]

                esdf_result, init_path_xy, opt_path_xy = update_trajectories(
                    args, updated_points, estimated_cam_matrix, vla_path, n_iter=20, time_session=False)
                if self.show_time_performance:
                    t4 = time.perf_counter()
                    self.get_logger().info(f"update_trajectories took {(t4 - t3) * 1000:.1f} ms")

                chosen_waypoint = pick_waypoint(
                    opt_path_xy,
                    target_distance=self.lookahead_distance,
                )
                self.get_logger().info(f"chosen_waypoint {chosen_waypoint}")

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

        self.inference_count += 1
        elapsed = time.perf_counter() - self.inference_start_time

        if elapsed >= 1.0:
            inference_rate = self.inference_count / elapsed
            self.get_logger().info(
                f"Inference rate: {inference_rate:.2f} Hz "
                f"({self.inference_count} in {elapsed:.2f}s)"
            )

            self.inference_count = 0
            self.inference_start_time = time.perf_counter()
        # print(f"image queue {len(self.image_queue)} chosen waypoint: {chosen_waypoint}")

        # reached_goal = self.closest_node == self.goal_node
        # goal_reached_msg = Bool()
        # goal_reached_msg.data = bool(reached_goal)
        # self.goal_pub.publish(goal_reached_msg)

        # if reached_goal:
        #     print("Reached goal! Stopping...")

def main(args: argparse.Namespace):
    rclpy.init()
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
    parser.add_argument("--parallel-inference", type=bool, default=0,
                        help="Run depth and bounding-box inference concurrently on separate CUDA streams.")
    parser.add_argument("--h-min", type=float, default=0.5, help="Minimum kept height in meters.")
    parser.add_argument("--h-max", type=float, default=1.5, help="Maximum kept height in meters.")
    parser.add_argument("--x-min", type=float, default=0.0, help="Minimum forward extent in meters.")
    parser.add_argument("--x-max", type=float, default=7.0, help="Maximum forward extent in meters.")
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
    args = parser.parse_args()
    main(args)
