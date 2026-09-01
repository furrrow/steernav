from __future__ import annotations
import time
import argparse
from argparse import Namespace

import cv2
from cv_bridge import CvBridge
import numpy as np
import torch
import yaml
from numpy import dtype, ndarray
from numpy.typing import NDArray
from PIL import Image as PILImage
from torch import Tensor

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, Float32MultiArray
from nav_msgs.msg import Path
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy, QoSHistoryPolicy

from custom_utils.esdf_utils import visualize_path_esdf
from custom_utils.io_utils import load_calibration

import matplotlib
matplotlib.use("Agg")
from moge.model.v2 import MoGeModel
from steer_dummy_path import update_trajectories


class SteeringNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('Steering_Node')

        self.buffer_size = None
        self.image_queue = []
        self.image_timestamp_queue = []
        self.waypoint_queue = []
        self.waypoint_timestamp_queue = []

        # CONSTANTS
        robot_name = 'ghost'
        parent_dir = "/home/jim/Projects/steernav"
        # parent_dir = "/workspace"
        DEPLOY_CONFIG_PATH = f"{parent_dir}/steernav/config/deployment.yaml"
        MODEL_CONFIG_PATH = "config/models.yaml"
        CAMERA_MATRIX_DIR = f"{parent_dir}/steernav/camera_matrix.json"
        self.distance_cutoff = 10
        with open(DEPLOY_CONFIG_PATH, "r") as f:
            deploy_config = yaml.safe_load(f)
        self.rate = deploy_config["frame_rate"]
        self.waypoint_idx = deploy_config['waypoint_idx']
        robot_config = deploy_config[robot_name]
        self.max_v = robot_config["max_v"]
        self.max_w = robot_config["max_w"]
        self.original_img_size = (robot_config["img_w"], robot_config["img_h"])  # (1280, 720)
        self.shrink_img_size = (robot_config["shrink_w"], robot_config["shrink_h"])  # (640, 480)
        self.dt = 1 / self.rate
        self.reached_goal = False

        # ROS Topics
        IMAGE_TOPIC = robot_config['image_topic']
        self.compressed_img_topic = True if "compressed" in IMAGE_TOPIC else False
        print(f"IMAGE_TOPIC: {IMAGE_TOPIC} compressed_img_topic: {self.compressed_img_topic}")
        POLICY_PATH_TOPIC = robot_config['policy_path_topic']
        STEERED_WAYPOINT_TOPIC = robot_config['steered_waypoint_topic']
        SAMPLED_ACTIONS_TOPIC = robot_config['sampled_actions_topic']
        REACHED_GOAL_TOPIC = robot_config['reached_goal_topic']
        OVERLAY_TOPIC = robot_config['overlay_topic']

        # load model parameters
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_params = yaml.safe_load(f)

        self.buffer_size = model_params["buffer_size"]
        self.cam_matrix, self.dist_coeffs, self.T_base_from_cam = load_calibration(CAMERA_MATRIX_DIR)
        self.T_cam_from_base = np.linalg.inv(self.T_base_from_cam)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        model_name = model_params['depth_model_name']
        self.depth_model = MoGeModel.from_pretrained(model_name).to(self.device).eval()

        # ROS 2 Topics
        msg_type = CompressedImage if self.compressed_img_topic else Image
        self.image_sub = self.create_subscription(
            msg_type, IMAGE_TOPIC, self.img_callback_obs,
            qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                   history=QoSHistoryPolicy.KEEP_LAST,
                                   depth=10))
        self.policy_waypoint_sub = self.create_subscription(
            Path, POLICY_PATH_TOPIC, self.waypoint_callback_obs,
            qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                   history=QoSHistoryPolicy.KEEP_LAST,
                                   depth=10))
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
        self.goal_pub = self.create_publisher(Bool, REACHED_GOAL_TOPIC, 1)
        self.timer = self.create_timer(1.0 / self.rate, lambda: self.run_steering_loop(args))

        # self.imsave_timer = self.create_timer(1, lambda: self.save_images_and_actions())

        print("Waiting for image observations...")

        self.model_params = model_params

        self.closest_node = 0
        self.br = CvBridge()

    def img_callback_obs(self, msg: Image):
        self.get_logger().info("Reached Image callback!")
        if self.compressed_img_topic:
            self.obs_img = self.br.compressed_imgmsg_to_cv2(msg)
        else:
            self.obs_img = self.br.imgmsg_to_cv2(msg)
        # Original camera timestamp
        self.obs_img_timestamp = msg.header.stamp
        self.obs_img = PILImage.fromarray(cv2.cvtColor(self.obs_img, cv2.COLOR_BGR2RGB))
        if self.obs_img.size != self.shrink_img_size:
            self.obs_img = self.obs_img.resize(self.shrink_img_size)
            print(f"resizing image from {self.obs_img.size} to {self.shrink_img_size}")

        if self.buffer_size is not None:
            if len(self.image_queue) >= self.buffer_size + 1:
                self.image_queue.pop(0)
                self.image_timestamp_queue.pop(0)
            self.image_queue.append(self.obs_img)
            self.image_timestamp_queue.append(self.obs_img_timestamp)

    def waypoint_callback_obs(self, msg: Path):
        self.get_logger().info("Reached waypoiont callback!")
        waypoint_stamp = msg.header.stamp
        waypoints = [
            (
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y
            )
            for pose_stamped in msg.poses
        ]
        if self.buffer_size is not None:
            if len(self.waypoint_queue) >= self.buffer_size + 1:
                self.waypoint_queue.pop(0)
                self.waypoint_timestamp_queue.pop(0)
            self.waypoint_queue.append(np.array(waypoints))
            self.waypoint_timestamp_queue.append(waypoint_stamp)

    def find_closest_stamp(self, target_stamp):
        if len(self.image_queue) == 0:
            self.get_logger().warn("Steering Node: image_queue empty!")
            return 0
        timestamps = np.array([t.sec + (t.nanosec * 1e-9) for t in self.image_timestamp_queue])
        timestamp_diff = timestamps - (target_stamp.sec + target_stamp.nanosec * 1e-9)
        # self.get_logger().info(f"target_stamp {target_stamp}")
        # self.get_logger().info(f"self.image_timestamp_queue {self.image_timestamp_queue}")
        # self.get_logger().info(f"timestamp_diff {timestamp_diff}")
        closest_idx = np.argmin(np.abs(timestamp_diff))
        return closest_idx

    def run_steering_loop(self, args):
        chosen_waypoint = np.zeros(2)
        if (len(self.image_queue) > self.buffer_size) and (len(self.waypoint_queue) > 0):
            latest_waypoint_timestamp = self.waypoint_timestamp_queue[-1]
            last_waypoint = self.waypoint_queue[-1]
            closest_idx = self.find_closest_stamp(latest_waypoint_timestamp)
            closest_stamp = self.image_timestamp_queue[closest_idx]
            sec_diff = closest_stamp.sec - latest_waypoint_timestamp.sec + (closest_stamp.nanosec - latest_waypoint_timestamp.nanosec) * 1e-9
            self.get_logger().info(f"closest_idx {closest_idx}, closest_stamp - latest_waypoint_timestamp: {sec_diff:.6f} sec")

            obs_image = np.array(self.image_queue[closest_idx])
            input_image = torch.from_numpy(obs_image).to(self.device).permute(2, 0, 1).float().div_(255.0)
            vla_path = np.array(last_waypoint)

            model_output = self.depth_model.infer(input_image)
            moge_points = model_output['points'].cpu().numpy()
            estimated_cam_matrix = model_output['intrinsics'].cpu().numpy()
            points_input = moge_points.astype(np.float32, copy=True)
            points_input[~model_output["mask"].cpu().numpy().astype(bool)] = np.nan
            depth = model_output['depth'].cpu().numpy()

            esdf_result, init_path_xy, opt_path_xy = update_trajectories(
                args, points_input, estimated_cam_matrix, vla_path, time_session=False)
            if self.original_img_size != self.shrink_img_size:
                original_frame = cv2.resize(obs_image, dsize=self.original_img_size,
                                            interpolation=cv2.INTER_CUBIC)
            else:
                original_frame = obs_image
            esdf_surface = visualize_path_esdf(depth=depth, rgb=original_frame,
                                               result=esdf_result, cam_matrix=self.cam_matrix,
                                               T_cam_from_base=self.T_cam_from_base,
                                               before_path=init_path_xy, after_path=opt_path_xy,
                                               idx=0, args=args)
            out_msg = self.br.cv2_to_imgmsg(np.array(esdf_surface), encoding="rgb8")
            self.trajectory_visual_pub.publish(out_msg)
            chosen_waypoint = opt_path_xy[self.waypoint_idx]

        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.flatten().tolist()
        self.steered_waypoint_pub.publish(waypoint_msg)
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
        rclpy.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pipeline to adjust a dummy path accoring to a depth-map ESDF."
    )
    parser.add_argument("--h-min", type=float, default=0.5, help="Minimum kept height in meters.")
    parser.add_argument("--h-max", type=float, default=1.5, help="Maximum kept height in meters.")
    parser.add_argument("--x-min", type=float, default=0.0, help="Minimum forward extent in meters.")
    parser.add_argument("--x-max", type=float, default=20.0, help="Maximum forward extent in meters.")
    parser.add_argument("--y-min", type=float, default=-5.0, help="Minimum lateral extent in meters.")
    parser.add_argument("--y-max", type=float, default=5.0, help="Maximum lateral extent in meters.")
    parser.add_argument("--resolution", type=float, default=0.10, help="Grid resolution in meters per cell.")
    parser.add_argument("--sensor-x", type=float, default=0.0, help="Sensor x location in map frame.")
    parser.add_argument("--sensor-y", type=float, default=0.0, help="Sensor y location in map frame.")
    parser.add_argument("--camera-height", type=float, default=1.0, help="AGL, in meters")
    parser.add_argument("--esdf-height-scale", type=float, default=1.8)
    parser.add_argument("--esdf-height-clip-m", type=float, default=2.0)
    parser.add_argument("--esdf-projected-smooth-sigma", type=float, default=1.5)
    parser.add_argument("--esdf-projected-color-percentile", type=float, default=80.0)
    args = parser.parse_args()
    main(args)
