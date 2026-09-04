from __future__ import annotations
import time
import argparse
from argparse import Namespace

import cv2
from cv_bridge import CvBridge
import numpy as np
import torch
import yaml
from transformers import AutoProcessor, AutoModelForCausalLM

from PIL import Image as PILImage
from torch import Tensor

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, Float32MultiArray
from nav_msgs.msg import Path, Odometry
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy, QoSHistoryPolicy
import tf2_ros
from geometry_msgs.msg import Vector3Stamped

from custom_utils.esdf_utils import visualize_path, debug_visualize
from custom_utils.io_utils import load_calibration, filter_unwanted_results

import matplotlib
matplotlib.use("Agg")
from moge.model.v2 import MoGeModel
import supervision as sv
from steer_dummy_path import update_trajectories
from custom_utils.pointcloud_utils import update_points


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
        CAMERA_MATRIX_DIR = f"{parent_dir}/steernav/old_cam_matrix.json"
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
        self.detection_queue = []
        self.detection_queue_len = 20
        self.robot_velocity_base = np.zeros(3, dtype=np.float64)
        self.robot_angular_velocity_base = np.zeros(3, dtype=np.float64)
        self.dt = 1 / self.rate
        self.reached_goal = False

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

        # ROS 2 Topics
        msg_type = CompressedImage if self.compressed_img_topic else Image
        self.image_sub = self.create_subscription(
            msg_type, IMAGE_TOPIC, self.img_callback_obs,
            qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                   history=QoSHistoryPolicy.KEEP_LAST,
                                   depth=10))
        self.odom_sub = self.create_subscription(
            Odometry, ODOM_TOPIC, self.odom_callback_obs,
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
        # self.obs_img = cv2.cvtColor(self.obs_img, cv2.COLOR_BGR2RGB)
        self.obs_img = PILImage.fromarray(self.obs_img)
        if self.obs_img.size != self.shrink_img_size:
            self.obs_img = self.obs_img.resize(self.shrink_img_size)
            print(f"resizing image from {self.obs_img.size} to {self.shrink_img_size}")

        if self.buffer_size is not None:
            if len(self.image_queue) >= self.buffer_size + 1:
                self.image_queue.pop(0)
                self.image_timestamp_queue.pop(0)
            self.image_queue.append(self.obs_img)
            self.image_timestamp_queue.append(self.obs_img_timestamp)

    def odom_callback_obs(self, msg: Odometry):
        self.get_logger().info("Reached Odom callback!")
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

    def get_robot_velocity_camera(self, camera_frame):

        velocity = Vector3Stamped()

        velocity.header.frame_id = "base_link"
        velocity.header.stamp = self.get_clock().now().to_msg()

        velocity.vector.x = self.robot_velocity_base[0]
        velocity.vector.y = self.robot_velocity_base[1]
        velocity.vector.z = self.robot_velocity_base[2]

        try:
            velocity_camera = self.tf_buffer.transform(
                velocity,
                camera_frame,
                timeout=rclpy.duration.Duration(seconds=0.1),
            )

        except Exception as e:
            self.get_logger().warn(
                f"Could not transform robot velocity to camera frame: {e}"
            )
            return None

        return np.array([
            velocity_camera.vector.x,
            velocity_camera.vector.y,
            velocity_camera.vector.z,
        ])

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

            depth_model_output = self.depth_model.infer(input_image)
            moge_points = depth_model_output['points'].cpu().numpy()
            estimated_cam_matrix = depth_model_output['intrinsics'].cpu().numpy()
            points_input = moge_points.astype(np.float32, copy=True)
            points_input[~depth_model_output["mask"].cpu().numpy().astype(bool)] = np.nan
            depth = depth_model_output['depth'].cpu().numpy()

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
            bbox_result = obj_detect_result[self.task_prompt]
            bbox_result = filter_unwanted_results(bbox_result, obs_image.shape[1], obs_image.shape[0])
            bbox_only = [bbox for bbox, label in zip(bbox_result['bboxes'], bbox_result['labels'])]
            if len(bbox_only) > 0:
                dummy_confidence = np.ones(len(bbox_only)) * 0.7
                sv_detection = sv.Detections(xyxy=np.array(bbox_only), confidence=dummy_confidence)
                detections = self.tracker.update_with_detections(sv_detection)
                pos_dict = {}
                for box, id in zip(detections.xyxy, detections.tracker_id):
                    x1, y1, x2, y2 = box
                    box_3d_pts = points_input[int(y1):int(y2), int(x1):int(x2)]  # (120, 40, 3)
                    if box_3d_pts.size == 0:
                        continue
                    pts_flat = box_3d_pts.reshape(-1, 3)  # (N, 3)
                    valid_mask = ~np.isnan(pts_flat).any(axis=1) & (pts_flat != 0).any(axis=1)  # (N,)
                    valid_pts = pts_flat[valid_mask]  # (N, 3)
                    median_3d = np.median(valid_pts, axis=0)
                    pos_dict[id] = median_3d
                    print(f"detect id {id} median loc: {median_3d}")
                # coopting the data field since it is unused.
                detections.data = pos_dict
                # pred_color = plot_bbox(frame_rgb, bbox_result, detections.tracker_id, show_plot=False, return_img=True)
            else:
                # pred_color = frame_rgb
                detections = sv.Detections(xyxy=np.array([[0, 0, 0, 0]]), confidence=np.array([0.7]),
                                           tracker_id=np.array([0]), data={})
            # detection queue
            self.detection_queue.append(detections)
            if len(self.detection_queue) > self.detection_queue_len:
                self.detection_queue.pop(0)

            robot_velocity_camera = self.get_robot_velocity_camera(
                "camera_color_optical_frame"
            )
            if robot_velocity_camera is None:
                robot_velocity_camera=np.array([0, 0, 0.0])
            updated_points = update_points(points_input, self.detection_queue,
                                           robot_velocity_camera=robot_velocity_camera,
                                           time_incr=0.5, time_look_ahead=1.0)

            esdf_result, init_path_xy, opt_path_xy = update_trajectories(
                args, updated_points, estimated_cam_matrix, vla_path, time_session=False)
            if self.original_img_size != self.shrink_img_size:
                original_frame = cv2.resize(obs_image, dsize=self.original_img_size,
                                            interpolation=cv2.INTER_CUBIC)
            else:
                original_frame = obs_image
            esdf_surface = visualize_path(depth=depth, rgb=obs_image,
                                          esdf_result=esdf_result, bbox_result=bbox_result,
                                          cam_matrix=self.cam_matrix,
                                          T_cam_from_base=self.T_cam_from_base,
                                          before_path=init_path_xy, after_path=opt_path_xy,
                                          idx=0, args=args)
            # esdf_surface = debug_visualize(depth=depth, rgb=original_frame,
            #                                    result=esdf_result, cam_matrix=self.cam_matrix,
            #                                    T_cam_from_base=self.T_cam_from_base,
            #                                    before_path=init_path_xy, after_path=opt_path_xy,
            #                                    idx=0, args=args)
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
    parser.add_argument("--x-max", type=float, default=15.0, help="Maximum forward extent in meters.")
    parser.add_argument("--y-min", type=float, default=-5.0, help="Minimum lateral extent in meters.")
    parser.add_argument("--y-max", type=float, default=5.0, help="Maximum lateral extent in meters.")
    parser.add_argument("--resolution", type=float, default=0.10, help="Grid resolution in meters per cell.")
    parser.add_argument("--sensor-x", type=float, default=0.0, help="Sensor x location in map frame.")
    parser.add_argument("--sensor-y", type=float, default=0.0, help="Sensor y location in map frame.")
    parser.add_argument("--camera-height", type=float, default=1.0, help="AGL, in meters")
    parser.add_argument("--img_w", type=int, default=1280, help="resize img width to correctly overlay path")
    parser.add_argument("--img_h", type=int, default=720, help="resize img height to correctly overlay path")
    parser.add_argument("--esdf-height-scale", type=float, default=1.8)
    parser.add_argument("--esdf-height-clip-m", type=float, default=2.0)
    parser.add_argument("--esdf-projected-smooth-sigma", type=float, default=1.5)
    parser.add_argument("--esdf-projected-color-percentile", type=float, default=80.0)
    args = parser.parse_args()
    main(args)
