import ast
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
ROS_INFERENCE_PATH = ROOT / "steernav" / "ros_inference.py"
ROBOT_CONFIG_PATH = ROOT / "steernav" / "config" / "robot.yaml"


def load_ros_inference_functions(*names):
    tree = ast.parse(ROS_INFERENCE_PATH.read_text())
    function_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
    ]
    module = ast.Module(body=function_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(ROS_INFERENCE_PATH), "exec"), namespace)
    return tuple(namespace[name] for name in names)


(
    transform_robot_twist_to_camera,
    quaternion_to_rotation_matrix,
    rotation_vector_to_matrix,
    interpolate_pose_samples,
    pose_at_timestamp,
    individual_detection_mask,
) = load_ros_inference_functions(
    "transform_robot_twist_to_camera",
    "quaternion_to_rotation_matrix",
    "rotation_vector_to_matrix",
    "interpolate_pose_samples",
    "pose_at_timestamp",
    "individual_detection_mask",
)


def test_pose_at_timestamp_extrapolates_trailing_odometry_in_body_frame():
    half_yaw = np.pi / 4.0
    samples = [(
        10.0,
        np.array([1.0, 2.0, 0.0]),
        np.array([0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)]),
    )]

    pose = pose_at_timestamp(
        samples,
        [(10.0, np.array([2.0, 0.0, 0.0]))],
        [(10.0, np.zeros(3))],
        10.1,
        max_extrapolation=0.25,
    )

    np.testing.assert_allclose(pose[:3, 3], [1.0, 2.2, 0.0], atol=1e-7)
    np.testing.assert_allclose(
        pose[:3, :3], quaternion_to_rotation_matrix(samples[0][2]), atol=1e-7
    )


def test_pose_at_timestamp_rejects_extrapolation_past_limit():
    samples = [(10.0, np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))]

    assert pose_at_timestamp(
        samples,
        [(10.0, np.zeros(3))],
        [(10.0, np.zeros(3))],
        10.3,
        max_extrapolation=0.25,
    ) is None


def test_husky_has_complete_runtime_configuration():
    config = yaml.safe_load(ROBOT_CONFIG_PATH.read_text())
    husky = config["husky"]
    required = {
        "max_v",
        "max_w",
        "lookahead_distance",
        "robot_radius",
        "image_topic",
        "odom_topic",
        "policy_path_topic",
        "steered_waypoint_topic",
        "sampled_actions_topic",
        "overlay_topic",
        "camera_calibration",
    }

    assert required <= husky.keys()
    assert husky["image_topic"].endswith("/compressed")
    assert husky["odom_topic"] == "/a200_0648/platform/odom/filtered"
    assert (ROOT / "steernav" / husky["camera_calibration"]).is_file()


def test_planar_husky_yaw_has_no_lever_arm_velocity_for_vertical_camera_offset():
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, 0.76]

    linear_camera, angular_camera = transform_robot_twist_to_camera(
        linear_velocity_base=np.array([0.2, 0.0, 0.0]),
        angular_velocity_base=np.array([0.0, 0.0, 0.3]),
        T_base_from_cam=transform,
    )

    np.testing.assert_allclose(linear_camera, [0.2, 0.0, 0.0])
    np.testing.assert_allclose(angular_camera, [0.0, 0.0, 0.3])


def test_camera_lever_arm_is_included_for_nonvertical_offset():
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 0.0, 0.0]

    linear_camera, _ = transform_robot_twist_to_camera(
        linear_velocity_base=np.zeros(3),
        angular_velocity_base=np.array([0.0, 0.0, 0.5]),
        T_base_from_cam=transform,
    )

    np.testing.assert_allclose(linear_camera, [0.0, 0.5, 0.0])


def test_individual_detection_mask_removes_group_box():
    boxes = np.array([
        [190.0, 0.2, 608.0, 330.0],
        [304.0, 0.2, 385.0, 330.0],
        [526.0, 0.2, 605.0, 320.0],
    ])

    np.testing.assert_array_equal(
        individual_detection_mask(boxes, (480, 640, 3)),
        [False, True, True],
    )


def test_individual_detection_mask_rejects_box_clipped_on_two_edges():
    boxes = np.array([
        [228.0, 0.2, 424.0, 479.3],
        [466.0, 0.2, 526.0, 295.0],
    ])

    np.testing.assert_array_equal(
        individual_detection_mask(boxes, (480, 640, 3)),
        [False, True],
    )
