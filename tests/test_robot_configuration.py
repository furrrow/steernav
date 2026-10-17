import ast
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
ROS_INFERENCE_PATH = ROOT / "steernav" / "ros_inference.py"
ROBOT_CONFIG_PATH = ROOT / "steernav" / "config" / "robot.yaml"


def load_transform_robot_twist_to_camera():
    tree = ast.parse(ROS_INFERENCE_PATH.read_text())
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "transform_robot_twist_to_camera"
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(ROS_INFERENCE_PATH), "exec"), namespace)
    return namespace["transform_robot_twist_to_camera"]


transform_robot_twist_to_camera = load_transform_robot_twist_to_camera()


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
