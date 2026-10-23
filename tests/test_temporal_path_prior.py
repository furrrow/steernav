import ast
from pathlib import Path

import numpy as np


def load_ros_inference_function(name):
    source_path = Path(__file__).resolve().parents[1] / "steernav" / "ros_inference.py"
    tree = ast.parse(source_path.read_text())
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace[name]


rotate_deformation = load_ros_inference_function(
    "rotate_path_deformation_between_base_frames"
)
def load_steering_function(name):
    source_path = Path(__file__).resolve().parents[1] / "steernav" / "steer_dummy_path.py"
    tree = ast.parse(source_path.read_text())
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace[name]


temporal_lateral_gradient = load_steering_function("temporal_lateral_gradient")
temporal_clearance_weights = load_steering_function("temporal_clearance_weights")


def planar_pose(x=0.0, y=0.0, yaw=0.0):
    pose = np.eye(4)
    pose[:2, :2] = [
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)],
    ]
    pose[:2, 3] = [x, y]
    return pose


def test_translation_does_not_change_deformation_vectors():
    deformation = np.array([[0.1, -0.2], [0.0, -0.5]])

    result = rotate_deformation(
        deformation,
        planar_pose(x=1.0, y=2.0),
        planar_pose(x=4.0, y=-3.0),
    )

    np.testing.assert_allclose(result, deformation)


def test_deformation_rotates_into_current_base_frame():
    deformation = np.array([[1.0, 0.0]])

    result = rotate_deformation(
        deformation,
        planar_pose(yaw=0.0),
        planar_pose(yaw=np.pi / 2.0),
    )

    np.testing.assert_allclose(result, [[0.0, -1.0]], atol=1e-12)


def test_temporal_gradient_penalizes_only_opposite_side_motion():
    baseline = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    reference = np.array([[1.0, -0.8], [2.0, -0.8], [3.0, 0.8]])
    path = np.array([[1.0, -0.2], [2.0, 0.3], [3.0, -0.4]])

    result = temporal_lateral_gradient(
        path,
        baseline,
        reference,
        np.ones(3),
    )

    np.testing.assert_allclose(
        result,
        [[0.0, 0.0], [0.0, 0.3], [0.0, -0.4]],
    )


def test_temporal_gradient_does_not_pull_farther_toward_prior_offset():
    baseline = np.array([[1.0, 0.0]])
    reference = np.array([[1.0, -0.8]])
    path = np.array([[1.0, -0.1]])

    result = temporal_lateral_gradient(path, baseline, reference, np.ones(1))

    np.testing.assert_array_equal(result, [[0.0, 0.0]])


def test_unsafe_reference_releases_temporal_momentum():
    result = temporal_clearance_weights(
        np.array([0.8, 0.5, -0.2]),
        minimum_clearance=0.5,
    )

    np.testing.assert_array_equal(result, [1.0, 0.0, 0.0])
