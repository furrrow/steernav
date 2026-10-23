import ast
from pathlib import Path

import numpy as np


def load_path_function(name):
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


constrain_path_turn_angles = load_path_function("constrain_path_turn_angles")


def turn_cosines(path):
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    return np.sum(segments[:-1] * segments[1:], axis=1) / (
        lengths[:-1] * lengths[1:]
    )


def test_limits_hairpin_to_configured_turn_angle():
    path = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.1, 0.1],
        [0.2, -0.5],
    ])
    original_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)

    result = constrain_path_turn_angles(path, min_turn_cosine=0.5)

    np.testing.assert_allclose(
        np.linalg.norm(np.diff(result, axis=0), axis=1),
        original_lengths,
    )
    assert np.all(turn_cosines(result) >= 0.5 - 1e-8)


def test_backward_sweep_preserves_final_two_waypoints():
    path = np.array([[0.0, 0.0], [1.0, 0.0], [0.1, 0.1], [0.2, -0.5]])

    result = constrain_path_turn_angles(
        path,
        min_turn_cosine=0.5,
        preserve_end=True,
    )

    np.testing.assert_array_equal(result[-2:], path[-2:])
    assert np.all(turn_cosines(result) >= 0.5 - 1e-8)
