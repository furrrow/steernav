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


constrain_path_to_clearance_prefix = load_path_function(
    "constrain_path_to_clearance_prefix"
)
safety_clearance_margin = load_path_function("safety_clearance_margin")


def test_clearance_gate_stops_at_first_occupied_or_unknown_waypoint():
    path = np.array([[0.3, 0.0], [0.6, 0.1], [0.9, 0.2], [1.2, 0.3]])

    occupied_result = constrain_path_to_clearance_prefix(
        path, np.array([0.4, 0.2, -0.1, 0.3])
    )
    unknown_result = constrain_path_to_clearance_prefix(
        path, np.array([0.4, 0.0, 0.2, 0.3])
    )

    np.testing.assert_allclose(
        occupied_result,
        [[0.3, 0.0], [0.6, 0.1], [0.6, 0.1], [0.6, 0.1]],
    )
    np.testing.assert_allclose(
        unknown_result,
        [[0.3, 0.0], [0.3, 0.0], [0.3, 0.0], [0.3, 0.0]],
    )


def test_nonpositive_first_clearance_commands_stop():
    path = np.array([[0.3, 0.0], [0.6, 0.1]])
    result = constrain_path_to_clearance_prefix(path, np.array([-0.01, 0.4]))
    np.testing.assert_array_equal(result, np.zeros_like(path))


def test_radius_dependent_margin_blocks_small_positive_clearance():
    path = np.array([[0.3, 0.0], [0.6, 0.1], [0.9, 0.2]])
    clearance = np.array([0.6, 0.12, 0.7])

    result = constrain_path_to_clearance_prefix(
        path,
        clearance,
        minimum_clearance=safety_clearance_margin(0.5),
    )

    np.testing.assert_allclose(
        result,
        [[0.3, 0.0], [0.3, 0.0], [0.3, 0.0]],
    )
