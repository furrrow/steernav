import ast
from pathlib import Path

import numpy as np
import pytest


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


constrain_path_segment_lengths = load_path_function("constrain_path_segment_lengths")


def test_caps_each_segment_while_preserving_first_waypoint():
    path = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])

    result = constrain_path_segment_lengths(path, max_segment_length=0.5)

    np.testing.assert_array_equal(result[0], path[0])
    np.testing.assert_allclose(result[1], [0.5, 0.0])
    assert np.all(np.linalg.norm(np.diff(result, axis=0), axis=1) <= 0.5 + 1e-8)


def test_backward_sweep_preserves_final_waypoint():
    path = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])

    result = constrain_path_segment_lengths(
        path,
        max_segment_length=0.5,
        preserve_end=True,
    )

    np.testing.assert_array_equal(result[-1], path[-1])
    assert np.all(np.linalg.norm(np.diff(result, axis=0), axis=1) <= 0.5 + 1e-8)


@pytest.mark.parametrize("limit", [-0.1, np.inf, np.nan])
def test_rejects_invalid_distance_limit(limit):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        constrain_path_segment_lengths(np.zeros((2, 2)), limit)
