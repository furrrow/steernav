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


constrain_path_forward_progress = load_path_function(
    "constrain_path_forward_progress"
)


def test_removes_backward_prefix_and_preserves_later_segments():
    path = np.array([
        [0.492, 0.035],
        [0.438, 0.027],
        [0.303, -0.276],
        [0.535, -0.595],
    ])
    later_segment = path[3] - path[2]

    result = constrain_path_forward_progress(path)

    assert np.all(np.diff(result[:, 0]) >= 0.0)
    np.testing.assert_array_equal(result[0], path[0])
    np.testing.assert_allclose(result[3] - result[2], later_segment)


def test_backward_sweep_preserves_endpoint():
    path = np.array([[0.5, 0.0], [0.4, 0.1], [0.3, 0.2]])

    result = constrain_path_forward_progress(path, preserve_end=True)

    assert np.all(np.diff(result[:, 0]) >= 0.0)
    np.testing.assert_array_equal(result[-1], path[-1])
