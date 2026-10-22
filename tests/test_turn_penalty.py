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


negative_turn_gradient = load_path_function("negative_turn_gradient")


def turn_loss(path):
    _, cosines = negative_turn_gradient(path)
    return 0.5 * np.sum(np.minimum(cosines, 0.0) ** 2)


def test_positive_turn_has_no_penalty_gradient():
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])

    gradient, cosines = negative_turn_gradient(path)

    np.testing.assert_allclose(cosines, [0.0])
    np.testing.assert_array_equal(gradient, np.zeros_like(path))


def test_negative_turn_gradient_matches_finite_difference():
    path = np.array([[0.0, 0.0], [1.0, 0.2], [0.4, 0.8]])
    analytical, cosines = negative_turn_gradient(path)
    numerical = np.zeros_like(path)
    step = 1e-6

    for point_index in range(len(path)):
        for axis in range(2):
            forward = path.copy()
            backward = path.copy()
            forward[point_index, axis] += step
            backward[point_index, axis] -= step
            numerical[point_index, axis] = (
                turn_loss(forward) - turn_loss(backward)
            ) / (2.0 * step)

    assert cosines[0] < 0.0
    np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-7)


def test_gradient_descent_makes_a_reversal_less_negative():
    path = np.array([[0.0, 0.0], [1.0, 0.2], [0.4, 0.8]])
    gradient, before = negative_turn_gradient(path)

    _, after = negative_turn_gradient(path - 0.1 * gradient)

    assert after[0] > before[0]
