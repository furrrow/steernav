"""Receiver diagnostics must describe, not change, pose selection."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('receiver_timing', ROOT/'steernav/receiver_timing.py')
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)
tree = ast.parse((ROOT/'steernav/ros_inference.py').read_text())
names = {'quaternion_to_rotation_matrix', 'rotation_vector_to_matrix',
         'interpolate_pose_samples', 'pose_at_timestamp'}
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SteeringNode')
functions += [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'get_base_pose_world']
ns = dict(np=np, time=__import__('time'), alignment_details=timing.alignment_details)
exec(compile(ast.Module(body=functions, type_ignores=[]), '<pose>', 'exec'), ns)


def sample(t):
    return (t, np.array([t, 0., 0.]), np.array([0., 0., 0., 1.]))


@pytest.mark.parametrize('mode,t,source', [
    ('extrapolate', 1., 'exact'), ('endpoint', 1.5, 'interpolated'),
    ('endpoint', .95, 'early_endpoint'), ('endpoint', 2.05, 'late_endpoint'),
    ('extrapolate', 2.15, 'extrapolated'), ('endpoint', 2.15, 'unavailable'),
    ('extrapolate', .95, 'unavailable'), ('extrapolate', 2.3, 'unavailable')])
def test_lookup_equivalence_and_sources(mode, t, source):
    records = []
    obj = SimpleNamespace(receiver_timing=True, odom_alignment=mode,
        odom_pose_buffer=[sample(1.), sample(2.)],
        odom_linear_velocity_buffer=[(2., np.ones(3))],
        odom_angular_velocity_buffer=[(2., np.zeros(3))],
        odom_pose_max_extrapolation=.25, _timing=lambda event, **kw: records.append(kw))
    actual = ns['get_base_pose_world'](obj, t)
    assert records[-1]['source'] == source
    obj.receiver_timing = None
    unlogged = ns['get_base_pose_world'](obj, t)
    if actual is None:
        assert unlogged is None
    else:
        np.testing.assert_array_equal(actual, unlogged)


def test_empty_and_missing_twist():
    assert timing.alignment_details([], [], [], 1., 'extrapolate', False)['reason'] == 'empty_buffer'
    assert timing.alignment_details([sample(1.)], [], [], 1.1, 'extrapolate', False)['reason'] == 'missing_twist'


def test_writer_strict_json_flush_and_no_overwrite(tmp_path):
    path = tmp_path/'timing.jsonl'
    writer = timing.ReceiverTiming(path)
    writer.emit('pose_query', value=float('nan'), episode_id='a')
    assert writer.close() is None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]['value'] is None
    assert rows[0]['monotonic_ns'] > 0
    assert rows[-1]['event'] == 'writer_summary'
    assert rows[-1]['dropped_records'] == 0
    with pytest.raises(FileExistsError):
        timing.ReceiverTiming(path)


def test_bounded_queue_drops_without_blocking(tmp_path, monkeypatch):
    # Stop the consumer deterministically to exercise overflow.
    monkeypatch.setattr(timing.ReceiverTiming, '_write', lambda self: None)
    writer = timing.ReceiverTiming(tmp_path/'bounded.jsonl', capacity=1)
    writer.emit('first')
    writer.emit('overflow')
    assert writer.queue.qsize() == 1
    assert writer.dropped == 1
    writer.close()
    writer.file.close()


def test_main_flushes_writer_when_tmux_sends_hup(tmp_path):
    handlers = {}
    def register(signum, handler):
        old = handlers.get(signum)
        handlers[signum] = handler
        return old
    fake_signal = SimpleNamespace(SIGHUP=1, SIGTERM=15, SIG_IGN=0, signal=register)
    path = tmp_path/'shutdown.jsonl'
    class Node:
        def __init__(self, args):
            self.receiver_timing=timing.ReceiverTiming(path)
            self.receiver_timing.emit('settings')
        def destroy_node(self):
            assert self.receiver_timing.close() is None
    def spin(node):
        handlers[1](1, None)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='main')
    namespace=dict(argparse=__import__('argparse'), signal=fake_signal, SteeringNode=Node,
        rclpy=SimpleNamespace(init=lambda **kw:None,spin=spin,ok=lambda:False))
    exec(compile(ast.Module(body=[main],type_ignores=[]),'<main>','exec'),namespace)
    namespace['main'](SimpleNamespace())
    assert json.loads(path.read_text().splitlines()[-1])['event']=='writer_summary'
