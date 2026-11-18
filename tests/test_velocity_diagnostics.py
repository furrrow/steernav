from types import SimpleNamespace
import numpy as np
import pytest
from custom_utils.pointcloud_utils import update_points


@pytest.mark.parametrize('speed,missing,reason', [
    (0., False, 'below_speed_gate'), (.5, False, 'projected'),
    (4., False, 'above_speed_gate'), (.5, True, 'missing_pose')])
def test_velocity_diagnostics_preserve_outputs_and_log_rejections(speed, missing, reason):
    times = [1.,1.1,1.2]
    detections = [SimpleNamespace(tracker_id=np.array([7]),
        data={7: np.array([speed*t,0.,5.])}, xyxy=np.array([[0,0,2,2]])) for t in times]
    poses = [np.eye(4) for _ in times]
    if missing: poses[1] = None
    points = np.ones((2,2,3)); points[...,2]=5.
    kwargs = dict(detection_timestamps=times, camera_poses_world=poses,
                  velocity_window=5, min_dynamic_speed=.25)
    expected, expected_motion = update_points(points.copy(), detections, **kwargs)
    records=[]
    actual, motion = update_points(points.copy(), detections,
        velocity_diagnostics=records.append, **kwargs)
    np.testing.assert_array_equal(actual,expected)
    assert len(motion) == len(expected_motion)
    assert records[0]['reason'] == reason
    assert records[0]['projection_applied'] == (reason=='projected')
    if not missing:
        np.testing.assert_allclose(records[0]['estimated_velocity_camera'],[speed,0.,0.],atol=1e-10)
    else:
        assert records[0]['selected_pose_available'] == [True,False,True]


def test_first_observation_is_reported_without_inventing_velocity():
    records=[]
    d=SimpleNamespace(tracker_id=np.array([2]),data={2:np.array([0.,0.,5.])})
    update_points(np.ones((2,2,3)),[d],detection_timestamps=[1.],velocity_diagnostics=records.append)
    assert records[0]['reason']=='insufficient_history'
    assert 'estimated_velocity_camera' not in records[0]


def test_supplied_velocity_diagnostics_preserve_authoritative_projection():
    points=np.ones((2,2,3));points[...,2]=5.
    d=SimpleNamespace(tracker_id=np.array([7,8]),
        data={7:np.array([0.,0.,5.]),8:np.array([1.,0.,5.])},
        xyxy=np.array([[0,0,2,2],[0,0,1,1]]))
    kwargs=dict(detection_timestamps=[1.],external_velocities_camera={7:np.array([.5,0.,0.])},allow_velocity_fallback=False)
    expected,_=update_points(points.copy(),[d],**kwargs)
    records=[]
    actual,_=update_points(points.copy(),[d],velocity_diagnostics=records.append,**kwargs)
    np.testing.assert_array_equal(actual,expected)
    assert [r['reason'] for r in records]==['external_projected','authoritative_unmatched']
    assert records[0]['estimated_velocity_camera']==[.5,0.,0.]
    assert 'estimated_velocity_camera' not in records[1]
