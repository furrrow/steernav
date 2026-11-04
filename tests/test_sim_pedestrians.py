import json

import numpy as np
import pytest

from steernav.sim_pedestrians import (
    advance_pedestrian_states,
    closest_pedestrian_snapshot,
    match_pedestrians_to_tracks,
    parse_pedestrian_state,
    pedestrian_states_in_camera,
)


def test_advance_pedestrian_states_to_image_time():
    pedestrians = {
        "A": (np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.0, -0.25]))
    }

    advanced = advance_pedestrian_states(pedestrians, 0.2)

    np.testing.assert_allclose(advanced["A"][0], [1.1, 2.0, 2.95])
    np.testing.assert_allclose(advanced["A"][1], [0.5, 0.0, -0.25])


def _payload(**overrides):
    message = {
        "episode_id": "run:episode:1",
        "stamp": {"sec": 12, "nanosec": 250_000_000},
        "frame_id": "odom",
        "pedestrians": [{
            "id": "/World/Characters/A",
            "position": [1.0, 2.0, 3.0],
            "velocity": [0.5, 0.0, -0.25],
        }],
    }
    message.update(overrides)
    return json.dumps(message)


def test_parse_pedestrian_state():
    timestamp, episode_id, pedestrians = parse_pedestrian_state(_payload())

    assert timestamp == 12.25
    assert episode_id == "run:episode:1"
    np.testing.assert_allclose(
        pedestrians["/World/Characters/A"][1], [0.5, 0.0, -0.25]
    )


@pytest.mark.parametrize(
    "override",
    [
        {"frame_id": "base_link"},
        {"stamp": {"sec": 1, "nanosec": 1_000_000_000}},
        {"pedestrians": [{"id": "A", "position": [0, 0], "velocity": [0, 0, 0]}]},
    ],
)
def test_parse_pedestrian_state_rejects_invalid_payload(override):
    with pytest.raises(ValueError):
        parse_pedestrian_state(_payload(**override))


def test_closest_snapshot_obeys_max_age():
    snapshots = [(1.0, {"a": 1}), (1.2, {"a": 2})]

    assert closest_pedestrian_snapshot(snapshots, 1.18, max_age=0.05) == snapshots[1]
    assert closest_pedestrian_snapshot(snapshots, 1.5, max_age=0.05) is None


def test_world_state_is_transformed_to_camera_without_ego_velocity():
    world_from_camera = np.eye(4)
    world_from_camera[:3, 3] = [1.0, 2.0, 3.0]
    pedestrians = {"A": (np.array([2.0, 2.0, 7.0]), np.array([0.5, 0.0, 0.0]))}

    camera = pedestrian_states_in_camera(pedestrians, world_from_camera)

    np.testing.assert_allclose(camera["A"][0], [1.0, 0.0, 4.0])
    np.testing.assert_allclose(camera["A"][1], [0.5, 0.0, 0.0])


def test_match_pedestrians_to_tracks_is_one_to_one():
    tracks = {
        7: np.array([0.1, 0.0, 4.0]),
        8: np.array([2.1, 0.0, 5.0]),
    }
    pedestrians = {
        "A": (np.array([0.0, 1.0, 4.0]), np.array([0.5, 0.0, 0.0])),
        "B": (np.array([2.0, 1.0, 5.0]), np.array([-0.5, 0.0, 0.0])),
    }

    matches = match_pedestrians_to_tracks(tracks, pedestrians, max_ground_distance=0.5)

    assert matches[7]["pedestrian_id"] == "A"
    assert matches[8]["pedestrian_id"] == "B"
    np.testing.assert_allclose(matches[7]["velocity_camera"], [0.5, 0.0, 0.0])


def test_match_pedestrians_to_tracks_maximizes_valid_matches():
    tracks = {
        1: np.array([0.0, 0.0, 4.0]),
        2: np.array([1.0, 0.0, 4.0]),
    }
    pedestrians = {
        "A": (np.array([0.9, 1.0, 4.0]), np.array([1.0, 0.0, 0.0])),
        "B": (np.array([1.9, 1.0, 4.0]), np.array([0.0, 0.0, 0.0])),
    }

    matches = match_pedestrians_to_tracks(
        tracks, pedestrians, max_ground_distance=1.0
    )

    assert matches[1]["pedestrian_id"] == "A"
    assert matches[2]["pedestrian_id"] == "B"
