"""Parsing, synchronization, and association for Isaac pedestrian state."""

from __future__ import annotations

import json
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


def parse_pedestrian_state(payload: str, expected_frame: str = "odom"):
    """Parse one JSON pedestrian snapshot into validated NumPy vectors."""
    message = json.loads(payload)
    if not isinstance(message, dict):
        raise ValueError("pedestrian state must be a JSON object")
    episode_id = message.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("episode_id must be a nonempty string")
    if message.get("frame_id") != expected_frame:
        raise ValueError(f"frame_id must be {expected_frame!r}")
    stamp = message.get("stamp")
    if not isinstance(stamp, dict):
        raise ValueError("stamp must be an object")
    sec = stamp.get("sec")
    nanosec = stamp.get("nanosec")
    if not isinstance(sec, int) or not isinstance(nanosec, int):
        raise ValueError("stamp sec and nanosec must be integers")
    if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
        raise ValueError("invalid pedestrian timestamp")

    pedestrians = {}
    entries = message.get("pedestrians")
    if not isinstance(entries, list):
        raise ValueError("pedestrians must be a list")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each pedestrian must be an object")
        pedestrian_id = entry.get("id")
        if not isinstance(pedestrian_id, str) or not pedestrian_id:
            raise ValueError("pedestrian id must be a nonempty string")
        if pedestrian_id in pedestrians:
            raise ValueError(f"duplicate pedestrian id {pedestrian_id!r}")
        position = np.asarray(entry.get("position"), dtype=np.float64)
        velocity = np.asarray(entry.get("velocity"), dtype=np.float64)
        if position.shape != (3,) or velocity.shape != (3,):
            raise ValueError("pedestrian position and velocity must have length 3")
        if not np.isfinite(position).all() or not np.isfinite(velocity).all():
            raise ValueError("pedestrian position and velocity must be finite")
        pedestrians[pedestrian_id] = (position, velocity)
    return sec + nanosec * 1e-9, episode_id, pedestrians


def closest_pedestrian_snapshot(snapshots: Iterable, timestamp, max_age=0.1):
    """Return the closest timestamped snapshot inside a bounded time gap."""
    snapshots = list(snapshots)
    if not snapshots:
        return None
    timestamp = float(timestamp)
    closest = min(snapshots, key=lambda sample: abs(float(sample[0]) - timestamp))
    if abs(float(closest[0]) - timestamp) > float(max_age):
        return None
    return closest


def advance_pedestrian_states(pedestrians_world, delta_time):
    """Propagate constant-velocity simulator states to another timestamp."""
    delta_time = float(delta_time)
    if not np.isfinite(delta_time):
        raise ValueError("delta_time must be finite")
    return {
        pedestrian_id: (
            np.asarray(position, dtype=np.float64)
            + np.asarray(velocity, dtype=np.float64) * delta_time,
            np.asarray(velocity, dtype=np.float64).copy(),
        )
        for pedestrian_id, (position, velocity) in pedestrians_world.items()
    }


def pedestrian_states_in_camera(pedestrians_world, world_from_camera):
    """Express world-frame pedestrian positions and velocities in camera axes."""
    world_from_camera = np.asarray(world_from_camera, dtype=np.float64)
    if world_from_camera.shape != (4, 4):
        raise ValueError("world_from_camera must have shape (4, 4)")
    camera_from_world = np.linalg.inv(world_from_camera)
    rotation_camera_from_world = camera_from_world[:3, :3]
    result = {}
    for pedestrian_id, (position_world, velocity_world) in pedestrians_world.items():
        position_h = np.r_[np.asarray(position_world, dtype=np.float64), 1.0]
        position_camera = (camera_from_world @ position_h)[:3]
        velocity_camera = rotation_camera_from_world @ np.asarray(
            velocity_world, dtype=np.float64
        )
        result[pedestrian_id] = (position_camera, velocity_camera)
    return result


def match_pedestrians_to_tracks(
    track_positions_camera,
    pedestrians_camera,
    max_ground_distance=1.0,
):
    """Globally match tracks and pedestrians using camera X/Z ground distance.

    The assignment maximizes the number of pairs inside the distance gate
    before minimizing their total distance.  A nearest-pair greedy pass can
    consume the only feasible pedestrian for one track even when the other
    track has a valid alternative.
    """
    valid_tracks = []
    for track_id, track_position in track_positions_camera.items():
        track_position = np.asarray(track_position, dtype=np.float64)
        if track_position.shape != (3,) or not np.isfinite(track_position).all():
            continue
        valid_tracks.append((int(track_id), track_position))

    valid_pedestrians = []
    for pedestrian_id, (pedestrian_position, pedestrian_velocity) in pedestrians_camera.items():
        pedestrian_position = np.asarray(pedestrian_position, dtype=np.float64)
        pedestrian_velocity = np.asarray(pedestrian_velocity, dtype=np.float64)
        if (
            pedestrian_position.shape != (3,)
            or pedestrian_velocity.shape != (3,)
            or not np.isfinite(pedestrian_position).all()
            or not np.isfinite(pedestrian_velocity).all()
            or pedestrian_position[2] <= 0.0
        ):
            continue
        valid_pedestrians.append(
            (pedestrian_id, pedestrian_position, pedestrian_velocity)
        )

    if not valid_tracks or not valid_pedestrians:
        return {}

    max_ground_distance = float(max_ground_distance)
    track_count = len(valid_tracks)
    pedestrian_count = len(valid_pedestrians)
    unmatched_cost = max_ground_distance * (track_count + 1)
    invalid_cost = unmatched_cost * (track_count + pedestrian_count + 1)
    costs = np.full(
        (track_count, pedestrian_count + track_count),
        unmatched_cost,
        dtype=np.float64,
    )
    distances = np.full((track_count, pedestrian_count), np.inf, dtype=np.float64)
    for track_index, (_, track_position) in enumerate(valid_tracks):
        for pedestrian_index, (_, pedestrian_position, _) in enumerate(
            valid_pedestrians
        ):
            distance = float(np.linalg.norm(
                track_position[[0, 2]] - pedestrian_position[[0, 2]]
            ))
            distances[track_index, pedestrian_index] = distance
            costs[track_index, pedestrian_index] = (
                distance if distance <= max_ground_distance else invalid_cost
            )

    matches = {}
    row_indices, column_indices = linear_sum_assignment(costs)
    for track_index, pedestrian_index in zip(row_indices, column_indices):
        if pedestrian_index >= pedestrian_count:
            continue
        distance = distances[track_index, pedestrian_index]
        if distance > max_ground_distance:
            continue
        track_id = valid_tracks[track_index][0]
        pedestrian_id, _, velocity = valid_pedestrians[pedestrian_index]
        matches[track_id] = {
            "pedestrian_id": pedestrian_id,
            "velocity_camera": velocity.copy(),
            "distance": distance,
        }
    return matches
