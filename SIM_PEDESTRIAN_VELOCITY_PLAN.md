# Isaac Sim Pedestrian Velocity Integration Plan

> **Branch status:** `sim-ground-truth` is an experimental offshoot of
> `momentum`. Simulator-specific integration is intentionally kept together
> here while the ROS payload and matching interface settle, so its history and
> implementation may be less polished than the main development branch.

## Goal

For Isaac Sim benchmark runs, use the simulated pedestrians' ground-truth motion instead of estimating their velocity from noisy image detections and monocular depth. Keep the existing visual estimator available for real robots and for benchmark comparisons.

Isaac Sim's animated pedestrians are not handled like rigid-body robots and do not currently expose the `get_linear_velocity()` path used for Nova Carter. The benchmark already reads each character's moving `SkelRoot` world transform for collision checks. The bridge should sample those exact transforms on simulation updates and compute the ground-truth velocity vector from consecutive samples:

```text
velocity_world = (position_world_now - position_world_previous) / delta_sim_time
```

The vector provides both speed and movement direction. Character orientation may also be published later if facing direction is needed, but it should not be used in place of movement direction.

## Data flow

```text
Isaac character SkelRoot transforms
              |
              v
Position difference at the simulation update rate
              |
              v
/isaacsim/pedestrians (world-frame position and velocity)
              |
              v
Timestamp synchronization with the camera image
              |
              v
World-to-camera transform and detection association
              |
              v
Depth points swept along ground-truth pedestrian velocity
```

## 1. Define the ROS payload

Start with `std_msgs/msg/String` carrying JSON. This avoids adding and building a custom ROS interface package during the first experiment and follows the existing `/isaacsim/episode` pattern.

Use the topic `/isaacsim/pedestrians` and the following schema:

```json
{
  "episode_id": "run:episode:pid",
  "stamp": {"sec": 42, "nanosec": 133333333},
  "frame_id": "odom",
  "pedestrians": [
    {
      "id": "/World/Characters/Character_38/SkelRoot",
      "position": [3.21, -1.08, 0.0],
      "velocity": [0.73, 0.12, 0.0]
    }
  ]
}
```

The timestamp must come from the same `current_time` value used for `/clock`, camera images, and odometry. Positions and velocities must both be expressed in the fixed `odom` world frame.

After the experiment is stable, replace the JSON payload with typed messages such as `PedestrianState` and `PedestrianStateArray`. The fields should remain equivalent: a standard header plus a stable ID, position, and velocity for every pedestrian.

## 2. Add pedestrian sampling to the TIC-VLA Isaac bridge

Modify `~/Projects/TIC-VLA/DynaNav/behavior/nova_carter_ros_bridge.py`:

1. Add a publisher for `/isaacsim/pedestrians`.
2. On play, clear all previous pedestrian samples so state cannot leak between episodes.
3. On each simulation update, enumerate active direct children of `/World/Characters`, using the same exclusions and `SkelRoot` lookup as the collision-metric code in `DynaNav/benchmark.py`.
4. Read each moving `SkelRoot` world translation.
5. Store the previous position and simulation timestamp by stable prim path.
6. Compute velocity only when the previous sample exists and `delta_sim_time` is positive and finite.
7. Publish zero velocity for the first sample, after a time discontinuity, or after an episode reset. Marking initial samples as invalid is another option for the later typed schema.
8. Remove cached entries for characters that disappear from the stage.
9. Publish one array message after all characters have been sampled.

Do not calculate velocity from wall-clock time. Pauses, slow rendering, and accelerated simulation make wall time differ from simulation time.

Use best-effort, keep-last depth 1 QoS. Pedestrian state is transient sensor data, and processing an old backlog would produce unsafe predictions.

Factor the existing character enumeration and `SkelRoot` lookup out of `DynaNav/benchmark.py` if it can be shared without coupling the bridge to benchmark state. Otherwise, reproduce the small traversal in the bridge and keep its exclusions identical.

## 3. Stabilize the simulator velocity signal

Use the actual transform displacement rather than route commands. Scripted route direction does not capture navigation turns, collision avoidance, animation transitions, or a character being blocked.

Apply only narrowly scoped validity handling:

- Ignore non-finite positions and velocities.
- Reset a character's history if the simulation timestamp moves backward or the interval is unexpectedly large.
- Clamp or reject physically impossible teleport velocities rather than sweeping obstacles across the map.
- Set very small speeds to zero to suppress transform quantization while a character is idle.

Avoid a long smoothing window initially. The simulator positions are ground truth, and excess smoothing would delay turns. If one-step differencing shows animation-root jitter, use a short two- or three-sample central/least-squares estimate and document the resulting latency.

## 4. Subscribe and buffer in SteerNav

Modify `steernav/ros_inference.py`:

1. Add an `--object-velocity-source` argument with choices `estimated` and `sim-ground-truth`; keep `estimated` as the default so real-robot behavior is unchanged.
2. Add an optional `--pedestrian-state-topic` argument defaulting to `/isaacsim/pedestrians`.
3. Create the pedestrian-state subscription only in `sim-ground-truth` mode.
4. Validate the JSON schema, frame name, finite numeric values, and episode ID in the callback.
5. Store a short timestamp-ordered buffer of complete pedestrian snapshots.
6. Clear this buffer, associations, and character history whenever an episode starts or resets.
7. At image processing time, select the snapshot closest to the image timestamp within a small configured tolerance. Prefer interpolation between surrounding snapshots if both are available.
8. Treat stale or missing simulator state explicitly; do not silently reuse it indefinitely.

The callback should remain lightweight. JSON parsing and validation are appropriate there, while transforms and detection association should happen in the steering loop for the selected image.

## 5. Transform ground-truth state into the camera frame

For each pedestrian snapshot selected at image time, use the timestamp-aligned camera pose already maintained by SteerNav.

```text
position_camera = R_camera_from_world * (position_world - camera_position_world)
velocity_camera = R_camera_from_world * velocity_world
```

Only rotate the absolute world-frame velocity. Do not apply the existing robot linear/angular ego-motion compensation to it; that compensation is needed only when recovering object motion from apparent camera-frame displacement.

Reject pedestrians behind the camera or outside the image before association. Use the camera intrinsics for the processed image to project each remaining ground-truth position into pixel coordinates.

## 6. Associate simulator pedestrians with visual detections

SteerNav still needs the detected person's depth points to represent the pedestrian's body and swept volume. Therefore, match simulator identities to the current person detections:

1. Project each ground-truth pedestrian center into the image.
2. Prefer a detection box containing that projected point.
3. When several boxes contain it, choose the box whose depth-derived 3-D median is closest to the ground-truth camera position.
4. When no box contains it, allow a bounded nearest-center match only if both pixel distance and 3-D distance pass conservative gates.
5. Enforce one-to-one assignment between simulator pedestrians and detections.
6. Leave unmatched detections on the visual estimator or treat them as static, according to the fallback policy below.

Record the matched simulator ID on the detection/track data for diagnostics. Do not assume ByteTrack IDs correspond to Isaac prim paths.

If the SkelRoot origin projects poorly because it is at the character's feet or hips, add a documented vertical offset or derive a stable body-center point from the character bounds.

## 7. Feed simulator velocity into obstacle projection

Extend `services/custom_utils/custom_utils/pointcloud_utils.py::update_points` so a current track can optionally carry an externally supplied camera-frame velocity.

For matched tracks in `sim-ground-truth` mode:

- Use the supplied velocity directly for dynamic/static classification and future displacement.
- Keep the current depth points and swept-volume construction.
- Skip visual position-history velocity fitting.
- Skip camera ego-motion compensation.
- Apply the existing plausible-speed and finite-value checks.

For `estimated` mode, preserve the current behavior exactly.

Prefer a clear optional input such as a dictionary keyed by tracker ID over placing unrelated values into positional arrays. For example:

```python
external_velocities_camera: dict[int, np.ndarray] | None
```

## 8. Define fallback behavior

Use the following initial policy:

- If a detection matches fresh simulator state, use ground-truth velocity.
- If it does not match, use the existing visual estimator once enough history exists.
- If simulator state is stale, fall back to the visual estimator and emit a rate-limited warning.
- If a simulator pedestrian has no detection, do not synthesize image points during the first implementation; there is no observed body point cloud to sweep.
- If the simulator reports a stationary pedestrian, retain the current depth obstacle without forward projection.

This hybrid behavior prevents a brief association miss from removing dynamic-obstacle handling. Add a stricter ground-truth-only mode later if it is useful for controlled ablations.

## 9. Add diagnostics

Under `--debug-object-motion`, log enough information to verify the complete chain:

- image and pedestrian-state timestamps and their difference;
- simulator ID, world position, and world velocity;
- transformed camera position and velocity;
- projected pixel coordinate;
- matched tracker ID and association distances;
- whether the selected source was `sim-ground-truth`, `estimated`, or `static`;
- final predicted displacement over the look-ahead horizon.

Overlay an arrow for each matched ground-truth velocity when visualization is enabled. This makes frame-axis or sign errors immediately visible.

## 10. Test the integration

Add unit tests in SteerNav for:

1. Parsing and rejecting malformed pedestrian messages.
2. Selecting or interpolating a snapshot at an image timestamp.
3. Rotating world-frame velocity into the camera frame without adding robot ego motion.
4. Projecting a ground-truth point and matching it one-to-one with detections.
5. Passing an external velocity through `update_points` and producing the expected swept displacement.
6. Falling back to estimated velocity for unmatched or stale state.
7. Clearing all simulator state at episode boundaries.

Add bridge-side tests in TIC-VLA for the pure pose-difference helper, including initial samples, stationary motion, normal motion, zero/negative time intervals, teleports, and resets. Keep USD stage traversal as a small integration check if it cannot be isolated cheaply.

Run a deterministic Isaac scenario containing one stationary person and pedestrians moving along known axes. Verify in the logs and overlay that:

- stationary speed remains near zero;
- velocity magnitude agrees with displacement over simulation time;
- camera-frame direction changes correctly as the robot turns;
- predicted sweeps point in the pedestrian's direction of travel;
- episode resets never reuse the prior episode's velocity;
- ground-truth mode removes the large spikes caused by detection/depth jitter.

Finally, compare benchmark results for `estimated` and `sim-ground-truth` using the same episodes and seeds. Report navigation metrics together with association coverage and the percentage of detections using each velocity source; otherwise, improvements cannot be attributed cleanly to ground-truth velocity.

## Recommended implementation order

1. Implement and log Isaac pose-derived velocities without publishing them.
2. Publish and inspect `/isaacsim/pedestrians` alongside camera and odometry timestamps.
3. Add SteerNav subscription, buffering, transforms, and debug overlay.
4. Implement detection association and measure its coverage.
5. Add external velocity support to `update_points` and enable swept projection.
6. Add fallback handling and episode resets.
7. Run deterministic validation scenarios.
8. Run the estimated-versus-ground-truth benchmark ablation.
9. Replace JSON with typed ROS messages if the interface will be retained.
