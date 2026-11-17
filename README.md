# SteerNav

SteerNav is a ROS 2 safety layer for a vision-language navigation policy. It
accepts the policy's local path, estimates static and moving obstacles from the
forward camera, deforms the path in a local ESDF, and publishes one safe
lookahead waypoint for the controller.

The main ROS entry point is [`steernav/ros_inference.py`](steernav/ros_inference.py).
Despite its current name, this node does not run the VLA. The VLA is a separate
process and supplies the nominal path that SteerNav modifies.

## Branch status

`sim-ground-truth` is the primary, most up-to-date development branch. Use it
for new work and benchmark runs. Despite its name, it supports both visually
estimated object velocities (`--object-velocity-source estimated`) and simulator
ground-truth velocities (`--object-velocity-source sim-ground-truth`); using this
branch does not require simulator velocities.

`momentum` is deprecated and retained for historical reference and reproducing
older experiments. New fixes and development should target `sim-ground-truth`.
Record the branch, commit, velocity source, robot radius, and safety margin when
comparing benchmark runs, since these settings have differed across versions.

## TIC-VLA / DynaNav workflow

DynaNav's `ticvla_ros` navigation method is the simulator-side ROS bridge. It
publishes observations and episode metadata and consumes velocity commands. A
VLA adapter, SteerNav, and the waypoint controller remain alive across DynaNav's
per-episode simulator child processes.

```text
DynaNav / Isaac Sim
  /isaacsim/camera/front/image_raw/compressed ──┬──> VLA adapter
  /isaacsim/odom ──────────────────────────────┼──> SteerNav
  /isaacsim/episode ───────────────────────────┼──> VLA, SteerNav, controller
                                               │
VLA adapter ── /sim_nova/policy_path ─────────> SteerNav
                                                    │
                                    depth + people detection
                                    tracking + motion prediction
                                    ESDF path deformation
                                    1 m arc-length lookahead
                                                    │
                         /sim_nova/steered_waypoint v
                                      episode-aware PD controller
                                                    │
                                  /isaacsim/cmd_vel v
                                           DynaNav bridge
```

All processes must use the same `ROS_DOMAIN_ID`. The path and waypoint use the
`base_link` convention in meters: `x` is forward and `y` is left. The VLA path
must contain finite `(x, y)` positions and should set `header.frame_id` to
`base_link`. SteerNav currently reads pose orientation as a planar heading, then
resamples the input path to eight equally spaced XY points before optimization.

The `sim_nova` profile in
[`steernav/config/robot.yaml`](steernav/config/robot.yaml) defines the complete
topic mapping:

| Direction | Topic | Type | Purpose |
| --- | --- | --- | --- |
| subscribe | `/isaacsim/camera/front/image_raw/compressed` | `sensor_msgs/CompressedImage` | RGB obstacle observation |
| subscribe | `/isaacsim/odom` | `nav_msgs/Odometry` | pose and base-frame twist used for motion compensation |
| subscribe | `/isaacsim/episode` | `std_msgs/String` | transient-local episode boundary |
| subscribe | `/sim_nova/policy_path` | `nav_msgs/Path` | nominal VLA path in `base_link` |
| publish | `/sim_nova/steered_waypoint` | `std_msgs/Float32MultiArray` | optimized `[x, y]` controller target |
| publish | `/sim_nova/path_overlay` | `sensor_msgs/Image` | optional debug visualization |
| controller publish | `/isaacsim/cmd_vel` | `geometry_msgs/Twist` | command consumed by DynaNav |

At every new episode ID, SteerNav clears images, paths, odometry, detections,
tracks, and motion history, then publishes a zero waypoint. It ignores duplicate
episode announcements and rejects episodes whose state is not `started` or whose
robot type is not `nova_carter`. The controller must also use episode control so
it clears its cached waypoint and publishes zero velocity until a fresh steered
waypoint arrives.

### Run a steered benchmark

Use the environments belonging to each repository. The VLA adapter must follow
`DynaNav/VLA_ADAPTER_GUIDE.md`: load its model once, take the instruction from
the episode message, reset temporal state on every new episode, and publish a
finite metric path on `/sim_nova/policy_path`.

```bash
# Terminal 1: DynaNav / Isaac environment
cd /home/jim/Projects/TIC-VLA
DynaNav/run_benchmark.sh DynaNav/configs/benchmark_single.yaml

# Terminal 2: the selected VLA's environment (example entry point)
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
uv run python inference/<model>_ros.py --robot sim_nova

# Terminal 3: SteerNav environment
cd /home/jim/Projects/steernav
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
uv run steernav/ros_inference.py --robot sim_nova \
  --object-velocity-source sim-ground-truth

# Terminal 4: SteerNav controller; --steer is required
cd /home/jim/Projects/steernav
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
./steernav/controller.sh \
  --robot sim_nova --steer --simulation-episode-control --dt 1.5 \
  --waypoint-timeout 5.0
```

Set `ROS_DOMAIN_ID` to the value used by DynaNav's `.env.testing`; `42` above is
only the value used in the current DynaNav examples. `--steer` is significant:
without it, the controller listens to `/sim_nova/waypoint`, which is the VLA's
unmodified baseline waypoint, and bypasses SteerNav.

Before allowing a full run, inspect one episode:

```bash
ros2 topic echo /isaacsim/episode std_msgs/msg/String \
  --qos-durability transient_local --once
ros2 topic hz /isaacsim/camera/front/image_raw/compressed
ros2 topic hz /isaacsim/odom
ros2 topic echo /isaacsim/pedestrians std_msgs/msg/String --once
ros2 topic echo /sim_nova/policy_path nav_msgs/msg/Path --once
ros2 topic echo /sim_nova/steered_waypoint \
  std_msgs/msg/Float32MultiArray --once
ros2 topic hz /isaacsim/cmd_vel
```

A forward nominal path should contain mostly positive `x` values. Verify that a
new episode causes zero velocity, followed by motion only after a new policy path
and steered waypoint arrive.

The waypoint timeout must be longer than the slowest complete SteerNav inference
cycle. The controller defaults to five seconds because the depth, grounding,
optimization, and visualization stages can collectively exceed the old one-second
timeout. If the controller repeatedly prints `Not receiving waypoint data`, measure
the rate of `/sim_nova/steered_waypoint` and set `--waypoint-timeout` above the
largest normal interval. A zero waypoint received from SteerNav still stops the
robot immediately.

## Current processing flow

`SteeringNode` loads MoGe and Florence-2 once at startup. Image, odometry, path,
and episode callbacks append to shared history buffers. A 15 Hz timer takes the
latest buffered image and path, runs depth and person grounding, updates
ByteTrack tracks, compensates tracked motion with timestamp-aligned odometry,
predicts obstacle motion, and calls `update_trajectories`. The optimized path is
then reduced to the configured arc-length lookahead waypoint and published.

For Isaac benchmarks, `--object-velocity-source sim-ground-truth` matches each
detected person to the timestamped simulator position and uses its world-frame
velocity after rotating it into camera axes. Simulator velocity is authoritative
in this mode: if simulator state is stale or a track is unmatched, SteerNav
retains the observed depth obstacle but suppresses its motion projection instead
of falling back to a visual velocity estimate.
`--no-velocity` skips Florence-2, tracking, and velocity adjustment, passing the
original depth points directly to `update_trajectories`. `--debug-motion` reports
the optimized path, segment lengths, turn cosines, temporal prior, and selected
waypoint. The temporal prior carries the previous path deformation into the
current robot frame and resists per-waypoint side reversals during every Adam
update without pulling farther toward the previous offset. The prior acts
where its native metric clearance satisfies the safety margin, and
the untruncated optimized path preserves the chosen side across safety stops.
The optimizer also projects each iteration to keep
waypoint x coordinates nondecreasing, preventing a truncated feasible path
from pointing back toward the robot. The final feasible-prefix gate requires
one robot radius of additional metric clearance beyond the inflated footprint.
Visualization is currently always on.

### Receiver timing diagnostics

Receiver diagnostics are disabled by default. Use
`--receiver-timing-log /absolute/path/to/new-file.jsonl` to record image/odometry
callback entry times, original header stamps and fallback flags, pose-query
brackets/source/gaps, raw odometry and frame-level projection/temporal availability.
The bounded asynchronous writer counts dropped records and refuses to overwrite
an existing capture. An orderly shutdown adds a `writer_summary`; check this and
loss counts before interpreting the log. Callback times are monotonic; message
stamps are ROS time. They must not be subtracted across clock domains.

The NaVILA launcher forwards `STEERNAV_RECEIVER_TIMING_LOG`. TIC-VLA's
`run_navila_odom_timing.bash` enables it alongside ROS bag recording and saves
run provenance. `scripts/summarize_receiver_timing.py` in TIC-VLA summarizes
matching-odometry callback order. A later callback does not establish DDS arrival
order or prove the message was queued at query time. Detailed per-track velocity
rejection reasons and shadow alignment are not implemented by this timing logger.
Alignment selection and defaults are unchanged.


For simulator-only shadow velocity diagnostics, add `--velocity-reference-log`
together with `--receiver-timing-log`. It requires estimated steered mode on
`sim_nova`. The node subscribes to simulator pedestrian states for logging only;
these velocities never replace the visual estimates. `visual_velocity` records
include fitted velocity before speed/evidence rejection, observation stamps,
projection status and explicit skip reasons. `track_observations` records preserve
detection/position coverage, including empty frames. The pinned custom_utils
working tree now has an optional `velocity_diagnostics` callback; preserve that
submodule change along with this node change.

NaVILA forwards `STEERNAV_VELOCITY_REFERENCE_LOG=true`. TIC-VLA's
`run_navila_velocity_comparison.bash` captures matched episode-12 runs under both
alignments, twice with reversed order, and compares against independently recorded
simulator states after each run. It uses later exact/bracketed odometry for the
reference camera pose; unmatched/ambiguous tracks and missing reference data are
reported as coverage exclusions. Both instantaneous and same-window reference
errors are reported. Simulator motion and geometric identity matching still have
uncertainty. Logging is opt-in; smoothing, gates and alignment defaults are unchanged.

### Optional tapered pedestrian sweep

The default velocity projection retains the full detected width at every prediction
step. Add `--taper-pedestrian-sweep` to `steernav/ros_inference.py` to reduce the
transverse ground-plane width linearly to zero at the prediction tip. This applies
to both estimated and simulator velocities. Height, longitudinal extent, velocity,
lookahead, and ESDF robot-radius clearance are unchanged. The finite footprint and
ESDF clearance mean the final occupied region is a tapered wedge, not an exact triangle.

This changes the geometry of the projected obstacle, rather than inflating or
smoothing the pedestrian velocity. At the current position the detected width is
preserved; farther along the motion direction the width decreases. The predicted
travel distance remains velocity multiplied by the prediction horizon. Tapering
therefore may reduce how much a sweep blocks the robot's path, but it cannot
correct an inaccurate speed or direction estimate. Zero ground-plane displacement
receives no taper.

For a future comparison, pair full-width and tapered runs separately for each
velocity source (`estimated` and `sim-ground-truth`) and odometry alignment
(`extrapolate` and `endpoint`). Keep the episode, prediction horizon, velocity
window, image buffering, robot radius, clearance margin, success threshold and
stopping rules fixed. Compare projection coverage and navigation outcomes alongside
velocity errors: tapering changes occupied geometry, so a navigation improvement
alone does not establish more accurate velocity estimation.

The flag is disabled by default. NaVILA launchers forward
`STEERNAV_TAPER_PEDESTRIAN_SWEEP=true` to the node. TIC-VLA's
`run_navila_velocity_comparison.bash` and `run_navila_odom_timing.bash` accept
`--taper-pedestrian-sweep` (or the environment variable); use
`--no-taper-pedestrian-sweep` to explicitly select the full-width baseline.
The setting survives tmux handoffs and is recorded in comparison plans, capture
provenance, analysis summaries and node `frame_result` records. Use a fresh batch
for each geometry setting so previous results remain intact. For example:

```bash
cd /home/jim/Projects/TIC-VLA
./run_navila_velocity_comparison.bash --velocity-source sim-ground-truth --taper-pedestrian-sweep
```

Without the option or environment override, existing runs retain full-width sweeps.


Timing capture shutdown now uses a verified owner handshake: the writer creates
`<log>.owner.json` with its process-start identity. The NaVILA launcher signals that
process and waits for its `writer_summary` and exit before killing the tmux session.
`<log>.shutdown.json` records success/failure of the handshake. The actual summary
contains attempted/written/dropped counters and clean-shutdown status; absent
summaries from older captures cannot be reconstructed. TIC-VLA captures now also
save the original benchmark JSON, `episode_outcomes.csv`, `termination_summary.json`
and `capture_status.json`. Episode termination records are separate from timing
writer shutdown records. Missing episode outcomes, missing reasons or incomplete
writer finalization cause a nonzero run status instead of a false completion.
