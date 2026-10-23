# SteerNav

SteerNav is a ROS 2 safety layer for a vision-language navigation policy. It
accepts the policy's local path, estimates static and moving obstacles from the
forward camera, deforms the path in a local ESDF, and publishes one safe
lookahead waypoint for the controller.

The main ROS entry point is [`steernav/ros_inference.py`](steernav/ros_inference.py).
Despite its current name, this node does not run the VLA. The VLA is a separate
process and supplies the nominal path that SteerNav modifies.

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
uv run steernav/ros_inference.py --robot sim_nova

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
