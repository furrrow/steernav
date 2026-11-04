# Geometry and Planning Utilities Migration Plan

## Goal

Create a small set of reusable robotics primitives in `custom_utils` while
keeping SteerNav-specific perception, prediction, and planning policy inside
the SteerNav project.

The migration must preserve online and offline behavior. It should improve
module ownership without changing numerical conventions, planner thresholds,
ROS topics, model behavior, or published outputs.

## Design principles

1. Shared utilities contain mechanisms, not application policy.
2. Shared geometry has no ROS, MoGe, Supervision, OpenCV, or SteerNav
   dependency.
3. Frame conventions are explicit in names and docstrings.
4. Public transforms use `target_from_source` naming.
5. Quaternions use XYZW ordering.
6. Poses use 4-by-4 homogeneous matrices unless a function is explicitly
   documented as planar.
7. Point arrays use shape `(..., 3)` and preserve their leading dimensions.
8. Old public names receive temporary compatibility aliases when they may be
   used by another project.
9. Each phase is independently testable and should leave the repository in a
   working state.

## Target module ownership

```text
services/custom_utils/custom_utils/
  geometry_utils.py       rigid transforms, rotations, poses, and twists
  planning_utils.py       generic path sampling and planar path operations
  pointcloud_utils.py     point-cloud filtering and point-cloud algorithms
  bev_grid.py             optional later split for BEV/raycasting primitives
  occupancy_utils.py      optional later split for occupancy and ESDF primitives

steernav/
  dynamic_obstacles.py    SteerNav tracking and motion-prediction policy
  goal_projection.py      SteerNav image-box-to-3D-goal policy
  trajectory_optimizer.py SteerNav ESDF path-deformation policy
  ros_inference.py        online ROS orchestration
  steer_dummy_path.py     offline orchestration
```

Dependency direction:

```text
custom_utils.geometry_utils
          ↑
custom_utils.planning_utils   custom_utils.pointcloud_utils
          ↑                         ↑
          └──────────── SteerNav modules ────────────┘
```

`custom_utils` must not import SteerNav.

## Shared versus SteerNav-specific functions

### Move to `custom_utils.geometry_utils`

From `services/custom_utils/custom_utils/pointcloud_utils.py`:

- `_normalize_vector` → `normalize_vector`
- `rotation_matrix_from_axis_angle`
- `transform_points`
- `transform_point_between_camera_poses` →
  `transform_point_between_poses`
- `compensate_camera_motion` → `compensate_sensor_motion`
- `camera_to_base_transform` → `optical_camera_to_base_transform`

From `steernav/geometry.py`:

- `quaternion_to_rotation_matrix` →
  `rotation_matrix_from_quaternion_xyzw`
- `interpolate_pose_samples`
- `transform_robot_twist_to_camera` → `transform_twist`
- the generic operation underlying
  `rotate_path_deformation_between_base_frames` →
  `rotate_vectors_between_poses`

Add small canonical primitives where useful:

- `invert_transform`
- `compose_transforms`
- `transform_point`

The shared API should not use camera-, robot-, or path-specific terminology
unless it implements a documented standard frame convention.

### Move to `custom_utils.planning_utils`

From `steernav/geometry.py`:

- `resample_path` → `resample_path_by_arc_length`
- `pick_waypoint` → `lookahead_point_along_path`

Reconcile these with the existing functions:

- `start_to_current`
- `odom_to_robot`

Where possible, implement planar path conversion using canonical transform
primitives instead of maintaining independent matrix conventions.

### Keep in shared `pointcloud_utils`

These are reusable point-cloud operations, not rigid geometry:

- ground-plane candidate selection and fitting
- ground-plane correction estimation
- height and region-of-interest filtering
- conversion of points to BEV indices
- pseudo-laserscan generation
- organized point-cloud occlusion processing
- point-cloud visibility ray tracing

Private helpers may remain beside the public operation that owns them.

### Candidates for later shared modules

Potential `bev_grid.py` contents:

- `BEVGrid`
- `world_to_grid`
- `bresenham_line`
- `rasterize_lines_vectorized`
- `rasterize_organized_occluders`
- `get_inbound_points`
- `raytrace_visibility_from_points`

Potential `occupancy_utils.py` contents:

- `compute_esdf_from_occupancy`
- `inflate_obstacles_via_esdf`
- `build_unknown_mask`
- `obstacle_and_unknown_cost`

These later splits are not prerequisites for the geometry migration.

### Keep specifically in SteerNav

Goal-projection policy:

- `bbox_to_pointmap_bounds`
- `_estimate_camera_goal_from_valid_mask`
- `_build_overlap_exclusion_mask`
- `estimate_goal_from_bbox_columns`

Dynamic-obstacle policy:

- `merge_points`
- `_sweep_point_patch`
- `update_points`

`update_points` depends on Supervision detections, tracker IDs, history queues,
missed-track handling, velocity thresholds, dynamic classification, prediction
horizon, and organized point-map mutation. Those are SteerNav behavior rather
than general geometry.

Low-level operations discovered while extracting these modules may be promoted
to `custom_utils` only when they have a project-independent API and tests.

## Canonical geometry API

The intended API is approximately:

```python
def normalize_vector(vector, eps=1e-8): ...

def rotation_matrix_from_axis_angle(axis, angle_radians): ...

def rotation_matrix_from_quaternion_xyzw(quaternion): ...

def invert_transform(target_from_source): ...

def compose_transforms(target_from_intermediate, intermediate_from_source): ...

def transform_points(target_from_source, points_source): ...

def transform_point(target_from_source, point_source): ...

def transform_point_between_poses(
    point_source,
    world_from_source,
    world_from_target,
): ...

def rotate_vectors_between_poses(
    vectors_source,
    world_from_source,
    world_from_target,
): ...

def transform_twist(
    linear_velocity_source,
    angular_velocity_source,
    target_from_source,
): ...

def interpolate_pose_samples(
    pose_samples,
    timestamp,
    max_endpoint_gap=0.1,
): ...

def compensate_sensor_motion(
    observed_relative_velocity,
    object_position_sensor,
    sensor_linear_velocity,
    sensor_angular_velocity,
): ...
```

Before implementation, verify and document whether `transform_twist` maps the
twist to the target-frame origin or merely re-expresses vector components. The
current SteerNav implementation includes the sensor lever arm and therefore
does both an origin shift and a basis change.

## Migration phases

### Phase 0: Establish a baseline

1. Run the complete test suite and record the result.
2. Add characterization tests for any function that will move but is not
   already covered.
3. Record current function signatures and repository-wide call sites.
4. Confirm whether external projects import public names from `custom_utils`.

Gate:

- The baseline suite passes.
- Every moved function has direct numerical tests.

### Phase 1: Add canonical geometry primitives

1. Create `custom_utils/geometry_utils.py`.
2. Move or reimplement the generic functions listed above without changing
   their numerical behavior.
3. Add tests covering:
   - identity transforms
   - translation and rotation
   - arbitrary leading point-array dimensions
   - inverse and composition consistency
   - quaternion normalization and invalid inputs
   - pose interpolation endpoints and SLERP
   - vector rotation without translation
   - twist lever-arm behavior
   - sensor-motion compensation
4. Keep compatibility wrappers in the old modules initially.

Gate:

- New and existing geometry tests pass.
- New geometry utilities import only NumPy and standard-library modules.

### Phase 2: Migrate `pointcloud_utils` callers

1. Import canonical functions from `geometry_utils`.
2. Replace internal calls to the old local implementations.
3. Preserve old public names as aliases with deprecation comments if external
   callers may exist.
4. Remove private duplicate implementations after all local callers migrate.

Gate:

- Point-cloud and ESDF tests pass unchanged.
- Repository search finds no duplicate transform mathematics.

### Phase 3: Migrate SteerNav geometry callers

1. Update `ros_inference.py` to use canonical geometry functions.
2. Replace the local path-deformation rotation with
   `rotate_vectors_between_poses` or a thin SteerNav semantic wrapper.
3. Update dynamic-obstacle pose alignment to use
   `transform_point_between_poses`.
4. Remove corresponding functions from `steernav/geometry.py` once all callers
   are migrated.

Gate:

- ROS-independent tests pass.
- Online modules compile.
- Numerical outputs match characterization tests.

### Phase 4: Migrate generic path primitives

1. Add `resample_path_by_arc_length` and
   `lookahead_point_along_path` to `planning_utils.py`.
2. Update SteerNav callers and tests.
3. Decide whether existing names receive compatibility aliases.
4. Remove `steernav/geometry.py` if it becomes empty; otherwise rename it to
   reflect any remaining SteerNav-specific responsibility.

Gate:

- Path resampling and lookahead tests pass.
- No application module owns duplicate path primitives.

### Phase 5: Move SteerNav policy out of `custom_utils`

1. Create `steernav/dynamic_obstacles.py`.
2. Move `merge_points`, `_sweep_point_patch`, and `update_points` into it.
3. Create `steernav/goal_projection.py` if the goal-projection functions are
   still used by SteerNav.
4. Update imports in online and offline entry points.
5. Keep shared mathematical helpers in `custom_utils`.

Gate:

- Dynamic-obstacle tests pass against the new SteerNav module.
- `custom_utils` no longer depends on Supervision for generic point-cloud use,
  unless another shared module explicitly requires it.

### Phase 6: Correct the online/offline planning boundary

The current worktree moved the complete former `steer_dummy_path.py`
implementation into `trajectory_optimizer.py` and left an offline wrapper.
That was a mechanical rename rather than the intended extraction.

Correct it by:

1. Restoring offline model inference, stream handling, visualization, CLI
   parsing, and `main()` to `steer_dummy_path.py`.
2. Keeping only ESDF path-deformation policy and its direct helpers in
   `trajectory_optimizer.py`.
3. Having both `ros_inference.py` and `steer_dummy_path.py` import the shared
   optimizer.
4. Ensuring neither application entry point imports the other.

Gate:

- Both entry points compile.
- Offline invocation remains backward compatible.
- `trajectory_optimizer.py` has no model-loading, stream, CLI, or video-output
  responsibility.

### Phase 7: Optional BEV and occupancy cleanup

Only after the geometry and ownership migrations stabilize:

1. Consider extracting BEV grid/raycasting primitives.
2. Consider extracting occupancy and ESDF primitives.
3. Keep `pointcloud_to_esdf_pipeline` as an application composition until its
   policy choices are explicit configuration.

This phase should be a separate change set.

## Compatibility strategy

For public functions that may be consumed outside this repository, retain a
temporary alias in the original module:

```python
from .geometry_utils import transform_point_between_poses


def transform_point_between_camera_poses(
    point_camera,
    world_from_camera,
    target_world_from_camera,
):
    """Compatibility wrapper; use transform_point_between_poses."""
    return transform_point_between_poses(
        point_camera,
        world_from_camera,
        target_world_from_camera,
    )
```

Do not silently change argument order while renaming. Compatibility wrappers
must preserve the original signature.

Remove aliases only after checking all known consuming projects.

## Validation checklist

Run after every phase:

```bash
UV_CACHE_DIR=/tmp/steernav-uv-cache uv run python -m pytest
UV_CACHE_DIR=/tmp/steernav-uv-cache uv run python -m py_compile \
  steernav/ros_inference.py \
  steernav/steer_dummy_path.py \
  steernav/trajectory_optimizer.py
git diff --check
```

Also verify when ROS is available:

- The online node starts and loads both models.
- Episode reset clears temporal state.
- Static-scene optimized paths match the pre-migration behavior.
- Dynamic-obstacle projected points match recorded fixtures.
- Published waypoint values match within an agreed numerical tolerance.
- The offline pipeline produces equivalent paths and visualizations.

## Non-goals

This migration does not initially:

- change planner weights or thresholds
- change ESDF construction behavior
- change detector or tracker behavior
- introduce new ROS message types
- redesign configuration
- optimize runtime performance
- split every large utility module at once

Those changes should be evaluated independently after module ownership is
clear.
