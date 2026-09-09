#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

#uv run "$SCRIPT_DIR/../services/custom_utils/custom_utils/kinematic_controller_odom.py"
#uv run "$SCRIPT_DIR/../services/custom_utils/custom_utils/planner_dwa_ros2.py"
uv run "$SCRIPT_DIR/../services/custom_utils/custom_utils/kinematic_controller_naive.py"