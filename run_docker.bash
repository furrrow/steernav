#!/usr/bin/env bash
set -euo pipefail

CYCLONEDDS_CONFIG="$HOME/.config/cyclonedds-robot.xml"

[[ -f "$CYCLONEDDS_CONFIG" ]] || {
    echo "Missing CycloneDDS config: $CYCLONEDDS_CONFIG" >&2
    exit 1
}

docker run -it --rm \
  --name steernav \
  --user "$(id -u):$(id -g)" \
  --net=host \
  --gpus all \
  -e DISPLAY="${DISPLAY:-}" \
  -e HOME=/home/ros \
  -e ROS_LOCALHOST_ONLY=0 \
  -e ROS_DISABLE_SHARED_MEMORY=1 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e CYCLONEDDS_URI=file:///etc/cyclonedds-robot.xml \
  -e XDG_CACHE_HOME=/home/ros/.cache \
  -e MPLCONFIGDIR=/home/ros/.cache/matplotlib \
  -v "$HOME:/home/ros" \
  -v "$(pwd):/workspace/steernav" \
  -v "$CYCLONEDDS_CONFIG:/etc/cyclonedds-robot.xml:ro" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v "$HOME/.ssh:/home/ros/.ssh:ro" \
  -v "$HOME/.cache/uv:/home/ros/.cache/uv" \
  steernav \
  bash