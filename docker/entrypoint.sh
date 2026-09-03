#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

if [[ -f /workspace/steernav/setup.bash ]]; then
    source /workspace/steernav/setup.bash
fi

exec "$@"