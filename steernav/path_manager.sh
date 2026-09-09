#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

uv run "$SCRIPT_DIR/../services/custom_utils/custom_utils/path_manager.py"