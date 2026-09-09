#!/bin/bash

# bag_name
BAG_NAME=${1:-rosbag_record_name}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -z "$1" ]; then
	read -p "Enter bag name to be recorded: " BAG_NAME
	if [ -z "$BAG_NAME" ]; then
		echo "Bag name required, exiting..."
		exit 1
	fi
else
	BAG_NAME=$1
fi

OUTPUT_NAME="${BAG_NAME}_${TIMESTAMP}"
echo "Recording rosbag: $OUTPUT_NAME"

ros2 bag record \
	-o "$OUTPUT_NAME" \
	/a200_0648/cmd_vel \
	/a200_0648/platform/odom/filtered \
	/husky/policy_path \
	/husky/waypoint \
	/husky/steered_waypoint \
	/camera/camera/color/image_raw/compressed \
	/camera/camera/color/camera_info \
	/path \
	/started \
	/next_goal \
	/req_goal \
	/tf \
	/tf_static \




