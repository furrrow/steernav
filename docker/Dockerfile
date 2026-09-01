FROM osrf/ros:humble-desktop

# 1. Install basic tools + CycloneDDS RMW
RUN apt-get update && apt-get install -y \
    curl ca-certificates wget git build-essential \
    python3-numpy vim openssh-client tmux \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# 2. Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# 3. Set CycloneDDS as default RMW
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# 4. Working directory
WORKDIR /workspace

# 5. Entry point: source ROS + workspace
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]