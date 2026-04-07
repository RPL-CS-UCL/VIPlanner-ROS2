# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# SPDX-License-Identifier: BSD-3-Clause
# Ported to ROS2 by RPL-CS-UCL

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("viplanner_ros2")
    default_config = os.path.join(pkg_share, "config", "viplanner.yaml")

    return LaunchDescription([
        Node(
            package="viplanner_ros2",
            executable="viplanner_node",
            name="viplanner_node",
            output="screen",
            parameters=[default_config],
        ),
    ])
