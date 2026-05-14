# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Ported from ROS Noetic to ROS2 Humble by RPL-CS-UCL

# python
import os
import time
import warnings
from typing import Optional, Tuple

import cv2
import cv_bridge
import numpy as np
import PIL

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor


import scipy.spatial.transform as stf
import tf2_ros
import tf2_geometry_msgs
import torch

from geometry_msgs.msg import Point, PointStamped, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Joy
from std_msgs.msg import Float32, Header, Int16
from visualization_msgs.msg import Marker

# VIPlanner
from .vip_inference import VIPlannerInference

warnings.filterwarnings("ignore")

# conversion matrix from ROS camera convention (z-forward, y-down, x-right)
# to robotics convention (x-forward, y-left, z-up)
ROS_TO_ROBOTICS_MAT = stf.Rotation.from_euler("XYZ", [-90, 0, -90], degrees=True).as_matrix()
CAMERA_FLIP_MAT = stf.Rotation.from_euler("XYZ", [180, 0, 0], degrees=True).as_matrix()


class VIPlannerNode(Node):
    """VIPlanner ROS2 Node Class"""

    debug: bool = False

    def __init__(self):
        super().__init__("viplanner_node")

        # ── Declare all parameters ──────────────────────────────────
        # planning
        self.declare_parameter("main_freq", 5)
        self.declare_parameter("image_flip", True)
        self.declare_parameter("conv_dist", 0.5)
        self.declare_parameter("max_depth", 10.0)
        self.declare_parameter("overlap_ratio_thres", 0.7)
        self.declare_parameter("depth_zero_ratio_thres", 0.7)
        # networks
        self.declare_parameter("model_save", "models")
        self.declare_parameter("m2f_cfg_file", "")
        self.declare_parameter("m2f_model_path", "")
        # ROS topics
        self.declare_parameter("depth_topic", "/rgbd_camera/depth/image")
        self.declare_parameter("depth_info_topic", "/depth_camera_front_upper/depth/camera_info")
        self.declare_parameter("rgb_topic", "/wide_angle_camera_front/image_raw/compressed")
        self.declare_parameter("rgb_info_topic", "/wide_angle_camera_front/camera_info")
        self.declare_parameter("goal_topic", "/mp_waypoint")
        self.declare_parameter("path_topic", "/viplanner/path")
        self.declare_parameter("m2f_timer_topic", "/viplanner/m2f_timer")
        self.declare_parameter("depth_uint_type", False)
        self.declare_parameter("compressed", True)
        self.declare_parameter("mount_cam_frame", "")
        # frame ids
        self.declare_parameter("robot_id", "base")
        self.declare_parameter("world_id", "odom")
        # fear reaction
        self.declare_parameter("is_fear_act", True)
        self.declare_parameter("buffer_size", 10)
        self.declare_parameter("angular_thread", 0.3)
        self.declare_parameter("track_dist", 0.5)
        # smart joystick
        self.declare_parameter("joyGoal_scale", 0.5)

        # ── Read parameters into a simple namespace ─────────────────
        self.cfg = self._read_params()
        self._last_wait_log_time = 0.0
        self._log_effective_config()

        # init planner algo class
        self.vip_algo = VIPlannerInference(self.cfg)
        self.get_logger().info(
            "Model mode sem=%s rgb=%s device=%s"
            % (
                self.vip_algo.train_cfg.sem,
                self.vip_algo.train_cfg.rgb,
                getattr(self.vip_algo, "_device", "unknown"),
            )
        )

        if self.vip_algo.train_cfg.sem:
            try:
                from .m2f_inference import Mask2FormerInference
            except ImportError as exc:
                raise RuntimeError(
                    "Semantic mode requires the mmdet/mmcv stack to be installed in the image."
                ) from exc
            # init semantic network
            self.m2f_inference = Mask2FormerInference(
                config_file=self.cfg.m2f_config_path,
                checkpoint_file=self.cfg.m2f_model_path,
            )
            self.m2f_timer_data = Float32()
            self.m2f_timer_pub = self.create_publisher(Float32, self.cfg.m2f_timer_topic, 10)

        # init transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # init bridge
        self.bridge = cv_bridge.CvBridge()

        # init flags
        self.is_goal_init = False
        self.ready_for_planning_depth = False
        self.ready_for_planning_rgb_sem = False
        self.is_goal_processed = False
        self.is_smartjoy = False
        self.goal_cam_frame_set = False
        self.init_goal_trans = True

        # planner status
        self.planner_status = Int16()
        self.planner_status.data = 0

        # fear reaction
        self.fear_buffter = 0
        self.is_fear_reaction = False

        # process time
        self.vip_timer_data = Float32()
        self.vip_timer_pub = self.create_publisher(Float32, "/viplanner/timer", 10)

        # depth and rgb image message
        self.depth_header = Header()
        self.rgb_header = Header()
        self.depth_img: np.ndarray = None
        self.depth_pose: np.ndarray = None
        self.sem_rgb_img: np.ndarray = None
        self.sem_rgb_odom: np.ndarray = None
        self.pix_depth_cam_frame: np.ndarray = None

        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            Image,
            self.cfg.depth_topic,
            self.depthCallback,
            sensor_qos,
        )

        if self.vip_algo.train_cfg.sem or self.vip_algo.train_cfg.rgb:
            if self.cfg.compressed:
                self.create_subscription(
                    CompressedImage,
                    self.cfg.rgb_topic,
                    self.imageCallbackCompressed,
                    sensor_qos,
                )
            else:
                self.create_subscription(
                    Image,
                    self.cfg.rgb_topic,
                    self.imageCallback,
                    sensor_qos,
                )
        else:
            self.ready_for_planning_rgb_sem = True

        # subscribe to further topics
        self.goal_world_frame: PointStamped = None
        self.goal_robot_frame: PointStamped = None
        self.create_subscription(PointStamped, self.cfg.goal_topic, self.goalCallback, 10)
        self.create_subscription(Joy, "/joy", self.joyCallback, 10)

        # camera info subscribers
        self.K_depth: np.ndarray = np.zeros((3, 3))
        self.K_rgb: np.ndarray = np.zeros((3, 3))
        self.depth_intrinsics_init: bool = False
        self.rgb_intrinsics_init: bool = False
        self.create_subscription(CameraInfo, self.cfg.depth_info_topic, self.depthCamInfoCallback, 10)
        self.create_subscription(CameraInfo, self.cfg.rgb_info_topic, self.rgbCamInfoCallback, 10)

        # publish effective goal pose and marker with max distance (circle) and direct line (line)
        self.crop_goal_pub = self.create_publisher(PointStamped, "viplanner/visualization/crop_goal", 1)
        self.marker_circ_pub = self.create_publisher(Marker, "viplanner/visualization/odom_circle", 1)
        self.marker_line_pub = self.create_publisher(Marker, "viplanner/visualization/goal_line", 1)
        self.marker_circ: Marker = None
        self.marker_line: Marker = None
        self.max_goal_distance: float = 10.0
        self._init_markers()

        # planning status topics
        self.status_pub = self.create_publisher(Int16, "/viplanner/status", 10)

        # path topics
        self.path_pub = self.create_publisher(Path, self.cfg.path_topic, 10)
        self.path_viz_pub = self.create_publisher(Path, self.cfg.path_topic + "_viz", 10)
        self.fear_path_pub = self.create_publisher(Path, self.cfg.path_topic + "_fear", 10)

        # viz semantic image
        self.m2f_pub = self.create_publisher(CompressedImage, "/viplanner/sem_image/compressed", 3)

        # ── Timer-based main loop ───────────────────────────────────
        main_freq = self.cfg.main_freq
        self.timer = self.create_timer(1.0 / main_freq, self.spin_once)

        self.get_logger().info("VIPlanner Ready.")

    # ─── Parameter helpers ──────────────────────────────────────────

    def _read_params(self):
        """Read all declared parameters into a simple namespace object."""

        class Cfg:
            pass

        cfg = Cfg()
        cfg.main_freq = self.get_parameter("main_freq").value
        cfg.image_flip = self.get_parameter("image_flip").value
        cfg.conv_dist = self.get_parameter("conv_dist").value
        cfg.max_depth = self.get_parameter("max_depth").value
        cfg.overlap_ratio_thres = self.get_parameter("overlap_ratio_thres").value
        cfg.depth_zero_ratio_thres = self.get_parameter("depth_zero_ratio_thres").value

        cfg.model_save = self.get_parameter("model_save").value
        cfg.m2f_config_path = self.get_parameter("m2f_cfg_file").value
        cfg.m2f_model_path = self.get_parameter("m2f_model_path").value

        cfg.depth_topic = self.get_parameter("depth_topic").value
        cfg.depth_info_topic = self.get_parameter("depth_info_topic").value
        cfg.rgb_topic = self.get_parameter("rgb_topic").value
        cfg.rgb_info_topic = self.get_parameter("rgb_info_topic").value
        cfg.goal_topic = self.get_parameter("goal_topic").value
        cfg.path_topic = self.get_parameter("path_topic").value
        cfg.m2f_timer_topic = self.get_parameter("m2f_timer_topic").value
        cfg.depth_uint_type = self.get_parameter("depth_uint_type").value
        cfg.compressed = self.get_parameter("compressed").value

        mount_cam = self.get_parameter("mount_cam_frame").value
        cfg.mount_cam_frame = mount_cam if mount_cam else None

        cfg.robot_id = self.get_parameter("robot_id").value
        cfg.world_id = self.get_parameter("world_id").value

        cfg.is_fear_act = self.get_parameter("is_fear_act").value
        cfg.buffer_size = self.get_parameter("buffer_size").value
        cfg.angular_thread = self.get_parameter("angular_thread").value
        cfg.track_dist = self.get_parameter("track_dist").value
        cfg.joyGoal_scale = self.get_parameter("joyGoal_scale").value

        return cfg

    def _log_effective_config(self):
        self.get_logger().info(
            "Configured topics depth=%s depth_info=%s rgb=%s rgb_info=%s goal=%s path=%s compressed=%s"
            % (
                self.cfg.depth_topic,
                self.cfg.depth_info_topic,
                self.cfg.rgb_topic,
                self.cfg.rgb_info_topic,
                self.cfg.goal_topic,
                self.cfg.path_topic,
                self.cfg.compressed,
            )
        )
        self.get_logger().info(
            "Configured frames robot_id=%s world_id=%s mount_cam_frame=%s"
            % (
                self.cfg.robot_id,
                self.cfg.world_id,
                self.cfg.mount_cam_frame,
            )
        )

    def _log_waiting_state(self):
        now = time.time()
        missing = []
        if not self.ready_for_planning_depth:
            missing.append(f"depth image on {self.cfg.depth_topic}")
        if not self.ready_for_planning_rgb_sem:
            missing.append(f"rgb/sem image on {self.cfg.rgb_topic}")
        if not self.is_goal_init:
            missing.append(f"goal on {self.cfg.goal_topic}")
        if self.is_goal_init and not self.goal_cam_frame_set:
            missing.append("goal-to-camera transform")
        if missing:
            self.get_logger().warning(
                "Waiting for planner inputs: %s" % ", ".join(missing),
                throttle_duration_sec=2.0,
            )
        self._last_wait_log_time = now

    # ─── Main planning loop (called by timer) ──────────────────────

    def spin_once(self):
        if not all((
            self.ready_for_planning_rgb_sem,
            self.ready_for_planning_depth,
            self.is_goal_init,
            self.goal_cam_frame_set,
        )):
            self._log_waiting_state()
            return

        # copy current data
        cur_depth_image = self.depth_img.copy()
        cur_depth_pose = self.depth_pose.copy()

        if self.vip_algo.train_cfg.sem or self.vip_algo.train_cfg.rgb:
            cur_rgb_pose = self.sem_rgb_odom.copy()
            cur_rgb_image = self.sem_rgb_img.copy()
            time_warp = 0.0
        else:
            time_warp = 0.0
            self.time_sem = 0.0

        # project goal
        goal_cam_frame = self.goalProjection(cur_depth_pose=cur_depth_pose)

        # Network Planning
        start = time.time()
        if self.vip_algo.train_cfg.sem or self.vip_algo.train_cfg.rgb:
            waypoints, fear = self.vip_algo.plan(cur_depth_image, cur_rgb_image, goal_cam_frame)
        else:
            waypoints, fear = self.vip_algo.plan_depth(cur_depth_image, goal_cam_frame)
        time_planner = time.time() - start

        start = time.time()

        # transform waypoint to robot frame
        waypoints = (self.cam_rot @ waypoints.T).T + self.cam_offset

        # publish time
        self.vip_timer_data.data = time_planner * 1000
        self.vip_timer_pub.publish(self.vip_timer_data)

        # check goal less than convergence range
        if (
            (np.sqrt(goal_cam_frame[0][0] ** 2 + goal_cam_frame[0][1] ** 2) < self.cfg.conv_dist)
            and self.is_goal_processed
            and (not self.is_smartjoy)
        ):
            self.is_goal_init = False
            if self.planner_status.data == 0:
                self.planner_status.data = 1
                self.status_pub.publish(self.planner_status)
            self.get_logger().info("Goal Arrived")

        # check for path with high risk (=fear) path
        if fear > 0.7:
            self.is_fear_reaction = True
            is_track_ahead = self.isForwardTraking(waypoints)
            self.fearPathDetection(fear, is_track_ahead)
            if self.is_fear_reaction:
                self.get_logger().warning("current path prediction is invalid.", throttle_duration_sec=2.0)
                if self.planner_status.data == 0:
                    self.planner_status.data = -1
                    self.status_pub.publish(self.planner_status)

        # publish path
        self.pubPath(waypoints, self.is_goal_init)

        time_other = time.time() - start
        total_time = round(time_warp + time_planner + time_other, 4)
        self.get_logger().debug(
            f"Path predicted in {total_time}s"
            f"  warp: {round(time_warp, 4)}s"
            f"  planner: {round(time_planner, 4)}s"
            f"  other: {round(time_other, 4)}s"
        )

    # ─── GOAL PROJECTION ────────────────────────────────────────────

    def goalProjection(self, cur_depth_pose: np.ndarray):
        cur_goal_robot_frame = np.array([
            self.goal_robot_frame.point.x,
            self.goal_robot_frame.point.y,
            self.goal_robot_frame.point.z,
        ])
        cur_goal_world_frame = np.array([
            self.goal_world_frame.point.x,
            self.goal_world_frame.point.y,
            self.goal_world_frame.point.z,
        ])

        if np.linalg.norm(cur_goal_robot_frame[:2]) > self.max_goal_distance:
            cur_goal_robot_frame[:2] = (
                cur_goal_robot_frame[:2]
                / np.linalg.norm(cur_goal_robot_frame[:2])
                * (self.max_goal_distance / 2)
            )
            crop_goal = PointStamped()
            crop_goal.header.stamp = self.depth_header.stamp
            crop_goal.header.frame_id = self.cfg.robot_id
            crop_goal.point.x = float(cur_goal_robot_frame[0])
            crop_goal.point.y = float(cur_goal_robot_frame[1])
            crop_goal.point.z = float(cur_goal_robot_frame[2])
            self.crop_goal_pub.publish(crop_goal)

            # update markers
            self.marker_circ.color.a = 0.1
            self.marker_circ.pose.position = Point(
                x=float(cur_depth_pose[0]),
                y=float(cur_depth_pose[1]),
                z=float(cur_depth_pose[2]),
            )
            self.marker_circ_pub.publish(self.marker_circ)
            self.marker_line.points = []
            self.marker_line.points.append(
                Point(x=float(cur_depth_pose[0]), y=float(cur_depth_pose[1]), z=float(cur_depth_pose[2]))
            )
            self.marker_line.points.append(
                Point(x=float(cur_goal_world_frame[0]), y=float(cur_goal_world_frame[1]), z=float(cur_goal_world_frame[2]))
            )
            self.marker_line_pub.publish(self.marker_line)
        else:
            self.marker_circ.color.a = 0.0
            self.marker_circ_pub.publish(self.marker_circ)
            self.marker_line.points = []
            self.marker_line_pub.publish(self.marker_line)

        goal_cam_frame = self.cam_rot.T @ (cur_goal_robot_frame - self.cam_offset).T
        return torch.tensor(goal_cam_frame, dtype=torch.float32)[None, ...]

    def _init_markers(self):
        if isinstance(self.vip_algo.train_cfg.data_cfg, list):
            self.max_goal_distance = self.vip_algo.train_cfg.data_cfg[0].max_goal_distance
        else:
            self.max_goal_distance = self.vip_algo.train_cfg.data_cfg.max_goal_distance

        # setup circle marker
        self.marker_circ = Marker()
        self.marker_circ.header.frame_id = self.cfg.world_id
        self.marker_circ.type = Marker.SPHERE
        self.marker_circ.action = Marker.ADD
        self.marker_circ.scale.x = self.max_goal_distance
        self.marker_circ.scale.y = self.max_goal_distance
        self.marker_circ.scale.z = 0.01
        self.marker_circ.color.a = 0.1
        self.marker_circ.color.r = 0.0
        self.marker_circ.color.g = 0.0
        self.marker_circ.color.b = 1.0
        self.marker_circ.pose.orientation.w = 1.0

        # setup line marker
        self.marker_line = Marker()
        self.marker_line.header.frame_id = self.cfg.world_id
        self.marker_line.type = Marker.LINE_STRIP
        self.marker_line.action = Marker.ADD
        self.marker_line.scale.x = 0.1
        self.marker_line.color.a = 1.0
        self.marker_line.color.r = 0.0
        self.marker_line.color.g = 0.0
        self.marker_line.color.b = 1.0
        self.marker_line.pose.orientation.w = 1.0

    # ─── PATH PUB, GOAL and ODOM SUB and FEAR DETECTION ─────────────

    def pubPath(self, waypoints, is_goal_init=True):
        if not is_goal_init:
            # empty path crashes CMU pathfollower
            return
        
        poses = []
        for p in waypoints:
            pose = PoseStamped()
            pose.pose.position.x = float(p[0])
            pose.pose.position.y = float(p[1])
            poses.append(pose)

        # Wait for the transform from base frame to odom frame
        trans = None
        try:
            trans = self.tf_buffer.lookup_transform(
                self.cfg.world_id,
                self.cfg.robot_id,
                Time.from_msg(self.depth_header.stamp),
                Duration(seconds=1.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            self.get_logger().error(
                f"Failed to lookup transform from {self.cfg.robot_id} to {self.cfg.world_id}"
            )
            return

        if trans is None:
            return

        # Transform each pose from base to odom frame
        transformed_poses = []
        transformed_poses_np = np.zeros(waypoints.shape)
        for idx, pose in enumerate(poses):
            transformed_pose = tf2_geometry_msgs.do_transform_pose_stamped(
                PoseStamped(header=self.depth_header, pose=pose.pose), trans
            )
            transformed_poses.append(transformed_pose)
            transformed_poses_np[idx] = np.array([
                transformed_pose.pose.position.x,
                transformed_pose.pose.position.y,
                transformed_pose.pose.position.z,
            ])

        success, curr_depth_cam_odom_pose = self.poseCallback(
            self.depth_header.frame_id, Time()
        )

        # remove all waypoints already passed
        front_poses = np.linalg.norm(
            transformed_poses_np - transformed_poses_np[0], axis=1
        ) > np.linalg.norm(
            curr_depth_cam_odom_pose[:3] - transformed_poses_np[0]
        )
        poses = [pose for idx, pose in enumerate(poses) if front_poses[idx]]
        transformed_poses = [pose for idx, pose in enumerate(transformed_poses) if front_poses[idx]]

        # add header
        base_path = Path()
        base_fear_path = Path()
        odom_path = Path()

        base_path.header.frame_id = base_fear_path.header.frame_id = self.cfg.robot_id
        odom_path.header.frame_id = self.cfg.world_id
        base_path.header.stamp = base_fear_path.header.stamp = odom_path.header.stamp = self.depth_header.stamp

        if self.is_fear_reaction:
            base_fear_path.poses = poses
            base_path.poses = poses[:1]
        else:
            base_path.poses = poses
        odom_path.poses = transformed_poses

        self.fear_path_pub.publish(base_fear_path)
        self.path_pub.publish(base_path)
        self.path_viz_pub.publish(odom_path)

    def fearPathDetection(self, fear, is_forward):
        if fear > 0.5 and is_forward:
            if not self.is_fear_reaction:
                self.fear_buffter = self.fear_buffter + 1
        elif self.is_fear_reaction:
            self.fear_buffter = self.fear_buffter - 1
        if self.fear_buffter > self.cfg.buffer_size:
            self.is_fear_reaction = True
        elif self.fear_buffter <= 0:
            self.is_fear_reaction = False

    def isForwardTraking(self, waypoints):
        xhead = np.array([1.0, 0])
        phead = None
        for p in waypoints:
            if np.linalg.norm(p[0:2]) > self.cfg.track_dist:
                phead = p[0:2] / np.linalg.norm(p[0:2])
                break
        if np.all(phead is not None) and phead.dot(xhead) > 1.0 - self.cfg.angular_thread:
            return True
        return False

    def joyCallback(self, joy_msg):
        if joy_msg.buttons[4] > 0.9:
            self.get_logger().info("Switch to Smart Joystick mode ...")
            self.is_smartjoy = True
            self.fear_buffter = 0
            self.is_fear_reaction = False
        if self.is_smartjoy:
            if np.sqrt(joy_msg.axes[3] ** 2 + joy_msg.axes[4] ** 2) < 1e-3:
                self.fear_buffter = 0
                self.is_fear_reaction = False
                self.is_goal_init = False
            else:
                joy_goal = PointStamped()
                joy_goal.header.frame_id = self.cfg.robot_id
                joy_goal.point.x = float(joy_msg.axes[4] * self.cfg.joyGoal_scale)
                joy_goal.point.y = float(joy_msg.axes[3] * self.cfg.joyGoal_scale)
                joy_goal.point.z = 0.0
                joy_goal.header.stamp = self.get_clock().now().to_msg()
                self.goal_pose = joy_goal
                self.is_goal_init = True
                self.is_goal_processed = False

    def goalCallback(self, msg):
        self.get_logger().info("Received a new goal")
        self.goal_pose = msg
        self.is_smartjoy = False
        self.is_goal_init = True
        self.is_goal_processed = False
        self.fear_buffter = 0
        self.is_fear_reaction = False
        self.planner_status.data = 0

    # ─── RGB IMAGE AND DEPTH CALLBACKS ──────────────────────────────

    def poseCallback(
        self, frame_id: str, img_stamp, target_frame_id: Optional[str] = None
    ) -> Tuple[bool, np.ndarray]:
        target_frame_id = target_frame_id if target_frame_id else self.cfg.world_id
        try:
            if self.cfg.mount_cam_frame is None:
                transform = self.tf_buffer.lookup_transform(
                    target_frame_id, frame_id, img_stamp, Duration(seconds=4.0)
                )
            else:
                frame_id = self.cfg.mount_cam_frame
                transform = self.tf_buffer.lookup_transform(
                    target_frame_id,
                    self.cfg.mount_cam_frame,
                    img_stamp,
                    Duration(seconds=4.0),
                )
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            pose = np.array([
                translation.x, translation.y, translation.z,
                rotation.x, rotation.y, rotation.z, rotation.w,
            ])
            return True, pose
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        )as e:
            self.get_logger().error(
                f"Pose Fail to transfer {frame_id} into {target_frame_id} frame: "
                f"{type(e).__name__}: {e}"
            )
        return False, np.zeros(7)

    def imageCallback(self, rgb_msg: Image):
        self.get_logger().debug(f"Received rgb image {rgb_msg.header.frame_id}")

        success, pose = self.poseCallback(
            rgb_msg.header.frame_id, Time.from_msg(rgb_msg.header.stamp)
        )
        if not success:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except cv_bridge.CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        if self.vip_algo.train_cfg.sem:
            image = self.semPrediction(image)

        self.sem_rgb_odom = pose
        self.sem_rgb_img = image
        self.ready_for_planning_rgb_sem = True

        # publish the semantic image
        if self.vip_algo.train_cfg.sem:
            image_pub = cv2.resize(image, (480, 360))
            image_pub = cv2.cvtColor(image_pub, cv2.COLOR_RGB2BGR)
            success, compressed_image = cv2.imencode(".jpg", image_pub)
            if not success:
                self.get_logger().error("Failed to compress semantic image")
                return
            sem_msg = CompressedImage()
            sem_msg.header = rgb_msg.header
            sem_msg.format = "jpeg"
            sem_msg.data = compressed_image.tobytes()
            self.m2f_pub.publish(sem_msg)

    def imageCallbackCompressed(self, rgb_msg: CompressedImage):
        self.get_logger().debug(
            f"Received rgb image {rgb_msg.header.frame_id}"
        )

        self.rgb_header = rgb_msg.header

        success, pose = self.poseCallback(
            rgb_msg.header.frame_id, Time.from_msg(rgb_msg.header.stamp)
        )
        if not success:
            return

        try:
            rgb_arr = np.frombuffer(rgb_msg.data, np.uint8)
            image = cv2.imdecode(rgb_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(str(e))
            return

        if self.vip_algo.train_cfg.sem:
            image = self.semPrediction(image)

        self.sem_rgb_img = image
        self.sem_rgb_odom = pose
        self.ready_for_planning_rgb_sem = True

        # publish the semantic image
        if self.vip_algo.train_cfg.sem:
            image_pub = cv2.resize(image, (480, 360))
            image_pub = cv2.cvtColor(image_pub, cv2.COLOR_RGB2BGR)
            success, compressed_image = cv2.imencode(".jpg", image_pub)
            if not success:
                self.get_logger().error("Failed to compress semantic image")
                return
            sem_msg = CompressedImage()
            sem_msg.header = rgb_msg.header
            sem_msg.format = "jpeg"
            sem_msg.data = compressed_image.tobytes()
            self.m2f_pub.publish(sem_msg)

    def semPrediction(self, image):
        start = time.time()
        image = self.m2f_inference.predict(image)
        self.time_sem = time.time() - start
        self.m2f_timer_data.data = float(self.time_sem * 1000)
        self.m2f_timer_pub.publish(self.m2f_timer_data)
        return image

    def depthCallback(self, depth_msg: Image):
        self.get_logger().debug(
            f"Received depth image {depth_msg.header.frame_id}"
        )

        self.depth_header = depth_msg.header
        success, self.depth_pose = self.poseCallback(
            depth_msg.header.frame_id, Time.from_msg(depth_msg.header.stamp)
        )
        if not success:
            return

        # DEPTH Image — use cv_bridge instead of ros_numpy
        try:
            image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except cv_bridge.CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        image = image.astype(np.float32)
        image[~np.isfinite(image)] = 0
        if self.cfg.depth_uint_type:
            image = image / 1000.0
        image[image > self.cfg.max_depth] = 0.0
        if self.cfg.image_flip:
            image = PIL.Image.fromarray(image)
            self.depth_img = np.array(image.transpose(PIL.Image.Transpose.ROTATE_180))
        else:
            self.depth_img = image

        # transform goal into robot frame and world frame
        if self.is_goal_init:
            if self.goal_pose.header.frame_id != self.cfg.robot_id:
                try:
                    trans = self.tf_buffer.lookup_transform(
                        self.cfg.robot_id,
                        self.goal_pose.header.frame_id,
                        Time.from_msg(self.depth_header.stamp),
                        Duration(seconds=1.0),
                    )
                    self.goal_robot_frame = tf2_geometry_msgs.do_transform_point(self.goal_pose, trans)
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ):
                    self.get_logger().error(
                        f"Goal: Fail to transfer {self.goal_pose.header.frame_id} into {self.cfg.robot_id}"
                    )
                    return
            else:
                self.goal_robot_frame = self.goal_pose

            if self.goal_pose.header.frame_id != self.cfg.world_id:
                try:
                    trans = self.tf_buffer.lookup_transform(
                        self.cfg.world_id,
                        self.goal_pose.header.frame_id,
                        Time.from_msg(self.depth_header.stamp),
                        Duration(seconds=1.0),
                    )
                    self.goal_world_frame = tf2_geometry_msgs.do_transform_point(self.goal_pose, trans)
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ):
                    self.get_logger().error(
                        f"Goal: Fail to transfer {self.goal_pose.header.frame_id} into {self.cfg.world_id}"
                    )
                    return
            else:
                self.goal_world_frame = self.goal_pose

            # get static transform and cam offset
            if self.init_goal_trans:
                if self.cfg.mount_cam_frame is None:
                    success, tf_robot_depth = self.poseCallback(
                        self.depth_header.frame_id,
                        Time.from_msg(self.depth_header.stamp),
                        self.cfg.robot_id,
                    )
                else:
                    success, tf_robot_depth = self.poseCallback(
                        self.cfg.mount_cam_frame,
                        Time.from_msg(self.depth_header.stamp),
                        self.cfg.robot_id,
                    )

                if not success:
                    return
                self.cam_offset = tf_robot_depth[0:3]
                self.cam_rot = stf.Rotation.from_quat(tf_robot_depth[3:7]).as_matrix() @ ROS_TO_ROBOTICS_MAT
                if not self.cfg.image_flip:
                    self.cam_rot = self.cam_rot @ CAMERA_FLIP_MAT
                if self.debug:
                    print(
                        "CAM ROT",
                        stf.Rotation.from_matrix(self.cam_rot).as_euler("xyz", degrees=True),
                    )
                self.init_goal_trans = False

            self.goal_cam_frame_set = True

        # declare ready for planning
        self.ready_for_planning_depth = True
        self.is_goal_processed = True

    # ─── Camera Info Callbacks ──────────────────────────────────────

    def depthCamInfoCallback(self, cam_info_msg: CameraInfo):
        if not self.depth_intrinsics_init:
            self.get_logger().info("Received depth camera info")
            self.K_depth = np.array(cam_info_msg.k).reshape(3, 3)
            self.depth_intrinsics_init = True

    def rgbCamInfoCallback(self, cam_info_msg: CameraInfo):
        if not self.rgb_intrinsics_init:
            self.get_logger().info("Received rgb camera info")
            self.K_rgb = np.array(cam_info_msg.k).reshape(3, 3)
            self.rgb_intrinsics_init = True


def main(args=None):
    rclpy.init(args=args)
    node = VIPlannerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
