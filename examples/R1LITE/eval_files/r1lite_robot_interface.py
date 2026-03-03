#!/usr/bin/env python3
"""
R1LITE Robot Interface for Real-World Deployment

This module provides a ROS2-based interface for interacting with the R1LITE
dual-arm robot and its three cameras (head, left wrist, right wrist).

Features:
- Subscribe to 3 camera topics and cache latest images
- Subscribe to robot state (EE poses and gripper states)
- Publish delta actions to robot control topics
- Thread-safe access to observations
- Integration with R1LITE inference pipeline
"""

import threading
import time
import math
from typing import Optional, Tuple, List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped


class SimpleRotation:
    """Lightweight rotation class for quaternion operations."""

    def __init__(self, quat):
        # quat is [x, y, z, w]
        self._quat = np.array(quat, dtype=np.float64)
        norm = np.linalg.norm(self._quat)
        if norm > 1e-6:
            self._quat /= norm
        else:
            self._quat = np.array([0.0, 0.0, 0.0, 1.0])

    def as_quat(self):
        return self._quat

    def __mul__(self, other):
        # Quaternion multiplication
        x1, y1, z1, w1 = self._quat
        x2, y2, z2, w2 = other._quat

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return SimpleRotation([x, y, z, w])

    @staticmethod
    def from_quat(quat):
        return SimpleRotation(quat)

    @staticmethod
    def from_euler(seq, angles, degrees=False):
        """Convert euler angles to quaternion (only supports 'xyz')."""
        if seq != "xyz":
            raise NotImplementedError("Only xyz supported")

        rx, ry, rz = angles
        if degrees:
            rx, ry, rz = np.radians([rx, ry, rz])

        cx = math.cos(rx * 0.5)
        sx = math.sin(rx * 0.5)
        cy = math.cos(ry * 0.5)
        sy = math.sin(ry * 0.5)
        cz = math.cos(rz * 0.5)
        sz = math.sin(rz * 0.5)

        x = sx * cy * cz + cx * sy * sz
        y = cx * sy * cz - sx * cy * sz
        z = cx * cy * sz + sx * sy * cz
        w = cx * cy * cz - sx * sy * sz

        return SimpleRotation([x, y, z, w])


class R1LITERobotInterface(Node):
    """
    ROS2 interface for R1LITE dual-arm robot.

    This class handles:
    - Image acquisition from 3 cameras
    - Robot state monitoring (EE poses and grippers)
    - Action execution (delta EE control)
    """

    def __init__(
        self,
        node_name: str = "r1lite_robot_interface",
        publish_rate: float = 10.0,
    ):
        super().__init__(node_name)

        # QoS profile for best effort communication
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE

        # Camera image subscribers
        self.sub_head_img = self.create_subscription(
            Image,
            "/hdas/camera_head/head/color/image_raw",
            self._head_img_callback,
            qos,
        )
        self.sub_left_img = self.create_subscription(
            Image,
            "/hdas/camera_wrist_left/color/image_raw",
            self._left_img_callback,
            qos,
        )
        self.sub_right_img = self.create_subscription(
            Image,
            "/hdas/camera_wrist_right/color/image_raw",
            self._right_img_callback,
            qos,
        )

        # Robot state subscribers
        self.sub_left_pose = self.create_subscription(
            PoseStamped,
            "/motion_control/pose_ee_arm_left",
            self._left_pose_callback,
            qos,
        )
        self.sub_right_pose = self.create_subscription(
            PoseStamped,
            "/motion_control/pose_ee_arm_right",
            self._right_pose_callback,
            qos,
        )
        self.sub_left_gripper = self.create_subscription(
            JointState,
            "/motion_target/target_position_gripper_left",
            self._left_gripper_callback,
            qos,
        )
        self.sub_right_gripper = self.create_subscription(
            JointState,
            "/motion_target/target_position_gripper_right",
            self._right_gripper_callback,
            qos,
        )

        # Action publishers
        self.pub_left_pose = self.create_publisher(
            PoseStamped,
            "/motion_target/target_pose_arm_left",
            qos,
        )
        self.pub_right_pose = self.create_publisher(
            PoseStamped,
            "/motion_target/target_pose_arm_right",
            qos,
        )
        self.pub_left_gripper = self.create_publisher(
            JointState,
            "/motion_target/target_position_gripper_left",
            qos,
        )
        self.pub_right_gripper = self.create_publisher(
            JointState,
            "/motion_target/target_position_gripper_right",
            qos,
        )

        # Cached observations (thread-safe)
        self.lock = threading.Lock()
        self.head_image: Optional[np.ndarray] = None
        self.left_image: Optional[np.ndarray] = None
        self.right_image: Optional[np.ndarray] = None
        self.left_pose: Optional[PoseStamped] = None
        self.right_pose: Optional[PoseStamped] = None
        self.left_gripper_value: float = 0.0
        self.right_gripper_value: float = 0.0

        # Current EE state for delta integration
        self.left_ee_pos = np.array([0.059, -0.335, 0.343])
        self.right_ee_pos = np.array([0.0585, -0.335, 0.334])
        self.left_ee_rot = SimpleRotation.from_quat([0.0, 0.0, 0.0, 1.0])
        self.right_ee_rot = SimpleRotation.from_quat([0.0, 0.0, 0.0, 1.0])

        self.publish_rate = publish_rate
        self.frame_id = "base_link"
        self.euler_seq = "xyz"

        self.get_logger().info(f"R1LITE Robot Interface initialized")
        self.get_logger().info(f"Publish rate: {publish_rate} Hz")

    # ========== Camera Callbacks ==========

    def _head_img_callback(self, msg: Image):
        """Callback for head camera image."""
        with self.lock:
            self.head_image = self._image_msg_to_numpy(msg)

    def _left_img_callback(self, msg: Image):
        """Callback for left wrist camera image."""
        with self.lock:
            self.left_image = self._image_msg_to_numpy(msg)

    def _right_img_callback(self, msg: Image):
        """Callback for right wrist camera image."""
        with self.lock:
            self.right_image = self._image_msg_to_numpy(msg)

    # ========== State Callbacks ==========

    def _left_pose_callback(self, msg: PoseStamped):
        """Callback for left arm EE pose."""
        with self.lock:
            self.left_pose = msg
            # Update current position for delta integration
            self.left_ee_pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ])
            self.left_ee_rot = SimpleRotation.from_quat([
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ])

    def _right_pose_callback(self, msg: PoseStamped):
        """Callback for right arm EE pose."""
        with self.lock:
            self.right_pose = msg
            # Update current position for delta integration
            self.right_ee_pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ])
            self.right_ee_rot = SimpleRotation.from_quat([
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ])

    def _left_gripper_callback(self, msg: JointState):
        """Callback for left gripper state."""
        with self.lock:
            if msg.position:
                # Gripper value in [0, 100], normalize to [0, 1]
                self.left_gripper_value = float(msg.position[0]) / 100.0

    def _right_gripper_callback(self, msg: JointState):
        """Callback for right gripper state."""
        with self.lock:
            if msg.position:
                # Gripper value in [0, 100], normalize to [0, 1]
                self.right_gripper_value = float(msg.position[0]) / 100.0

    # ========== Helper Methods ==========

    @staticmethod
    def _image_msg_to_numpy(msg: Image) -> np.ndarray:
        """Convert ROS Image message to numpy array."""
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        )

    # ========== Public API ==========

    def get_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get latest images from all three cameras.

        Returns:
            Tuple of (head_image, left_image, right_image), each (H, W, 3) uint8
            Returns None for any camera if no image received yet
        """
        with self.lock:
            return self.head_image, self.left_image, self.right_image

    def get_head_image(self) -> Optional[np.ndarray]:
        """
        Get latest head camera image.

        Returns:
            head_image: (H, W, 3) uint8 array, or None if no image received yet
        """
        with self.lock:
            return self.head_image

    def get_left_image(self) -> Optional[np.ndarray]:
        """
        Get latest left wrist camera image.

        Returns:
            left_image: (H, W, 3) uint8 array, or None if no image received yet
        """
        with self.lock:
            return self.left_image

    def get_right_image(self) -> Optional[np.ndarray]:
        """
        Get latest right wrist camera image.

        Returns:
            right_image: (H, W, 3) uint8 array, or None if no image received yet
        """
        with self.lock:
            return self.right_image

    def get_left_gripper(self) -> float:
        """
        Get current left gripper value.

        Returns:
            gripper_value: float in [0, 1], where 0 is fully open and 1 is fully closed
        """
        with self.lock:
            return self.left_gripper_value

    def get_right_gripper(self) -> float:
        """
        Get current right gripper value.

        Returns:
            gripper_value: float in [0, 1], where 0 is fully open and 1 is fully closed
        """
        with self.lock:
            return self.right_gripper_value

    def get_robot_state(self) -> np.ndarray:
        """
        Get current robot state as 16-dim array.

        Format: [left_xyz(3), left_euler(3), left_gripper(1),
                 right_xyz(3), right_euler(3), right_gripper(1)]

        Returns:
            state: (16,) array
        """
        with self.lock:
            # Convert quaternions to euler angles
            left_quat = self.left_ee_rot.as_quat()
            right_quat = self.right_ee_rot.as_quat()

            left_euler = self._quat_to_euler(left_quat)
            right_euler = self._quat_to_euler(right_quat)

            # Binarize gripper values (0 or 1)
            left_grip_binary = 1.0 if self.left_gripper_value > 0.5 else 0.0
            right_grip_binary = 1.0 if self.right_gripper_value > 0.5 else 0.0

            state = np.concatenate([
                self.left_ee_pos,      # 3
                left_euler,            # 3
                [left_grip_binary],    # 1
                self.right_ee_pos,     # 3
                right_euler,           # 3
                [right_grip_binary],   # 1
            ])

            return state.astype(np.float32)

    def send_delta_action(self, action: np.ndarray) -> None:
        """
        Send delta action to robot.

        Args:
            action: (14,) array with format:
                    [left_xyz(3), left_euler(3), left_gripper(1),
                     right_xyz(3), right_euler(3), right_gripper(1)]
        """
        # Parse action
        left_delta_pos = action[0:3]
        left_delta_euler = action[3:6]
        left_gripper = action[6]
        right_delta_pos = action[7:10]
        right_delta_euler = action[10:13]
        right_gripper = action[13]

        with self.lock:
            # Integrate deltas
            new_left_pos = self.left_ee_pos + left_delta_pos
            new_right_pos = self.right_ee_pos + right_delta_pos

            left_rot_delta = SimpleRotation.from_euler(
                self.euler_seq, left_delta_euler, degrees=False
            )
            right_rot_delta = SimpleRotation.from_euler(
                self.euler_seq, right_delta_euler, degrees=False
            )

            new_left_rot = self.left_ee_rot * left_rot_delta
            new_right_rot = self.right_ee_rot * right_rot_delta

            # Update internal state
            self.left_ee_pos = new_left_pos
            self.right_ee_pos = new_right_pos
            self.left_ee_rot = new_left_rot
            self.right_ee_rot = new_right_rot

        # Publish commands
        self._publish_pose_command(
            new_left_pos, new_left_rot, left_gripper,
            self.pub_left_pose, self.pub_left_gripper, ""
        )
        self._publish_pose_command(
            new_right_pos, new_right_rot, right_gripper,
            self.pub_right_pose, self.pub_right_gripper, ""
        )

    def _publish_pose_command(
        self,
        pos: np.ndarray,
        rot: SimpleRotation,
        gripper_val: float,
        pub_pose,
        pub_gripper,
        gripper_joint_name: str,
    ):
        """Publish pose and gripper command."""
        # Pose message
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])

        quat = rot.as_quat()
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        pub_pose.publish(msg)

        # Gripper message
        g = float(max(0.0, min(1.0, gripper_val)))
        g_100 = float(100.0 * g)

        g_msg = JointState()
        g_msg.header.stamp = msg.header.stamp
        g_msg.name = [gripper_joint_name] if gripper_joint_name else []
        g_msg.position = [g_100]
        pub_gripper.publish(g_msg)

    @staticmethod
    def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to euler angles (xyz sequence)."""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    # ========== Pose Get Functions ==========

    def get_left_pose_euler(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get left arm pose in euler angle representation.

        Returns:
            Tuple of (position, euler_angles, gripper_value)
            - position: (3,) array [x, y, z] in meters
            - euler_angles: (3,) array [roll, pitch, yaw] in radians
            - gripper_value: float in [0, 1]
        """
        with self.lock:
            pos = self.left_ee_pos.copy()
            euler = self._quat_to_euler(self.left_ee_rot.as_quat())
            gripper = self.left_gripper_value
        return pos, euler, gripper

    def get_right_pose_euler(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get right arm pose in euler angle representation.

        Returns:
            Tuple of (position, euler_angles, gripper_value)
            - position: (3,) array [x, y, z] in meters
            - euler_angles: (3,) array [roll, pitch, yaw] in radians
            - gripper_value: float in [0, 1]
        """
        with self.lock:
            pos = self.right_ee_pos.copy()
            euler = self._quat_to_euler(self.right_ee_rot.as_quat())
            gripper = self.right_gripper_value
        return pos, euler, gripper

    def get_left_pose_quat(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get left arm pose in quaternion representation.

        Returns:
            Tuple of (position, quaternion, gripper_value)
            - position: (3,) array [x, y, z] in meters
            - quaternion: (4,) array [x, y, z, w]
            - gripper_value: float in [0, 1]
        """
        with self.lock:
            pos = self.left_ee_pos.copy()
            quat = self.left_ee_rot.as_quat().copy()
            gripper = self.left_gripper_value
        return pos, quat, gripper

    def get_right_pose_quat(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get right arm pose in quaternion representation.

        Returns:
            Tuple of (position, quaternion, gripper_value)
            - position: (3,) array [x, y, z] in meters
            - quaternion: (4,) array [x, y, z, w]
            - gripper_value: float in [0, 1]
        """
        with self.lock:
            pos = self.right_ee_pos.copy()
            quat = self.right_ee_rot.as_quat().copy()
            gripper = self.right_gripper_value
        return pos, quat, gripper

    # ========== Pose Goto Functions ==========

    def goto_left_pose_euler(
        self,
        position: np.ndarray,
        euler_angles: np.ndarray,
        gripper_value: float,
    ) -> None:
        """
        Command left arm to go to target pose (euler representation).

        Args:
            position: (3,) array [x, y, z] in meters
            euler_angles: (3,) array [roll, pitch, yaw] in radians
            gripper_value: float in [0, 1]
        """
        rot = SimpleRotation.from_euler(self.euler_seq, euler_angles, degrees=False)

        with self.lock:
            self.left_ee_pos = np.array(position, dtype=np.float64)
            self.left_ee_rot = rot

        self._publish_pose_command(
            self.left_ee_pos,
            self.left_ee_rot,
            gripper_value,
            self.pub_left_pose,
            self.pub_left_gripper,
            "",
        )

    def goto_right_pose_euler(
        self,
        position: np.ndarray,
        euler_angles: np.ndarray,
        gripper_value: float,
    ) -> None:
        """
        Command right arm to go to target pose (euler representation).

        Args:
            position: (3,) array [x, y, z] in meters
            euler_angles: (3,) array [roll, pitch, yaw] in radians
            gripper_value: float in [0, 1]
        """
        rot = SimpleRotation.from_euler(self.euler_seq, euler_angles, degrees=False)

        with self.lock:
            self.right_ee_pos = np.array(position, dtype=np.float64)
            self.right_ee_rot = rot

        self._publish_pose_command(
            self.right_ee_pos,
            self.right_ee_rot,
            gripper_value,
            self.pub_right_pose,
            self.pub_right_gripper,
            "",
        )

    def goto_left_pose_quat(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        gripper_value: float,
    ) -> None:
        """
        Command left arm to go to target pose (quaternion representation).

        Args:
            position: (3,) array [x, y, z] in meters
            quaternion: (4,) array [x, y, z, w]
            gripper_value: float in [0, 1]
        """
        rot = SimpleRotation.from_quat(quaternion)

        with self.lock:
            self.left_ee_pos = np.array(position, dtype=np.float64)
            self.left_ee_rot = rot

        self._publish_pose_command(
            self.left_ee_pos,
            self.left_ee_rot,
            gripper_value,
            self.pub_left_pose,
            self.pub_left_gripper,
            "",
        )

    def goto_right_pose_quat(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        gripper_value: float,
    ) -> None:
        """
        Command right arm to go to target pose (quaternion representation).

        Args:
            position: (3,) array [x, y, z] in meters
            quaternion: (4,) array [x, y, z, w]
            gripper_value: float in [0, 1]
        """
        rot = SimpleRotation.from_quat(quaternion)

        with self.lock:
            self.right_ee_pos = np.array(position, dtype=np.float64)
            self.right_ee_rot = rot

        self._publish_pose_command(
            self.right_ee_pos,
            self.right_ee_rot,
            gripper_value,
            self.pub_right_pose,
            self.pub_right_gripper,
            "",
        )

    # ========== Status Check ==========

    def is_ready(self) -> bool:
        """Check if all observations are available."""
        with self.lock:
            return (
                self.head_image is not None
                and self.left_image is not None
                and self.right_image is not None
                and self.left_pose is not None
                and self.right_pose is not None
            )

    def wait_for_ready(self, timeout: float = 10.0) -> bool:
        """
        Wait until all observations are available.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if ready, False if timeout
        """
        start_time = time.time()
        rate = self.create_rate(10)  # 10 Hz check rate

        while rclpy.ok():
            if self.is_ready():
                self.get_logger().info("Robot interface ready!")
                return True

            if time.time() - start_time > timeout:
                self.get_logger().warn(
                    f"Timeout waiting for observations. "
                    f"head={self.head_image is not None}, "
                    f"left={self.left_image is not None}, "
                    f"right={self.right_image is not None}, "
                    f"left_pose={self.left_pose is not None}, "
                    f"right_pose={self.right_pose is not None}"
                )
                return False

            rclpy.spin_once(self, timeout_sec=0.1)

        return False
