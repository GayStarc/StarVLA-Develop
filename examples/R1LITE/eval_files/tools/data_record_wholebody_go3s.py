#!/usr/bin/env python3
"""
ROS2 data recorder for GO 3S + robot motion topics.
Joystick controls (subscribes /joy_right):
- X button: start recording
- Y button: stop recording
"""

import os
import subprocess
import threading
import time
import json
from datetime import datetime
from signal import SIGINT
import bisect
import math
from typing import cast

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Joy
import logging

import h5py
from PIL import Image
import rosbag2_py


def quat_to_rot_matrix(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n == 0.0:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - s * (yy + zz), s * (xy - wz), s * (xz + wy)],
            [s * (xy + wz), 1.0 - s * (xx + zz), s * (yz - wx)],
            [s * (xz - wy), s * (yz + wx), 1.0 - s * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rot_matrix_to_euler_xyz(m):
    sy = math.sqrt(m[0, 0] * m[0, 0] + m[1, 0] * m[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(m[2, 1], m[2, 2])
        y = math.atan2(-m[2, 0], sy)
        z = math.atan2(m[1, 0], m[0, 0])
    else:
        x = math.atan2(-m[1, 2], m[1, 1])
        y = math.atan2(-m[2, 0], sy)
        z = 0.0
    return np.array([x, y, z], dtype=np.float64)


class DataRecordWholebodyGo3S(Node):
    @staticmethod
    def get_logger_custom(name: str = "default") -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def __init__(self):
        super().__init__("data_record_wholebody_go3s")

        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        self.joy_subscription = self.create_subscription(
            Joy,
            "/joy_right",
            self.joy_callback,
            qos_profile,
        )

        self.is_recording = False
        self.record_base_dir = os.path.expanduser(
            os.environ.get("RECORD_BASE_DIR", "~/GalaxeaDataset")
        )
        self.record_db3_dir = os.path.join(self.record_base_dir, "db3")
        self.record_h5_dir = os.path.join(self.record_base_dir, "h5")
        self.record_process = None

        self.record_command = (
            "ros2 bag record --qos-profile-overrides-path "
            "/home/robot/Desktop/WristInsta360_Project/qos_record.yaml "
            "--max-cache-size 1073741824 "
            "-o {}/{}_{} "
            "/hdas/camera_head/head/color/image_raw "
            "/hdas/camera_head/head/depth/image_rect_raw "
            "/hdas/camera_wrist_left/color/image_raw "
            "/hdas/camera_wrist_right/color/image_raw "
            "/motion_control/pose_ee_arm_left "
            "/motion_control/pose_ee_arm_right "
            "/motion_target/target_position_gripper_left "
            "/motion_target/target_position_gripper_right "
        )

        self.lock = threading.Lock()
        self.record_id = 0
        self.custom_logger = self.get_logger_custom("DataRecordWholebodyGo3S")

        self.strict_tolerance_s = 0.033
        self.head_topic = "/hdas/camera_head/head/color/image_raw"
        self.left_topic = "/hdas/camera_wrist_left/color/image_raw"
        self.right_topic = "/hdas/camera_wrist_right/color/image_raw"
        self.left_pose_topic = "/motion_control/pose_ee_arm_left"
        self.right_pose_topic = "/motion_control/pose_ee_arm_right"
        self.left_gripper_topic = "/motion_target/target_position_gripper_left"
        self.right_gripper_topic = "/motion_target/target_position_gripper_right"

        self.custom_logger.info("DataRecordWholebodyGo3S initialized.")
        self.custom_logger.info(
            "Joystick controls: X=start recording, Y=stop recording"
        )

    def stop_record(self):
        time.sleep(2)
        try:
            if self.record_process is not None:
                os.killpg(self.record_process.pid, SIGINT)
                self.record_process.wait()
                self.record_process = None
                self.custom_logger.info("Stopped recording.")
        except Exception as e:
            self.custom_logger.error(f"Error stopping recording: {e}")
        self.is_recording = False

        try:
            self.export_to_hdf5()
        except Exception as e:
            self.custom_logger.error(f"Export to h5 failed: {e}")

    def start_record(self):
        if self.record_process is None:
            record_folder = os.path.join(
                self.record_db3_dir, datetime.now().strftime("%Y%m%d")
            )
            if not os.path.exists(record_folder):
                os.makedirs(record_folder)

            try:
                time.sleep(2)
                cmd = self.record_command.format(
                    record_folder,
                    self.record_id,
                    datetime.now().strftime("%Y%m%d_%H%M%S"),
                )
                self.record_process = subprocess.Popen(
                    cmd,
                    shell=True,
                    preexec_fn=os.setsid,
                )
                self.custom_logger.info(f"Started recording to folder: {record_folder}")
                self.is_recording = True
                self.record_id += 1
            except Exception as e:
                self.custom_logger.error(f"Error starting recording: {e}")
                self.record_process = None
                self.is_recording = False

    def joy_callback(self, msg: Joy):
        buttons = {}
        buttons["X"] = msg.buttons[2] if len(msg.buttons) > 2 else 0
        buttons["Y"] = msg.buttons[3] if len(msg.buttons) > 3 else 0

        with self.lock:
            if self.is_recording:
                if buttons["Y"] == 1:
                    self.stop_record()
            else:
                if buttons["X"] == 1:
                    self.start_record()

    def _closest_index(self, stamps, target):
        if not stamps:
            return None
        idx = bisect.bisect_left(stamps, target)
        if idx == 0:
            return 0
        if idx == len(stamps):
            return len(stamps) - 1
        before = stamps[idx - 1]
        after = stamps[idx]
        return idx if abs(after - target) < abs(target - before) else idx - 1

    def _ns_to_sec(self, nsec):
        return nsec / 1e9

    def export_to_hdf5(self):
        bag_dir = os.path.join(self.record_db3_dir, datetime.now().strftime("%Y%m%d"))
        bag_prefix = f"{self.record_id - 1}_" if self.record_id > 0 else ""
        latest_dir = None
        if os.path.exists(bag_dir):
            candidates = [
                os.path.join(bag_dir, d)
                for d in os.listdir(bag_dir)
                if os.path.isdir(os.path.join(bag_dir, d))
            ]
            if candidates:
                latest_dir = max(candidates, key=os.path.getmtime)
        if latest_dir is None:
            self.custom_logger.error("No bag directory found to export")
            return

        storage_options = rosbag2_py.StorageOptions(
            uri=latest_dir, storage_id="sqlite3"
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        type_map = {}
        for info in reader.get_all_topics_and_types():
            type_map[info.name] = info.type

        buffers = {
            t: []
            for t in [
                self.head_topic,
                self.left_topic,
                self.right_topic,
                self.left_pose_topic,
                self.right_pose_topic,
                self.left_gripper_topic,
                self.right_gripper_topic,
            ]
        }

        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if topic not in buffers:
                continue
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            buffers[topic].append((timestamp, msg))

        head = buffers[self.head_topic]
        if not head:
            self.custom_logger.error("No head images found")
            return

        def stamps_for(topic):
            return [t for t, _ in buffers[topic]]

        left_stamps = stamps_for(self.left_topic)
        right_stamps = stamps_for(self.right_topic)
        left_pose_stamps = stamps_for(self.left_pose_topic)
        right_pose_stamps = stamps_for(self.right_pose_topic)
        left_grip_stamps = stamps_for(self.left_gripper_topic)
        right_grip_stamps = stamps_for(self.right_gripper_topic)

        frames = []
        dropped = 0

        for i, (t_head, head_msg) in enumerate(head):
            idx_left = self._closest_index(left_stamps, t_head)
            idx_right = self._closest_index(right_stamps, t_head)
            idx_lp = self._closest_index(left_pose_stamps, t_head)
            idx_rp = self._closest_index(right_pose_stamps, t_head)
            idx_lg = self._closest_index(left_grip_stamps, t_head)
            idx_rg = self._closest_index(right_grip_stamps, t_head)

            idxs = [idx_left, idx_right, idx_lp, idx_rp, idx_lg, idx_rg]
            if any(idx is None for idx in idxs):
                dropped += 1
                continue

            idx_left = int(cast(int, idx_left))
            idx_right = int(cast(int, idx_right))
            idx_lp = int(cast(int, idx_lp))
            idx_rp = int(cast(int, idx_rp))
            idx_lg = int(cast(int, idx_lg))
            idx_rg = int(cast(int, idx_rg))

            deltas = [
                abs(left_stamps[idx_left] - t_head),
                abs(right_stamps[idx_right] - t_head),
                abs(left_pose_stamps[idx_lp] - t_head),
                abs(right_pose_stamps[idx_rp] - t_head),
                abs(left_grip_stamps[idx_lg] - t_head),
                abs(right_grip_stamps[idx_rg] - t_head),
            ]
            if any(self._ns_to_sec(d) > self.strict_tolerance_s for d in deltas):
                dropped += 1
                continue

            left_img = buffers[self.left_topic][idx_left][1]
            right_img = buffers[self.right_topic][idx_right][1]
            left_pose = buffers[self.left_pose_topic][idx_lp][1]
            right_pose = buffers[self.right_pose_topic][idx_rp][1]
            left_grip = buffers[self.left_gripper_topic][idx_lg][1]
            right_grip = buffers[self.right_gripper_topic][idx_rg][1]

            frame = {
                "image_head": np.frombuffer(head_msg.data, dtype=np.uint8).reshape(
                    head_msg.height, head_msg.width, -1
                ),
                "image_left": np.frombuffer(left_img.data, dtype=np.uint8).reshape(
                    left_img.height, left_img.width, -1
                ),
                "image_right": np.frombuffer(right_img.data, dtype=np.uint8).reshape(
                    right_img.height, right_img.width, -1
                ),
                "left_state": np.array(
                    [
                        left_pose.pose.position.x,
                        left_pose.pose.position.y,
                        left_pose.pose.position.z,
                        left_pose.pose.orientation.x,
                        left_pose.pose.orientation.y,
                        left_pose.pose.orientation.z,
                        left_pose.pose.orientation.w,
                        1.0
                        if (float(left_grip.position[0]) if left_grip.position else 0.0)
                        > 90.0
                        else 0.0,
                    ],
                    dtype=np.float64,
                ),
                "right_state": np.array(
                    [
                        right_pose.pose.position.x,
                        right_pose.pose.position.y,
                        right_pose.pose.position.z,
                        right_pose.pose.orientation.x,
                        right_pose.pose.orientation.y,
                        right_pose.pose.orientation.z,
                        right_pose.pose.orientation.w,
                        1.0
                        if (
                            float(right_grip.position[0])
                            if right_grip.position
                            else 0.0
                        )
                        > 90.0
                        else 0.0,
                    ],
                    dtype=np.float64,
                ),
            }

            frames.append(frame)

        actions = []
        for i in range(len(frames) - 1):
            l0 = frames[i]["left_state"]
            l1 = frames[i + 1]["left_state"]
            r0 = frames[i]["right_state"]
            r1 = frames[i + 1]["right_state"]

            l_dp = l1[:3] - l0[:3]
            r_dp = r1[:3] - r0[:3]

            l_q0 = l0[3:7]
            l_q1 = l1[3:7]
            r_q0 = r0[3:7]
            r_q1 = r1[3:7]

            l_r0 = quat_to_rot_matrix(l_q0)
            l_r1 = quat_to_rot_matrix(l_q1)
            r_r0 = quat_to_rot_matrix(r_q0)
            r_r1 = quat_to_rot_matrix(r_q1)

            l_rdiff = l_r0.T @ l_r1
            r_rdiff = r_r0.T @ r_r1

            l_de = rot_matrix_to_euler_xyz(l_rdiff)
            r_de = rot_matrix_to_euler_xyz(r_rdiff)

            l_action = np.concatenate([l_dp, l_de, [l1[7]]]).astype(np.float64)
            r_action = np.concatenate([r_dp, r_de, [r1[7]]]).astype(np.float64)

            actions.append({"left_action": l_action, "right_action": r_action})

        frames_out = []
        for i in range(len(actions)):
            frame = frames[i].copy()
            frame.update(actions[i])
            frames_out.append(frame)

        if not frames_out:
            self.custom_logger.error("No valid frames after alignment")
            return

        # --- H5 Save Logic ---
        h5_date_dir = os.path.join(
            self.record_h5_dir, datetime.now().strftime("%Y%m%d")
        )
        os.makedirs(h5_date_dir, exist_ok=True)
        out_path = os.path.join(h5_date_dir, os.path.basename(latest_dir) + ".h5")

        # Prepare datasets
        img_head_data = []
        img_left_data = []
        img_right_data = []
        state_left_data = []
        state_right_data = []
        action_left_data = []
        action_right_data = []

        target_size_wrist = (854, 480)

        for f in frames_out:
            # Head: Keep native resolution (e.g. 640x480)
            img_head_data.append(f["image_head"])

            # Wrist Left: Resize to 854x480
            im_l = Image.fromarray(f["image_left"])
            if im_l.size != target_size_wrist:
                im_l = im_l.resize(target_size_wrist, Image.BILINEAR)
            img_left_data.append(np.array(im_l))

            # Wrist Right: Resize to 854x480
            im_r = Image.fromarray(f["image_right"])
            if im_r.size != target_size_wrist:
                im_r = im_r.resize(target_size_wrist, Image.BILINEAR)
            img_right_data.append(np.array(im_r))

            state_left_data.append(f["left_state"])
            state_right_data.append(f["right_state"])
            action_left_data.append(f["left_action"])
            action_right_data.append(f["right_action"])

        try:
            with h5py.File(out_path, "w") as f:
                obs = f.create_group("observations")
                imgs = obs.create_group("images")

                # Use gzip compression, chunks=True for better storage
                imgs.create_dataset(
                    "head",
                    data=np.array(img_head_data),
                    compression="gzip",
                    compression_opts=4,
                    chunks=True,
                )
                imgs.create_dataset(
                    "left",
                    data=np.array(img_left_data),
                    compression="gzip",
                    compression_opts=4,
                    chunks=True,
                )
                imgs.create_dataset(
                    "right",
                    data=np.array(img_right_data),
                    compression="gzip",
                    compression_opts=4,
                    chunks=True,
                )

                state = obs.create_group("state")
                state.create_dataset("left", data=np.array(state_left_data))
                state.create_dataset("right", data=np.array(state_right_data))

                act = f.create_group("actions")
                act.create_dataset("left", data=np.array(action_left_data))
                act.create_dataset("right", data=np.array(action_right_data))

            total = len(head)
            kept = len(frames_out)
            dropped_total = total - kept
            self.custom_logger.info(
                f"Aligned frames saved: {out_path} | total={total} kept={kept} dropped={dropped_total}"
            )
        except Exception as e:
            self.custom_logger.error(f"Failed to write H5 file: {e}")
            return

        self.update_dataset_json(out_path, kept)

    def update_dataset_json(self, h5_path, frame_count):
        """Append the new episode info to dataset.json in RECORD_BASE_DIR"""
        json_path = os.path.join(self.record_base_dir, "dataset.json")

        # Define structure metadata
        # Note: Image shapes depend on what was actually saved.
        # We assume standard capture, but ideally we'd read it from the file we just wrote.
        # For efficiency, we hardcode the expected structure logic.
        structure = {
            "observations": {
                "images": {
                    "head": ["N", "H_native", "W_native", 3],  # Native (e.g. 480x640)
                    "left": ["N", 480, 854, 3],  # Resized
                    "right": ["N", 480, 854, 3],  # Resized
                },
                "state": {"left": ["N", 8], "right": ["N", 8]},
            },
            "actions": {"left": ["N", 7], "right": ["N", 7]},
        }

        entry = {
            "episode_id": os.path.basename(h5_path).replace("_aligned_frames.h5", ""),
            "file_path": h5_path,
            "relative_path": os.path.relpath(h5_path, self.record_base_dir),
            "length": frame_count,
            "fps": 30,
            "timestamp": datetime.now().isoformat(),
            "structure": structure,
        }

        data_list = []
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data_list = json.load(f)
                    if not isinstance(data_list, list):
                        data_list = []
            except Exception as e:
                self.custom_logger.warn(
                    f"Could not read existing dataset.json: {e}. Creating new."
                )

        data_list.append(entry)

        try:
            with open(json_path, "w") as f:
                json.dump(data_list, f, indent=2)
            self.custom_logger.info(f"Updated dataset registry: {json_path}")
        except Exception as e:
            self.custom_logger.error(f"Failed to update dataset.json: {e}")

    def destroy_node(self):
        if self.is_recording:
            self.stop_record()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    data_recorder = DataRecordWholebodyGo3S()

    try:
        rclpy.spin(data_recorder)
    except KeyboardInterrupt:
        data_recorder.custom_logger.info(
            "Keyboard interrupt received, shutting down..."
        )
    finally:
        data_recorder.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
