#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import numpy as np
import h5py
import time
import os
import argparse
import math

# from scipy.spatial.transform import Rotation as R # REMOVED due to numpy version mismatch
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState


# --- Custom Rotation Class to replace Scipy ---
class SimpleRotation:
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
        # q = [x, y, z, w]
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
        # Only supports 'xyz' for now as used in the code
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

        # XYZ order quaternion construction
        # q = qx * qy * qz
        # For xyz order:
        # qx = sin(rx/2) cos(ry/2) cos(rz/2) + cos(rx/2) sin(ry/2) sin(rz/2)
        # qy = cos(rx/2) sin(ry/2) cos(rz/2) - sin(rx/2) cos(ry/2) sin(rz/2)
        # qz = cos(rx/2) cos(ry/2) sin(rz/2) + sin(rx/2) sin(ry/2) cos(rz/2)
        # qw = cos(rx/2) cos(ry/2) cos(rz/2) - sin(rx/2) sin(ry/2) sin(rz/2)

        # Correct formula for 'xyz' extrinsic (static frame)
        # which matches standard RPY roll-pitch-yaw

        x = sx * cy * cz + cx * sy * sz
        y = cx * sy * cz - sx * cy * sz
        z = cx * cy * sz + sx * sy * cz
        w = cx * cy * cz - sx * sy * sz

        return SimpleRotation([x, y, z, w])


# Alias R to SimpleRotation
R = SimpleRotation
# ---------------------------------------------

# NOTE: This robot's r1_gripper_controller accepts JointState with name=[] and
# position in [0, 100] (observed from mobiman_tabletop_tele_node).
LEFT_GRIPPER_JOINT = ""
RIGHT_GRIPPER_JOINT = ""

EULER_SEQ = "xyz"
FRAME_ID = "base_link"
PUBLISH_RATE = 10.0


class H5DataLoader:
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        self.file = h5py.File(path, "r")
        self.len = self.file["actions"]["left"].shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {
            "left_action": self.file["actions"]["left"][idx],
            "right_action": self.file["actions"]["right"][idx],
        }

    def close(self):
        self.file.close()


class DeltaIntegrationReplayNodeH5(Node):
    def __init__(self, data_path):
        super().__init__("delta_integration_replay_node_h5")

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE

        self.pub_l_pose = self.create_publisher(
            PoseStamped, "/motion_target/target_pose_arm_left", qos
        )
        self.pub_r_pose = self.create_publisher(
            PoseStamped, "/motion_target/target_pose_arm_right", qos
        )

        self.pub_l_gripper = self.create_publisher(
            JointState, "/motion_target/target_position_gripper_left", qos
        )
        self.pub_r_gripper = self.create_publisher(
            JointState, "/motion_target/target_position_gripper_right", qos
        )

        self.loader = H5DataLoader(data_path)
        self.total_frames = len(self.loader)

        # Initial positions (same as replay_eef.py)
        self.l_curr_pos = np.array([0.059, -0.335, 0.343])
        self.r_curr_pos = np.array([0.0585, -0.335, 0.334])

        self.l_curr_rot = R.from_quat([0.0, 0.0, 0.0, 1.0])
        self.r_curr_rot = R.from_quat([0.0, 0.0, 0.0, 1.0])

        self.current_idx = 0
        self.timer = self.create_timer(1.0 / PUBLISH_RATE, self.timer_callback)

        self.get_logger().info(f"Replay Node Started. Frames: {self.total_frames}")
        self.get_logger().info("System Reset to Zero. Integrating Actions...")
        time.sleep(1.0)

    def calculate_next_state(self, curr_pos, curr_rot, action_vec):
        delta_pos = action_vec[:3]
        next_pos = curr_pos + delta_pos

        delta_euler = action_vec[3:6]
        rot_diff = R.from_euler(EULER_SEQ, delta_euler, degrees=False)
        next_rot = curr_rot * rot_diff

        target_gripper = action_vec[6]
        return next_pos, next_rot, target_gripper

    def publish_command(
        self, pos, rot, gripper_val, pub_pose, pub_gripper, gripper_joint_name: str
    ):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID

        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])

        quat = rot.as_quat()
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        pub_pose.publish(msg)

        g = float(max(0.0, min(1.0, gripper_val)))
        g_100 = float(100.0 * g)

        g_msg = JointState()
        g_msg.header.stamp = msg.header.stamp
        g_msg.name = [gripper_joint_name] if gripper_joint_name else []
        g_msg.position = [g_100]
        pub_gripper.publish(g_msg)

    def timer_callback(self):
        if self.current_idx >= self.total_frames:
            self.get_logger().info("Finished.")
            self.timer.cancel()
            self.loader.close()
            return

        frame_data = self.loader[self.current_idx]

        self.l_curr_pos, self.l_curr_rot, l_g = self.calculate_next_state(
            self.l_curr_pos, self.l_curr_rot, frame_data["left_action"]
        )
        self.publish_command(
            self.l_curr_pos,
            self.l_curr_rot,
            l_g,
            self.pub_l_pose,
            self.pub_l_gripper,
            gripper_joint_name=LEFT_GRIPPER_JOINT,
        )

        self.r_curr_pos, self.r_curr_rot, r_g = self.calculate_next_state(
            self.r_curr_pos, self.r_curr_rot, frame_data["right_action"]
        )
        self.publish_command(
            self.r_curr_pos,
            self.r_curr_rot,
            r_g,
            self.pub_r_pose,
            self.pub_r_gripper,
            gripper_joint_name=RIGHT_GRIPPER_JOINT,
        )

        if self.current_idx % 30 == 0:
            self.get_logger().info(
                f"Frame {self.current_idx}/{self.total_frames}\n"
                f"  Left XYZ : [{self.l_curr_pos[0]:.4f}, {self.l_curr_pos[1]:.4f}, {self.l_curr_pos[2]:.4f}]\n"
                f"  Right XYZ: [{self.r_curr_pos[0]:.4f}, {self.r_curr_pos[1]:.4f}, {self.r_curr_pos[2]:.4f}]"
            )

        self.current_idx += 1


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description="Replay motions from H5 file.")
    parser.add_argument("file", help="Path to the .h5 file to replay")

    # Handle ROS args by stripping them out if passed via ros2 run
    # but argparse might conflict with rclpy args.
    # Safe way: use non-conflicting parsing or just sys.argv
    # We'll use a simple manual check if argparse fails, or just rely on the user passing the file.

    # Actually, rclpy.init(args=args) strips ros args.
    # We should parse our args from the remainder.
    # But for simplicity, we can just look at sys.argv if not using ros2 run standard way.

    # Let's try to find the file arg in sys.argv
    import sys

    file_path = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--") and arg.endswith(".h5"):
            file_path = arg
            break

    if not file_path:
        # Auto-discovery logic
        base_dir = os.path.expanduser(
            os.environ.get("RECORD_BASE_DIR", "~/GalaxeaDataset")
        )
        h5_root = os.path.join(base_dir, "h5")
        found = False

        if os.path.exists(h5_root):
            # Sort date directories by modification time (newest first)
            date_dirs = [
                os.path.join(h5_root, d)
                for d in os.listdir(h5_root)
                if os.path.isdir(os.path.join(h5_root, d))
            ]
            date_dirs.sort(key=os.path.getmtime, reverse=True)

            for d_dir in date_dirs:
                h5_files = [
                    os.path.join(d_dir, f)
                    for f in os.listdir(d_dir)
                    if f.endswith(".h5")
                ]
                if h5_files:
                    # Pick newest file in this directory
                    file_path = max(h5_files, key=os.path.getmtime)
                    print(f"No file specified. Auto-selected latest H5: {file_path}")
                    found = True
                    break

        if not found:
            print("Usage: python3 replay_h5.py <path_to_h5_file>")
            print(f"No H5 files found in {h5_root} for auto-selection.")
            return

    node = DeltaIntegrationReplayNodeH5(file_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
