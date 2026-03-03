#!/usr/bin/env python3
"""
Example script demonstrating pose control functions.

This script shows how to:
1. Get current robot poses (euler and quaternion)
2. Command robot to target poses
3. Execute simple motion sequences
"""

import rclpy
import time
import numpy as np

from r1lite_robot_interface import R1LITERobotInterface


def main():
    # Initialize ROS2
    rclpy.init()

    # Create robot interface
    robot = R1LITERobotInterface(
        node_name="example_pose_control",
        publish_rate=10.0,
    )

    print("[EXAMPLE] Waiting for robot interface to be ready...")
    if not robot.wait_for_ready(timeout=10.0):
        print("[EXAMPLE] ERROR: Robot interface not ready!")
        robot.destroy_node()
        rclpy.shutdown()
        return

    print("[EXAMPLE] Robot interface ready!")
    print("[EXAMPLE] Starting pose control demonstration...\n")

    # Spin once to update state
    rclpy.spin_once(robot, timeout_sec=0.1)

    # ========== Example 1: Get current poses ==========
    print("=" * 60)
    print("Example 1: Getting current robot poses")
    print("=" * 60)

    # Get poses in euler representation
    left_pos, left_euler, left_grip = robot.get_left_pose_euler()
    right_pos, right_euler, right_grip = robot.get_right_pose_euler()

    print("\nLeft arm (Euler):")
    print(f"  Position: x={left_pos[0]:.4f}, y={left_pos[1]:.4f}, z={left_pos[2]:.4f}")
    print(f"  Euler: roll={left_euler[0]:.4f}, pitch={left_euler[1]:.4f}, yaw={left_euler[2]:.4f}")
    print(f"  Gripper: {left_grip:.2f}")

    print("\nRight arm (Euler):")
    print(f"  Position: x={right_pos[0]:.4f}, y={right_pos[1]:.4f}, z={right_pos[2]:.4f}")
    print(f"  Euler: roll={right_euler[0]:.4f}, pitch={right_euler[1]:.4f}, yaw={right_euler[2]:.4f}")
    print(f"  Gripper: {right_grip:.2f}")

    # Get poses in quaternion representation
    left_pos_q, left_quat, _ = robot.get_left_pose_quat()
    right_pos_q, right_quat, _ = robot.get_right_pose_quat()

    print("\nLeft arm (Quaternion):")
    print(f"  Quaternion: x={left_quat[0]:.4f}, y={left_quat[1]:.4f}, z={left_quat[2]:.4f}, w={left_quat[3]:.4f}")

    print("\nRight arm (Quaternion):")
    print(f"  Quaternion: x={right_quat[0]:.4f}, y={right_quat[1]:.4f}, z={right_quat[2]:.4f}, w={right_quat[3]:.4f}")

    time.sleep(2)

    # ========== Example 2: Move to target pose (small movement) ==========
    print("\n" + "=" * 60)
    print("Example 2: Moving left arm up by 1cm")
    print("=" * 60)

    # Get current left arm pose
    left_pos, left_euler, left_grip = robot.get_left_pose_euler()
    print(f"\nCurrent left arm position: {left_pos}")

    # Create target pose (move up 1cm)
    target_pos = left_pos.copy()
    target_pos[2] += 0.01  # Move up 1cm

    print(f"Target left arm position: {target_pos}")
    print("Sending command...")

    # Command robot to target pose
    robot.goto_left_pose_euler(target_pos, left_euler, left_grip)

    # Wait for motion
    time.sleep(2)

    # Verify new position
    rclpy.spin_once(robot, timeout_sec=0.1)
    new_pos, _, _ = robot.get_left_pose_euler()
    print(f"New left arm position: {new_pos}")
    print(f"Position change: {new_pos - left_pos}")

    time.sleep(1)

    # ========== Example 3: Return to original pose ==========
    print("\n" + "=" * 60)
    print("Example 3: Returning left arm to original position")
    print("=" * 60)

    print("Sending command to return...")
    robot.goto_left_pose_euler(left_pos, left_euler, left_grip)

    time.sleep(2)

    # Verify position
    rclpy.spin_once(robot, timeout_sec=0.1)
    final_pos, _, _ = robot.get_left_pose_euler()
    print(f"Final left arm position: {final_pos}")
    print(f"Distance from original: {np.linalg.norm(final_pos - left_pos):.6f} m")

    # ========== Example 4: Gripper control ==========
    print("\n" + "=" * 60)
    print("Example 4: Gripper control")
    print("=" * 60)

    left_pos, left_euler, left_grip = robot.get_left_pose_euler()
    print(f"\nCurrent left gripper: {left_grip:.2f}")

    # Close gripper
    print("Closing left gripper...")
    robot.goto_left_pose_euler(left_pos, left_euler, 1.0)
    time.sleep(1)

    rclpy.spin_once(robot, timeout_sec=0.1)
    _, _, new_grip = robot.get_left_pose_euler()
    print(f"Left gripper after close command: {new_grip:.2f}")

    # Open gripper
    print("Opening left gripper...")
    robot.goto_left_pose_euler(left_pos, left_euler, 0.0)
    time.sleep(1)

    rclpy.spin_once(robot, timeout_sec=0.1)
    _, _, new_grip = robot.get_left_pose_euler()
    print(f"Left gripper after open command: {new_grip:.2f}")

    # ========== Cleanup ==========
    print("\n" + "=" * 60)
    print("Demonstration completed!")
    print("=" * 60)

    robot.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
