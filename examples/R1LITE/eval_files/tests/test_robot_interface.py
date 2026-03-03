#!/usr/bin/env python3
"""
Test script for R1LITE robot interface.

This script tests the robot interface by:
1. Initializing the interface
2. Waiting for observations
3. Displaying camera images in windows
4. Printing current state and poses
5. Optionally sending test actions
"""

import rclpy
import time
import numpy as np
import argparse
import cv2

from r1lite_robot_interface import R1LITERobotInterface


def main():
    parser = argparse.ArgumentParser(description="Test R1LITE robot interface")
    parser.add_argument(
        "--send-actions",
        action="store_true",
        help="Send small test actions to robot (use with caution!)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Test duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable camera image display windows",
    )
    parser.add_argument(
        "--test-pose-control",
        action="store_true",
        help="Test pose get/goto functions (use with caution!)",
    )
    args = parser.parse_args()

    # Initialize ROS2
    rclpy.init()

    # Create robot interface
    robot = R1LITERobotInterface(
        node_name="test_robot_interface",
        publish_rate=10.0,
    )

    print("[TEST] Waiting for robot interface to be ready...")
    if not robot.wait_for_ready(timeout=10.0):
        print("[TEST] ERROR: Robot interface not ready!")
        robot.destroy_node()
        rclpy.shutdown()
        return

    print("[TEST] Robot interface ready!")

    # Create display windows if enabled
    if not args.no_display:
        cv2.namedWindow("Head Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Left Wrist Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Right Wrist Camera", cv2.WINDOW_NORMAL)
        print("[TEST] Display windows created")

    # Test pose get functions
    if args.test_pose_control:
        print("\n[TEST] Testing pose get functions...")
        left_pos, left_euler, left_grip = robot.get_left_pose_euler()
        right_pos, right_euler, right_grip = robot.get_right_pose_euler()
        print(f"  Left arm (euler): pos={left_pos}, euler={left_euler}, grip={left_grip}")
        print(f"  Right arm (euler): pos={right_pos}, euler={right_euler}, grip={right_grip}")

        left_pos_q, left_quat, left_grip_q = robot.get_left_pose_quat()
        right_pos_q, right_quat, right_grip_q = robot.get_right_pose_quat()
        print(f"  Left arm (quat): pos={left_pos_q}, quat={left_quat}, grip={left_grip_q}")
        print(f"  Right arm (quat): pos={right_pos_q}, quat={right_quat}, grip={right_grip_q}")

    print(f"[TEST] Running test for {args.duration} seconds...")
    print("[TEST] Press 'q' in any window to quit early")

    start_time = time.time()
    step = 0

    while rclpy.ok() and (time.time() - start_time) < args.duration:
        # Spin once to process callbacks
        rclpy.spin_once(robot, timeout_sec=0.01)

        # Get observations
        head_img, left_img, right_img = robot.get_images()
        state = robot.get_robot_state()

        # Display images
        if not args.no_display:
            if head_img is not None:
                # Convert RGB to BGR for OpenCV display
                head_bgr = cv2.cvtColor(head_img, cv2.COLOR_RGB2BGR)
                # Add text overlay
                cv2.putText(head_bgr, f"Step: {step}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Head Camera", head_bgr)

            if left_img is not None:
                left_bgr = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
                cv2.putText(left_bgr, f"Left Wrist", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Left Wrist Camera", left_bgr)

            if right_img is not None:
                right_bgr = cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR)
                cv2.putText(right_bgr, f"Right Wrist", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Right Wrist Camera", right_bgr)

            # Check for 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[TEST] User requested quit")
                break

        # Print status every second
        if step % 10 == 0:
            print(f"\n[Step {step}] Observations:")
            print(f"  Head image: {head_img.shape if head_img is not None else 'None'}")
            print(f"  Left image: {left_img.shape if left_img is not None else 'None'}")
            print(f"  Right image: {right_img.shape if right_img is not None else 'None'}")
            print(f"  State shape: {state.shape}")
            print(f"  Left EE: xyz={state[0:3]}, euler={state[3:6]}, gripper={state[6]:.2f}")
            print(f"  Right EE: xyz={state[7:10]}, euler={state[10:13]}, gripper={state[13]:.2f}")

        # Test pose control functions
        if args.test_pose_control and step == 50:
            print("\n[TEST] Testing goto_pose functions (moving slightly)...")
            # Get current poses
            left_pos, left_euler, left_grip = robot.get_left_pose_euler()
            right_pos, right_euler, right_grip = robot.get_right_pose_euler()

            # Move left arm slightly up (5mm)
            new_left_pos = left_pos.copy()
            new_left_pos[2] += 0.005
            robot.goto_left_pose_euler(new_left_pos, left_euler, left_grip)
            print(f"  Commanded left arm to move up 5mm")

            # Move right arm slightly up (5mm)
            new_right_pos = right_pos.copy()
            new_right_pos[2] += 0.005
            robot.goto_right_pose_euler(new_right_pos, right_euler, right_grip)
            print(f"  Commanded right arm to move up 5mm")

        # Optionally send test actions
        if args.send_actions and step > 0:
            # Send very small delta actions (almost zero motion)
            action = np.zeros(14, dtype=np.float32)
            # Small random noise for testing
            action[0:3] = np.random.randn(3) * 0.001  # left xyz
            action[7:10] = np.random.randn(3) * 0.001  # right xyz

            robot.send_delta_action(action)

            if step % 10 == 0:
                print(f"  Sent test action (small deltas)")

        step += 1
        time.sleep(0.1)  # 10 Hz

    print(f"\n[TEST] Test completed after {step} steps")
    print("[TEST] Shutting down...")

    # Cleanup
    if not args.no_display:
        cv2.destroyAllWindows()
        print("[TEST] Display windows closed")

    robot.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
