"""
Example usage of R1LITE inference interface for real-world deployment.

This script demonstrates how to:
1. Initialize the model client
2. Initialize robot interface
3. Get images and state from robot
4. Run inference and get actions
5. Send actions to robot
"""

import numpy as np
import yaml
import rclpy
import time

from r1lite_inference import R1LITEModelClient, create_model_client
from r1lite_robot_interface import R1LITERobotInterface


def main():
    # Initialize ROS2
    rclpy.init()

    # Load config
    with open("deploy_policy.yml", "r") as f:
        config = yaml.safe_load(f)

    # Create model client
    model = create_model_client(config)

    # Create robot interface
    robot = R1LITERobotInterface(
        node_name="r1lite_inference_node",
        publish_rate=10.0,
    )

    # Wait for robot to be ready
    print("[R1LITE] Waiting for robot interface to be ready...")
    if not robot.wait_for_ready(timeout=10.0):
        print("[R1LITE] ERROR: Robot interface not ready!")
        robot.destroy_node()
        rclpy.shutdown()
        return

    # Task instruction
    instruction = "Pick up the red cube and place it on the plate"
    model.reset(instruction)

    print(f"[R1LITE] Task: {instruction}")
    print("[R1LITE] Starting inference loop...")

    # Main control loop
    max_steps = 300
    for step in range(max_steps):
        # Spin once to process callbacks
        rclpy.spin_once(robot, timeout_sec=0.01)

        # Get observations from robot
        head_img, left_img, right_img = robot.get_images()
        state = robot.get_robot_state()

        # Check if all observations are available
        if head_img is None or left_img is None or right_img is None:
            print(f"[Step {step}] Waiting for images...")
            time.sleep(0.1)
            continue

        images = [head_img, left_img, right_img]

        # Run inference
        action = model.step(images, state, instruction)

        # action format: [left_xyz(3), left_euler(3), left_gripper(1),
        #                 right_xyz(3), right_euler(3), right_gripper(1)]
        if step % 10 == 0:
            print(f"[Step {step}] Action: {action}")

        # Send action to robot
        robot.send_delta_action(action)

        # Check if task is done (implement your own logic)
        if check_task_done():
            print("[R1LITE] Task completed!")
            break

        # Control loop rate
        time.sleep(1.0 / 10.0)  # 10 Hz

    print("[R1LITE] Inference loop finished.")

    # Cleanup
    robot.destroy_node()
    rclpy.shutdown()


def check_task_done() -> bool:
    """
    Check if task is completed.

    TODO: Implement your own task completion logic here.
    Examples:
    - Check if object reached target position
    - Check if gripper state changed
    - Use vision-based detection
    """
    return False


if __name__ == "__main__":
    main()
