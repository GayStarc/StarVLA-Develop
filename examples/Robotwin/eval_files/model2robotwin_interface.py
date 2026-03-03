import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2 as cv
import json_numpy
import numpy as np
from scipy.spatial.transform import Rotation

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.tools import read_mode_config

try:
    from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None

np.set_printoptions(precision=4, suppress=True, linewidth=10000)


def wxyz_to_xyzw(quat):
    """Convert quaternion from wxyz to xyzw."""
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def xyzw_to_wxyz(quat):
    """Convert quaternion from xyzw to wxyz."""
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


# Joint layout for delta: only arm joints use delta, gripper uses absolute
# Format: left_arm, left_gripper, right_arm, right_gripper
LEFT_ARM_DOF = 6
LEFT_GRIPPER_DOF = 1
RIGHT_ARM_DOF = 6
RIGHT_GRIPPER_DOF = 1


def quaternion_to_euler(quat_wxyz):
    """Convert quaternion (w, x, y, z) to euler angles (roll, pitch, yaw)."""
    return Rotation.from_quat(wxyz_to_xyzw(quat_wxyz)).as_euler('xyz', degrees=False)


def euler_to_quaternion(euler):
    """Convert euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)."""
    return xyzw_to_wxyz(Rotation.from_euler('xyz', euler, degrees=False).as_quat())


def _extract_joint_positions(observation: Dict[str, Any]) -> Optional[np.ndarray]:
    """Extract joint positions from observation["joint_action"].

    Expected layout: {"left_arm": ndarray, "left_gripper": scalar/ndarray,
                      "right_arm": ndarray, "right_gripper": scalar/ndarray, ...}
    Returns: [left_arm, left_gripper, right_arm, right_gripper] concatenated.
    """
    joint_action = observation.get("joint_action")
    if not isinstance(joint_action, dict):
        return None

    left_arm = joint_action.get("left_arm")
    right_arm = joint_action.get("right_arm")
    if left_arm is None or right_arm is None:
        return None

    parts = [np.asarray(left_arm)]
    left_gripper = joint_action.get("left_gripper")
    if left_gripper is not None:
        parts.append(np.asarray(left_gripper).reshape(-1))
    parts.append(np.asarray(right_arm))
    right_gripper = joint_action.get("right_gripper")
    if right_gripper is not None:
        parts.append(np.asarray(right_gripper).reshape(-1))
    return np.concatenate(parts)


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble: bool = False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="127.0.0.1",
        port=5694,
        use_delta: bool = False,
        use_euler: bool = False,
        use_joint: bool = True,
        use_state: bool = True,
    ) -> None:

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.use_delta = use_delta
        self.use_euler = use_euler
        self.use_joint = use_joint
        self.use_state = use_state

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)
        self.state_norm_stats = (
            self.get_state_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
            if self.use_state
            else None
        )
        self.raw_actions = None

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None

    def step(
        self,
        example: dict,
        step: int = 0,
    ) -> np.ndarray:
        state = example.get("state", None)
        if self.use_state and state is not None:
            state = self.normalize_state(state, self.state_norm_stats)
            example["state"] = state.reshape(1, -1)
        elif not self.use_state and "state" in example:
            del example["state"]

        task_description = example.get("lang", None)
        images = example["image"]

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        images = [self._resize_image(image) for image in images]
        example["image"] = images
        vla_input = {
            "examples": [example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        action_chunk_size = self.action_chunk_size

        if step % action_chunk_size == 0 or self.raw_actions is None:
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"]  # B, chunk, D
            except KeyError:
                print(f"Response data: {response}")
                raise KeyError(f"Key 'normalized_actions' not found in response data: {response['data'].keys()}")

            normalized_actions = normalized_actions[0]
            print(f"Normalized Actions: {normalized_actions}")
            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats
            )

        action_idx = step % action_chunk_size
        if action_idx >= len(self.raw_actions):
            pass

        current_action = self.raw_actions[action_idx]
        # current_action = current_action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]
        return current_action

    @staticmethod
    def normalize_state(state: dict[str, np.ndarray], state_norm_stats: Dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Normalize the state
        """
        # mask = [True, True, True, True, True, True, True, True, True, True, True, True, False, False]
        mask = [True, True, True, True, True, True, False, True, True, True, True, True, True, False] # Joint
        mask = np.array(mask, dtype=bool)
        state_high, state_low = np.array(state_norm_stats["max"]), np.array(state_norm_stats["min"])
        normalized_state = np.where(
            mask,
            (state - state_low) / (state_high - state_low) * 2 - 1,
            state,
        )
        normalized_state = np.where(~mask, (normalized_state > 0.5).astype(normalized_state.dtype), normalized_state)
        return normalized_state

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)

        # Continuous dims (mask=True): min-max unnormalize
        # Binary dims (mask=False, e.g. gripper): threshold at 0.1 to get 0/1, matching training transform
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            (normalized_actions > 0.5).astype(np.float64),
        )

        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_state_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["state"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            if len(norm_stats) == 1:
                unnorm_key = next(iter(norm_stats.keys()))
            else:
                unnorm_key = next(iter(norm_stats.keys()))

        if unnorm_key not in norm_stats:
            unnorm_key = next(iter(norm_stats.keys()))

        return unnorm_key


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    use_delta = usr_args.get("use_delta", False)
    use_euler = usr_args.get("use_euler", False)
    use_joint = usr_args.get("use_joint", True)
    use_state = usr_args.get("use_state", True)

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        use_delta=use_delta,
        use_euler=use_euler,
        use_joint=use_joint,
        use_state=use_state,
    )


def reset_model(model):
    model.reset(task_description="")


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _apply_joint_delta(raw_action: np.ndarray, current_qpos: np.ndarray) -> np.ndarray:
    """Apply delta to arm joints while keeping gripper values absolute.

    Expected layout: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)].
    Falls back to full-vector delta if the action length doesn't match.
    """
    expected_len = LEFT_ARM_DOF + LEFT_GRIPPER_DOF + RIGHT_ARM_DOF + RIGHT_GRIPPER_DOF
    if len(raw_action) < expected_len:
        return current_qpos + raw_action

    action = np.array(raw_action, dtype=np.float64)
    l_arm = slice(0, LEFT_ARM_DOF)
    r_arm_start = LEFT_ARM_DOF + LEFT_GRIPPER_DOF
    r_arm = slice(r_arm_start, r_arm_start + RIGHT_ARM_DOF)

    action[l_arm] = current_qpos[l_arm] + raw_action[l_arm]
    action[r_arm] = current_qpos[r_arm] + raw_action[r_arm]
    return action


def _compute_arm_target(
    raw_xyz: np.ndarray,
    raw_rot: np.ndarray,
    current_endpose: np.ndarray,
    use_delta: bool,
    use_euler: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute target xyz and quaternion (wxyz) for a single arm."""
    current_xyz = current_endpose[0:3]
    current_quat_wxyz = current_endpose[3:7]

    if use_delta:
        target_xyz = current_xyz + raw_xyz
        if use_euler:
            current_euler = quaternion_to_euler(current_quat_wxyz)
            target_quat = euler_to_quaternion(current_euler + raw_rot)
        else:
            target_quat = xyzw_to_wxyz(
                (Rotation.from_quat(wxyz_to_xyzw(raw_rot))
                 * Rotation.from_quat(wxyz_to_xyzw(current_quat_wxyz))).as_quat()
            )
    else:
        target_xyz = raw_xyz
        target_quat = euler_to_quaternion(raw_rot) if use_euler else raw_rot

    return target_xyz, target_quat


def _parse_dual_arm_action(raw_action: np.ndarray, rot_dim: int):
    """Split raw action into per-arm (xyz, rot, gripper) tuples.

    Returns ((left_xyz, left_rot, left_gripper),
             (right_xyz, right_rot, right_gripper)).
    """
    left_xyz = raw_action[0:3]
    left_rot = raw_action[3:3 + rot_dim]
    left_gripper = raw_action[3 + rot_dim]

    right_start = 3 + rot_dim + 1
    right_xyz = raw_action[right_start:right_start + 3]
    right_rot = raw_action[right_start + 3:right_start + 3 + rot_dim]
    right_gripper = raw_action[right_start + 3 + rot_dim]

    return (left_xyz, left_rot, left_gripper), (right_xyz, right_rot, right_gripper)


def _build_state(model, observation) -> np.ndarray:
    """Build the state vector depending on joint vs. pose control mode."""
    if model.use_joint:
        state = _extract_joint_positions(observation)
        if state is None:
            logging.warning("use_joint enabled but no joint positions found in observation; using empty state.")
            state = np.array([])
        return state

    # Pose control: state = [left_endpose(7), left_gripper(1), right_endpose(7), right_gripper(1)]
    endpose = observation["endpose"]
    return np.concatenate([
        endpose["left_endpose"], [endpose["left_gripper"]],
        endpose["right_endpose"], [endpose["right_gripper"]],
    ])


def _execute_action(TASK_ENV, model, raw_action, observation):
    """Dispatch raw_action to the environment as joint (qpos) or pose (ee) control."""
    if model.use_joint:
        if model.use_delta:
            current_qpos = _extract_joint_positions(observation)
            if current_qpos is None:
                logging.warning("use_joint+use_delta but no joint positions found; using absolute joint action.")
                action = raw_action
            else:
                action = _apply_joint_delta(raw_action, current_qpos)
        else:
            action = raw_action
        TASK_ENV.take_action(action, action_type='qpos')
        return

    # Pose control
    endpose = observation["endpose"]
    rot_dim = 3 if model.use_euler else 4
    (l_xyz, l_rot, l_grip), (r_xyz, r_rot, r_grip) = _parse_dual_arm_action(raw_action, rot_dim)

    left_target_xyz, left_target_quat = _compute_arm_target(
        l_xyz, l_rot, endpose["left_endpose"], model.use_delta, model.use_euler)
    right_target_xyz, right_target_quat = _compute_arm_target(
        r_xyz, r_rot, endpose["right_endpose"], model.use_delta, model.use_euler)

    # Format: [left_xyz(3), left_quat(4), left_gripper(1),
    #          right_xyz(3), right_quat(4), right_gripper(1)]
    action = np.concatenate([
        left_target_xyz, left_target_quat, [l_grip],
        right_target_xyz, right_target_quat, [r_grip],
    ])
    TASK_ENV.take_action(action, action_type='ee')


# ---------------------------------------------------------------------------
# Main eval entry point
# ---------------------------------------------------------------------------

def eval(TASK_ENV, model, observation):
    instruction = TASK_ENV.get_instruction()

    # Prepare images: [head, left, right] to match training order
    obs_cameras = observation["observation"]
    images = [obs_cameras["head_camera"]["rgb"],
              obs_cameras["left_camera"]["rgb"],
              obs_cameras["right_camera"]["rgb"]]

    example = {"lang": str(instruction), "image": images}
    if model.use_state:
        example["state"] = _build_state(model, observation)

    raw_action = model.step(example, step=TASK_ENV.take_action_cnt)
    print(f"Model Action: {raw_action}")

    _execute_action(TASK_ENV, model, raw_action, observation)

