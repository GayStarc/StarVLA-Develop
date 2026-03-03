"""
R1LITE Real-World Dual-Arm Robot Inference Interface

Input:
- 3 images: head, left arm, right arm (224x224)
- state: 16-dim robot state

Output:
- action: 14-dim delta action (delta_ee)
  Format: [left_xyz(3), left_euler(3), left_gripper(1), right_xyz(3), right_euler(3), right_gripper(1)]
"""

from pathlib import Path
from typing import Optional
from collections import deque

import numpy as np
import cv2 as cv

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.tools import read_mode_config

try:
    from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None


class R1LITEModelClient:
    """Model client for R1LITE dual-arm robot inference."""

    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 5694,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 4,
        action_ensemble: bool = False,
        action_ensemble_horizon: int = 3,
        adaptive_ensemble_alpha: float = 0.1,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.unnorm_key = unnorm_key
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size

        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon

        self.task_description = None
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon, self.adaptive_ensemble_alpha
            )
        else:
            self.action_ensembler = None

        # Load normalization stats
        self.action_norm_stats = self._get_action_stats(self.unnorm_key, policy_ckpt_path)
        self.action_chunk_size = self._get_action_chunk_size(policy_ckpt_path)
        self.state_norm_stats = self._get_state_stats(self.unnorm_key, policy_ckpt_path)

        self.raw_actions = None
        self.step_count = 0

        print(f"[R1LITE] Initialized with unnorm_key: {unnorm_key}")
        print(f"[R1LITE] Action chunk size: {self.action_chunk_size}")

    def reset(self, task_description: str = "") -> None:
        """Reset the model client for a new episode."""
        self.task_description = task_description
        if self.action_ensemble and self.action_ensembler:
            self.action_ensembler.reset()
        self.raw_actions = None
        self.step_count = 0

    def step(
        self,
        images: list[np.ndarray],
        state: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        """
        Run one inference step.

        Args:
            images: List of 3 images [head, left, right], each (H, W, 3) uint8
            state: Robot state array of shape (16,)
            instruction: Task instruction string

        Returns:
            action: Delta action array of shape (14,)
        """
        if instruction != self.task_description:
            self.reset(instruction)

        # Resize images
        images = [self._resize_image(img) for img in images]

        # Normalize state
        normalized_state = self._normalize_state(state, self.state_norm_stats)

        # Prepare input
        example = {
            "lang": str(instruction),
            "image": images,
            "state": normalized_state.reshape(1, -1),
        }

        vla_input = {
            "examples": [example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        # Get action from model (with action chunking)
        if self.step_count % self.action_chunk_size == 0 or self.raw_actions is None:
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"]  # (B, chunk, D)
            except KeyError:
                print(f"[R1LITE] Response error: {response}")
                raise KeyError(f"Key 'normalized_actions' not found: {response['data'].keys()}")

            normalized_actions = normalized_actions[0]  # (chunk, D)
            self.raw_actions = self._unnormalize_actions(
                normalized_actions, self.action_norm_stats
            )

        # Select current action from chunk
        action_idx = self.step_count % self.action_chunk_size
        if action_idx < len(self.raw_actions):
            current_action = self.raw_actions[action_idx]
        else:
            current_action = self.raw_actions[-1]

        self.step_count += 1

        return current_action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    @staticmethod
    def _normalize_state(state: np.ndarray, state_norm_stats: dict) -> np.ndarray:
        """Normalize state using min-max normalization."""
        # Mask for continuous vs discrete values (gripper)
        # state: [left_xyz(3), left_euler(3), left_gripper(1), right_xyz(3), right_euler(3), right_gripper(1), ...]
        mask = np.ones(len(state), dtype=bool)
        mask[6] = False   # left gripper
        mask[13] = False  # right gripper

        state_high = np.array(state_norm_stats["max"])
        state_low = np.array(state_norm_stats["min"])

        normalized_state = np.where(
            mask,
            (state - state_low) / (state_high - state_low + 1e-8) * 2 - 1,
            state,
        )
        # Binarize gripper values
        normalized_state = np.where(
            ~mask,
            (normalized_state > 0.5).astype(normalized_state.dtype),
            normalized_state,
        )
        return normalized_state

    @staticmethod
    def _unnormalize_actions(
        normalized_actions: np.ndarray, action_norm_stats: dict
    ) -> np.ndarray:
        """Unnormalize actions from [-1, 1] to original scale."""
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["min"], dtype=bool)
        )
        action_high = np.array(action_norm_stats["max"])
        action_low = np.array(action_norm_stats["min"])

        normalized_actions = np.clip(normalized_actions, -1, 1)

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions

    @staticmethod
    def _get_action_stats(unnorm_key: str, policy_ckpt_path: str) -> dict:
        """Load action normalization statistics."""
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = R1LITEModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def _get_state_stats(unnorm_key: str, policy_ckpt_path: str) -> dict:
        """Load state normalization statistics."""
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = R1LITEModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["state"]

    @staticmethod
    def _get_action_chunk_size(policy_ckpt_path: str) -> int:
        """Get action chunk size from model config."""
        model_config, _ = read_mode_config(policy_ckpt_path)
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    @staticmethod
    def _check_unnorm_key(norm_stats: dict, unnorm_key: Optional[str]) -> str:
        """Check and return valid unnorm key."""
        if unnorm_key is None or unnorm_key not in norm_stats:
            unnorm_key = next(iter(norm_stats.keys()))
        return unnorm_key


def create_model_client(config: dict) -> R1LITEModelClient:
    """Create R1LITE model client from config dict."""
    return R1LITEModelClient(
        policy_ckpt_path=config["policy_ckpt_path"],
        unnorm_key=config.get("unnorm_key"),
        host=config.get("host", "127.0.0.1"),
        port=config.get("port", 5694),
        image_size=config.get("image_size", [224, 224]),
        use_ddim=config.get("use_ddim", True),
        num_ddim_steps=config.get("num_ddim_steps", 4),
    )
