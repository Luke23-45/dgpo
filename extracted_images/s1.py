# FILE: utils/action_wrappers.py

from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AbsoluteJointToDeltaJointWrapper(VecEnvWrapper):
    """
    Correct, robust VecEnv wrapper that translates policy absolute-joint outputs
    into delta joint commands.

    FINAL VERSION - This version correctly handles the interaction with VecNormalize
    by explicitly fetching the unnormalized "original" observation for its

    internal calculations, solving the unit mismatch bug.
    """
    def __init__(self,
                 venv: VecNormalize,
                 policy_outputs_are_normalized: bool = False,
                 joint_slice: slice = slice(8, 15), # For 7-DoF arm: ee_pose(7)+gripper(1) -> joints start at 8
                 n_arm_joints: int = 7,
                 safety_clip: float = 1.0,
                 debug: bool = False):
        # This wrapper MUST be applied to a VecNormalize instance to function correctly.
        assert isinstance(venv, VecNormalize), "This wrapper must wrap a VecNormalize environment."
        super().__init__(venv)

        self.policy_outputs_are_normalized = bool(policy_outputs_are_normalized)
        self.joint_slice = joint_slice
        self.n_arm_joints = n_arm_joints
        self.safety_clip = float(safety_clip)
        self.debug = bool(debug)

        # Auto-detect action scaling from the base environment
        base_env = self.venv.envs[0] if hasattr(self.venv, "envs") else self.venv
        self.action_scaling = getattr(base_env.unwrapped, "ACTION_SCALING_FACTOR", 0.05)

        if self.policy_outputs_are_normalized:
            try:
                self.joint_low = base_env.unwrapped.joint_lower_limits[:self.n_arm_joints]
                self.joint_high = base_env.unwrapped.joint_upper_limits[:self.n_arm_joints]
            except (AttributeError, IndexError):
                raise RuntimeError("policy_outputs_are_normalized=True but joint bounds could not be auto-detected.")

        logger.info(
            "AbsoluteJointToDeltaJointWrapper initialized: "
            f"action_scaling={self.action_scaling}, joint_slice={self.joint_slice}"
        )

    def step_async(self, actions: np.ndarray):
        # `actions` is the absolute joint pose from the policy (num_envs, 8)
        
        # --- THE DEFINITIVE FIX ---
        # We use the official SB3 method to get the unnormalized observation.
        original_obs = self.venv.get_original_obs()
        current_qpos = original_obs["proprio"][:, self.joint_slice]
        
        # Slice the policy's action into arm and gripper parts
        arm_targets = actions[:, :self.n_arm_joints]
        gripper_targets = actions[:, self.n_arm_joints:]

        if self.policy_outputs_are_normalized:
            # Denormalize from [-1,1] to [low, high]
            arm_targets_actual = self.joint_low + 0.5 * (arm_targets + 1.0) * (self.joint_high - self.joint_low)
        else:
            arm_targets_actual = arm_targets

        # Perform the translation in real-world units (radians)
        required_arm_action = (arm_targets_actual - current_qpos) / self.action_scaling
        
        # Recombine and clip for safety
        final_actions = np.concatenate([required_arm_action, gripper_targets], axis=1)
        clipped_actions = np.clip(final_actions, -self.safety_clip, self.safety_clip)
        
        # Send the final, correct delta action to the underlying environment
        self.venv.step_async(clipped_actions)

    def reset(self):
        # Pass through, this is handled by the VecNormalize wrapper.
        return self.venv.reset()

    def step_wait(self):
        # Pass through, this is also handled by the VecNormalize wrapper.
        return self.venv.step_wait()