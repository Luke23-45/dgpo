# In file: utils/obs_adapters.py
import numpy as np
from typing import Dict, Any, List
from gymnasium import spaces 
import gymnasium as gym
from typing import Tuple
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper ,VecNormalize
import logging
logger = logging.getLogger(__name__)

def build_octo_observation(
    env_obs: Dict[str, Any],
    history_len: int = 2
) -> Dict[str, np.ndarray]:
    """
    Builds a single-timestep observation dictionary that is strictly compliant
    with the octo-small-1.5 model's expected schema. This is the single
    source of truth for creating the final dictionary structure the model expects.
    """
    T = int(max(1, history_len))
    B = 1
    
    # --- 1. Build the INNER "observations" dictionary ---
    octo_obs = {}

    # Image Primary
    if "image_primary" in env_obs:
        img = np.asarray(env_obs["image_primary"])
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        octo_obs["image_primary"] = np.repeat(img[np.newaxis, np.newaxis, ...], T, axis=1)
    
    # Image Wrist (with robust placeholder)
    if "image_wrist" in env_obs:
        wrist = np.asarray(env_obs["image_wrist"])
        if wrist.ndim == 3 and wrist.shape[0] in (1, 3):
            wrist = np.transpose(wrist, (1, 2, 0))
        octo_obs["image_wrist"] = np.repeat(wrist[np.newaxis, np.newaxis, ...], T, axis=1)
    else:
        octo_obs["image_wrist"] = np.zeros((B, T, 128, 128, 3), dtype=np.uint8)

    # Timestep
    t0 = int(np.asarray(env_obs.get("timestep", 0), dtype=np.int32).reshape(()))
    octo_obs["timestep"] = np.full((B, T), t0, dtype=np.int32)
    proprio_key = "internal_full_proprio" if "internal_full_proprio" in env_obs else "proprio"
    if proprio_key in env_obs:
        proprio = np.asarray(env_obs[proprio_key]).astype(np.float32)
        # The model expects a key named "proprio", so we standardize it here.
        octo_obs["proprio"] = np.repeat(proprio[np.newaxis, np.newaxis, ...], T, axis=1)
    # Task Completed (with correct 4D padding)
    tc_vec = np.zeros(4, dtype=np.float32)
    tc_vec[0] = float(np.asarray(env_obs.get("task_completed", 0.0)).item())
    octo_obs["task_completed"] = np.tile(tc_vec.reshape(1, 1, -1), (B, T, 1))
    
    # Padding Masks
    pad = np.ones((B, T), dtype=bool)
    nested_pad = {k: pad for k in octo_obs}
    octo_obs["pad_mask_dict"] = nested_pad
    
    octo_obs["timestep_pad_mask"] = pad


    return octo_obs


def build_octo_batch_from_list(obs_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Takes a LIST of environment observations and creates an efficient, single
    OCTO-compliant batch for inference.
    """
    if not obs_list:
        return {}
    
    # Use the perfected single-obs adapter on each observation in the list
    octo_obs_list = [build_octo_observation(obs, history_len=1) for obs in obs_list]
    
    # Collate the list of flat dictionaries into a single flat batch dictionary
    final_batch = {}
    if not octo_obs_list:
        return final_batch

    # Iterate over all keys in the first (sample) dictionary
    for key in octo_obs_list[0].keys():
        if key == "pad_mask_dict":
            # Handle the nested pad_mask_dict separately
            final_batch[key] = {}
            for sub_key in octo_obs_list[0][key].keys():
                final_batch[key][sub_key] = np.vstack(
                    [d[key][sub_key] for d in octo_obs_list]
                )
        else:
            final_batch[key] = np.vstack([d[key] for d in octo_obs_list])
            
    return final_batch


class OctoToSB3Adapter(gym.ObservationWrapper):
    """
    A comprehensive wrapper to convert OCTO-native observation dictionaries
    into a format compatible with Stable Baselines 3. It performs:
    1. Key Filtering: Drops specified keys (e.g., 'pad_mask_dict').
    2. Image Transposition: Converts HWC images to CHW format.
    3. Flattening: Flattens any remaining nested dictionaries.
    4. Sanitization: Ensures all non-image data is float32.
    """
    def __init__(self, env, keys_to_drop: Tuple[str, ...] = ("pad_mask_dict", "internal_full_proprio")):
        super().__init__(env)
        self.keys_to_drop = set(keys_to_drop) # Use a set for faster lookups

        # Define the final, flattened observation space for SB3 with correct dtypes.
        new_obs_space = {}
        for key, space in self.env.observation_space.spaces.items():
            if key in self.keys_to_drop:
                continue
            
            # Case 1: The space is an image (3D Box, uint8).
            # We transpose its shape and explicitly keep its dtype as uint8.
            # This is the PRIMARY FIX for the numpy memory allocation error.
            if isinstance(space, spaces.Box) and len(space.shape) == 3 and space.dtype == np.uint8:
                new_shape = (space.shape[2], space.shape[0], space.shape[1])
                new_obs_space[key] = spaces.Box(
                    low=0, high=255, shape=new_shape, dtype=np.uint8
                )
            
            # Case 2: The space is anything else.
            # We trust the original environment's definition and pass it through,
            # ensuring maximum robustness and compatibility. We just handle the
            # case of scalar values, which we reshape to (1,).
            else:
                shape = space.shape if space.shape != () else (1,)
                new_obs_space[key] = spaces.Box(
                    low=space.low, high=space.high, shape=shape, dtype=space.dtype
                )
        
        self.observation_space = spaces.Dict(new_obs_space)

    def observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Transforms the observation to match the declared observation_space.
        """
        new_obs = {}
        for key, value in obs.items():
            # Skip any keys that are meant to be dropped.
            if key in self.keys_to_drop:
                continue

            # Check if the key is actually in our target space. This is a safety check.
            if key not in self.observation_space.spaces:
                continue

            space = self.observation_space.spaces[key]
            
            # Case 1: The space is an image. Transpose from HWC to CHW.
            if isinstance(space, spaces.Box) and len(space.shape) == 3 and space.dtype == np.uint8:
                new_obs[key] = np.transpose(value, (2, 0, 1))
            
            # Case 2: It's a scalar that needs reshaping.
            elif isinstance(space, spaces.Box) and space.shape == (1,):
                 # Ensure it's a numpy array and has the correct shape.
                new_obs[key] = np.asarray(value, dtype=space.dtype).reshape(1)
            
            # Case 3: Any other array. Just ensure it has the correct dtype.
            else:
                new_obs[key] = np.asarray(value, dtype=space.dtype)
                
        return new_obs


class VecOctoToSB3Adapter(VecEnvWrapper):
    """
    A vectorized observation wrapper that correctly adapts observations from a
    `VecEnv` for use with a Stable Baselines 3 MultiInputPolicy.

    This is the vectorized equivalent of the `OctoToSB3Adapter`. It correctly
    handles batched observations from multiple parallel environments.
    """
    def __init__(self, venv: VecEnv, keys_to_drop: Tuple[str, ...] = ("pad_mask_dict", "internal_full_proprio", "expert_fsm_state", "expert_source")):
        super().__init__(venv)
        self.keys_to_drop = set(keys_to_drop)
        
        # Modify the observation space of the vectorized environment.
        new_obs_space = {}
        original_space = self.venv.observation_space.spaces
        
        for key, space in original_space.items():
            if key in self.keys_to_drop:
                continue
                
            # Image transposition (HWC -> CHW)
            if isinstance(space, spaces.Box) and len(space.shape) == 3 and space.dtype == np.uint8:
                new_shape = (space.shape[2], space.shape[0], space.shape[1])
                new_obs_space[key] = spaces.Box(
                    low=0, high=255, shape=new_shape, dtype=np.uint8
                )
            # Preserve other spaces
            else:
                shape = space.shape if space.shape != () else (1,)
                new_obs_space[key] = spaces.Box(
                    low=space.low, high=space.high, shape=shape, dtype=space.dtype
                )
                
        self.observation_space = spaces.Dict(new_obs_space)

    def reset(self):
        # The base `venv` reset returns a dict of (n_envs, H, W, C) arrays
        obs = self.venv.reset()
        return self._process_obs(obs)

    def step_wait(self):
        # `step_wait` returns obs, rewards, dones, infos
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._process_obs(obs), rewards, dones, infos

    def _process_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        new_obs = {}
        for key, value in obs.items():
            if key in self.keys_to_drop:
                continue

            if key not in self.observation_space.spaces:
                continue
                
            space = self.observation_space.spaces[key]
            
            # Image transposition: now handles a batch of images (N, H, W, C) -> (N, C, H, W)
            if isinstance(space, spaces.Box) and len(space.shape) == 3 and space.dtype == np.uint8:
                new_obs[key] = np.transpose(value, (0, 3, 1, 2))
            
            # Scalar reshaping: now handles a batch (N,) -> (N, 1)
            elif isinstance(space, spaces.Box) and space.shape == (1,):
                new_obs[key] = np.asarray(value, dtype=space.dtype).reshape(self.num_envs, 1)
                
            else:
                new_obs[key] = np.asarray(value, dtype=space.dtype)
                
        return new_obs


# In utils/obs_adapters.py

class AbsoluteJointToDeltaJointWrapper(VecEnvWrapper):
    """
    Robust VecEnv wrapper that translates policy absolute-joint outputs
    into delta joint commands expected by a 'delta' PandaEnv.
    """
    def __init__(self,
                 venv: VecEnv,
                 action_scaling: float,
                 safety_clip: float = 1.0,
                 debug: bool = False):
        super().__init__(venv)
        self.action_scaling = float(action_scaling)
        self.safety_clip = float(safety_clip)
        self.debug = bool(debug)
        self._last_obs = None

        try:
            action_dim = int(self.action_space.shape[-1])
            self.expected_joints = action_dim - 1
        except Exception:
            self.expected_joints = 7
            logger.warning("Could not infer action_dim; defaulting expected_joints=7")

        self.joint_slice = slice(0, self.expected_joints)
        self._is_normalized = isinstance(self.venv, VecNormalize)

    def _find_get_original_obs_handle(self):
        curr = self.venv
        while curr is not None:
            if hasattr(curr, "get_original_obs"):
                return curr
            if hasattr(curr, "venv"):
                curr = getattr(curr, "venv")
            elif hasattr(curr, "env"):
                curr = getattr(curr, "env")
            else:
                curr = None
        return None

    def _get_latest_physical_obs(self):
        getter_obj = self._find_get_original_obs_handle()
        if getter_obj is not None:
            return getter_obj.get_original_obs()
        
        if self._last_obs is not None:
            return self._last_obs
        
        raise RuntimeError("AbsoluteJointToDeltaJointWrapper: no physical observation available.")

    def reset(self):
        self._last_obs = self.venv.reset()
        return self._last_obs


    def step_async(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = np.expand_dims(actions, 0)

        # Correctly get the un-normalized "physical" observation.
        if self._is_normalized:
            obs_phys = self.venv.get_original_obs()
        else:
            obs_phys = self._last_obs
        
        current_proprio = np.asarray(obs_phys["proprio"], dtype=np.float32)
        if current_proprio.ndim == 1:
            current_proprio = np.expand_dims(current_proprio, 0)

        current_qpos = current_proprio[:, self.joint_slice]

        arm_targets_actual = actions[:, :self.expected_joints]
        gripper_targets = actions[:, self.expected_joints:]

        # Calculate the required physical delta (in radians)
        required_physical_delta = arm_targets_actual - current_qpos

        # Convert the physical delta to a normalized delta using the scaling factor
        normalized_delta = required_physical_delta / self.action_scaling

        # Form the final action with the normalized delta
        final_actions = np.concatenate([normalized_delta, gripper_targets], axis=1)
        final_actions = np.clip(final_actions, -self.safety_clip, self.safety_clip)

        self.venv.step_async(final_actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self._last_obs = obs
        return obs, rewards, dones, infos