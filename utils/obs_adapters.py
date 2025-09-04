# In file: utils/obs_adapters.py
import numpy as np
from typing import Dict, Any, List
from gymnasium import spaces 
import gymnasium as gym
from typing import Tuple
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
        self.keys_to_drop = keys_to_drop

        # Define the final, flattened observation space for SB3
        flat_spaces = {}
        original_space = self.env.observation_space.spaces
        
        for key, space in original_space.items():
            if key in self.keys_to_drop:
                continue
            
            # Handle image transposition (HWC -> CHW)
            if isinstance(space, spaces.Box) and len(space.shape) == 3 and space.dtype == np.uint8:
                new_shape = (space.shape[2], space.shape[0], space.shape[1])
                flat_spaces[key] = spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)
            # Handle other Box spaces (proprio, timestep)
            elif isinstance(space, spaces.Box):
                shape = space.shape if space.shape != () else (1,)
                flat_spaces[key] = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
            else:
                flat_spaces[key] = space

        self.observation_space = spaces.Dict(flat_spaces)

    def observation(self, obs):
        new_obs = {}
        for key, value in obs.items():
            if key in self.keys_to_drop:
                continue

            # Transpose images
            if key in self.observation_space.spaces and isinstance(self.observation_space.spaces[key], spaces.Box) and len(value.shape) == 3:
                new_obs[key] = np.transpose(value, (2, 0, 1)).astype(np.uint8)
            # Sanitize other values
            elif key in self.observation_space.spaces:
                arr = np.asarray(value)
                if arr.shape == ():
                    arr = arr.reshape(1)
                new_obs[key] = arr.astype(np.float32)

        return new_obs
