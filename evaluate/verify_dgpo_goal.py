import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import gymnasium as gym
from envs.panda_env import PandaEnv
from envs.dgpo_env_wrapper import DGPOEnvWrapper
from omegaconf import OmegaConf
import os

def main():
    # 1. Setup Config
    cfg = OmegaConf.create({
        "environment": {
            "xml_path": "envs/panda_pick_place.xml",
            "urdf_path": "urdf/panda_mujoco_kinematics.urdf"
        },
        "expert": {
            "object_size": [0.04, 0.04, 0.04],
            "grasp_width": 0.6,
            "hover_height": 0.15,
            "grasp_offset_z": 0.025
        },
        "training": {
            "use_policy_blending": False
        }
    })

    # 2. Init Env
    print("Initializing Environment...")
    base_env = PandaEnv(
        xml_path=cfg.environment.xml_path,
        render_mode="rgb_array", 
        control_mode="delta"
    )
    env = DGPOEnvWrapper(base_env, cfg)
    
    # 3. Reset
    print("Resetting Env (Seed 46)...")
    obs, info = env.reset(seed=46) # Same seed as test_visualize_goal.py default
    
    # 4. Get Goal Image
    if 'goal_img' in info:
        goal_img = info['goal_img']
        print(f"Goal Image Shape: {goal_img.shape}")
        
        # 5. Save
        save_path = "dgpo_verified_goal.png"
        Image.fromarray(goal_img).save(save_path)
        print(f"Saved {save_path}")
    else:
        print("ERROR: 'goal_img' not found in info dict!")
    
    env.close()

if __name__ == "__main__":
    main()
