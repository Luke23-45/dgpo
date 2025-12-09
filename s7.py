import sys
import os
import cv2
import hydra
import mujoco
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from PIL import Image
from contextlib import contextmanager

# --- Project Imports setup ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv

# ==============================================================================
# 1. THE SUSPECT LOGIC (Copied EXACTLY from your Eval Script)
# ==============================================================================
@contextmanager
def render_physics_safe_goal_suspect(env, target_pos_world):
    """
    The exact logic used in the failing evaluation.
    It moves the robot to 'Home' before rendering the goal.
    """
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    saved_ctrl = env.data.ctrl.copy()
    
    try:
        # 1. Move Robot to "Home" (Clear view of object)
        home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        env.data.qpos[:7] = home_qpos
        env.data.qpos[7:] = 0.04 # Open Grippers
        
        # 2. Teleport Object to Goal
        obj_jnt_adr = env.model.jnt_qposadr[env.object_joint_id]
        current_quat = env.data.qpos[obj_jnt_adr+3 : obj_jnt_adr+7].copy()
        
        env.data.qpos[obj_jnt_adr : obj_jnt_adr+3] = target_pos_world
        env.data.qpos[obj_jnt_adr+3 : obj_jnt_adr+7] = current_quat
        
        # 3. Zero Velocities
        env.data.qvel[:] = 0.0
        
        # 4. Propagate Kinematics
        mujoco.mj_forward(env.model, env.data)
        yield
        
    finally:
        # 5. Restore
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        env.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)

# ==============================================================================
# 2. MAIN TEST RUNNER
# ==============================================================================
@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_semantic_planner_config")
def main(cfg: DictConfig):
    print("--- Starting Goal Image Forensics ---")
    
    # 1. Setup Environment
    xml_path = cfg.get("xml_path", "envs/panda_pick_place.xml")
    env = PandaEnv(
        xml_path=xml_path,
        control_mode='delta',
        render_mode="rgb_array",
        action_scaling_factor=0.5
    )
    
    # 2. Use the specific seed from your logs
    # Log said: seed - 1103648435
    test_seeds = [1103648435] 
    
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)
    
    for seed in test_seeds:
        print(f"Testing Seed: {seed}")
        obs, _ = env.reset(seed=seed)
        
        # A. Capture START State (What the robot sees now)
        img_start = env.render()
        img_start_bgr = cv2.cvtColor(img_start, cv2.COLOR_RGB2BGR)
        
        # B. Capture GOAL State (Using the suspect logic)
        with render_physics_safe_goal_suspect(env, obs['goal_pos_world']):
            img_goal = env.render()
        
        img_goal_bgr = cv2.cvtColor(img_goal, cv2.COLOR_RGB2BGR)
        
        # C. Create Comparison Image
        # Stack vertically: Top = Start, Bottom = Goal
        comparison = np.vstack([img_start_bgr, img_goal_bgr])
        
        # Add labels
        cv2.putText(comparison, f"START (Seed {seed})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "GOAL (Generated)", (10, 254), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        filename = output_dir / f"forensics_seed_{seed}.jpg"
        cv2.imwrite(str(filename), comparison)
        print(f"Saved forensics image to: {filename}")
        
    print("\nDONE. Please inspect 'debug_output/forensics_seed_1103648435.jpg'.")
    print("CHECK: Does the 'GOAL' image show the robot holding the object?")
    print("       Or is the robot far away/invisible?")

if __name__ == "__main__":
    main()