# FILE: scripts/visualize_expert_trajectory.py (DEFINITIVE VERSION)

import argparse
import logging
import os
import cv2
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.getcwd())

from envs.panda_env import PandaEnv
from utils.scripted_expert import ExpertConfig, ObjectProfile, ScriptedExpert
from utils.ik_solver import IKSolver

# --- Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
DEFAULT_URDF_PATH = "urdf/panda_mujoco_kinematics.urdf"
DEFAULT_XML_PATH = "envs/panda_pick_place.xml"
DEFAULT_OUTPUT_DIR = "videos"

def generate_expert_video(output_path: str, urdf_path: str, xml_path: str, seed: int, max_steps: int, scale: float):
    logger.info("--- Starting Expert Trajectory Visualization ---")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- 1. Initialize all components with a single, consistent configuration ---
    env = PandaEnv(xml_path=xml_path, control_mode="delta")
    expert = ScriptedExpert(object_profile=ObjectProfile(size=np.array([0.04,0.04,0.04]), grasp_width_normalized=0.6))
    ik_solver = IKSolver(urdf_path=urdf_path)
    
    # --- 2. Perform the definitive, correct reset sequence ---
    obs, _ = env.reset(seed=seed)
    env.set_object_size(expert.object.size)
    expert.reset()
    ik_solver.reset_controller_state()

    # --- 3. Parameter calculation ---
    N_SUBSTEPS = 20
    effective_dt = env.model.opt.timestep * N_SUBSTEPS
    max_dq = env.ACTION_SCALING_FACTOR / effective_dt
    arm_joint_ids = np.arange(7)
    logger.info(f"Controller Params: max_dq={max_dq:.2f}, scale={env.ACTION_SCALING_FACTOR}")

    video_writer = None
    try:
        frame_rgb = env.render()
        height, width, _ = frame_rgb.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        # --- 4. Run the single episode loop ---
        for _ in tqdm(range(env.max_episode_steps), desc="Generating Episode"):
            expert_obs = env.get_expert_obs()
            target_ee_pose, grip = expert.get_target_pose(expert_obs)
            
            delta = ik_solver.compute_delta_action(
                target_ee_pose=target_ee_pose,
                model=env.model, data=env.data, ee_site_id=env.ee_site_id,
                joint_qpos_indices=arm_joint_ids, effective_dt=effective_dt, max_dq=max_dq,
            )
            action = np.concatenate([delta, [grip]])
            
            # Write frame before stepping
            frame_rgb = env.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_bgr, f"State: {expert.get_state()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 100), 2)
            video_writer.write(frame_bgr)
            
            obs, _, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated or expert.is_done():
                break

    finally:
        if video_writer:
            video_writer.release()
        env.close()

    success = expert.was_successful()
    if success:
        logger.info(f"--- ✅ Video of successful trajectory saved to {output_path} ---")
    else:
        logger.warning(f"--- ⚠️ Trajectory failed. Video may show a failed attempt. Final state: {expert.get_state()} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video of one expert trajectory.")
    parser.add_argument("--output", type=str, default=os.path.join(DEFAULT_OUTPUT_DIR, "expert_trajectory.mp4"))
    parser.add_argument("--urdf", type=str, default=DEFAULT_URDF_PATH)
    parser.add_argument("--xml", type=str, default=DEFAULT_XML_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps for the episode.")
    parser.add_argument("--scale", type=float, default=0.5, help="Action scaling factor for the environment.")
    
    args = parser.parse_args()

    generate_expert_video(
        output_path=args.output, urdf_path=args.urdf, xml_path=args.xml,
        seed=args.seed, max_steps=args.max_steps, scale=args.scale
    )