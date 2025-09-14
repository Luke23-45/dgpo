# In file: scripts/analyze_octo_expert.py
"""
Definitive Analysis Script for OCTO Foundation Model Performance.

This script runs a single episode using the reliable ScriptedExpert to generate
a ground-truth trajectory. At every step, it simultaneously queries the OCTO
foundation model for its own predicted action from the same observation.

It produces a detailed report and a comparison video to quantitatively and
qualitatively measure the OCTO model's performance in your environment.

The output provides the ground-truth evidence for:
1. The positional error (in cm) between OCTO's prediction and the expert's target.
2. The rotational error (in degrees) between the two.
3. A side-by-side video visualizing the ScriptedExpert's target (green sphere)
   vs. the OCTO model's prediction (blue sphere).
"""
import argparse
import json
import logging
import time
from pathlib import Path
import sys

import cv2
import jax
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# --- Project Imports ---
sys.path.append(str(Path(__file__).resolve().parent.parent))

from envs.panda_env import PandaEnv
from models.bc_policy import BCNet
from octo.model.octo_model import OctoModel
from utils.obs_adapters import build_octo_observation
from utils.scripted_expert import ScriptedExpert, ObjectProfile
from utils.ik_solver import IKSolver

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("ANALYZE_OCTO_EXPERT")


# --- Helper Functions ---

def draw_sphere(renderer, pos, color, radius=0.015):
    """Helper to add a colored sphere marker to the MuJoCo scene."""
    # Add a new geom to the scene for visualization
    renderer.model.ngeom += 1
    mujoco.mj_addGeom(
        renderer.model,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0, 0]),
        pos.astype(np.float64),
        None,
        -1,  # Not part of a body
        np.array(color, dtype=np.float32)
    )

def quat_distance_degrees(q1_xyzw, q2_xyzw):
    """Calculate the angular distance between two quaternions in degrees."""
    r1 = R.from_quat(q1_xyzw)
    r2 = R.from_quat(q2_xyzw)
    # The magnitude of the angle of the relative rotation
    return np.rad2deg((r1.inv() * r2).magnitude())

def postprocess_octo_action(raw_action: np.ndarray, current_ee_pose: np.ndarray) -> np.ndarray:
    """Applies the same safety heuristics used in the data generation to the raw OCTO output."""
    pose = np.array(raw_action, dtype=np.float32)
    
    # Z-flip heuristic
    if (pose[2] < 0.0) and (current_ee_pose[2] > 0.1):
        pose[2] *= -1.0
        
    # Normalize quaternion
    q = pose[3:7]
    norm = np.linalg.norm(q)
    if norm > 1e-6:
        pose[3:7] = q / norm
        
    return pose


def main(args: argparse.Namespace):
    log.info("--- Starting OCTO Foundation Model Analysis Script ---")

    # --- 1. Setup ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"octo_analysis_report_seed{args.seed}.json"
    video_path = output_dir / f"octo_comparison_video_seed{args.seed}.mp4"

    log.info(f"Analysis report will be saved to: {report_path}")
    log.info(f"Comparison video will be saved to: {video_path}")

    # --- 2. Initialize All Components ---
    log.info("Initializing Environment, Experts, and OCTO Model...")
    env = PandaEnv(xml_path=args.xml_path)
    ik_solver = IKSolver(urdf_path=args.urdf_path)
    
    object_profile = ObjectProfile(size=np.array([0.04, 0.04, 0.04]), grasp_width_normalized=0.6)
    scripted_expert = ScriptedExpert(object_profile)
    
    octo_model = OctoModel.load_pretrained(args.octo_model_name)
    octo_task = octo_model.create_tasks(texts=[args.instruction])
    jax_key = jax.random.PRNGKey(args.seed)

    log.info("All components initialized successfully.")

    # --- 3. Setup Video Writer and Scene Renderer ---
    frame = env.render()
    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))
    
    analysis_results = []
    
    # --- 4. Run One Ground-Truth Episode ---
    scripted_expert.reset()
    obs, _ = env.reset(seed=args.seed)

    try:
        for step in range(env.max_episode_steps):
            # --- A: Get Ground-Truth Action from ScriptedExpert ---
            expert_obs = env.get_expert_obs()
            ground_truth_pose, gripper_action = scripted_expert.get_target_pose(
                expert_obs["ee_pose_world"], expert_obs["object_pos_world"],
                expert_obs["object_orn_world"], expert_obs["goal_pos_world"],
                expert_obs["is_grasped"]
            )
            
            # --- B: Get Predicted Action from OCTO Model ---
            # Prepare the observation exactly as the training pipeline would
            octo_obs_input = build_octo_observation(obs)
            jax_key, subkey = jax.random.split(jax_key)
            raw_octo_action = octo_model.sample_actions(octo_obs_input, octo_task, rng=subkey)
            
            # Post-process the raw action
            octo_predicted_pose = postprocess_octo_action(raw_octo_action[0, 0, :7], expert_obs["ee_pose_world"])

            # --- C: Calculate and Record Metrics ---
            pos_error_m = np.linalg.norm(ground_truth_pose[:3] - octo_predicted_pose[:3])
            rot_error_deg = quat_distance_degrees(ground_truth_pose[3:7], octo_predicted_pose[3:7])
            
            step_report = {
                "step": step,
                "expert_fsm_state": scripted_expert.get_state(),
                "ground_truth_pose": ground_truth_pose.tolist(),
                "octo_predicted_pose": octo_predicted_pose.tolist(),
                "position_error_cm": pos_error_m * 100,
                "rotation_error_deg": rot_error_deg,
            }
            analysis_results.append(step_report)
            log.info(f"Step {step} | State: {scripted_expert.get_state()} | Pos Error: {pos_error_m*100:.2f} cm | Rot Error: {rot_error_deg:.2f} deg")

            # --- D: Visualize ---
            # In the video, a GREEN sphere shows the ScriptedExpert's target.
            # A BLUE sphere shows the OCTO model's prediction.
            draw_sphere(env.renderer, ground_truth_pose[:3], [0, 1, 0, 0.7])  # Green for ground truth
            draw_sphere(env.renderer, octo_predicted_pose[:3], [0, 0, 1, 0.7]) # Blue for OCTO
            
            frame_rgb = env.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            
            # --- E: Step the environment using the GROUND-TRUTH action ---
            # This ensures we get a full, successful trajectory to evaluate against.
            base_pos, base_quat = env.get_base_pose()
            R_world_base = R.from_quat(base_quat)
            pos_in_base = R_world_base.inv().apply(ground_truth_pose[:3] - base_pos)
            rot_in_base = R_world_base.inv() * R.from_quat(ground_truth_pose[3:7])
            target_pose_base = np.concatenate([pos_in_base, rot_in_base.as_quat()])
            
            current_joints = expert_obs["internal_full_proprio"][:7]
            arm_action = ik_solver.compute_action(target_pose_base, current_joints)
            final_action = np.concatenate([arm_action, [gripper_action]])
            
            obs, _, terminated, truncated, _ = env.step(final_action)
            
            if scripted_expert.is_done() or terminated or truncated:
                log.info("Ground-truth episode finished.")
                break
                
    finally:
        # --- 5. Save Report and Cleanup ---
        avg_pos_error = np.mean([r["position_error_cm"] for r in analysis_results])
        avg_rot_error = np.mean([r["rotation_error_deg"] for r in analysis_results])
        
        summary = {
            "summary_metrics": {
                "average_position_error_cm": avg_pos_error,
                "average_rotation_error_deg": avg_rot_error,
                "max_position_error_cm": np.max([r["position_error_cm"] for r in analysis_results]),
                "min_position_error_cm": np.min([r["position_error_cm"] for r in analysis_results]),
            },
            "step_by_step_report": analysis_results
        }
        
        log.info(f"\n--- ANALYSIS SUMMARY ---\n"
                 f"  Average Positional Error: {avg_pos_error:.2f} cm\n"
                 f"  Average Rotational Error: {avg_rot_error:.2f} degrees\n"
                 f"--------------------------")
        
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=4)
        
        log.info("Releasing video writer and closing environment...")
        video_writer.release()
        env.close()
        log.info("Analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze OCTO model performance against a scripted expert.")
    
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml")
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf")
    parser.add_argument("--octo_model_name", type=str, default="hf://rail-berkeley/octo-small-1.5")
    parser.add_argument("--instruction", type=str, default="pick up the red block")
    parser.add_argument("--output_dir", type=str, default="analysis_reports")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)