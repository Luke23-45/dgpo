# FILE: scripts/verify_dataset.py
#
# Definitive, SOTA, End-to-End Data Verification Script.
#
# This script provides an apples-to-apples comparison of a saved trajectory
# from an LMDB dataset against a freshly generated "live" trajectory using the
# exact same seed. It produces both a quantitative (CSV) and qualitative
# (side-by-side video) report of any discrepancies, serving as the final
# certification of data integrity.
#

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Generator, Tuple
import yaml
import cv2
import numpy as np
from tqdm import tqdm

# --- Project Imports ---
# Ensure the project root is in the Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from utils.ik_solver import IKSolver
from utils.scripted_expert import ExpertConfig, ObjectProfile, ScriptedExpert
# We reuse the SoAEpisodeLoader from our visualization script
from scripts.visualize_dataset import SoAEpisodeLoader

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s",
)
logger = logging.getLogger("verify_dataset")


def generate_live_trajectory(
    cfg: Dict[str, Any],
    seed: int
) -> Generator[Dict[str, Any], None, None]:
    """
    Runs a single, deterministic episode in a live environment and yields the
    full expert observation dictionary at each timestep.
    """
    logger.info(f"--- Starting LIVE re-simulation for seed {seed} ---")
    
    env = PandaEnv(xml_path=cfg["xml_path"], control_mode='delta')
    expert_cfg = ExpertConfig(**cfg.get("expert_config", {}))
    object_profile = ObjectProfile(
        size=np.array(cfg.get("object_size")),
        grasp_width_normalized=float(cfg.get("grasp_width"))
    )
    expert = ScriptedExpert(object_profile=object_profile, cfg=expert_cfg)
    ik_solver = IKSolver(urdf_path=cfg["urdf_path"])
    
    # Deterministic Reset
    obs, _ = env.reset(seed=seed)
    env.set_object_size(object_profile.size)
    expert.reset()
    ik_solver.reset_controller_state()

    # Controller parameters needed for action computation
    N_SUBSTEPS = 20
    effective_dt = env.model.opt.timestep * N_SUBSTEPS
    arm_joint_ids = np.arange(7)
    max_dq = env.ACTION_SCALING_FACTOR / effective_dt

    for step in range(env.max_episode_steps):
        expert_obs = env.get_expert_obs()
        expert_obs['expert_state'] = expert.get_state()
        
        # Yield the observation BEFORE the step is taken, to match saved data
        yield expert_obs

        target_ee_pose, gripper_action = expert.get_target_pose(expert_obs)

        delta_arm_action = ik_solver.compute_delta_action(
            target_ee_pose=target_ee_pose,
            model=env.model, data=env.data, ee_site_id=env.ee_site_id,
            joint_qpos_indices=arm_joint_ids,
            effective_dt=effective_dt, max_dq=max_dq
        )
        final_action = np.concatenate([delta_arm_action, [gripper_action]])

        obs, _, terminated, truncated, _ = env.step(final_action)
        
        if terminated or truncated or expert.is_done():
            # Yield the final observation state
            final_expert_obs = env.get_expert_obs()
            final_expert_obs['expert_state'] = expert.get_state()
            yield final_expert_obs
            logger.info(f"Live simulation finished at step {step}. Final state: {expert.get_state()}")
            break
    
    env.close()
    logger.info("--- Live re-simulation complete ---")


def find_episode_by_seed(loader: SoAEpisodeLoader, target_seed: int) -> Tuple[int, Dict[str, Any]]:
    """Finds the index and data for the first episode matching a given seed."""
    for i in range(len(loader)):
        ep_meta = loader.episode_metadata[i]
        if ep_meta.get("seed") == target_seed:
            logger.info(f"Found episode for seed {target_seed} at index {i}.")
            return i, loader.get_episode(i)
    raise ValueError(f"Could not find an episode with seed {target_seed} in the dataset.")


def format_array_for_csv(arr: np.ndarray) -> str:
    """Formats a numpy array into a compact, readable string for CSV logging."""
    return np.array2string(arr, precision=6, separator=',', suppress_small=True)



def main(args):
    """Main orchestration function for the verification process."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Load Configuration from Data Generation ---
    try:
        with open(args.gen_config, "r") as f:
            gen_cfg = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Generation config not found at {args.gen_config}. Cannot run deterministic simulation.")
        return
        
    # --- 2. Load the Saved Episode from the LMDB Dataset ---
    try:
        loader = SoAEpisodeLoader(args.demo_path)
        ep_idx, saved_episode = find_episode_by_seed(loader, args.seed)
        saved_obs_list = saved_episode["obs_list"]
        logger.info(f"Successfully loaded saved trajectory for seed {args.seed}. Length: {len(saved_obs_list)}")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load saved episode: {e}")
        return
    finally:
        if 'loader' in locals() and loader:
            loader.close()

    # --- 3. Generate the "Live" Trajectory ---
    live_obs_generator = generate_live_trajectory(gen_cfg, args.seed)
    
    # --- 4. Setup Logging and Video Writing ---
    video_path = output_dir / f"verification_seed_{args.seed}.mp4"
    csv_path = output_dir / f"verification_log_seed_{args.seed}.csv"

    # Get frame dimensions from the first frame of the saved data
    h, w, _ = saved_obs_list[0]['image_primary'].shape
    # Video will be side-by-side, so width is doubled
    video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w * 2, h))

    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "step", "expert_state_live", "expert_state_saved", "state_match",
        "proprio_error", "ee_pose_error",
        "live_proprio_qpos", "saved_proprio_qpos",
        "live_ee_pose", "saved_ee_pose"
    ])
    
    error_history = {"proprio": [], "ee_pose": []}

    # --- 5. The Synchronized Comparison Loop ---
    logger.info(f"Starting synchronized comparison. Writing video to {video_path} and log to {csv_path}")
    comparison_len = min(len(saved_obs_list), 500) # Cap comparison length for safety
    
    for t in tqdm(range(comparison_len), desc="Comparing Timesteps"):
        try:
            live_obs = next(live_obs_generator)
        except StopIteration:
            logger.warning(f"Live generator stopped early at step {t}. Ending comparison.")
            break
        
        saved_obs = saved_obs_list[t]
        
        # --- Quantitative Comparison ---
        proprio_err = np.linalg.norm(live_obs['proprio'] - saved_obs['proprio'])
        ee_pose_err = np.linalg.norm(live_obs['ee_pose_world'] - saved_obs['ee_pose_world'])
        state_match = live_obs['expert_state'] == saved_obs.get('expert_state', 'N/A')
        
        error_history["proprio"].append(proprio_err)
        error_history["ee_pose"].append(ee_pose_err)
        saved_expert_state = saved_obs.get('expert_state', 'N/A')
        state_match = live_obs['expert_state'] == saved_expert_state

        csv_writer.writerow([
            t,
            live_obs['expert_state'],
            saved_expert_state,
            state_match,
            f"{proprio_err:.8f}",
            f"{ee_pose_err:.8f}",
            format_array_for_csv(live_obs['proprio'][:7]), # Log qpos part of proprio
            format_array_for_csv(saved_obs['proprio'][:7]),
            format_array_for_csv(live_obs['ee_pose_world']),
            format_array_for_csv(saved_obs['ee_pose_world'])
        ])
        
        # --- Qualitative Comparison (Video Frame) ---
        live_frame = cv2.cvtColor(live_obs['image_primary'], cv2.COLOR_RGB2BGR)
        saved_frame = cv2.cvtColor(saved_obs['image_primary'], cv2.COLOR_RGB2BGR)
        
        # Add labels to each frame
        cv2.putText(live_frame, "LIVE SIMULATION", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(saved_frame, "SAVED FROM LMDB", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1, cv2.LINE_AA)
        
        # Create side-by-side composite frame
        composite_frame = np.concatenate((live_frame, saved_frame), axis=1)
        
        # Add overlay text with diagnostics
        cv2.putText(composite_frame, f"Step: {t}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(composite_frame, f"State Match: {state_match}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if state_match else (0, 0, 255), 2)
        cv2.putText(composite_frame, f"Proprio Error: {proprio_err:.4f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(composite_frame, f"EE Pose Error: {ee_pose_err:.4f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        video_writer.write(composite_frame)

    # --- 6. Finalization and Reporting ---
    logger.info("Comparison loop finished. Finalizing outputs.")
    video_writer.release()
    csv_file.close()

    avg_proprio_err = np.mean(error_history["proprio"]) if error_history["proprio"] else 0
    avg_ee_pose_err = np.mean(error_history["ee_pose"]) if error_history["ee_pose"] else 0
    
    logger.info("="*50)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Average Proprioception Error: {avg_proprio_err:.6f}")
    logger.info(f"Average End-Effector Pose Error: {avg_ee_pose_err:.6f}")

    if avg_proprio_err < 1e-4 and avg_ee_pose_err < 1e-4:
        logger.info("✅ VERIFICATION PASSED: The saved data is a near-perfect match to the live simulation.")
    else:
        logger.warning("❌ VERIFICATION FAILED: Significant discrepancies found between saved and live data.")
        logger.warning("Check the generated CSV and video for details. This may indicate non-determinism in the simulation or data saving process.")
    logger.info("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SOTA end-to-end data verification script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "demo_path", type=str,
        help="Path to the .lmdb dataset file to verify."
    )
    parser.add_argument(
        "--gen-config", type=str, required=True,
        help="Path to the original `gen_dataset_config.yaml` used to create the data."
    )
    parser.add_argument(
        "--seed", type=int, required=True, default=42,
        help="The specific episode seed to verify."
    )
    parser.add_argument(
        "--output-dir", type=str, default="verification/run",
        help="Directory to save the output video and CSV log."
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second for the output comparison video."
    )
    
    main(parser.parse_args())


# python -m s10 "C:\Users\Hellx\Documents\Programming\python\Project\redhot\data\validation\validation_dataset.lmdb" --gen-config "configs\gen_dataset_config.yaml" --seed 702