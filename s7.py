# FILE: debug_scripts/1_verify_bc_policy.py

"""
Robust verification script for a trained Behavioral Cloning (BC) policy.

This script loads a BC model checkpoint and runs it in the PandaEnv for a
set number of episodes to empirically verify its performance.

Key Features:
- Explicit Pass/Fail criteria for each episode (Grasp + Lift).
- Automatic video recording of each episode for visual inspection.
- Detailed per-step logging of critical state variables.
- Clean, readable output and a final summary report.
- Handles all dependencies and potential errors gracefully.
"""

import argparse
import logging
import sys
from pathlib import Path
import time

# --- Dependency Check ---
# This check is critical for ensuring videos can be saved.
try:
    import gymnasium as gym
    import moviepy
except ImportError as e:
    print("=" * 80)
    print(f"ERROR: A required library is missing: {e.name}")
    print("Please install the necessary libraries by running:")
    print("pip install gymnasium moviepy")
    print("=" * 80)
    sys.exit(1)

import numpy as np
import torch

# --- Project Imports ---
# Add project root to path to allow relative imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from envs.panda_env import PandaEnv
from models.bc_policy import BCNet
from scripts.arun_experiment import load_bc_checkpoint

# Configure logging for a cleaner output
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("bc_verifier")


def verify_bc_policy(
    bc_model_path: str,
    run_name: str,
    num_episodes: int = 5,
    lift_threshold: float = 0.45,
):
    """
    Loads a BC policy and runs deterministic rollouts to verify its ability
    to perform the pick-and-place task.
    """
    logger.info("=" * 80)
    logger.info("--- DEBUG SCRIPT 1: BEHAVIORAL CLONING POLICY VERIFICATION ---")
    logger.info(f"Loading model from: {bc_model_path}")
    logger.info("=" * 80)

    # --- 1. Setup Environment ---
    video_dir = project_root / "debug_videos" / run_name
    video_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📹 Videos will be saved to: {video_dir}")

    env = PandaEnv(
        xml_path="envs/panda_pick_place.xml",
        control_mode='absolute',
        render_mode="rgb_array"
    )
    # Wrap the environment for video recording.
    env = gym.wrappers.RecordVideo(env, video_folder=str(video_dir), name_prefix=f"{run_name}-ep")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. Load the BC Model ---
    try:
        state_dict = load_bc_checkpoint(bc_model_path, device)
        action_dim = env.action_space.shape[0]
        model = BCNet(n_actions=action_dim).to(device)
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode for deterministic output
        logger.info(f"✅ Successfully loaded BC model onto {device}.")
    except Exception as e:
        logger.error(f"\n❌ [ERROR] Failed to load BC model: {e}", exc_info=True)
        env.close()
        return

    # --- 3. Run Verification Episodes ---
    successful_episodes = 0
    try:
        for i in range(num_episodes):
            logger.info(f"\n--- Episode {i+1}/{num_episodes} ---")
            obs, _ = env.reset()
            terminated = truncated = False
            step = 0
            is_grasped_in_ep = False
            is_lifted_in_ep = False
            max_gripper_cmd = -1.0
            
            while not (terminated or truncated):
                # Prepare observation for the model
                obs_for_policy = {
                    "image_primary": torch.from_numpy(obs["image_primary"]).to(device).permute(2, 0, 1).unsqueeze(0),
                    "proprio": torch.from_numpy(obs["proprio"]).to(device).unsqueeze(0),
                }

                # Get deterministic action from the BC policy
                with torch.no_grad():
                    action_tensor = model(obs_for_policy)
                action = action_tensor.squeeze(0).cpu().numpy()

                obs, _, terminated, truncated, _ = env.step(action)
                step += 1

                # Log key metrics
                ee_pos = obs['ee_pose_world'][:3]
                cube_pos = obs['object_pos_world']
                dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
                is_grasped = obs['is_grasped'][0]
                cube_z_height = cube_pos[2]
                gripper_cmd = action[-1] # Gripper action is the last element
                max_gripper_cmd = max(max_gripper_cmd, gripper_cmd)

                if is_grasped:
                    is_grasped_in_ep = True
                if is_grasped and cube_z_height > lift_threshold:
                    is_lifted_in_ep = True

                log_msg = (f"  Step {step:03d} | Dist: {dist_to_cube:.4f} | "
                           f"Grasped: {is_grasped} | Cube Z: {cube_z_height:.3f} | "
                           f"Gripper Cmd: {gripper_cmd:+.3f}")
                print(log_msg, end="\n")

                # --- BUG FIX APPLIED HERE ---
                # Access max_episode_steps through the unwrapped environment
                if step >= env.unwrapped.max_episode_steps:
                    truncated = True

            print() # Newline after the episode finishes

            # --- 4. Evaluate Episode Outcome ---
            if is_grasped_in_ep and is_lifted_in_ep:
                logger.info(f"✅ SUCCESS: Episode completed. Grasped and lifted the object.")
                successful_episodes += 1
            else:
                logger.info(f"❌ FAILURE: Episode finished without a successful grasp and lift.")
                logger.info(f"  - Achieved Grasp: {is_grasped_in_ep}")
                logger.info(f"  - Achieved Lift: {is_lifted_in_ep}")
                logger.info(f"  - Max Gripper Cmd: {max_gripper_cmd:.3f}")
    
    finally:
        # --- 5. Final Summary and Cleanup ---
        # The env.close() call triggers the final video to be saved.
        env.close()
        logger.info("\n" + "=" * 80)
        logger.info("--- VERIFICATION SUMMARY ---")
        success_rate = (successful_episodes / num_episodes) * 100
        logger.info(f"Success Rate: {successful_episodes}/{num_episodes} ({success_rate:.1f}%)")
        if success_rate >= 50: # Use >= for clarity
            logger.info("✅ Result: The BC policy is considered functional. Ready for RL.")
        else:
            logger.info("❌ Result: The BC policy is NOT functional. Do not proceed to RL.")
        logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust BC Policy Verification Script")
    parser.add_argument(
        "--bc_model_path",
        type=str,
        required=True,
        help="Path to the .pth BC model checkpoint (e.g., artifacts/my_run/checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="bc_verification",
        help="A name for this verification run, used for the video directory."
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run for verification."
    )
    args = parser.parse_args()

    # Create a unique run name with a timestamp to avoid overwriting videos
    run_name_with_timestamp = f"{args.run_name}_{time.strftime('%Y%m%d-%H%M%S')}"

    verify_bc_policy(
        args.bc_model_path,
        run_name=run_name_with_timestamp,
        num_episodes=args.num_episodes
    )