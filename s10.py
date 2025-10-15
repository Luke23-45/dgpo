# FILE: scripts/test_expert_dataset_vs_compare_delta.py (FINAL VERSION WITH VIDEO)

import logging
from pathlib import Path
import numpy as np
import sys
import argparse
from tqdm import tqdm
import cv2  # <-- ADD CV2 IMPORT
import mujoco
sys.path.append(str(Path(__file__).resolve().parent.parent))

from envs.panda_env import PandaEnv
from utils.scripted_expert import ScriptedExpert, ExpertConfig, ObjectProfile
from utils.ik_solver import IKSolver
from utils.expert_dataset import ExpertDataset

# --- Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s")
log = logging.getLogger("DIAGNOSTIC")
OUT_DIR = Path("diagnostic_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Consistent Configuration (Single Source of Truth) ---
MAX_EPISODE_STEPS = 500
ACTION_SCALING_FACTOR = 0.5
OBJECT_PROFILE_INSTANCE = ObjectProfile(size=np.array([0.04, 0.04, 0.04]), grasp_width_normalized=0.6)
EXPERT_CONFIG_INSTANCE = ExpertConfig()

# --- Monkey-Patching for State History Tracking (Unchanged) ---
def add_state_history_to_expert():
    if hasattr(ScriptedExpert, "get_state_history"):
        return
    log.info("Patching ScriptedExpert to track state history for diagnostics...")
    original_reset = ScriptedExpert.reset
    def patched_reset(self):
        original_reset(self)
        self._state_history = []
        self._wait_counter_history = []
    original_get_target_pose = ScriptedExpert.get_target_pose
    def patched_get_target_pose(self, expert_obs):
        self._state_history.append(self._state)
        self._wait_counter_history.append(self._wait_counter)
        return original_get_target_pose(self, expert_obs)
    ScriptedExpert.reset = patched_reset
    ScriptedExpert.get_target_pose = patched_get_target_pose
    ScriptedExpert.get_state_history = lambda self: self._state_history
    ScriptedExpert.get_wait_counter_history = lambda self: self._wait_counter_history

# --- START OF MODIFICATIONS ---
def run_ground_truth_trajectory(expert, seed: int, urdf_path: str, xml_path: str, video_writer: cv2.VideoWriter = None):
    """Generates the single 'ground truth' trajectory AND saves a video if a writer is provided."""
    log.info(f"--- [GROUND TRUTH] Running trajectory with seed={seed} ---")
    
    env = PandaEnv(
        xml_path=xml_path, control_mode="delta"
    )
    ik_solver = IKSolver(urdf_path=urdf_path)
    
    obs, _ = env.reset(seed=seed)
    env.set_object_size(expert.object.size)
    expert.reset()
    ik_solver.reset_controller_state()
    
    N_SUBSTEPS = 20
    effective_dt = env.model.opt.timestep * N_SUBSTEPS
    max_dq = env.ACTION_SCALING_FACTOR / effective_dt
    arm_joint_ids = np.arange(7)

    trajectory_data = []
    try:
        for step in tqdm(range(env.max_episode_steps), desc="[GROUND TRUTH] Generating"):
            # --- START OF PATCH 1 of 2: Save the FULL simulation state ---
            # Instead of just the observation dict, we save everything needed to perfectly reconstruct the state.
            state_snapshot = {
                "obs": env.get_expert_obs(),
                "qpos": env.data.qpos.copy(),
                "qvel": env.data.qvel.copy(),
                "time": env.data.time,
            }
            # --- END OF PATCH 1 of 2 ---

            target_ee_pose, grip = expert.get_target_pose(state_snapshot["obs"])
            
            delta = ik_solver.compute_delta_action(
                target_ee_pose=target_ee_pose, model=env.model, data=env.data, ee_site_id=env.ee_site_id,
                joint_qpos_indices=arm_joint_ids, effective_dt=effective_dt, max_dq=max_dq,
            )
            action = np.concatenate([delta, [grip]])
            
            state_snapshot["action"] = action # Add the action to the snapshot
            trajectory_data.append(state_snapshot)

            # Render frame and write to video
            if video_writer:
                frame_rgb = env.render()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_bgr, f"State: {expert.get_state()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 100), 2)
                cv2.putText(frame_bgr, "[GROUND TRUTH RUN]", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
                video_writer.write(frame_bgr)
            
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc or expert.is_done():
                break
    finally:
        env.close()

    success = expert.was_successful()
    log.info(f"[GROUND TRUTH] Run finished. Success: {success}. Steps: {len(trajectory_data)}.")
    return trajectory_data, success

def main(args: argparse.Namespace):
    add_state_history_to_expert()
    
    video_path = OUT_DIR / f"diagnostic_video_seed{args.seed}.mp4"
    log.info(f"Will save diagnostic video to: {video_path}")
    
    # Initialize video writer
    # We need a dummy env to get the frame size
    dummy_env = PandaEnv(xml_path=args.xml_path)
    frame = dummy_env.render()
    height, width, _ = frame.shape
    dummy_env.close()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))

    # We create ONE expert instance to be used for the ground truth run
    ground_truth_expert = ScriptedExpert(object_profile=OBJECT_PROFILE_INSTANCE, cfg=EXPERT_CONFIG_INSTANCE)
    
    try:
        gt_trajectory, gt_success = run_ground_truth_trajectory(
            ground_truth_expert, args.seed, args.urdf_path, args.xml_path, video_writer
        )
    finally:
        video_writer.release()
        log.info("Video writer released.")

    if not gt_success:
        log.error("Ground truth trajectory failed. Cannot perform a meaningful comparison. Video may show the failure.")
        return

    log.info(f"✅ Ground truth video saved successfully to {video_path}")
    
    # --- The rest of the script for numerical comparison remains the same ---
    # ...
    log.info("\n--- [DATASET] Simulating ExpertDataset Logic as an Observer ---")
    
    dataset = ExpertDataset(
        urdf_path=args.urdf_path,
        env_xml_path=args.xml_path,
        base_seed=args.seed,
        action_scaling_factor=ACTION_SCALING_FACTOR,
        scripted_cfg=EXPERT_CONFIG_INSTANCE,
        object_size=OBJECT_PROFILE_INSTANCE.size.tolist(),
        object_grasp_width=OBJECT_PROFILE_INSTANCE.grasp_width_normalized,
    )
    dataset._init_worker_state()

    dataset_actions = []
    effective_dt = dataset._env.model.opt.timestep * 20

    for i, step_data in enumerate(tqdm(gt_trajectory, desc="[DATASET] Observing")):
        
        # Manually set the dataset's internal simulation state to perfectly
        # match the ground truth step using the saved full state.
        dataset._env.data.time = step_data["time"]
        dataset._env.data.qpos[:] = step_data["qpos"]
        dataset._env.data.qvel[:] = step_data["qvel"]
        # We must call mj_forward to ensure all derived values (like sensor data, site positions) are updated.
        mujoco.mj_forward(dataset._env.model, dataset._env.data)

        # Sync the expert's internal FSM state
        # The observation used here MUST be the one from the ground truth step data.
        true_obs = step_data["obs"]
        dataset._scripted_expert._state = ground_truth_expert.get_state_history()[i]
        dataset._scripted_expert._wait_counter = ground_truth_expert.get_wait_counter_history()[i]
        
        # Call the dataset's core logic function on the true observation
        _, ds_action, _ = dataset._generate_one(true_obs)
        dataset_actions.append(ds_action)

    # --- Final Comparison ---
    log.info("="*60)
    log.info("          FINAL DIAGNOSTIC COMPARISON")
    log.info("="*60)

    max_action_diff, divergence_step = 0.0, -1
    for i in range(len(gt_trajectory)):
        diff = np.linalg.norm(gt_trajectory[i]["action"] - dataset_actions[i])
        if diff > max_action_diff:
            max_action_diff, divergence_step = diff, i
            
    log.info(f"Max action difference: {max_action_diff:.8f} (found at step {divergence_step})")

    if max_action_diff < 1e-6:
        log.info("✅✅✅ PASSED: The ExpertDataset's core logic is bit-for-bit identical to the ground truth.")
    else:
        log.error("❌❌❌ FAILED: Discrepancy found. The ExpertDataset logic is not identical.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Definitive diagnostic for ExpertDataset logic.")
    parser.add_argument("--seed", type=int, default=815, help="Seed for the simulation.")
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf")
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml")
    args = parser.parse_args()
    
    main(args)