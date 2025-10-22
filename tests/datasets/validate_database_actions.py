# FILE: scripts/validate_database_actions.py
# (Definitive Auditor to Compare Database Actions vs. Live Replay)

import logging
from pathlib import Path
import numpy as np
import sys
import argparse
from tqdm import tqdm
import cv2
import json
import lmdb
import pickle
import mujoco

sys.path.append(str(Path(__file__).resolve().parent.parent))

# --- Local Imports (from your project) ---
from envs.panda_env import PandaEnv
from utils.scripted_expert import ScriptedExpert, ExpertConfig, ObjectProfile
from utils.ik_solver import IKSolver

# --- Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s")
log = logging.getLogger("DB_VALIDATOR")
OUT_DIR = Path("diagnostic_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Configuration (Copied from previous script for consistency) ---
OBJECT_PROFILE_INSTANCE = ObjectProfile(size=np.array([0.04, 0.04, 0.04]), grasp_width_normalized=0.6)
EXPERT_CONFIG_INSTANCE = ExpertConfig()

# --- SOTA Database Loader (borrowed from validate_transformation.py) ---
class SoAEpisodeLoader:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        log.info(f"Initializing SOTA Loader for DB: {self.db_path}")
        self.env = lmdb.open(str(db_path), readonly=True, lock=False, readahead=False, subdir=False)
        
        index_path = self.db_path.parent / f"{self.db_path.stem}_index.json"
        with open(index_path, 'r') as f:
            self.index_data = json.load(f)
        self.episode_metadata = self.index_data["episodes"]
        log.info(f"[SOTA Loader] Indexed {len(self.episode_metadata)} episodes from JSON.")

    def find_episode_by_seed(self, seed: int) -> int:
        log.info(f"Searching for episode with seed={seed}...")
        for i, meta in enumerate(self.episode_metadata):
            if meta.get("seed") == seed:
                log.info(f"Found episode for seed {seed} at index {i}.")
                return i
        raise ValueError(f"Could not find an episode with seed={seed} in the database index.")

    def _get_modality(self, key, compression, dtype, shape):
        with self.env.begin() as txn:
            blob = txn.get(key.encode('ascii'))
        if compression == 'raw':
            return np.frombuffer(blob, dtype=np.dtype(dtype)).reshape(shape)
        elif compression in ('jpeg', 'png'):
            byte_list = pickle.loads(blob)
            images = [cv2.cvtColor(cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for b in byte_list]
            return np.stack(images)
        raise ValueError(f"Unknown compression {compression}")

    def get_episode(self, index: int) -> dict:
        ep_meta = self.episode_metadata[index]
        modalities = {}
        for name, meta in ep_meta["modalities"].items():
            modalities[name] = self._get_modality(meta['key'], meta['compression'], meta['dtype'], meta['shape'])
        return modalities

    def close(self):
        self.env.close()

# --- Live Trajectory Replay Function ---
def replay_live_trajectory(seed: int, urdf_path: str, xml_path: str):
    """Generates a trajectory live, returning actions and rendered frames."""
    log.info(f"--- [LIVE REPLAY] Running trajectory with seed={seed} ---")
    
    env = PandaEnv(xml_path=xml_path, control_mode="delta")
    ik_solver = IKSolver(urdf_path=urdf_path)
    expert = ScriptedExpert(object_profile=OBJECT_PROFILE_INSTANCE, cfg=EXPERT_CONFIG_INSTANCE)
    
    obs, _ = env.reset(seed=seed)
    env.set_object_size(expert.object.size)
    expert.reset()
    ik_solver.reset_controller_state()
    
    N_SUBSTEPS = 20
    effective_dt = env.model.opt.timestep * N_SUBSTEPS
    max_dq = env.ACTION_SCALING_FACTOR / effective_dt
    arm_joint_ids = np.arange(7)

    live_actions = []
    live_frames = []
    try:
        for step in tqdm(range(env.max_episode_steps), desc="[LIVE REPLAY] Generating"):
            # Get the full, rich observation for the expert
            expert_obs = env.get_expert_obs()
            target_ee_pose, grip = expert.get_target_pose(expert_obs)
            
            delta = ik_solver.compute_delta_action(
                target_ee_pose=target_ee_pose, model=env.model, data=env.data, ee_site_id=env.ee_site_id,
                joint_qpos_indices=arm_joint_ids, effective_dt=effective_dt, max_dq=max_dq,
            )
            action = np.concatenate([delta, [grip]])
            live_actions.append(action)

            # Render frame for the video
            frame_rgb = env.render()
            live_frames.append(frame_rgb)
            
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc or expert.is_done():
                break
    finally:
        env.close()

    log.info(f"[LIVE REPLAY] Run finished. Success: {expert.was_successful()}. Steps: {len(live_actions)}.")
    return np.array(live_actions), live_frames

def main(args: argparse.Namespace):
    # --- 1. Load Episode Data From Database ---
    db_loader = SoAEpisodeLoader(args.sota_db_path)
    try:
        ep_idx = db_loader.find_episode_by_seed(args.seed)
        db_episode_data = db_loader.get_episode(ep_idx)
    except (ValueError, KeyError) as e:
        log.error(f"Failed to load episode from database: {e}")
        return
    finally:
        db_loader.close()
        
    db_actions = db_episode_data['actions']
    db_images = db_episode_data['image_primary']

    # --- 2. Generate Live Trajectory for Comparison ---
    live_actions, live_frames = replay_live_trajectory(args.seed, args.urdf_path, args.xml_path)

    # --- 3. Sanity Check and Numerical Comparison ---
    log.info("\n" + "="*60)
    log.info("          FINAL DIAGNOSTIC COMPARISON")
    log.info("="*60)

    if len(db_actions) != len(live_actions):
        log.error(f"❌❌❌ FATAL: Trajectory length mismatch! DB={len(db_actions)}, Live={len(live_actions)}.")
        log.error("This indicates a non-determinism issue in the expert's termination logic.")
        return

    action_diffs = np.linalg.norm(db_actions - live_actions, axis=1)
    max_action_diff = np.max(action_diffs)
    divergence_step = np.argmax(action_diffs)
    
    log.info(f"Max action difference: {max_action_diff:.8f} (found at step {divergence_step})")

    if max_action_diff < 1e-6:
        log.info("✅✅✅ PASSED: The actions in the database are bit-for-bit identical to a live deterministic replay.")
    else:
        log.error("❌❌❌ FAILED: Discrepancy found. The database actions do not match the live replay.")

    # --- 4. Generate Side-by-Side Diagnostic Video ---
    video_path = OUT_DIR / f"db_vs_live_validation_seed{args.seed}.mp4"
    log.info(f"Generating comparison video: {video_path}")
    
    height, width, _ = db_images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Video width is doubled to accommodate side-by-side frames
    video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width * 2, height))

    try:
        for t in tqdm(range(len(db_actions)), desc="Generating Video"):
            frame_db = db_images[t]
            frame_live = live_frames[t]

            # Convert to BGR for OpenCV
            frame_db_bgr = cv2.cvtColor(frame_db, cv2.COLOR_RGB2BGR)
            frame_live_bgr = cv2.cvtColor(frame_live, cv2.COLOR_RGB2BGR)

            # Add text overlays
            cv2.putText(frame_db_bgr, "[FROM DATABASE]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            cv2.putText(frame_live_bgr, "[LIVE REPLAY]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 100), 2)
            
            diff_text = f"Action Diff: {action_diffs[t]:.6f}"
            color = (100, 255, 100) if action_diffs[t] < 1e-6 else (100, 100, 255)
            cv2.putText(frame_live_bgr, diff_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Combine frames and write to video
            combined_frame = np.hstack([frame_db_bgr, frame_live_bgr])
            video_writer.write(combined_frame)
    finally:
        video_writer.release()
        log.info(f"✅ Comparison video saved successfully to {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate database actions against a live replay.")
    parser.add_argument("sota_db_path", type=str, help="Path to the SOTA LMDB file to validate.")
    parser.add_argument("--seed", type=int, default=815, help="Seed of the episode to validate.")
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf")
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml")
    args = parser.parse_args()
    
    main(args)

"""
python -m s10 data\training\sota_dataset\expert_training_run_99914b93.lmdb  --seed 49
"""