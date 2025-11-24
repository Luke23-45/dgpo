# FILE: scripts/visualize_dataset_episode.py

"""
Dataset Visualizer.
Renders a video of a specific episode from the training dataset (LMDB).
Overlays Ground Truth data (Phase, Gripper, Advantage) on the video 
to verify data integrity.
"""

import argparse
import cv2
import logging
import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.semantic_planner_dataset import SemanticPlannerDataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Viz")

def overlay_text(img, text, pos, color=(0, 255, 0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to training.lmdb")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize")
    parser.add_argument("--output", type=str, default="viz_output.mp4", help="Output filename")
    args = parser.parse_args()

    # 1. Load Dataset (Raw Mode)
    # We use use_aug=False because we want to see the clean data
    log.info(f"Loading dataset: {args.dataset}")
    ds = SemanticPlannerDataset(args.dataset, use_aug=False)
    
    # 2. Extract Episode Metadata
    reader = ds.expert_reader
    if args.episode >= reader.get_num_episodes():
        log.error(f"Episode {args.episode} out of bounds (Max: {reader.get_num_episodes()-1})")
        return

    ep_meta = reader.episode_metadata[args.episode]
    ep_len = ep_meta['length']
    log.info(f"Visualizing Episode {args.episode} (Length: {ep_len})")

    # 3. Access Raw Modalities via LRU Cache helper
    # This bypasses the __getitem__ transforms to get raw uint8 images
    def get_mod(name):
        if name not in ep_meta["modalities"]: return None
        meta = ep_meta["modalities"][name]
        return reader._get_full_modality_array(
            key=meta["key"], compression=meta["compression"],
            dtype_str=meta["dtype"], shape_list=tuple(meta["shape"])
        )

    # Load full arrays for this episode
    images = get_mod("image_primary")
    
    # Try to load SOTA explicit keys first, fallback to others
    phases = get_mod("gt_phase")
    if phases is None: phases = get_mod("task_phases") # fallback
    
    grippers = get_mod("gt_gripper")
    if grippers is None: grippers = get_mod("is_grasped") # fallback

    advs = get_mod("advantages")

    if images is None:
        log.error("No image_primary found in dataset.")
        return

    # 4. Video Setup
    H, W, C = images[0].shape
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))

# 5. Frame Loop
    for t in tqdm(range(ep_len)):
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(images[t], cv2.COLOR_RGB2BGR)

        # Data Extraction with .item() for safety
        # 'phases' is usually (T,), but 'grippers' might be (T, 1) based on your writer logic
        p_val = phases[t] if phases is not None else -1
        if isinstance(p_val, np.ndarray): p_val = p_val.item()
            
        g_val = grippers[t] if grippers is not None else -1.0
        if isinstance(g_val, np.ndarray): g_val = g_val.item()

        a_val = advs[t] if advs is not None else 0.0
        if isinstance(a_val, np.ndarray): a_val = a_val.item()

        # Determine Text Color based on Gripper Logic
        # 1.0 = Closed (Green), 0.0 = Open (Yellow)
        status_color = (0, 255, 0) if g_val > 0.5 else (0, 255, 255)

        # Overlay Info
        overlay_text(img, f"Step: {t}", (10, 20), (255, 255, 255))
        overlay_text(img, f"Phase: {int(p_val)}", (10, 40), (255, 255, 255))
        overlay_text(img, f"Grip Label: {g_val:.1f}", (10, 60), status_color)
        overlay_text(img, f"Advantage: {a_val:.2f}", (10, 80), (200, 200, 255))

        writer.write(img)


    writer.release()
    log.info(f"Video saved to {args.output}")
    
    # Verify Gripper Stats for this episode
    if grippers is not None:
        closed_steps = np.sum(grippers > 0.5)
        log.info(f"Episode Gripper Stats: Closed for {closed_steps} / {ep_len} steps")

if __name__ == "__main__":
    main()



# python -m s4 --dataset "C:\Users\Hellx\Documents\Programming\python\Project\redhot\data\final_training_set\training_set.lmdb" --episode 0 --output "check_ep0.mp4"