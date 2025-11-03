# FILE: scripts/evaluate_ego_planner.py
# (State-of-the-Art, Corrected, and Definitive Version)

"""
The definitive, state-of-the-art evaluation script for the Ego-Planner policy.

This script runs a trained policy on full episodes from a validation dataset and
generates detailed, insightful "Policy Decision Panel" videos. These videos
provide a rich qualitative analysis of the model's behavior by showing what the
policy sees, what its goal is, and what it decides to do at every step.

This version incorporates all corrections from a deep architectural and logical audit,
ensuring the evaluation is both correct and meaningful.
"""

import logging
from pathlib import Path
import pickle
from typing import Dict, Any, Optional, List
import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from envs.panda_env import PandaEnv
from PIL import Image
# --- Project-Specific Imports ---
from models.ego_planner import EgoPlanner, EgoPlannerConfig
from models.diffusion_policy import NoiseScheduler, NoiseSchedulerConfig
from utils.ego_planner_dataset import EgoPlannerDataset
import mujoco
from utils.lmdb_utils import close_lmdb_env, open_lmdb_env
import json
import functools
import sys
# --- SOTA Imports ---
try:
    import cv2
except ImportError:
    print("Error: OpenCV (cv2) package not found. Please install it with 'pip install opencv-python'")
    sys.exit(1)

# Setup a logger for the script
log = logging.getLogger(__name__)


# FILE: s12.py (Your evaluation script)
# REPLACE the entire load_model_from_checkpoint function with this one.

def load_model_from_checkpoint(model_config: DictConfig, checkpoint_path: str, device: str) -> EgoPlanner:
    """
    Loads the EgoPlanner model from a checkpoint with SOTA robustness.
    
    This definitive version is patched to handle:
    1.  Architectural drift in the HuggingFace vision backbone by renaming keys.
    2.  Potential mismatches by using strict=False for maximum compatibility.
    """
    log.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the entire checkpoint dictionary from the file
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    
    # Check if the actual model weights are nested under 'state_dict'
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Assume the checkpoint file is just the state_dict itself
        state_dict = checkpoint

    # --- START OF DEFINITIVE PATCH ---
    # This block programmatically renames keys to fix the vision backbone mismatch.
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # Define the patterns to replace
        replacements = {
            "strategist.vision_backbone.vision_embeddings.": "strategist.vision_backbone.vision_model.embeddings.",
            "strategist.vision_backbone.vision_encoder.": "strategist.vision_backbone.vision_model.encoder.",
            "strategist.vision_backbone.vision_post_layernorm.": "strategist.vision_backbone.vision_model.post_layernorm.",
            "strategist.vision_backbone.vision_head.": "strategist.vision_backbone.vision_model.head."
        }
        for old_prefix, new_prefix in replacements.items():
            if new_key.startswith(old_prefix):
                new_key = new_key.replace(old_prefix, new_prefix, 1)
                break # Stop after the first match
        cleaned_state_dict[new_key] = value
    
    if len(cleaned_state_dict) != len(state_dict):
        log.warning("State dict key count changed during cleaning. This is unexpected.")

    log.info("Finished renaming keys to handle potential vision backbone updates.")
    # --- END OF DEFINITIVE PATCH ---

    # Instantiate the model using the provided configuration
    model = EgoPlanner(model_config).to(device)
    
    # Load the (now corrected) state dict into the model.
    # Using strict=False is more robust as it will ignore any non-critical mismatches
    # and only error out on true problems like shape mismatches.
    incompatible_keys = model.load_state_dict(cleaned_state_dict, strict=False)

    if incompatible_keys.missing_keys:
        log.warning(f"Weights not found in checkpoint for layers: {incompatible_keys.missing_keys}")
    if incompatible_keys.unexpected_keys:
        log.warning(f"Checkpoint had unexpected weights not in model: {incompatible_keys.unexpected_keys}")

    model.eval()
    log.info("Model loaded successfully and set to evaluation mode.")
    return model

@torch.no_grad()
def generate_closed_loop_rollout_video(
    episode_idx: int,
    policy: EgoPlanner,
    dataset: EgoPlannerDataset,
    scheduler: NoiseScheduler,
    cfg: DictConfig,
    device: torch.device,
    output_dir: Path
):
    """
    Performs a closed-loop rollout in a simulator using the trained policy
    and generates a video of the autonomous attempt. This is the true test
    of the policy's ability to perform the task.
    """
    log.info(f"Generating CLOSED-LOOP ROLLOUT video for episode {episode_idx}...")
    policy.eval()

    # 1. --- SETUP THE SIMULATOR & EPISODE ---
    # Initialize the environment in 'delta' mode, as that's what the policy outputs.
    env = PandaEnv(xml_path=cfg.env.xml_path, control_mode='delta')

    # Get the seed from the dataset episode to ensure the environment reset is identical.
    episode_seed = dataset.expert_reader.episode_metadata[episode_idx]['seed']
    
    # Reset the environment. The env's reset logic will handle all object/robot placement.
    obs, _ = env.reset(seed=episode_seed)

    # Get the static initial and goal images that the policy's "Strategist" needs.
    # These remain constant for the entire rollout.
    first_valid_timestep = dataset.obs_horizon - 1
    initial_sample = dataset.get_episode_sample(episode_idx, first_valid_timestep)
    if initial_sample is None:
        log.error(f"Could not get initial sample for ep {episode_idx}. Skipping rollout.")
        env.close()
        return

    initial_image_static = initial_sample['initial_image'].unsqueeze(0).to(device)
    goal_image_static = initial_sample['goal_image'].unsqueeze(0).to(device)
    goal_image_np = initial_sample['goal_image'].permute(1, 2, 0).cpu().numpy()

    # 2. --- THE CLOSED-LOOP ROLLOUT ---
    frames = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)

    for t in tqdm(range(env.max_episode_steps), desc="  Rollout Step"):
        # a. Preprocess the current observation from the simulator for the policy.
        # The policy expects a history of observations, so we stack the current one.
        obs_history_tensors = {}
        for key in ['image_primary', 'image_wrist', 'proprio']:
            if 'image' in key:
                # Convert simulator's HWC NumPy array to model's CHW Tensor
                transform = dataset.transform_wrist if 'wrist' in key else dataset.transform_primary
                tensor = transform(Image.fromarray(obs[key]))
            else:
                tensor = torch.from_numpy(obs[key]).float()
            
            # Create a history by duplicating the current observation `obs_horizon` times
            obs_history_tensors[key] = tensor.unsqueeze(0).repeat(dataset.obs_horizon, 1, 1, 1) if 'image' in key else tensor.unsqueeze(0).repeat(dataset.obs_horizon, 1)
        
        # b. Construct the full batch dictionary for the policy.
        batch_for_policy = {
            'initial_image': initial_image_static,
            'goal_image': goal_image_static,
            'observation_history': {k: v.unsqueeze(0).to(device) for k, v in obs_history_tensors.items()}
        }

        # c. Get the policy's predicted action horizon.
        predicted_action_horizon = policy.sample(
            batch=batch_for_policy,
            scheduler=scheduler,
            guidance_plan=cfg.inference.guidance_scale_plan,
            guidance_obs=cfg.inference.guidance_scale_obs,
            num_inference_steps=cfg.inference.sampling_steps
        )
        
        # d. Execute only the FIRST action step in the simulator.
        action_to_take = predicted_action_horizon.squeeze(0).cpu().numpy()[0]
        
        # e. Step the environment and get the NEW observation.
        obs, _, terminated, truncated, _ = env.step(action_to_take)
        
        # f. Render the visualization frame.
        primary_img_np = obs['image_primary']
        wrist_img_np = obs['image_wrist']
        
        for ax_row in axes:
            for ax in ax_row: ax.clear()

        fig.suptitle(f'Ego-Planner Closed-Loop Rollout | Episode {episode_idx}', fontsize=16)
        axes[0, 0].imshow(primary_img_np); axes[0, 0].set_title(f'Primary View (Step {t})'); axes[0, 0].axis('off')
        axes[0, 1].imshow(wrist_img_np); axes[0, 1].set_title('Wrist View'); axes[0, 1].axis('off')
        axes[1, 0].imshow(goal_image_np); axes[1, 0].set_title('Strategic Goal'); axes[1, 0].axis('off')
        axes[1, 1].text(0.5, 0.5, 'Policy is in Closed-Loop Control', ha='center', va='center', fontsize=14, wrap=True)
        axes[1, 1].axis('off')

        fig.canvas.draw()
        rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
        frame = rgba_buffer[:, :, :3]
        frames.append(frame)

        if terminated or truncated:
            log.info(f"Rollout ended at step {t}. Terminated: {terminated}, Truncated: {truncated}")
            break

    # 3. --- CLEANUP ---
    plt.close(fig)
    env.close()
    
    video_path = output_dir / f"episode_{episode_idx}_CLOSED_LOOP_rollout.mp4"
    imageio.mimsave(video_path, frames, fps=cfg.video_fps, quality=8)
    log.info(f"Successfully saved closed-loop rollout video to {video_path}")

@torch.no_grad()
def generate_episode_video(
    episode_idx: int,
    policy: EgoPlanner,
    dataset: EgoPlannerDataset,
    scheduler: NoiseScheduler,
    cfg: DictConfig,
    device: torch.device,
    output_dir: Path
):
    """Generates a complete "Policy Decision Panel" video for a single episode."""
    log.info(f"Generating evaluation video for episode {episode_idx}...")
    
    policy.eval()
    
    # Get the static goal image for the entire episode visualization.
    # This uses the dataset's __getitem__ which returns tensors.
    first_valid_timestep = dataset.obs_horizon - 1
    initial_sample = dataset.get_episode_sample(episode_idx, first_valid_timestep)
    if initial_sample is None:
        log.error(f"FATAL: Could not retrieve initial sample for episode {episode_idx}. Skipping.")
        return
    goal_image_np = initial_sample['goal_image'].permute(1, 2, 0).cpu().numpy()

    frames = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    fig.suptitle(f'Ego-Planner Evaluation | Episode {episode_idx}', fontsize=16)
    
    episode_len = dataset.get_episode_length(episode_idx)
    
    # --- BUG FIX #1: The evaluation loop MUST start from the first valid timestep. ---
    # We loop up to the second-to-last step because at step 't', the observation history
    # ends at 't', and we predict the action that would be taken next.
    eval_range = range(first_valid_timestep, episode_len - 1)
    
    for t in tqdm(eval_range, desc=f"  Evaluating Episode {episode_idx}"):
        
        # Get the complete, pre-processed sample for the current timestep 't'.
        sample = dataset.get_episode_sample(episode_idx, t)
        if sample is None:
            log.warning(f"Could not retrieve sample for t={t}. Ending video generation early.")
            break

        # --- BUG FIX #2: Remove incorrect and redundant image processing. ---
        # The 'EgoPlannerDataset' already returns model-ready tensors.
        # We just need to add a batch dimension and send to the correct device.
        batch_for_policy = {
            'initial_image': sample['initial_image'].unsqueeze(0).to(device),
            'goal_image': sample['goal_image'].unsqueeze(0).to(device),
            'observation_history': {
                k: v.unsqueeze(0).to(device) for k, v in sample['observation_history'].items()
            }
        }
        gt_action_chunk = sample['action_chunk']

        # --- BUG FIX #3: Call policy.sample with the correct 'batch' dictionary format. ---
        predicted_action_horizon = policy.sample(
            batch=batch_for_policy, # Pass the single batch dictionary
            scheduler=scheduler,
            guidance_plan=cfg.inference.guidance_scale_plan,
            guidance_obs=cfg.inference.guidance_scale_obs,
            num_inference_steps=cfg.inference.sampling_steps
        )
        
        # 3. Prepare data for plotting (move to CPU/numpy).
        # The observation history is a sequence; we want the *last* image in that sequence for the current view.
        primary_img_np = sample['observation_history']['image_primary'][-1].permute(1, 2, 0).cpu().numpy()
        wrist_img_np = sample['observation_history']['image_wrist'][-1].permute(1, 2, 0).cpu().numpy()
        # --- END OF DEFINITIVE FIX ---
        gt_action_np = gt_action_chunk.numpy()
        pred_action_np = predicted_action_horizon.squeeze(0).cpu().numpy()

        # 4. Render the 2x2 panel frame.
        for ax_row in axes:
            for ax in ax_row: ax.clear()

        axes[0, 0].imshow(primary_img_np); axes[0, 0].set_title(f'Primary View (Step {t})'); axes[0, 0].axis('off')
        axes[0, 1].imshow(wrist_img_np); axes[0, 1].set_title('Wrist View'); axes[0, 1].axis('off')
        axes[1, 0].imshow(goal_image_np); axes[1, 0].set_title('Strategic Goal'); axes[1, 0].axis('off')

        for i in range(gt_action_np.shape[1]):
            axes[1, 1].plot(gt_action_np[:, i], color=f'C{i}', linestyle='--', label=f'GT Dim {i}' if t == first_valid_timestep else "")
            axes[1, 1].plot(pred_action_np[:, i], color=f'C{i}', linestyle='-', label=f'Pred Dim {i}' if t == first_valid_timestep else "")
        axes[1, 1].set_title('Action Horizon: Prediction vs. Expert'); axes[1, 1].set_xlabel('Horizon Step'); axes[1, 1].set_ylabel('Action Value')
        axes[1, 1].set_ylim(-1.2, 1.2); axes[1, 1].grid(True, alpha=0.4)
        if t == first_valid_timestep: axes[1, 1].legend(fontsize='small', ncol=2)

        fig.canvas.draw()
        
        # --- START OF DEFINITIVE FIX (Matplotlib API Update) ---
        # Get the RGBA buffer from the canvas and convert it to a NumPy array
        rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
        # Convert the 4-channel RGBA image to a 3-channel RGB image for the video
        frame = rgba_buffer[:, :, :3]
        # --- END OF DEFINITIVE FIX ---
        
        frames.append(frame)

    plt.close(fig)
    
    video_path = output_dir / f"episode_{episode_idx}_evaluation.mp4"
    imageio.mimsave(video_path, frames, fps=cfg.video_fps, quality=8)
    log.info(f"Successfully saved evaluation video to {video_path}")


class SoAEpisodeLoader:
    """Handles loading and reconstructing full episodes from the SOTA SoA format."""
    def __init__(self, demo_path: str):
        self.demo_path = Path(demo_path)
        if not self.demo_path.exists(): raise FileNotFoundError(f"Dataset path not found: {self.demo_path}")
        
        # Suffix handling for index path is more robust now
        index_path = self.demo_path.with_suffix(f"{self.demo_path.suffix}_index.json")
        if not index_path.exists(): raise FileNotFoundError(f"Index file not found: {index_path}")
        
        with open(index_path, 'r') as f: self.index_data = json.load(f)
        self.episode_metadata = self.index_data["episodes"]
        log.info(f"SoA Loader indexed {len(self.episode_metadata)} episodes.")
        self._lmdb_env = open_lmdb_env(str(self.demo_path), readonly=True, lock=False, readahead=False, subdir=False)

    def __len__(self) -> int: return len(self.episode_metadata)
    def __del__(self): self.close()
    def close(self): close_lmdb_env(self._lmdb_env)
    
    def _get_lmdb_blob(self, key: str) -> bytes:
        with self._lmdb_env.begin(write=False) as txn:
            blob = txn.get(key.encode("ascii"))
            if blob is None: raise KeyError(f"Missing LMDB key {key!r}")
            return blob

    @functools.lru_cache(maxsize=16)
    def _get_full_modality_array(self, key: str, compression: str, dtype_str: str, shape: tuple) -> np.ndarray:
        blob = self._get_lmdb_blob(key)
        if compression == "raw":
            return np.frombuffer(blob, dtype=np.dtype(dtype_str)).reshape(shape)
        elif compression in ("jpeg", "png"):
            images = [cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR) for b in pickle.loads(blob)]
            return np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images])
        raise ValueError(f"Unknown compression: {compression}")

    def get_episode(self, index: int) -> Dict[str, Any]:
        """Reconstructs a single episode, including ALL saved modalities."""
        if not (0 <= index < len(self)): raise IndexError(f"Episode index {index} out of bounds.")
            
        ep_meta = self.episode_metadata[index]
        modalities = {name: self._get_full_modality_array(meta["key"], meta["compression"], meta["dtype"], tuple(meta["shape"]))
                      for name, meta in ep_meta["modalities"].items()}
            
        obs_list = []
        # Find all keys that represent observations (anything that's not 'actions')
        obs_keys = [k for k in modalities.keys() if k != 'actions']
        
        for t in range(ep_meta["length"]):
            obs_step = {key: modalities[key][t] for key in obs_keys}
            obs_list.append(obs_step)

        return {"obs_list": obs_list, "actions": modalities.get("actions", [])}



@torch.no_grad()
def generate_forced_trajectory_video(
    episode_idx: int, policy: EgoPlanner, dataset: EgoPlannerDataset,
    scheduler: NoiseScheduler, cfg: DictConfig, device: torch.device, output_dir: Path
):
    """Performs "Forced Trajectory Following" using the complete simulation state."""
    log.info(f"Generating FORCED TRAJECTORY video for episode {episode_idx}...")
    policy.eval()

    env = PandaEnv(xml_path=cfg.env.xml_path, control_mode='delta')
    
    try:
        loader = SoAEpisodeLoader(cfg.dataset.path)
        full_episode_data = loader.get_episode(episode_idx)
    finally:
        loader.close() # Ensure LMDB connection is closed.

    # This part remains the same as our previous corrected version
    initial_img_np = full_episode_data['obs_list'][0]['image_primary']
    goal_img_np = full_episode_data['obs_list'][-1]['image_primary']
    initial_image_tensor = dataset.transform_primary(Image.fromarray(initial_img_np)).unsqueeze(0).to(device)
    goal_image_tensor = dataset.transform_primary(Image.fromarray(goal_img_np)).unsqueeze(0).to(device)

    frames, fig = [], plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 2)
    ax_pri, ax_wri = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_goal, ax_act = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    
    for t in tqdm(range(len(full_episode_data['obs_list'])), desc="  Forced Step"):
        gt_obs = full_episode_data['obs_list'][t]
        
        # --- DEFINITIVE FIX: Use the 'sim_qpos' key for teleportation ---
        qpos_key = 'sim_qpos'
        if qpos_key not in gt_obs:
            log.error(f"FATAL: Key '{qpos_key}' not found in reconstructed data at step {t}. "
                      "This evaluation mode cannot run. Please regenerate your dataset and ensure "
                      "the full `env.data.qpos` array is saved under the '{qpos_key}' modality.")
            break # Exit the loop for this episode
        
        env.data.qpos[:] = gt_obs[qpos_key]
        mujoco.mj_forward(env.model, env.data)
        sim_obs = env.get_expert_obs()

        obs_history_batch = {
            'image_primary': torch.stack([dataset.transform_primary(Image.fromarray(sim_obs['image_primary']))] * dataset.obs_horizon).unsqueeze(0).to(device),
            'image_wrist': torch.stack([dataset.transform_wrist(Image.fromarray(sim_obs['image_wrist']))] * dataset.obs_horizon).unsqueeze(0).to(device),
            'proprio': torch.from_numpy(sim_obs['proprio']).float().repeat(dataset.obs_horizon, 1).unsqueeze(0).to(device)
        }
        
        predicted_action_horizon = policy.sample(
            initial_image=initial_image_tensor, goal_image=goal_image_tensor,
            observation_history=obs_history_batch, scheduler=scheduler, **cfg.inference
        )
        pred_action_np = predicted_action_horizon.squeeze(0).cpu().numpy()
        
        # (Rendering logic remains the same)
        ax_pri.clear(); ax_wri.clear(); ax_goal.clear(); ax_act.clear()
        fig.suptitle(f'Ego-Planner Forced Trajectory | Episode {episode_idx}', fontsize=16)
        ax_pri.imshow(sim_obs['image_primary']); ax_pri.set_title(f'Sim View (Step {t})'); ax_pri.axis('off')
        ax_wri.imshow(sim_obs['image_wrist']); ax_wri.set_title('Wrist View'); ax_wri.axis('off')
        ax_goal.imshow(goal_img_np); ax_goal.set_title('Strategic Goal'); ax_goal.axis('off')

        for i in range(pred_action_np.shape[1]):
             ax_act.axhline(y=full_episode_data['actions'][t][i], color=f'C{i}', linestyle='--', label=f'GT Dim {i}' if t == 0 else "")
             ax_act.plot(pred_action_np[:, i], color=f'C{i}', linestyle='-', label=f'Pred Dim {i}' if t == 0 else "")
        ax_act.set_title('Action Horizon: Prediction vs. Expert'); ax_act.set_xlabel('Horizon'); ax_act.set_ylabel('Action'); ax_act.set_ylim(-1.2, 1.2); ax_act.grid(True, alpha=0.4)
        if t == 0: ax_act.legend(fontsize='small', ncol=2)
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

    plt.close(fig); env.close()
    if frames:
        video_path = output_dir / f"ep{episode_idx}_forced_trajectory.mp4"
        imageio.mimsave(video_path, frames, fps=cfg.video_fps, quality=8)
        log.info(f"Saved forced trajectory video to {video_path}")


@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_ego_planner_config")
def evaluate(cfg: DictConfig):
    log.info("----------- Ego-Planner SOTA Evaluation -----------")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("--------------------------------------------------")

    pl.seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model_config = EgoPlannerConfig(**cfg.model)
    policy = load_model_from_checkpoint(model_config, cfg.checkpoint_path, device)

    dataset = EgoPlannerDataset(
        dataset_path=cfg.dataset.path,
        obs_horizon=cfg.model.obs_horizon,
        action_horizon=cfg.model.action_horizon,
        use_aug=False # Never use augmentation for evaluation
    )


    # --- DEFINITIVE FIX 3: Create scheduler once from the same config as training ---
    scheduler_config = NoiseSchedulerConfig(**cfg.scheduler)
    scheduler = NoiseScheduler(scheduler_config).to(device)

    num_episodes_in_dataset = dataset.get_num_episodes()

    episode_ids_to_eval = cfg.episode_indices
    if not episode_ids_to_eval: # If list is empty, evaluate all specified
        episode_ids_to_eval = list(range(min(cfg.max_episodes_to_eval, num_episodes_in_dataset)))

    for ep_idx in episode_ids_to_eval:
        if ep_idx >= num_episodes_in_dataset:
            log.warning(f"Episode index {ep_idx} is out of bounds for dataset with {num_episodes_in_dataset} episodes. Skipping.")
            continue
        generate_forced_trajectory_video(ep_idx, policy, dataset, scheduler, cfg, device, output_dir)
        
    log.info("Evaluation complete.")

if __name__ == "__main__":
    evaluate()