# FILE: scripts/evaluate_policy.py
# (State-of-the-Art, Self-Contained Policy Evaluation Script)

import argparse
import logging
import random
import time
from pathlib import Path
from collections import deque
from typing import Dict, Any, Tuple
from tqdm import tqdm 
from typing import Deque
import numpy as np
import torch
import cv2
from omegaconf import OmegaConf, DictConfig
import os
# --- Project Imports (ensure these paths are correct) ---
from envs.panda_env import PandaEnv
from models.diffusion_policy import DiffusionPolicy, NoiseSchedulerConfig
from utils.rl_reward_wrapper import AdvancedRewardWrapper, AdvancedRewardConfig

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- SOTA Helper: Observation History Buffer (copied for self-containment) ---
class ObsHistoryBuffer:
    def __init__(self, history_len: int, obs_space):
        self.history_len = history_len
        self.keys = list(obs_space.keys())
        self.buffers: Dict[str, Deque[np.ndarray]] = {
            key: deque(maxlen=history_len) for key in self.keys
        }
    def reset(self, obs: Dict[str, np.ndarray]):
        for key in self.keys:
            self.buffers[key].clear()
            # Pre-fill with the first observation
            for _ in range(self.history_len):
                self.buffers[key].append(obs[key])
    def append(self, obs: Dict[str, np.ndarray]):
        for key in self.keys:
            self.buffers[key].append(obs[key])
    def get_stacked(self) -> Dict[str, np.ndarray]:
        return {key: np.stack(self.buffers[key], axis=0) for key in self.keys}


def load_policy_and_config(ckpt_path: Path, device: torch.device) -> Tuple[DiffusionPolicy, DictConfig]:
    """
    Loads a policy and its configuration from a checkpoint file.
    Robustly handles both pre-training and RL fine-tuning checkpoints.
    """
    log.info(f"Loading checkpoint from: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # --- 1. Extract Configuration ---
    # The config key is different between pre-training ('config') and RL ('config_dict')
    if 'config' in payload:
        train_cfg_dict = payload['config']
    elif 'config_dict' in payload:
        train_cfg_dict = payload['config_dict']
    else:
        raise KeyError("Checkpoint is missing 'config' or 'config_dict' payload.")
    
    # Create a clean OmegaConf object
    train_cfg = OmegaConf.create(train_cfg_dict)
    
    # --- 2. Reconstruct Model Architecture ---
    model_cfg = train_cfg.model
    scheduler_cfg = train_cfg.scheduler
    
    # We need proprio_dim. Let's create a temporary env to get it.
    temp_env = PandaEnv(xml_path=train_cfg.environment.xml_path)
    proprio_dim = temp_env.proprio_dim
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()

    policy = DiffusionPolicy(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        H_o=model_cfg.observation_horizon,
        H_a=model_cfg.action_horizon,
        image_feat_dim=model_cfg.image_feat_dim,
        scheduler_cfg=NoiseSchedulerConfig(**scheduler_cfg),
        d_model=model_cfg.d_model,
        denoiser_layers=model_cfg.denoiser_layers,
        denoiser_heads=model_cfg.denoiser_heads,
        device=device
    )

    # --- 3. Load Weights ---
    # The weights key is different between pre-training ('policy_state_dict') and RL ('actor_state_dict')
    if 'policy_state_dict' in payload:
        weights = payload['policy_state_dict']
    elif 'actor_state_dict' in payload:
        log.info("RL checkpoint detected. Loading weights from 'actor_state_dict'.")
        # We need to strip the 'diffusion_policy.' prefix from the RL actor's state dict
        rl_weights = payload['actor_state_dict']
        weights = {k.replace('diffusion_policy.', ''): v for k, v in rl_weights.items()}
    else:
        raise KeyError("Checkpoint is missing 'policy_state_dict' or 'actor_state_dict'.")

    policy.load_state_dict(weights)
    
    # Load EMA weights if they exist (common in pre-training)
    if 'ema_state_dict' in payload and payload['ema_state_dict'] is not None:
        log.info("Loading EMA weights into the model for evaluation.")
        policy.ema.load_state_dict(payload['ema_state_dict'])
        # Copy EMA weights to the main model for inference
        policy.load_state_dict(policy.ema.ema_model.state_dict())

    policy.to(device)
    policy.eval()
    log.info("Policy successfully reconstructed and loaded.")
    
    return policy, train_cfg

def set_seed(seed: int):
    """Sets the seed for all relevant random number generators for reproducibility."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    log.info(f"Global seed set to {seed}")
def run_evaluation(cfg: DictConfig):
    """Main evaluation function."""
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    
    # --- 1. Load Policy and its Training Config ---
    try:
        policy, train_cfg = load_policy_and_config(Path(cfg.ckpt_path), device)
    except Exception as e:
        log.critical(f"Failed to load policy. Error: {e}", exc_info=True)
        return

    # --- 2. Setup Evaluation Environment ---
    log.info("Setting up evaluation environment...")
    env = PandaEnv(
        xml_path=cfg.environment.xml_path,
        render_mode="rgb_array" # We need pixels for video
    )
    # Apply reward wrapper to get success metric, but we won't use the reward itself
    env = AdvancedRewardWrapper(env, reward_cfg=AdvancedRewardConfig())
    
    # --- 3. Initialize History Buffer ---
    try:
        observation_horizon = train_cfg.model.observation_horizon
        log.info(f"Using observation horizon from loaded config: {observation_horizon}")
    except Exception:
        log.error("Could not find 'model.observation_horizon' in the loaded checkpoint config. Aborting.")
        return

    history = ObsHistoryBuffer(observation_horizon, env.observation_space)
    
    # --- 4. Setup Video Recording ---
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"evaluation_rollouts.mp4"
    
    first_obs, _ = env.reset(seed=cfg.seed)
    H, W, _ = first_obs['image_primary'].shape
    video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 20, (W, H))

    # --- 5. Run Rollouts ---
    all_successes = []
    all_ep_lengths = []
    
    for i in range(cfg.eval.n_rollouts):
        log.info(f"--- Starting Rollout {i+1}/{cfg.eval.n_rollouts} ---")
        
        # Reset env and history buffer for the new episode
        current_seed = cfg.seed + i
        obs, info = env.reset(seed=current_seed)
        history.reset(obs)
        
        ep_length = 0
        ep_success = False
        
        for step in tqdm(range(cfg.eval.max_rollout_steps), desc=f"Rollout {i+1}"):
            # Prepare observation history for the policy
            obs_history_np = history.get_stacked()
            obs_history_torch = {
                k: torch.from_numpy(v).unsqueeze(0).to(device).float()
                for k, v in obs_history_np.items()
            }
            
            # Get action from the policy
            with torch.inference_mode():
                action_chunk = policy.sample(
                    obs_history_torch,
                    guidance_scale=cfg.eval.guidance_scale,
                    steps=cfg.eval.sampling_steps,
                    use_ema=True # Assume EMA weights are already loaded
                )
            
            # Take the first action from the predicted sequence
            action = action_chunk[0, 0].cpu().numpy()

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update history and record frame
            history.append(obs)
            frame_rgb = env.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Annotate frame
            success_text = f"SUCCESS: {info.get('is_success', False)}"
            text_color = (0, 255, 0) if info.get('is_success') else (255, 255, 255)
            cv2.putText(frame_bgr, success_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            cv2.putText(frame_bgr, f"Step: {step}", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            video_writer.write(frame_bgr)
            
            ep_length += 1
            if done:
                ep_success = info.get('is_success', False)
                break
        
        all_successes.append(ep_success)
        all_ep_lengths.append(ep_length)
        log.info(f"Rollout {i+1} finished. Success: {ep_success}, Length: {ep_length}")
    
    video_writer.release()
    env.close()

    # --- 6. Print Final Report ---
    success_rate = np.mean(all_successes) if all_successes else 0.0
    avg_ep_length = np.mean(all_ep_lengths) if all_ep_lengths else 0.0
    
    print("\n" + "="*50)
    log.info("--- 📊 Evaluation Report 📊 ---")
    print("="*50)
    log.info(f"Checkpoint Path: {cfg.ckpt_path}")
    log.info(f"Total Rollouts: {cfg.eval.n_rollouts}")
    log.info(f"Success Rate: {success_rate:.2%}")
    log.info(f"Average Episode Length: {avg_ep_length:.1f} steps")
    log.info(f"Videos of all rollouts saved to: {video_path}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Diffusion Policy.")
    parser.add_argument("ckpt_path", type=str, help="Path to the model checkpoint (.pth) file.")
    parser.add_argument("--output-dir", type=str, default="videos/evaluation", help="Directory to save evaluation videos.")
    parser.add_argument("--n-rollouts", type=int, default=2, help="Number of evaluation episodes to run.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for evaluation.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (e.g., 'cpu' or 'cuda').")
    # Add other parameters or link to a Hydra config
    args = parser.parse_args()

    # Create a minimal OmegaConf object from argparse for compatibility
    cfg = OmegaConf.create({
        "ckpt_path": args.ckpt_path,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "device": args.device,
        "eval": {
            "n_rollouts": args.n_rollouts,
            "max_rollout_steps": 500,
            "sampling_steps": 50,
            "guidance_scale": 1.5,
        },
        "environment": {
            "xml_path": "envs/panda_pick_place.xml"
        }
    })
    
    run_evaluation(cfg)


if __name__ == "__main__":
    main()

def ty():
    """
    python -m evaluate.evaluate_policy "C:\Users\Hellx\Documents\Programming\python\Project\dgpo\outputs\rl_finetune\my_first_rl_run\2025-10-28_23-41-13\checkpoints\interrupted_checkpoint.pth"
    """
    pass