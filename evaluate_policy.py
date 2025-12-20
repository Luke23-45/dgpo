import gymnasium as gym
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from envs.panda_env import PandaEnv
from envs.dgpo_env_wrapper import make_dgpo_env
from models.vision_transformer import VisionBackbone
from models.actor import Actor
# Import DGPO Policy if available/separate, otherwise assume Actor structure matches

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.ckpt_path = Path(checkpoint_path)
        
        # Load Config (Placeholder - ideally load from yaml)
        # For now, we assume a standard dict or load from the checkpoint if stored
        self.cfg = self._load_config(config_path)
        
        # Initialize Environment
        self.env = PandaEnv(
            render_mode="rgb_array",
            control_mode="delta",
            action_scaling_factor=0.022 # Match Training
        )
        
        # Initialize Model
        self.model = self._load_model()
        self.model.eval()
        
    def _load_config(self, path: str):
        # TODO: Load actual YAML
        return {} 
        
    def _load_model(self):
        log.info(f"Loading checkpoint from {self.ckpt_path}...")
        payload = torch.load(self.ckpt_path, map_location=self.device)
        
        # Detect if this is a BC checkpoint (state_dict only) or DGPO (dict with optimizer etc)
        if "policy_state_dict" in payload:
            state_dict = payload["policy_state_dict"]
            model_type = "DGPO"
        elif "state_dict" in payload:
             state_dict = payload["state_dict"]
             model_type = "BC/AWR"
        else:
            # Assume raw state dict
            state_dict = payload
            model_type = "Raw"
            
        log.info(f"Detected Checkpoint Type: {model_type}")
        
        # Initialize Architecture (Must match Training)
        # This is hardcoded for now based on known params. Ideally configured.
        backbone = VisionBackbone(
            backbone_name="resnet18", # Configurable?
            feature_dim=768
        )
        
        model = Actor(
            backbone=backbone,
            proprio_dim=22, # 7+7+2+6
            action_dim=8,   # 7 joint + 1 gripper
            chunk_size=10,  # Standard chunk
            hidden_dim=512
        ).to(self.device)
        
        # Handle Key Mismatches (e.g. 'module.' prefix from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") 
            new_state_dict[name] = v
            
        try:
            model.load_state_dict(new_state_dict)
            log.info("Model loaded successfully.")
        except Exception as e:
            log.warning(f"Strict load failed: {e}. Trying non-strict...")
            model.load_state_dict(new_state_dict, strict=False)
            
        return model

    @torch.no_grad()
    def evaluate(self, num_episodes: int = 50):
        log.info(f"Starting Evaluation: {num_episodes} Episodes")
        successes = []
        episode_lengths = []
        
        for i in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            step = 0
            
            # Action Chunking State
            action_queue = []
            
            while not done:
                # 1. Process Observation
                img = self._process_image(obs["image_primary"])
                proprio = torch.from_numpy(obs["proprio"]).float().unsqueeze(0).to(self.device)
                
                # 2. Inference (if queue empty)
                if not action_queue:
                    # Construct dummy goal/prev (Matches training input structure)
                    # For BC/AWR, we typically just need curr_img and proprio
                    # But the ACT architecture might expect specific keys
                    
                    # Assuming standard Policy forward signature: (image, proprio) -> Action Chunk
                    pred_chunk = self.model(img, proprio) # Shape (1, Chunk, Dim)
                    action_queue = pred_chunk.squeeze(0).cpu().numpy().tolist()
                    
                # 3. Execute Action
                if action_queue:
                    action = np.array(action_queue.pop(0))
                else:
                    action = np.zeros(8) # Fallback
                
                # Scaling is handled inside Env if configured, or here?
                # Training used unscaled actions [-1, 1], Env scales them.
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                step += 1
                
                if done:
                    # Check Success (Defined by Env)
                    # PandaEnv usually returns 'success' in info or we check goal distance
                    obj_pos = obs['object_pos_world']
                    goal_pos = obs['goal_pos_world']
                    dist = np.linalg.norm(obj_pos - goal_pos)
                    is_success = dist < 0.05
                    
                    successes.append(is_success)
                    episode_lengths.append(step)
                    log.info(f"Ep {i+1}: {'SUCCESS' if is_success else 'FAIL'} (Len {step})")
                    break
                    
        success_rate = np.mean(successes) * 100
        avg_len = np.mean(episode_lengths)
        
        log.info("="*30)
        log.info(f"RESULTS ({self.ckpt_path.name})")
        log.info(f"Success Rate: {success_rate:.2f}%")
        log.info(f"Avg Length:   {avg_len:.1f}")
        log.info("="*30)
        
        return success_rate, avg_len

    def _process_image(self, img_np):
        # Resize to 224x224, Normalize, ToTensor
        # TODO: Use same transform as training
        import cv2
        img = cv2.resize(img_np, (224, 224))
        img = img.transpose(2, 0, 1) # HWC -> CHW
        img = img / 255.0
        img = (img - 0.5) / 0.5
        return torch.from_numpy(img).float().unsqueeze(0).to(self.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to run")
    parser.add_argument("--config", type=str, default="configs/train_dgpo_config.yaml", help="Path to config")
    args = parser.parse_args()
    
    evaluator = Evaluator(args.config, args.ckpt)
    evaluator.evaluate(args.episodes)
