import os
import sys
import torch
import hydra
import logging
import numpy as np
import cv2
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(os.getcwd())

from train.train_dgpo_robust import DGPOTrainer

# Setup logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("CLOSED_LOOP_DIAG")

@hydra.main(config_path="./configs", config_name="train_dgpo_config", version_base="1.2")
def main(cfg):
    print("\n==================================================")
    print(" 🎬 CLOSED-LOOP FAILURE ANALYSIS TOOL")
    print("==================================================\n")

    # 1. Force Closed Loop (Alpha = 1.0)
    cfg.num_envs = 1
    cfg.training.use_policy_blending = True 
    
    # 2. Checkpoint Logic
    ckpt_path = cfg.checkpoint.resume_from
    if ckpt_path is None or str(ckpt_path).lower() == "null" or str(ckpt_path).lower() == "none":
        print("ℹ️ Using Initialized Policy (BC Baseline).")
    elif not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}. Using BC Baseline.")
    else:
        print(f"Loading: {ckpt_path}")

    # 3. Initialize Trainer
    try:
        trainer = DGPOTrainer(cfg)
        if ckpt_path and os.path.exists(ckpt_path):
            trainer._load_checkpoint(ckpt_path)
    except Exception as e:
        print(f"Failed to init: {e}")
        return

    # 4. Run Evaluation Episode
    env = trainer.envs
    obs, info = env.reset()
    
    debug_dir = Path("debug_output/closed_loop")
    debug_dir.mkdir(parents=True, exist_ok=True)
    video_path = debug_dir / "failure_mode.mp4"
    
    print(f"\nRecording to: {video_path}")
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        20, (256, 128) # stacked prev/curr or similar
    )
    
    # Force Policy Control
    trainer.blending_schedule.get_alpha = lambda x: 1.0 # 100% Autonomy
    
    done = False
    step = 0
    divergences = []
    
    # Reset internal state
    if hasattr(trainer, 'temporal_ensembles') and len(trainer.temporal_ensembles) > 0:
        trainer.temporal_ensembles[0].reset()
    prev_img = obs['image_primary'] # (1,3,128,128)
    
    print("\n[STARTING EPISODE]...")
    while not done and step < 200:
        # Get Action
        # Prepare inputs
        with torch.no_grad():
            # Convert to tensor
            curr_img_t = torch.from_numpy(obs['image_primary']).float().to(trainer.device) / 255.0
            prev_img_t = torch.from_numpy(prev_img).float().to(trainer.device) / 255.0
            goal_img_t = torch.from_numpy(info['goal_img']).float().to(trainer.device) / 255.0
            goal_img_t = torch.from_numpy(info['goal_img']).float().to(trainer.device) / 255.0
            
            # Handle Proprio Key Variance
            if 'proprioception' in obs:
                proprio_t = torch.from_numpy(obs['proprioception']).float().to(trainer.device)
            elif 'proprio_state' in obs:
                proprio_t = torch.from_numpy(obs['proprio_state']).float().to(trainer.device)
            elif 'proprio' in obs:
                proprio_t = torch.from_numpy(obs['proprio']).float().to(trainer.device)
            else:
                print(f"❌ KEYS FOUND: {list(obs.keys())}")
                raise KeyError("Could not find proprioception key")
            
            # Forward
            batch_input = {
                "prev_image": prev_img_t,
                "curr_image": curr_img_t,
                "goal_image": goal_img_t,
                "curr_proprio": proprio_t
            }
            dist = trainer.policy(batch_input)
            raw_action = dist.mode() # Deterministic for debugging
            
            # Unnormalize ?? Trainer does raw output. Assuming model outputs normalized action.
            # But we need environment executable action.
            # Usually environment wrapper handles unnormalization if configured.
            # Let's assume raw_action is executable for now or check Trainer logic.
            # Trainer logic: action = raw_action (if no scaler).
            
            action = raw_action.cpu().numpy()
            
            # Ensembling
            # trainer.temporal_ensemble.add(action) # simplified
            # smoothed_action = trainer.temporal_ensemble.get_smoothed_action()
            
        # Step Env
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        
        # Calculate Divergence from Expert (Oracle)
        # Expert is in info['expert_pose']? Info is array for VectorEnv.
        # VectorEnv info is a tuple/dict structure.
        
        # Access expert from Wrapper via hidden attribute if possible, or assume info has it.
        # Dict info: info['final_info'] usually has it? Or vector env merges keys.
        expert_action = next_info.get('expert_pose', np.zeros_like(action))
        
        # Simple Euclidean Diff (Position Only: first 3 coords)
        # Action is 7d or 8d.
        # action shape (1, 8). expert shape (1, 7).
        pol_pos = action[0, :3]
        exp_pos = expert_action[0, :3]
        
        div = np.linalg.norm(pol_pos - exp_pos)
        divergences.append(div)
        
        # Log
        print(f"Step {step:03d} | Div: {div*100:.2f} cm | Reward: {reward[0]:.4f}")
        
        # Visualization
        # Stack images
        vis_img = np.hstack([obs['image_primary'][0], info['goal_img'][0]])
        vis_img = np.transpose(vis_img, (1, 2, 0)) # CHW -> HWC
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        writer.write(vis_img)
        
        prev_img = obs['image_primary']
        obs = next_obs
        info = next_info
        step += 1
        
    writer.release()
    env.close()
    
    avg_div = np.mean(divergences)
    print("\n--------------------------------------------------")
    print(f"🏁 EPISODE FINISHED")
    print(f"   Avg Execution Divergence: {avg_div*100:.2f} cm")
    print(f"   Max Execution Divergence: {np.max(divergences)*100:.2f} cm")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
