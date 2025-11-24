# FILE: scripts/evaluate_receding_horizon.py
import logging
import sys
import csv
import cv2
import hydra
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from omegaconf import DictConfig

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver 

logging.basicConfig(level=logging.INFO, format="%(asctime)s [RHC-EVAL] %(message)s")
log = logging.getLogger("RHCEval")

class EMASmoother:
    """Applies Exponential Moving Average to actions."""
    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha
        self.last_action = None
        
    def smooth(self, current_action: np.ndarray) -> np.ndarray:
        if self.last_action is None:
            self.last_action = current_action
            return current_action
        # Standard EMA: alpha * NEW + (1-alpha) * OLD
        smoothed = self.alpha * current_action + (1.0 - self.alpha) * self.last_action
        self.last_action = smoothed
        return smoothed
        
    def reset(self):
        self.last_action = None

class RHC_Evaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

        # 1. Load Model
        log.info(f"Loading Model: {self.cfg.checkpoint_path}")
        self.pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            self.cfg.checkpoint_path, map_location=self.device
        )
        self.model = self.pl_module.model.eval().to(self.device)
        
        # 2. Init Env
        # Fallback to default scaling if not in config
        scaling = cfg.env.get("action_scaling_factor", 0.5)
        xml_path = self.cfg.get("xml_path", "envs/panda_pick_place.xml")
        
        self.env = PandaEnv(
            xml_path=xml_path, 
            control_mode='delta', 
            render_mode="rgb_array", 
            action_scaling_factor=scaling
        )
        
        # 3. Init Solver & Sync
        self.ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS 
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt 
        
        log.info(f"Sync: dt={self.effective_dt:.4f}, max_dq={self.max_dq:.2f}, scale={scaling}")

        # 4. Vision Transform (With Safety Toggle)
        # IMPORTANT: Check if your training used Normalization. 
        # If unsure, set 'use_normalization: False' in config first.
        tf_steps = [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
        ]
        if self.cfg.get("use_normalization", False):
            log.info("Applying ImageNet Normalization [-1, 1]")
            tf_steps.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        else:
            log.info("Using Raw Tensor Normalization [0, 1]")
            
        self.transform = transforms.Compose(tf_steps)
        
        self.action_smoother = EMASmoother(alpha=self.cfg.get("ema_alpha", 0.8))
        self.prev_image_buffer = None

    def _process_image(self, img_array):
        pil_img = Image.fromarray(img_array)
        return self.transform(pil_img).unsqueeze(0).to(self.device)

    def run(self):
        csv_file = self.output_dir / "rhc_eval_stats.csv"
        
        # Video Writer Setup
        dummy_obs, _ = self.env.reset()
        h, w, _ = self.env.render().shape
        video_path = self.output_dir / "eval_replay.mp4"
        video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

        f_csv = open(csv_file, 'w', newline='')
        writer = csv.writer(f_csv)
        writer.writerow(["episode", "step", "success", "goal_dist_m", "grip_logit"])

        success_count = 0

        try:
            for ep_idx in tqdm(range(self.cfg.num_episodes), desc="Phase 2 RHC Eval"):
                seed = self.cfg.seed + ep_idx
                self.env.reset(seed=seed)
                obs = self.env.get_expert_obs()
                
                # Buffer Init (Warmup History)
                # We treat t=0 history as a copy of t=0 current (Zero velocity start)
                curr_img_tensor = self._process_image(obs['image_primary'])
                self.prev_image_buffer = curr_img_tensor.clone()
                self.action_smoother.reset()
                
                if 'goal_image' in obs and obs['goal_image'] is not None:
                    goal_tensor = self._process_image(obs['goal_image'])
                else:
                    log.error("Goal image missing!")
                    break
                
                episode_success = False
                
                for step in range(self.cfg.max_steps):
                    # 1. Prepare Batch
                    curr_img_tensor = self._process_image(obs['image_primary'])
                    proprio_tensor = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
                    
                    batch = {
                        'prev_image': self.prev_image_buffer,
                        'curr_image': curr_img_tensor,
                        'goal_image': goal_tensor,
                        'curr_proprio': proprio_tensor
                    }
                    self.prev_image_buffer = curr_img_tensor.clone()

                    # 2. Inference
                    with torch.no_grad():
                        out = self.model(batch)
                    
                    # 3. RHC Selection
                    # We take Index 0. If jittery, try changing to [0, 1]
                    chunk_pose = out['pose_chunk'].cpu().numpy()[0] 
                    chunk_grip = out['gripper_chunk'].cpu().numpy()[0]
                    
                    target_pose_world = chunk_pose[0]
                    target_grip_logit = chunk_grip[0].item()
                    
                    # 4. Smoothing & Action
                    target_pose_world = self.action_smoother.smooth(target_pose_world)
                    grip_action = -1.0 if target_grip_logit > 0.0 else 1.0

                    # 5. Execution
                    try:
                        delta_joints = self.ik_solver.compute_delta_action(
                            target_ee_pose=target_pose_world,
                            model=self.env.model,
                            data=self.env.data,
                            ee_site_id=self.env.ee_site_id,
                            joint_qpos_indices=np.arange(7),
                            effective_dt=self.effective_dt,
                            max_dq=self.max_dq
                        )
                    except:
                        delta_joints = np.zeros(7)

                    action = np.concatenate([delta_joints, [grip_action]])
                    obs, _, terminated, truncated, _ = self.env.step(action)

                    # 6. Vis & Log
                    dist = np.linalg.norm(obs['object_pos_world'] - obs['goal_pos_world'])
                    
                    # Success: Object near goal AND lifted (Z > 0.45)
                    # Note: Table is usually 0.40, Cube top is 0.42/0.44. 
                    if dist < 0.05 and obs['object_pos_world'][2] > 0.45:
                        episode_success = True

                    # Render
                    frame = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                    cv2.putText(frame, f"Grip: {target_grip_logit:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    video_writer.write(frame)
                    
                    writer.writerow([ep_idx, step, int(episode_success), f"{dist:.4f}", f"{target_grip_logit:.4f}"])

                    if episode_success or terminated or truncated:
                        break
                
                if episode_success: success_count += 1
                log.info(f"Ep {ep_idx}: {'SUCCESS' if episode_success else 'FAIL'}")

        finally:
            video_writer.release()
            f_csv.close()
            self.env.close()
            log.info(f"Phase 2 Success Rate: {(success_count/self.cfg.num_episodes)*100:.1f}%")

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate_rhc_config")
def main(cfg: DictConfig):
    evaluator = RHC_Evaluator(cfg)
    evaluator.run()

if __name__ == "__main__":
    main()