# FILE: evaluate_hierarchical_planner.py
# (v1.0 - Hierarchical Intent-Conditioned Evaluator)

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import csv
import cv2
import hydra
import mujoco
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# --- Project Imports ---
# Ensure these paths match your project structure
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from utils.ik_solver import IKSolver
# Assuming the model class from asaer.txt is saved in models/hierarchical_planner.py
# You might need to adjust this import based on where you saved the model file.
from models.semantic_planner import HierarchicalPlanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Hierarchical-Eval] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Hierarchical_Eval")

# --- 1. ROBUST SMOOTHER ---
class LightweightSmoother:
    """Exponential Moving Average for smoothing neural network jitter."""
    def __init__(self, alpha_pos=0.7, alpha_grip=0.5):
        self.alpha_pos = alpha_pos
        self.alpha_grip = alpha_grip
        self.smooth_pose = None
        self.smooth_grip_logit = 0.0
        self.gripper_closed = False

    def reset(self):
        self.smooth_pose = None
        self.smooth_grip_logit = 0.0
        self.gripper_closed = False

    def update(self, raw_pose: np.ndarray, raw_logit: float) -> Tuple[np.ndarray, float]:
        if self.smooth_pose is None:
            self.smooth_pose = raw_pose
            self.smooth_grip_logit = raw_logit
        else:
            # Interpolate Position/Rotation
            # Note: For quats, simple lerp is approximation; slerp is better but lerp is faster for small steps
            self.smooth_pose = (self.alpha_pos * raw_pose) + ((1 - self.alpha_pos) * self.smooth_pose)
            self.smooth_grip_logit = (self.alpha_grip * raw_logit) + ((1 - self.alpha_grip) * self.smooth_grip_logit)
        
        # Hysteresis for Gripper Stability (Schmitt Trigger)
        if not self.gripper_closed and self.smooth_grip_logit > 0.5:
            self.gripper_closed = True
        elif self.gripper_closed and self.smooth_grip_logit < -0.5:
            self.gripper_closed = False
            
        # -1.0 is Closed, 1.0 is Open in PandaEnv
        gripper_cmd = -1.0 if self.gripper_closed else 1.0
        return self.smooth_pose, gripper_cmd

# --- 2. EVALUATOR CLASS ---
class HierarchicalEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log.info(f"Loading Checkpoint: {self.cfg.checkpoint_path}")
        
        # Load Model (Handling PL wrapper if necessary, here assuming direct load for clarity)
        # If you used PL, use: ModelClass.load_from_checkpoint(path).model
        checkpoint = torch.load(self.cfg.checkpoint_path, map_location=self.device)
        
        # Reconstruct Config from checkpoint or hydra
        # Assuming 'hyper_parameters' key exists from PL, or constructing manually
        # strict reconstruction of the config based on asaer.txt requirements
        from models.hierarchical_planner import HierarchicalPlannerConfig
        model_cfg = HierarchicalPlannerConfig(
            proprio_dim=22, # Standard Panda Proprio
            vision_feature_dim=768,
            fusion_transformer_layers=6,
            chunk_size=self.cfg.get("chunk_size", 10),
            num_subgoals=self.cfg.get("num_subgoals", 3), # From asaer.txt logic
            use_intent_input=True # CRITICAL: This model uses intent
        )
        
        self.model = HierarchicalPlanner(model_cfg).to(self.device)
        # Load weights logic (depends on how you saved it, usually state_dict)
        if 'state_dict' in checkpoint:
            # Strip 'model.' prefix if coming from PL
            state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        
        # Initialize Env & IK
        self.env = PandaEnv(
            xml_path=self.cfg.get("xml_path", "envs/panda_pick_place.xml"),
            control_mode='delta',
            render_mode="rgb_array",
            action_scaling_factor=0.5 
        )
        self.ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")

        # --- PHYSICS SYNCHRONIZATION (The "Smashing" Fix) ---
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS 
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt
        log.info(f"Physics Sync: max_dq set to {self.max_dq:.4f}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.smoother = LightweightSmoother()
        self.prev_img_buffer = None

    def run(self):
        out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        video_file = out_dir / "eval_video.mp4"
        csv_file = out_dir / "hierarchical_telemetry.csv"
        
        # Extended CSV headers for Hierarchical Analysis
        csv_f = open(csv_file, 'w', newline='')
        writer = csv.writer(csv_f)
        writer.writerow([
            "episode", "step", "intent_input", "phase_pred", 
            "subgoal_deviation", "grip_logit", "ee_z", "target_z", 
            "dist_to_obj", "is_grasped", "success"
        ])
        
        dummy = self.env.render()
        h, w, _ = dummy.shape
        video_writer = cv2.VideoWriter(str(video_file), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        
        success_count = 0

        try:
            for ep_idx in tqdm(range(self.cfg.num_episodes), desc="Evaluating"):
                self.env.reset(seed=self.cfg.seed + ep_idx)
                obs = self.env.get_expert_obs()
                self.smoother.reset()
                
                # Heuristic State Variables
                heuristic_intent = 0 # 0: Approach
                grasp_latch_timer = 0
                
                # Buffers
                curr_img_pil = Image.fromarray(obs['image_primary'])
                goal_img_pil = Image.fromarray(obs['goal_image']) # Assuming goal image provided by env reset
                
                # Pre-process static goal
                goal_tensor = self.transform(goal_img_pil).unsqueeze(0).to(self.device)
                self.prev_img_buffer = self.transform(curr_img_pil).unsqueeze(0).to(self.device)
                
                episode_success = False
                
                for step in range(self.cfg.max_steps):
                    # 1. PERCEPTION
                    curr_img_pil = Image.fromarray(obs['image_primary'])
                    curr_tensor = self.transform(curr_img_pil).unsqueeze(0).to(self.device)
                    proprio = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
                    
                    # 2. HEURISTIC MANAGER (The "Pre-Frontal Cortex")
                    # Determines the High-Level INTENT to feed the Neural Network
                    ee_pos = obs['ee_pose_world']
                    obj_pos = obs['object_pos_world']
                    dist_xy = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
                    is_grasped = obs['is_grasped'][0] > 0.5
                    
                    # State Machine Logic
                    if is_grasped:
                        heuristic_intent = 2 # TRANSPORT (Blue)
                    elif dist_xy < 0.05 and ee_pos[2] < 0.46:
                        heuristic_intent = 1 # GRASP/LATCH (Red)
                        grasp_latch_timer += 1
                    else:
                        heuristic_intent = 0 # APPROACH (Green)
                        grasp_latch_timer = 0
                        
                    # Prepare Batch
                    batch = {
                        'prev_image': self.prev_img_buffer,
                        'curr_image': curr_tensor,
                        'goal_image': goal_tensor,
                        'curr_proprio': proprio,
                        # CRITICAL: Inject Intent
                        'intent': torch.tensor([heuristic_intent], device=self.device).long()
                    }
                    self.prev_img_buffer = curr_tensor.clone()
                    
                    # 3. INFERENCE
                    with torch.no_grad():
                        out = self.model(batch)
                    
                    # Unpack Hierarchical Outputs
                    # Shape: (1, Chunk, 8) -> 7 pose + 1 grip
                    action_chunk = out['action_chunk'].cpu().numpy()[0] 
                    # Shape: (1, Subgoals, 3) -> 3 pos
                    subgoal_chunk = out['subgoal_chunk'].cpu().numpy()[0] 
                    phase_logits = out['phase_logits'].cpu().numpy()[0]
                    
                    # 4. ACTION SELECTION (Receding Horizon)
                    # We look ahead to step 2 or 4 for smoother motion
                    lookahead = min(3, len(action_chunk) - 1)
                    raw_target_pose = action_chunk[lookahead, :7].copy()
                    raw_grip_logit = action_chunk[lookahead, 7].item()
                    
                    # 5. HIERARCHY CHECK (Telemetry)
                    # Calculate distance between low-level action and high-level subgoal
                    # Just for logging to see if the model is "sane"
                    # Taking last subgoal as roughly the target
                    subgoal_deviation = np.linalg.norm(raw_target_pose[:3] - subgoal_chunk[-1])

                    # 6. SAFETY GUARDRAILS (Physics)
                    # Clamp Z to prevent table smashing
                    raw_target_pose[2] = max(raw_target_pose[2], 0.405)
                    
                    # 7. EXECUTION (Smoothing + IK)
                    target_pose, gripper_cmd = self.smoother.update(raw_target_pose, raw_grip_logit)
                    
                    # Override gripper if in Transport Mode (Ratchet Logic)
                    if heuristic_intent == 2: 
                        gripper_cmd = -1.0
                    
                    try:
                        delta_joints = self.ik_solver.compute_delta_action(
                            target_ee_pose=target_pose,
                            model=self.env.model,
                            data=self.env.data,
                            ee_site_id=self.env.ee_site_id,
                            joint_qpos_indices=np.arange(7),
                            effective_dt=self.effective_dt,
                            max_dq=self.max_dq # Synced Physics
                        )
                    except Exception:
                        delta_joints = np.zeros(7)
                        
                    action = np.concatenate([delta_joints, [gripper_cmd]])
                    obs, _, terminated, truncated, _ = self.env.step(action)
                    
                    # 8. SUCCESS CHECK
                    goal_pos = obs['goal_pos_world']
                    obj_pos_now = obs['object_pos_world']
                    dist_goal = np.linalg.norm(obj_pos_now - goal_pos)
                    
                    if dist_goal < 0.05: # Object brought to goal
                        episode_success = True

                    # 9. LOGGING & VISUALIZATION
                    frame = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                    
                    # HUD Overlay
                    intent_map = {0: "APPROACH", 1: "GRASP", 2: "TRANSPORT", 3: "PLACE"}
                    color_map = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0)}
                    
                    hud_text = f"INTENT: {intent_map.get(heuristic_intent, 'UNK')}"
                    hud_color = color_map.get(heuristic_intent, (255,255,255))
                    
                    cv2.putText(frame, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)
                    cv2.putText(frame, f"Subgoal Dev: {subgoal_deviation:.3f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    video_writer.write(frame)
                    
                    writer.writerow([
                        ep_idx, step, heuristic_intent, np.argmax(phase_logits),
                        subgoal_deviation, f"{raw_grip_logit:.4f}", 
                        f"{ee_pos[2]:.4f}", f"{target_pose[2]:.4f}", 
                        f"{dist_xy:.4f}", is_grasped, episode_success
                    ])
                    
                    if episode_success or terminated or truncated:
                        break
                
                if episode_success:
                    success_count += 1
                log.info(f"Episode {ep_idx}: {'SUCCESS' if episode_success else 'FAIL'}")

        finally:
            video_writer.release()
            self.env.close()
            csv_f.close()
            rate = (success_count / self.cfg.num_episodes) * 100
            log.info(f"FINAL EVALUATION RESULT: {rate:.1f}% Success Rate")

@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_hierarchical_config")
def main(cfg: DictConfig):
    evaluator = HierarchicalEvaluator(cfg)
    evaluator.run()

if __name__ == "__main__":
    main()