# FILE: evaluate_hierarchical_planner.py
# (v3.0 - Robust SOTA Evaluator with Auto-Config)

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
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from utils.ik_solver import IKSolver
from models.semantic_planner import SemanticPlanner, SemanticPlannerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Semantic-Eval] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Semantic_Eval")

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
            self.smooth_pose = (self.alpha_pos * raw_pose) + ((1 - self.alpha_pos) * self.smooth_pose)
            self.smooth_grip_logit = (self.alpha_grip * raw_logit) + ((1 - self.alpha_grip) * self.smooth_grip_logit)
        
        # Hysteresis for Gripper Stability
        if not self.gripper_closed and self.smooth_grip_logit > 0.5:
            self.gripper_closed = True
        elif self.gripper_closed and self.smooth_grip_logit < -0.5:
            self.gripper_closed = False
            
        gripper_cmd = -1.0 if self.gripper_closed else 1.0
        return self.smooth_pose, gripper_cmd

# --- 2. EVALUATOR CLASS ---
class HierarchicalEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log.info(f"Loading Checkpoint: {self.cfg.checkpoint_path}")
        
        # Load Checkpoint
        checkpoint = torch.load(
            self.cfg.checkpoint_path, 
            map_location=self.device, 
            weights_only=False 
        )
        state_dict = checkpoint.get('state_dict', checkpoint)

        # [FIX 1] Auto-Detect Layer Count from Weights
        # This prevents the "4 vs 6 layer" lobotomy.
        max_layer_idx = 0
        for k in state_dict.keys():
            if "fusion_transformer.layers." in k:
                # Extract layer index: ...layers.5...
                try:
                    parts = k.split("fusion_transformer.layers.")[1].split(".")
                    idx = int(parts[0])
                    max_layer_idx = max(max_layer_idx, idx)
                except:
                    pass
        
        detected_layers = max_layer_idx + 1
        log.info(f"Auto-detected Fusion Transformer Layers: {detected_layers}")

        # 1. Construct Configuration
        model_cfg = SemanticPlannerConfig(
            proprio_dim=22,
            vision_backbone_model=self.cfg.get("vision_backbone", "google/siglip-base-patch16-224"),
            vision_feature_dim=768,
            fusion_transformer_layers=detected_layers, # [FIX] Use detected count
            num_task_phases=5, 
            phase_dropout_prob=0.0
        )

        # 2. Initialize Model
        self.model = SemanticPlanner(model_cfg).to(self.device)
        
        # 3. Surgical State Dict Loading (Robust)
        new_state_dict = {}
        for k, v in state_dict.items():
            # Strip PL prefix
            if k.startswith("model."):
                k = k[6:]
            
            # Filter buffers
            if "grip_pos_weight" in k: continue
                
            # [FIX] Robust Backbone Remapping
            # Handle both "vision_backbone.vision_model..." and "vision_backbone.vision_embeddings..."
            if k.startswith("vision_backbone."):
                if "vision_head" in k: continue # Skip unused heads
                
                # If checkpoint lacks 'vision_model' sub-module (common in wrappers), insert it
                if "vision_embeddings." in k:
                    k = k.replace("vision_embeddings.", "vision_model.embeddings.")
                if "vision_encoder." in k:
                    k = k.replace("vision_encoder.", "vision_model.encoder.")
                if "vision_post_layernorm." in k:
                    k = k.replace("vision_post_layernorm.", "vision_model.post_layernorm.")

            new_state_dict[k] = v

        # Load weights
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        
        # Critical Check
        critical_missing = [k for k in missing if "fusion_transformer" in k or "traj_head" in k]
        if critical_missing:
            log.error(f"CRITICAL MISSING KEYS: {critical_missing}")
            raise RuntimeError("Model weights incomplete! Checkpoint mismatch.")
        
        if len(unexpected) > 0:
            log.info(f"Unexpected keys (usually safe): {unexpected[:5]} ...")

        self.model.eval()
        
        # 4. Initialize Env & IK
        self.env = PandaEnv(
            xml_path=self.cfg.get("xml_path", "envs/panda_pick_place.xml"),
            control_mode='delta',
            render_mode="rgb_array",
            action_scaling_factor=0.5 
        )
        self.ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")

        # Physics Synchronization
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS 
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.smoother = LightweightSmoother()

    def run(self):
        out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        video_file = out_dir / "eval_video.mp4"
        csv_file = out_dir / "semantic_telemetry.csv"
        
        csv_f = open(csv_file, 'w', newline='')
        writer = csv.writer(csv_f)
        writer.writerow([
            "episode", "step", "intent_input", 
            "grip_logit", "ee_z", "target_z", 
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
                
                # Buffers
                curr_img_pil = Image.fromarray(obs['image_primary'])
                goal_img_pil = Image.fromarray(obs['goal_image'])
                goal_tensor = self.transform(goal_img_pil).unsqueeze(0).to(self.device)
                
                episode_success = False
                heuristic_intent = 0
                
                # [FIX 2] Grasp Latch Timer
                # Prevents flickering "Grasp" intent which confuses the model
                grasp_intent_frames = 0 
                
                for step in range(self.cfg.max_steps):
                    # --- 1. PERCEPTION ---
                    curr_img_pil = Image.fromarray(obs['image_primary'])
                    curr_tensor = self.transform(curr_img_pil).unsqueeze(0).to(self.device)
                    proprio = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
                    
                    # --- 2. HEURISTIC MANAGER ---
                    ee_pos = obs['ee_pose_world']
                    obj_pos = obs['object_pos_world']
                    dist_xy = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
                    is_grasped = obs['is_grasped'][0] > 0.5
                    
                    # State Machine
                    if is_grasped:
                        heuristic_intent = 2 # TRANSPORT
                        grasp_intent_frames = 0
                    # [FIX 3] Tightened Approach Threshold (0.05 -> 0.03)
                    # Ensures robot is truly aligned before triggering grasp phase
                    elif dist_xy < 0.03 and ee_pos[2] < 0.46:
                        heuristic_intent = 1 # GRASP
                        grasp_intent_frames += 1
                    else:
                        heuristic_intent = 0 # APPROACH
                        grasp_intent_frames = 0
                    
                    # Construct Batch
                    batch = {
                        'initial_image': curr_tensor, 
                        'goal_image': goal_tensor,
                        'current_proprio': proprio,
                        'task_phase': torch.tensor([heuristic_intent], device=self.device).long()
                    }
                    
                    # --- 3. INFERENCE ---
                    with torch.no_grad():
                        out = self.model(batch)

                    # Unpack
                    raw_target_pose = out['pose'].cpu().numpy()[0]
                    raw_grip_logit = out['gripper_logit'].item()

                    # --- 4. SAFETY ---
                    raw_target_pose[2] = max(raw_target_pose[2], 0.405)
                    
                    # Smoothing
                    target_pose, gripper_cmd = self.smoother.update(raw_target_pose, raw_grip_logit)
                    
                    # [FIX 4] Grasp Assist
                    # If Heuristic says GRASP and we've been trying for a while, force close
                    if heuristic_intent == 1 and grasp_intent_frames > 10:
                         # Encourage closing if model is hesitant
                         if raw_grip_logit > -2.0: gripper_cmd = -1.0
                    
                    # Transport Lock
                    if heuristic_intent == 2: 
                        gripper_cmd = -1.0
                    
                    # --- 5. EXECUTION ---
                    try:
                        delta_joints = self.ik_solver.compute_delta_action(
                            target_ee_pose=target_pose,
                            model=self.env.model,
                            data=self.env.data,
                            ee_site_id=self.env.ee_site_id,
                            joint_qpos_indices=np.arange(7),
                            effective_dt=self.effective_dt,
                            max_dq=self.max_dq
                        )
                    except Exception:
                        delta_joints = np.zeros(7)
                        
                    action = np.concatenate([delta_joints, [gripper_cmd]])
                    obs, _, terminated, truncated, _ = self.env.step(action)
                    
                    # --- 6. SUCCESS CHECK ---
                    goal_pos = obs['goal_pos_world']
                    obj_pos_now = obs['object_pos_world']
                    dist_goal = np.linalg.norm(obj_pos_now - goal_pos)
                    
                    if dist_goal < 0.05: 
                        episode_success = True

                    # --- 7. LOGGING ---
                    frame = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                    intent_map = {0: "APPROACH", 1: "GRASP", 2: "TRANSPORT"}
                    color_map = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0)}
                    hud_text = f"PHASE: {intent_map.get(heuristic_intent, 'UNK')}"
                    cv2.putText(frame, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_map.get(heuristic_intent, (255,255,255)), 2)
                    video_writer.write(frame)
                    
                    writer.writerow([
                        ep_idx, step, heuristic_intent, 
                        f"{raw_grip_logit:.4f}", f"{ee_pos[2]:.4f}", f"{target_pose[2]:.4f}", 
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

@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_semantic_planner_config")
def main(cfg: DictConfig):
    evaluator = HierarchicalEvaluator(cfg)
    evaluator.run()

if __name__ == "__main__":
    main()