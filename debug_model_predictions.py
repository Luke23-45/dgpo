#!/usr/bin/env python3
"""
Quick diagnostic script to understand what the BC model is predicting.

This will help confirm if the model is predicting:
A) Absolute world-frame target poses (CORRECT behavior)
B) Poses close to current EE pose (hovering - WRONG behavior)
C) Something else

Run: python debug_model_predictions.py
"""

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from envs.panda_env import PandaEnv
from train.train_semantic_planner import SemanticPlannerLightningModule
import mujoco

def main():
    # Config
    BC_CHECKPOINT = "checkpoints/bc_backup_epoch_088.ckpt"  # Adjust path as needed
    
    # Check if checkpoint exists locally
    for path in [
        BC_CHECKPOINT,
        "/content/drive/MyDrive/pda/bc/bc_backup_epoch_088.ckpt", 
    ]:
        if Path(path).exists():
            BC_CHECKPOINT = path
            break
    
    if not Path(BC_CHECKPOINT).exists():
        print(f"ERROR: Cannot find checkpoint at {BC_CHECKPOINT}")
        print("Please provide correct path to BC checkpoint")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {BC_CHECKPOINT}")
    
    # Load model
    pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
        BC_CHECKPOINT, map_location=device
    )
    model = pl_module.model.eval().to(device)
    
    # Image transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Initialize environment
    env = PandaEnv(xml_path="envs/panda_pick_place.xml", control_mode="delta")
    
    # Run multiple episodes with different seeds
    for seed in [42, 123, 456]:
        print(f"\n{'='*60}")
        print(f"Episode with seed {seed}")
        print('='*60)
        
        obs, _ = env.reset(seed=seed)
        
        # Get ground truth positions
        ee_site_id = env.ee_site_id
        ee_pos_actual = env.data.site_xpos[ee_site_id].copy()
        
        obj_jnt_adr = env.model.jnt_qposadr[env.object_joint_id]
        obj_pos = env.data.qpos[obj_jnt_adr : obj_jnt_adr + 3].copy()
        
        goal_pos = env.get_goal_pos_expert()
        
        print(f"\nGround Truth Positions:")
        print(f"  Current EE:  [{ee_pos_actual[0]:.3f}, {ee_pos_actual[1]:.3f}, {ee_pos_actual[2]:.3f}]")
        print(f"  Object Pos:  [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
        print(f"  Goal Pos:    [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}]")
        
        # Prepare images
        curr_img = obs['image_primary']
        
        # Render goal image (object at goal)
        saved_qpos = env.data.qpos.copy()
        saved_qvel = env.data.qvel.copy()
        
        home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        env.data.qpos[:7] = home_qpos
        env.data.qpos[7:9] = 0.04
        env.data.qpos[obj_jnt_adr : obj_jnt_adr + 3] = goal_pos
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        goal_img = env.render()
        
        # Restore state
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        mujoco.mj_forward(env.model, env.data)
        
        # Prepare model input
        curr_tensor = transform(Image.fromarray(curr_img)).unsqueeze(0).to(device)
        prev_tensor = curr_tensor.clone()  # At t=0, prev=curr
        goal_tensor = transform(Image.fromarray(goal_img)).unsqueeze(0).to(device)
        
        # Get proprio
        joint_qpos = env.data.qpos[:7].copy()
        joint_qvel = env.data.qvel[:7].copy()
        left_touch = env.data.sensordata[env.left_touch_sensor_id]
        right_touch = env.data.sensordata[env.right_touch_sensor_id]
        left_force_adr = env.model.sensor_adr[env.left_force_sensor_id]
        right_force_adr = env.model.sensor_adr[env.right_force_sensor_id]
        left_force = env.data.sensordata[left_force_adr : left_force_adr + 3]
        right_force = env.data.sensordata[right_force_adr : right_force_adr + 3]
        
        proprio = np.concatenate([
            joint_qpos, joint_qvel,
            np.array([left_touch, right_touch]),
            left_force, right_force
        ]).astype(np.float32)
        proprio_tensor = torch.from_numpy(proprio).float().unsqueeze(0).to(device)
        
        batch = {
            'prev_image': prev_tensor,
            'curr_image': curr_tensor,
            'goal_image': goal_tensor,
            'curr_proprio': proprio_tensor
        }
        
        # Model inference
        with torch.no_grad():
            outputs = model(batch)
        
        # Extract predictions
        pose_chunk = outputs['pose_chunk'][0].cpu().numpy()  # (K, 7)
        grip_chunk = outputs['gripper_chunk'][0].cpu().numpy()  # (K, 1)
        phase_logits = outputs['phase_logits'][0].cpu().numpy()
        
        print(f"\nModel Predictions (first step of chunk):")
        pred_pos = pose_chunk[0, :3]
        pred_quat = pose_chunk[0, 3:]
        grip = grip_chunk[0, 0]
        phase = np.argmax(phase_logits)
        
        print(f"  Predicted Pos:     [{pred_pos[0]:.3f}, {pred_pos[1]:.3f}, {pred_pos[2]:.3f}]")
        print(f"  Predicted Quat:    [{pred_quat[0]:.3f}, {pred_quat[1]:.3f}, {pred_quat[2]:.3f}, {pred_quat[3]:.3f}]")
        print(f"  Gripper Logit:     {grip:.2f} ({'CLOSE' if grip > 0 else 'OPEN'})")
        print(f"  Phase Prediction:  {phase}")
        
        # CRITICAL: Calculate how close prediction is to various reference points
        dist_to_current_ee = np.linalg.norm(pred_pos - ee_pos_actual)
        dist_to_object = np.linalg.norm(pred_pos - obj_pos)
        dist_to_goal = np.linalg.norm(pred_pos - goal_pos)
        
        print(f"\n⭐ Prediction Distances:")
        print(f"  Distance to Current EE: {dist_to_current_ee:.4f}m")
        print(f"  Distance to Object:     {dist_to_object:.4f}m")
        print(f"  Distance to Goal:       {dist_to_goal:.4f}m")
        
        # Interpretation
        print(f"\n📊 Analysis:")
        if dist_to_current_ee < 0.02:
            print(f"  ❌ PROBLEM: Prediction is VERY close to current EE ({dist_to_current_ee:.4f}m away)")
            print(f"     This means the model learned to predict 'where the EE is' not 'where to go'")
        elif dist_to_object < 0.05:
            print(f"  ✓ GOOD: Prediction is near the object ({dist_to_object:.4f}m away)")
        elif dist_to_goal < 0.05:
            print(f"  ✓ GOOD: Prediction is near the goal ({dist_to_goal:.4f}m away)")
        else:
            print(f"  ⚠ UNCLEAR: Prediction is not close to any key location")
        
        # Also show the full chunk to see if there's motion
        print(f"\nAction Chunk (10 steps) - Position only:")
        for i in range(min(10, len(pose_chunk))):
            pos = pose_chunk[i, :3]
            print(f"  Step {i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        # Check if chunk shows movement
        pos_deltas = np.diff(pose_chunk[:, :3], axis=0)
        max_delta = np.max(np.abs(pos_deltas))
        print(f"\nMax position delta within chunk: {max_delta:.4f}m")
        if max_delta < 0.005:
            print(f"  ❌ PROBLEM: Chunk shows almost NO movement - model is predicting static poses")
        else:
            print(f"  ✓ OK: Chunk shows some movement")
    
    env.close()
    print(f"\n{'='*60}")
    print("DIAGNOSTIC COMPLETE")
    print('='*60)


if __name__ == "__main__":
    main()
