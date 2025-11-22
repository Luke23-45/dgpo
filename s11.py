# FILE: run_deep_diagnostic.py
# (The "Nuclear Option" for Debugging)

import logging
import sys
import traceback
import shutil
from pathlib import Path
import hydra
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from omegaconf import DictConfig
from torchvision import transforms

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("DEEP_DIAG")

class DeepDiagnosticSuite:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path("diagnostic_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        log.info("\n" + "="*60)
        log.info("   🔬 DEEP DIAGNOSTIC SUITE v2.0 (Forensic Analysis)")
        log.info("   Output Folder: ./diagnostic_outputs")
        log.info("="*60)

    # ==================================================================
    # TEST 1: THE "RGB vs BGR" & NORMALIZATION CHECK
    # Goal: Prove the model isn't looking at "blue faces" or "grey goo".
    # ==================================================================
    def test_1_vision_integrity(self):
        log.info("\n[TEST 1/6] Vision Integrity (RGB/BGR & Normalization)...")
        try:
            env = PandaEnv(xml_path="envs/panda_pick_place.xml", render_mode="rgb_array")
            env.reset()
            
            # 1. Capture Raw Frame from MuJoCo
            raw_img = env.render() # shape (H, W, 3), uint8
            
            # Save RAW to prove MuJoCo is working
            cv2.imwrite(str(self.output_dir / "1_mujoco_raw_bgr.png"), cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR))
            
            # 2. Apply Transform
            # SigLIP / CLIP expects: Resize -> Tensor -> Norm(0.5)
            tf = transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            tensor_img = tf(Image.fromarray(raw_img))
            
            # 3. Statistical Analysis
            # If normalized to [0.5], range should be [-1.0, 1.0]
            min_v, max_v, mean_v = tensor_img.min().item(), tensor_img.max().item(), tensor_img.mean().item()
            
            log.info(f"   > Tensor Stats: Min={min_v:.2f}, Max={max_v:.2f} (Should be approx -1 to 1)")
            
            if min_v > -0.1 or max_v < 0.1:
                log.error("   ❌ FAILURE: Image is not normalized! Range is too small.")
                return False
                
            # 4. Reconstruct Image (The "Human Eye" Check)
            # Un-normalize: x * 0.5 + 0.5
            recon_tensor = (tensor_img * 0.5 + 0.5).clamp(0, 1)
            recon_np = (recon_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Save Reconstruction
            # If this looks blue/orange swapped, your Training Data was wrong.
            cv2.imwrite(str(self.output_dir / "1_tensor_reconstruction.png"), cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR))
            log.info(f"   ✅ SUCCESS: Vision pipeline valid. Check '{self.output_dir}/1_tensor_reconstruction.png' manually for color swaps.")
            env.close()
            return True
        except Exception:
            log.error(traceback.format_exc())
            return False

    # ==================================================================
    # TEST 2: MODEL SENSITIVITY (The "Brain Dead" Check)
    # Goal: Feed 2 different images. If output is identical, weights are dead.
    # ==================================================================
    def test_2_sensitivity(self):
        log.info("\n[TEST 2/6] Model Sensitivity (Brain Check)...")
        try:
            pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
                self.cfg.checkpoint_path, map_location=self.device
            )
            model = pl_module.model.eval().to(self.device)
            
            # Create Input A: Pure Black
            img_a = torch.zeros(1, 3, 224, 224).to(self.device)
            # Create Input B: Pure White
            img_b = torch.ones(1, 3, 224, 224).to(self.device)
            
            # Dummy props
            prop = torch.zeros(1, 22).to(self.device)
            phase = torch.tensor([0]).to(self.device)
            
            # Detect Arch
            is_v9 = hasattr(model, 'traj_head')
            
            # Construct Batch
            if is_v9:
                batch_a = {'prev_image': img_a, 'curr_image': img_a, 'goal_image': img_a, 'curr_proprio': prop}
                batch_b = {'prev_image': img_b, 'curr_image': img_b, 'goal_image': img_b, 'curr_proprio': prop}
            else: # v8
                batch_a = {'initial_image': img_a, 'goal_image': img_a, 'task_phase': phase, 'current_proprio': prop}
                batch_b = {'initial_image': img_b, 'goal_image': img_b, 'task_phase': phase, 'current_proprio': prop}
            
            with torch.no_grad():
                out_a = model(batch_a)
                out_b = model(batch_b)
                
            # Extract Pose
            if is_v9:
                pose_a = out_a['pose_chunk'][0, 0].cpu().numpy()
                pose_b = out_b['pose_chunk'][0, 0].cpu().numpy()
            else:
                pose_a = out_a['pose'][0].cpu().numpy()
                pose_b = out_b['pose'][0].cpu().numpy()
                
            diff = np.linalg.norm(pose_a - pose_b)
            log.info(f"   > Input A (Black) -> Pose: {np.round(pose_a[:3], 3)}")
            log.info(f"   > Input B (White) -> Pose: {np.round(pose_b[:3], 3)}")
            log.info(f"   > Difference: {diff:.6f}")
            
            if diff < 0.001:
                log.error("   ❌ FAILURE: Model output is CONSTANT regardless of input! Weights are likely uninitialized or collapsed.")
                return False
            else:
                log.info("   ✅ SUCCESS: Model reacts to visual changes.")
                return True
        except Exception:
            log.error(traceback.format_exc())
            return False

    # ==================================================================
    # TEST 3: COORDINATE FRAME VALIDATION
    # Goal: Ensure model predicts ABSOLUTE WORLD COORDINATES.
    # ==================================================================
    def test_3_coordinates(self):
        log.info("\n[TEST 3/6] Coordinate System Check...")
        try:
            # Logic: A valid Absolute Pose must be within the robot's workspace
            # Workspace X: [0.2, 0.8], Y: [-0.5, 0.5], Z: [0.0, 0.8]
            
            pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
                self.cfg.checkpoint_path, map_location=self.device
            )
            model = pl_module.model.eval().to(self.device)
            
            # Run on a real image
            env = PandaEnv(xml_path="envs/panda_pick_place.xml", render_mode="rgb_array")
            obs, _ = env.reset()
            tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            img = tf(Image.fromarray(obs['image_primary'])).unsqueeze(0).to(self.device)
            prop = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
            
            # Inference
            is_v9 = hasattr(model, 'traj_head')
            if is_v9:
                batch = {'prev_image': img, 'curr_image': img, 'goal_image': img, 'curr_proprio': prop}
                out = model(batch)
                pred_xyz = out['pose_chunk'][0, 0, :3].detach().cpu().numpy()
            else:
                batch = {'initial_image': img, 'goal_image': img, 'task_phase': torch.tensor([0]).to(self.device), 'current_proprio': prop}
                out = model(batch)
                pred_xyz = out['pose'][0, :3].detach().cpu().numpy()

            log.info(f"   > Predicted XYZ: {pred_xyz}")
            
            # Check Bounds
            if np.abs(pred_xyz[0]) < 0.1 and np.abs(pred_xyz[1]) < 0.1:
                log.warning("   ⚠️ WARNING: Predicted X/Y are very close to 0.")
                log.warning("      This usually means the model is predicting DELTAS (Relative Motion).")
                log.warning("      BUT your evaluator assumes ABSOLUTE coordinates.")
                log.warning("      -> FIX: Switch Evaluator to 'delta_mode' or Retrain model.")
                return False
            
            if pred_xyz[0] > 1.0 or pred_xyz[2] > 1.0:
                 log.error("   ❌ FAILURE: Prediction is outside physical workspace (> 1.0m). Normalization issue likely.")
                 return False
                 
            log.info("   ✅ SUCCESS: Coordinates look like valid Absolute positions.")
            return True
        except Exception:
            log.error(traceback.format_exc())
            return False

    # ==================================================================
    # TEST 4: LATENT SPACE HEALTH (NaN Check)
    # Goal: Catch exploding gradients or bad math.
    # ==================================================================
    def test_4_health(self):
        log.info("\n[TEST 4/6] Numerical Health (NaN Check)...")
        try:
            pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
                self.cfg.checkpoint_path, map_location=self.device
            )
            model = pl_module.model.eval().to(self.device)
            
            # Feed Garbage (Random Noise)
            noise = torch.randn(1, 3, 224, 224).to(self.device) * 10.0
            prop = torch.randn(1, 22).to(self.device)
            
            is_v9 = hasattr(model, 'traj_head')
            if is_v9:
                batch = {'prev_image': noise, 'curr_image': noise, 'goal_image': noise, 'curr_proprio': prop}
                out = model(batch)
                tensor_check = out['pose_chunk']
            else:
                batch = {'initial_image': noise, 'goal_image': noise, 'task_phase': torch.tensor([0]).to(self.device), 'current_proprio': prop}
                out = model(batch)
                tensor_check = out['pose']
                
            if torch.isnan(tensor_check).any() or torch.isinf(tensor_check).any():
                log.error("   ❌ FAILURE: Model produced NaNs/Infs on noise input. It is unstable.")
                return False
                
            log.info("   ✅ SUCCESS: Model is numerically stable even with garbage input.")
            return True
        except Exception:
            log.error(traceback.format_exc())
            return False

    # ==================================================================
    # TEST 5: PHYSICS & IK INTEGRATION
    # Goal: Ensure 100% that we can execute a move.
    # ==================================================================
    def test_5_physics_integration(self):
        log.info("\n[TEST 5/6] Physics & IK Integration...")
        try:
            env = PandaEnv(xml_path="envs/panda_pick_place.xml", render_mode="rgb_array")
            env.reset()
            ik = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")
            
            # Define a Safe Target (Home position: x=0.3, y=0, z=0.5)
            safe_target = np.array([0.3, 0.0, 0.5])
            
            # Configure max_dq correctly
            dt = env.model.opt.timestep * 20
            max_dq = (env.ACTION_SCALING_FACTOR / dt) * 2.0
            
            # Compute
            delta = ik.compute_delta_action(
                target_ee_pose=np.concatenate([safe_target, [0,1,0,0]]),
                model=env.model, data=env.data, ee_site_id=env.ee_site_id,
                joint_qpos_indices=np.arange(7),
                effective_dt=dt,
                max_dq=max_dq
            )
            
            # Apply
            action = np.zeros(8)
            action[:7] = delta
            env.step(action)
            
            # Check Result
            new_ee = env.get_ee_pose()[0][:3]
            dist = np.linalg.norm(new_ee - safe_target)
            
            log.info(f"   > Target: {safe_target}")
            log.info(f"   > Result: {np.round(new_ee, 3)}")
            log.info(f"   > Error:  {dist:.4f} m")
            
            # It won't be perfect in 1 step, but it should move towards it
            if np.linalg.norm(delta) < 0.0001:
                 log.error("   ❌ FAILURE: IK generated zero velocity.")
                 return False
                 
            log.info("   ✅ SUCCESS: Physics stack is fully operational.")
            env.close()
            return True
        except Exception:
            log.error(traceback.format_exc())
            return False

    # ==================================================================
    # TEST 6: ARCHITECTURE VERIFICATION
    # Goal: Print exactly what the user has vs what they think they have.
    # ==================================================================
    def test_6_arch_report(self):
        log.info("\n[TEST 6/6] Final Architecture Report...")
        try:
            pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
                self.cfg.checkpoint_path, map_location=self.device
            )
            model = pl_module.model
            
            is_v8 = hasattr(model, 'pose_head') and hasattr(model, 'task_phase_embedding')
            is_v9 = hasattr(model, 'traj_head') and hasattr(model, 'phase_head')
            
            log.info("   > Analyzing Checkpoint Structure...")
            
            if is_v8:
                log.info("   🟢 IDENTIFIED: v8.0 (Phase Input)")
                log.info("      - Inputs: Image, Goal, Phase, Proprio")
                log.info("      - Output: Single Pose (7D)")
            elif is_v9:
                log.info("   🔵 IDENTIFIED: v9.0 (History Input)")
                log.info("      - Inputs: Prev, Curr, Goal, Proprio")
                log.info("      - Output: Pose Chunk (10x7D)")
            else:
                log.error("   🔴 UNKNOWN ARCHITECTURE! Keys found:")
                log.info(f"      {model.state_dict().keys()}")
                return False
                
            return True
        except Exception:
            log.error(traceback.format_exc())
            return False


    def run_all(self):
        results = [
            self.test_1_vision_integrity(),
            self.test_2_sensitivity(),
            self.test_3_coordinates(),
            self.test_4_health(),
            self.test_5_physics_integration(),
            self.test_6_arch_report()
        ]
        
        log.info("\n" + "="*60)
        log.info("   DIAGNOSTIC SUMMARY")
        log.info("="*60)
        
        if all(results):
            log.info("   🎉 ALL SYSTEMS GREEN. The code is perfect.")
            log.info("   If the robot still fails, the issue is purely TRAINING DATA QUALITY.")
        else:
            log.error("   💀 CRITICAL FAILURES DETECTED. See logs above.")

@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_semantic_planner_config")
def main(cfg: DictConfig):
    suite = DeepDiagnosticSuite(cfg)
    suite.run_all()

if __name__ == "__main__":
    main()