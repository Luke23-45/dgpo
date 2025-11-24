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

# Project Imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SMART-EVAL] %(message)s")
log = logging.getLogger("SmartEval")

class TemporalEnsembler:
    """
    Stabilizes model predictions by averaging overlapping chunks.
    This prevents 'Grasp Blinking'.
    """
    def __init__(self, chunk_size=10):
        self.chunk_size = chunk_size
        # Buffer stores lists of predictions for future timesteps
        # Index 0 is current step, Index 1 is next step, etc.
        self.pose_buffer = [[] for _ in range(chunk_size)]
        self.grip_buffer = [[] for _ in range(chunk_size)]

    def update(self, new_pose_chunk, new_grip_chunk):
        """
        Args:
            new_pose_chunk: (K, 7)
            new_grip_chunk: (K, 1)
        """
        # Add new predictions to the buffer
        for t in range(self.chunk_size):
            if t < len(new_pose_chunk):
                self.pose_buffer[t].append(new_pose_chunk[t])
                self.grip_buffer[t].append(new_grip_chunk[t])

    def get_action(self):
        """
        Returns the averaged action for the CURRENT timestep.
        Then shifts the buffer for the next timestep.
        """
        # 1. Average predictions for the current timestep (index 0)
        current_poses = self.pose_buffer[0]
        current_grips = self.grip_buffer[0]
        
        if not current_poses:
            return None, None # Buffer empty

        # Average Position
        avg_pose = np.mean(np.stack(current_poses), axis=0)
        # Normalize quaternion part
        avg_pose[3:] /= np.linalg.norm(avg_pose[3:])
        
        # Average Gripper Logit
        avg_grip = np.mean(np.stack(current_grips))

        # 2. Shift Buffer (Time moves forward)
        # Index 1 becomes Index 0, etc.
        self.pose_buffer.pop(0)
        self.grip_buffer.pop(0)
        
        # Add empty list at the end for the new furthest horizon
        self.pose_buffer.append([])
        self.grip_buffer.append([])

        return avg_pose, avg_grip

class SmartEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        log.info(f"Loading Model: {self.cfg.checkpoint_path}")
        self.pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            self.cfg.checkpoint_path, map_location=self.device
        )
        self.model = self.pl_module.model.eval().to(self.device)
        
        # Use the chunk size the model was trained with
        self.chunk_size = self.pl_module.cfg.model.get("chunk_size", 10)
        
        # Init Env & Solver
        xml_path = self.cfg.get("xml_path", "envs/panda_pick_place.xml")
        self.env = PandaEnv(xml_path=xml_path, control_mode='delta', render_mode="rgb_array", action_scaling_factor=0.5)
        self.ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")
        
        # Sync Controller
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS 
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.ensembler = TemporalEnsembler(self.chunk_size)
        self.prev_image_buffer = None

    def _process_image(self, img_array):
        pil_img = Image.fromarray(img_array)
        return self.transform(pil_img).unsqueeze(0).to(self.device)

    def run(self):
        out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        video_file = out_dir / self.cfg.output_video_path
        csv_file = out_dir / "smart_eval_stats.csv"
        
        # Setup Logging
        obs, _ = self.env.reset()
        h, w, _ = self.env.render().shape
        video_writer = cv2.VideoWriter(str(video_file), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        f_csv = open(csv_file, 'w', newline='')
        writer = csv.writer(f_csv)
        writer.writerow(["episode", "step", "success", "grip_logit", "dist_to_goal"])

        success_count = 0

        try:
            for ep_idx in tqdm(range(self.cfg.num_episodes), desc="Smart Evaluation"):
                seed = self.cfg.seed + ep_idx
                self.env.reset(seed=seed)
                obs = self.env.get_expert_obs()
                
                # Reset Buffers
                curr_img_tensor = self._process_image(obs['image_primary'])
                self.prev_image_buffer = curr_img_tensor.clone()
                self.ensembler = TemporalEnsembler(self.chunk_size) # Clear ensemble buffer
                
                # Goal
                if 'goal_image' in obs and obs['goal_image'] is not None:
                    goal_tensor = self._process_image(obs['goal_image'])
                else:
                    goal_tensor = torch.zeros_like(curr_img_tensor)

                episode_success = False
                
                # Warmup: We need to fill the ensembler buffer before moving
                # We run inference K times to populate the first set of actions? 
                # Actually, we can just start and suffer latency, or duplicate the first chunk.
                # Strategy: Run inference once, fill buffer with shifted versions.
                
                for step in range(self.cfg.max_steps):
                    curr_img_tensor = self._process_image(obs['image_primary'])
                    proprio_tensor = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
                    
                    batch = {
                        'prev_image': self.prev_image_buffer,
                        'curr_image': curr_img_tensor,
                        'goal_image': goal_tensor,
                        'curr_proprio': proprio_tensor
                    }
                    self.prev_image_buffer = curr_img_tensor.clone()

                    with torch.no_grad():
                        out = self.model(batch)
                    
                    pose_chunk = out['pose_chunk'].cpu().numpy()[0] 
                    grip_chunk = out['gripper_chunk'].cpu().numpy()[0]
                    
                    # Update Ensembler
                    self.ensembler.update(pose_chunk, grip_chunk)
                    
                    # Get Averaged Action
                    target_pose, target_grip_logit = self.ensembler.get_action()
                    
                    # Hysteresis for Gripper Action (Simple Debounce)
                    # Prevents fluttering around 0.0
                    gripper_action = -1.0 if target_grip_logit > 0.0 else 1.0

                    # IK & Step
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

                    action = np.concatenate([delta_joints, [gripper_action]])
                    obs, _, terminated, truncated, _ = self.env.step(action)

                    # Success Check
                    dist = np.linalg.norm(obs['object_pos_world'] - obs['goal_pos_world'])
                    if dist < 0.05 and obs['object_pos_world'][2] > 0.41:
                        episode_success = True

                    # Render
                    frame = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                    cv2.putText(frame, f"Grip: {target_grip_logit:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    video_writer.write(frame)
                    writer.writerow([ep_idx, step, int(episode_success), f"{target_grip_logit:.4f}", f"{dist:.4f}"])

                    if episode_success or terminated or truncated:
                        break
                
                if episode_success: success_count += 1
                log.info(f"Episode {ep_idx}: {'SUCCESS' if episode_success else 'FAIL'}")

        finally:
            video_writer.release()
            f_csv.close()
            self.env.close()
            log.info(f"Smart Model Success Rate: {(success_count/self.cfg.num_episodes)*100:.1f}%")

@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_semantic_planner_config")
def main(cfg):
    evaluator = SmartEvaluator(cfg)
    evaluator.run()

if __name__ == "__main__":
    main()