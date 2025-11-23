# FILE: evaluate_semantic_planner.py
# (Optimized for v8.0 Disentangled Architecture)

import logging
import sys
import cv2
import hydra
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from omegaconf import DictConfig
from torchvision import transforms
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver

log = logging.getLogger("AWSP_Eval")
logging.basicConfig(level=logging.INFO)

# 1. STATE ESTIMATOR (The Oracle for v8.0)
class StateEstimator:
    def __init__(self):
        self.current_phase = 0
        self.prev_is_grasped = False
    def reset(self):
        self.current_phase = 0
        self.prev_is_grasped = False
    def update(self, obs):
        # Simple Hysteresis Machine
        is_grasped = obs['is_grasped'][0] > 0.5
        dist_obj = np.linalg.norm(obs['ee_pose_world'][:3] - obs['object_pos_world'])
        dist_goal = np.linalg.norm(obs['object_pos_world'] - obs['goal_pos_world'])

        if is_grasped:
            self.prev_is_grasped = True
            self.current_phase = 3 if dist_goal < 0.05 else 2 # 3=Place, 2=Transport
        else:
            if self.prev_is_grasped:
                self.current_phase = 4 if dist_goal < 0.05 else 0 # 4=Done, 0=Retry
                self.prev_is_grasped = False
            else:
                # 1=Grasp if close, 0=Reach otherwise
                self.current_phase = 1 if dist_obj < 0.12 else 0
        return self.current_phase

# 2. TRAJECTORY SMOOTHER (Essential for v8.0 stability)
class TrajectorySmoother:
    def __init__(self):
        self.pose = None
        self.grip = 0.0
    def reset(self):
        self.pose = None
        self.grip = 0.0
    def update(self, raw_pose, raw_grip):
        if self.pose is None:
            self.pose = raw_pose
            self.grip = raw_grip
        else:
            # Heavy smoothing on Pose (0.6), Light on Gripper (0.3)
            self.pose = 0.6 * raw_pose + 0.4 * self.pose
            self.grip = 0.3 * raw_grip + 0.7 * self.grip
        return self.pose, self.grip

# 3. EVALUATOR
class AWSPEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model
        log.info(f"Loading v8.0 Model: {self.cfg.checkpoint_path}")
        pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            self.cfg.checkpoint_path, map_location=self.device
        )
        self.model = pl_module.model.eval().to(self.device)
        
        # Env & IK
        xml = self.cfg.get("xml_path", 'envs/panda_pick_place.xml')
        self.env = PandaEnv(xml_path=xml, control_mode='delta', render_mode="rgb_array")
        self.ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")
        
        # PHYSICS FIX: Dynamic Speed Limit
        sim_steps = 20
        self.dt = self.env.model.opt.timestep * sim_steps
        self.max_dq = (self.env.ACTION_SCALING_FACTOR / self.dt) * 2.0
        
        # VISION FIX: Normalization
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        
        self.oracle = StateEstimator()
        self.smoother = TrajectorySmoother()

    def run(self):
        out = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "eval.mp4"
        video = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        
        successes = []
        for ep in tqdm(range(self.cfg.num_episodes)):
            self.env.reset(seed=self.cfg.seed + ep)
            obs = self.env.get_expert_obs()
            self.oracle.reset()
            self.smoother.reset()
            
            # Virtual Goal
            goal_img = self.tf(Image.fromarray(self._get_goal_render(obs))).unsqueeze(0).to(self.device)
            
            done = False
            for step in range(self.cfg.max_steps):
                # 1. Get Phase (The Cheat Code)
                phase = self.oracle.update(obs)
                
                # 2. Prepare Batch (v8.0 Style)
                img = self.tf(Image.fromarray(obs['image_primary'])).unsqueeze(0).to(self.device)
                prop = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
                
                batch = {
                    'initial_image': img,
                    'goal_image': goal_tensor if 'goal_tensor' in locals() else goal_img,
                    'task_phase': torch.tensor([phase], device=self.device),
                    'current_proprio': prop
                }
                
                # 3. Inference
                with torch.no_grad():
                    res = self.model(batch)
                
                # 4. Smooth & Control
                target, grip_logit = self.smoother.update(res['pose'].cpu().numpy()[0], res['gripper_logit'].item())
                grip_cmd = -1.0 if grip_logit > 2.0 else 1.0 # Hysteresis threshold
                
                try:
                    d_joints = self.ik_solver.compute_delta_action(
                        target_ee_pose=target,
                        model=self.env.model, data=self.env.data, ee_site_id=self.env.ee_site_id,
                        joint_qpos_indices=np.arange(7), effective_dt=self.dt, max_dq=self.max_dq
                    )
                except: d_joints = np.zeros(7)
                
                obs, _, _, _, _ = self.env.step(np.concatenate([d_joints, [grip_cmd]]))
                
                # 5. Success Check
                if np.linalg.norm(obs['object_pos_world'] - obs['goal_pos_world']) < 0.05 and obs['is_grasped'][0] > 0.5:
                    done = True
                    
                # Render
                img = self.env.render()
                if video.isOpened():
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.putText(img, f"Phase: {phase}", (10,30), 0, 0.6, (0,255,0), 2)
                    video.write(img)
                
                if done: break
            successes.append(done)
        
        video.release()
        self.env.close()
        print(f"Success Rate: {sum(successes)/len(successes)*100:.1f}%")

    def _get_goal_render(self, obs):
        # Quick dirty goal render
        bak = (self.env.data.qpos.copy(), self.env.data.qvel.copy(), self.env.data.ctrl.copy())
        try:
            oa = self.env.model.jnt_qposadr[self.env.object_joint_id]
            gp = obs['goal_pos_world'].copy(); gp[2] = 0.42
            self.env.data.qpos[oa:oa+3] = gp
            self.env.data.qpos[:7] = [0, -0.78, 0, -2.35, 0, 1.57, 0.78] # Stash robot
            mujoco.mj_forward(self.env.model, self.env.data)
            return self.env.render()
        finally:
            self.env.data.qpos[:], self.env.data.qvel[:], self.env.data.ctrl[:] = bak
            mujoco.mj_forward(self.env.model, self.env.data)

@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_semantic_planner_config")
def main(cfg): AWSPEvaluator(cfg).run()

if __name__ == "__main__": main()