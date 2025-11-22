# FILE: virtual_goal_verification.py
# (Definitive Goal Image Verification & Generation Logic)

import logging
import os
import sys
import cv2
import mujoco
import numpy as np
from contextlib import contextmanager
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from utils.ik_solver import IKSolver

# --- Configuration ---
XML_PATH = "envs/panda_pick_place.xml"
URDF_PATH = "urdf/panda_mujoco_kinematics.urdf"
OUTPUT_DIR = Path("virtual_goal_verification")
OUTPUT_DIR.mkdir(exist_ok=True)

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("GOAL_VERIFY")


def _mujoco_to_scipy(quat_wxyz: np.ndarray) -> np.ndarray:
    """Helper: Convert MuJoCo WXYZ -> Scipy XYZW."""
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

def calculate_retract_joints(env: PandaEnv, ik_solver: IKSolver, target_pos_world: np.ndarray, object_quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Calculates joint angles for the robot to hover above the goal,
    ALIGNED with the object's orientation (Expert behavior).
    """
    # 1. Define Ideal Position (Hover)
    hover_z = 0.55 
    target_pos = np.array([target_pos_world[0], target_pos_world[1], hover_z])
    
    # 2. Define Ideal Orientation (Aligned with Cube)
    # The Expert aligns the gripper faces with the cube faces.
    # Object Frame: Z is Up.
    # Gripper Frame: Z is Approach (Down).
    # To match faces: We take Object Rotation and rotate 180 deg around X-axis (flip upside down).
    q_obj = _mujoco_to_scipy(object_quat_wxyz)
    r_obj = R.from_quat(q_obj)
    
    # Apply 180 flip to point gripper down while keeping Yaw alignment
    r_target = r_obj * R.from_euler('x', 180, degrees=True)
    target_matrix = r_target.as_matrix()

    # 3. Transform World -> Robot Base Frame
    base_pos, base_quat_xyzw = env.get_base_pose()
    
    # Create transformation matrices
    R_base_world = R.from_quat(base_quat_xyzw).as_matrix()
    T_base_world = np.eye(4)
    T_base_world[:3, :3] = R_base_world
    T_base_world[:3, 3] = base_pos
    
    # Invert to get World -> Base
    T_world_base = np.linalg.inv(T_base_world)
    
    # Transform Target Position
    target_pos_homo = np.append(target_pos, 1.0)
    target_pos_in_base = (T_world_base @ target_pos_homo)[:3]
    
    # Transform Target Orientation
    target_rot_in_base = T_world_base[:3, :3] @ target_matrix

    # 4. Solve Inverse Kinematics
    current_joints = env.data.qpos[:7].copy()
    initial_guess = [0.0] * len(ik_solver.chain.links)
    for i, val in enumerate(current_joints):
        if i < len(ik_solver._active_idx):
            initial_guess[ik_solver._active_idx[i]] = val

    full_joints = ik_solver.chain.inverse_kinematics(
        target_position=target_pos_in_base,
        target_orientation=target_rot_in_base,
        orientation_mode="all",
        initial_position=initial_guess
    )
    
    final_joints = np.array([full_joints[i] for i in ik_solver._active_idx])
    low, high = env.get_action_space_limits()
    return np.clip(final_joints, low[:7], high[:7])

@contextmanager
def render_perfect_goal(env: PandaEnv, ik_solver: IKSolver, goal_pos_world: np.ndarray):
    """
    Context manager that teleports BOTH the object AND the robot
    to a mathematically perfect 'Task Complete' state.
    """
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    saved_ctrl = env.data.ctrl.copy()

    try:
        # --- A. Teleport Object ---
        obj_addr = env.model.jnt_qposadr[env.object_joint_id]
        current_obj_quat = env.data.qpos[obj_addr+3 : obj_addr+7].copy()
        
        # Perfect placement on table (Z ~ 0.42)
        perfect_obj_pos = goal_pos_world.copy()
        if perfect_obj_pos[2] < 0.41: 
            perfect_obj_pos[2] = 0.42 
            
        env.data.qpos[obj_addr : obj_addr+3] = perfect_obj_pos
        env.data.qpos[obj_addr+3 : obj_addr+7] = current_obj_quat 

        # --- B. Teleport Robot (Aligned with Object) ---
        # Pass the object's orientation to the calculator
        target_joints = calculate_retract_joints(env, ik_solver, perfect_obj_pos, current_obj_quat)
        
        env.data.qpos[:7] = target_joints
        
        # --- C. Set Gripper to Open ---
        env.data.qpos[7] = 0.04
        env.data.qpos[8] = 0.04

        # --- D. Stabilize ---
        env.data.qvel[:] = 0.0
        env.data.ctrl[:7] = target_joints
        env.data.ctrl[7] = 0.04
        
        mujoco.mj_forward(env.model, env.data)
        
        yield

    finally:
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        env.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)

def main():
    logger.info(f"--- Verifying SOTA Virtual Goals ---")
    
    # 1. Init Components
    env = PandaEnv(xml_path=XML_PATH, control_mode='delta', render_mode='rgb_array')
    ik_solver = IKSolver(urdf_path=URDF_PATH)
    
    logger.info(f"Environment and IK Solver loaded.")

    # 2. Run Tests
    for seed in range(1000, 1005):
        logger.info(f"Generating Goal for Seed {seed}...")
        
        # Reset
        env.reset(seed=seed)
        obs = env.get_expert_obs()
        
        # Capture Start (Current State)
        img_start = env.render(camera_name="fixed_camera")
        img_start = cv2.cvtColor(img_start, cv2.COLOR_RGB2BGR)
        
        # Capture Goal (The New Logic)
        goal_pos = obs['goal_pos_world']
        
        with render_perfect_goal(env, ik_solver, goal_pos):
            img_goal = env.render(camera_name="fixed_camera")
            
        img_goal = cv2.cvtColor(img_goal, cv2.COLOR_RGB2BGR)

        # Annotate
        cv2.putText(img_start, f"Seed {seed}: Start", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(img_goal, f"Seed {seed}: SOTA Goal", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

        # Stitch
        combined = np.hstack([img_start, img_goal])
        
        # Save
        outfile = OUTPUT_DIR / f"goal_check_{seed}.png"
        cv2.imwrite(str(outfile), combined)
        logger.info(f"Saved: {outfile}")
    
    logger.info(f"Verification complete. Check folder: {OUTPUT_DIR.absolute()}")
    env.close()

if __name__ == "__main__":
    main()