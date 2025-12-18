
import numpy as np
import mujoco
from PIL import Image
import os
import argparse

from envs.panda_env import PandaEnv
from utils.ik_solver import IKSolver
from utils.scripted_expert import ScriptedExpert, ObjectProfile
from scipy.spatial.transform import Rotation as R

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="goal_verification.png")
    args = parser.parse_args()

    # 1. Initialize Env
    env = PandaEnv(render_mode="rgb_array", control_mode="delta")
    obs, info = env.reset(seed=args.seed)
    
    # 2. Render Initial State
    initial_img = obs["image_primary"]
    
    # 3. Render Goal Image (Correct Logic)
    ik_solver = IKSolver(
        urdf_path="urdf/panda_mujoco_kinematics.urdf"
    )
    
    # --- GOAL RENDER LOGIC ---
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    saved_ctrl = env.data.ctrl.copy()
    
    HOVER_HEIGHT = 0.10
    
    # Instantiate temporary expert to use its alignment logic
    dummy_expert = ScriptedExpert(ObjectProfile(size=np.zeros(3), grasp_width_normalized=0.0))

    try:
        goal_pos_world = obs['goal_pos_world']
        goal_orn_world = obs['goal_orn_world']

        # A. Move object to goal POSE (Position + Orientation)
        # FIX: Lift object by half-height (0.02) to prevent sinking into table (Z=0.401).
        obj_addr = env.model.jnt_qposadr[env.object_joint_id]
        target_obj_pos = goal_pos_world + np.array([0.0, 0.0, 0.02])
        env.data.qpos[obj_addr:obj_addr+3] = target_obj_pos
        
        # Set Orientation: Convert xyzw (SciPy) -> wxyz (MuJoCo)
        goal_orn_wxyz = env._scipy_xyzw_to_mujoco_wxyz(goal_orn_world)
        env.data.qpos[obj_addr+3:obj_addr+7] = goal_orn_wxyz
        
        # B. Calculate Robot Target Pose (Goal Pos + Hover Z)
        # FIX: Also lift robot by half-height (0.02) so it hovers relative to 
        # the object's center, not the table surface.
        target_pos = goal_pos_world + np.array([0.0, 0.0, HOVER_HEIGHT + 0.02])
        
        # C. Calculate DYNAMIC Target Orientation
        # The expert aligns with the goal object. We calculate this alignment relative
        # to the goal orientation we just retrieved.
        # FIX: Expert uses [1, 0, 0, 0] (Rot X 180) as base. [0, 1, 0, 0] causes twisted arm.
        seed_downward_quat = np.array([1.0, 0.0, 0.0, 0.0])
        target_quat = dummy_expert._calculate_aligned_orientation(goal_orn_world, seed_downward_quat)
        
        target_pose_7d = np.concatenate([target_pos, target_quat])
        
        # D. Solve IK for Hover Pose
        # Seed with home position
        home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        
        goal_qpos = ik_solver.solve_ik_static(
            target_pose=target_pose_7d, 
            model=env.model, 
            data=env.data, 
            ee_site_id=env.ee_site_id,
            q0=home_qpos 
        )
        
        if goal_qpos is None:
            print("Warning: IK failed for goal image.")
            env.data.qpos[:7] = home_qpos # Fallback (shouldn't happen)
        else:
            env.data.qpos[:7] = goal_qpos

        # Open Gripper (Retract state has open gripper)
        env.data.qpos[7:9] = 0.04 
        
        # Forward Prop
        mujoco.mj_forward(env.model, env.data)
        
        # Render
        goal_img = env.render()
        
    finally:
        # Restore State (Optional, script ends anyway)
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        env.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)

    # 4. Save Side-by-Side
    combined = np.concatenate([initial_img, goal_img], axis=1)
    Image.fromarray(combined).save(args.save_path)
    print(f"Saved verification image to {args.save_path}")

if __name__ == "__main__":
    main()
