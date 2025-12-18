
import numpy as np
import mujoco
from PIL import Image
import os
import argparse
import logging
from scipy.spatial.transform import Rotation as R

from envs.panda_env import PandaEnv
from utils.ik_solver import IKSolver
from utils.scripted_expert import ScriptedExpert, ObjectProfile, ExpertConfig
from utils.expert_dataset import ExpertDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugGoal")

def render_goal_image_static(env, ik_solver, obs):
    """
    My current static implementation.
    """
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    saved_ctrl = env.data.ctrl.copy()
    
    HOVER_HEIGHT = 0.10
    
    # Instantiate temporary expert to use its alignment logic
    dummy_expert = ScriptedExpert(ObjectProfile(size=np.zeros(3), grasp_width_normalized=0.0))

    try:
        goal_pos_world = obs['goal_pos_world']
        goal_orn_world = obs['goal_orn_world']

        # 1. Move object to goal POSE
        obj_addr = env.model.jnt_qposadr[env.object_joint_id]
        env.data.qpos[obj_addr:obj_addr+3] = goal_pos_world
        
        # Set Orientation
        goal_orn_wxyz = env._scipy_xyzw_to_mujoco_wxyz(goal_orn_world)
        env.data.qpos[obj_addr+3:obj_addr+7] = goal_orn_wxyz
        
        # 2. Calculate Robot Target Pose
        target_pos = goal_pos_world + np.array([0.0, 0.0, HOVER_HEIGHT])
        
        # 3. Dynamic Orientation
        # EXPERT FINDING: The expert reaches [1,0,0,0] (Rot X 180) naturally.
        # [0,1,0,0] (Rot Y 180) causes twisted arm (X-axis flipped).
        seed_downward_quat = np.array([1.0, 0.0, 0.0, 0.0])
        target_quat = dummy_expert._calculate_aligned_orientation(goal_orn_world, seed_downward_quat)
        
        target_pose_7d = np.concatenate([target_pos, target_quat])
        
        # 4. Solve IK
        # Use a "Natural Table Hover" seed instead of Tucked Home.
        # Derived from Expert Ground Truth: [0, 0, 0, -1.5, 0, 1.5, 0.785]
        # This prevents "Twisted" IK solutions.
        natural_hover_seed = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.785])
        
        goal_qpos = ik_solver.solve_ik_static(
            target_pose=target_pose_7d, 
            model=env.model, 
            data=env.data, 
            ee_site_id=env.ee_site_id,
            q0=natural_hover_seed 
        )
        
        if goal_qpos is not None:
            env.data.qpos[:7] = goal_qpos
        
        # Open gripper
        env.data.qpos[7:9] = 0.04
        
        mujoco.mj_forward(env.model, env.data)
        img = env.render()
        
        return img, env.data.qpos[:7].copy(), target_quat

    finally:
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        env.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)


def run_expert_dynamic(seed):
    """
    Run the full expert trajectory using ExpertDataset to get the REAL goal state.
    """
    cfg = ExpertConfig()
    
    # Initialize Dataset Generator
    ds = ExpertDataset(
        urdf_path="urdf/panda_mujoco_kinematics.urdf",
        env_xml_path="envs/panda_pick_place.xml",
        base_seed=seed,
        max_episodes_per_epoch=1,
        scripted_cfg=cfg,
        object_size=(0.04, 0.04, 0.04),
        object_grasp_width=0.8, # normalized
        action_scaling_factor=1.0, # Match env? Default is 0.5 in generate_dataset
        yield_full_obs=True
    )
    
    # Generate 1 episode
    print("Running ExpertDataset generation...")
    for _ in ds:
        pass # trigger generation
        
    if not ds.episodes:
        raise RuntimeError("ExpertDataset failed to produce an episode.")
        
    ep = ds.episodes[0]
    last_obs = ep['obs_list'][-1]
    
    # Extract Data
    img = last_obs['image_primary'] # Assuming this key exists in obs_list
    
    # Wait, ExpertDataset might store compressed images or raw.
    # Check if 'image_primary' is in obs_list.
    # Assuming standard return.
    
    # We also need qpos. Does obs_list have it?
    # Usually obs contains 'joint_pos' or similar.
    # Looking at ExpertDataset code: full_obs is stored.
    # 'proprio' is usually stored.
    # But we can recover qpos from the 'expert_target_pose' or just look at 'proprio'.
    
    # Let's verify what keys are available.
    # For debug, we trust 'proprio' which is usually joint pos + gripper + ...
    # Or we can look at the 'expert_states' in ep.
    
    # Actually, we can get the RAW qpos if we modify ExpertDataset to save it, 
    # OR we just rely on the image for visual comparison.
    # 'proprio' [7 joints, gripper(?)]
    
    qpos = last_obs['proprio'][:7]
    
    return img, qpos, last_obs.get('expert_target_pose', None)


def main():
    env = PandaEnv(render_mode="rgb_array", control_mode="delta")
    ik_solver_static = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")
    
    SEED = 42
    
    # 1. Dynamic (Ground Truth)
    print("Generating Dynamic Expert Goal...")
    dynamic_img, dynamic_qpos, dynamic_target = run_expert_dynamic(SEED)
    
    # 2. Static (My Logic)
    # BE CAREFUL: run_expert_dynamic runs an Env internally.
    # We need to reset OUR env to the SAME seed to get the SAME goal pose.
    print("Generating Static Goal...")
    env.reset(seed=SEED)
    obs = env.get_expert_obs()
    
    # IMPORTANT: Ensure the goal pose is identical.
    # ExpertDataset(base_seed=SEED) initializes its internal env with seed=SEED.
    # So the *first* episode should match env.reset(seed=SEED).
    
    static_img, static_qpos, static_target_quat = render_goal_image_static(env, ik_solver_static, obs)
    
    # 3. Compare
    print("--- Comparison ---")
    print(f"Static QPos:  {np.round(static_qpos, 3)}")
    print(f"Dynamic QPos: {np.round(dynamic_qpos, 3)}")
    print(f"Diff QPos:    {np.linalg.norm(static_qpos - dynamic_qpos):.4f}")
    
    # Extract Dynamic Quat (from expert_target_pose if available, else from proprio/FK? Proprio only has qpos usually)
    # Actually, dynamic_target (3rd return) is what we extracted.
    if dynamic_target is not None:
        dyn_quat = dynamic_target[3:]
        print(f"Static Quat:  {np.round(static_target_quat, 3)}")
        print(f"Dynamic Quat: {np.round(dyn_quat, 3)}")
        
        # Angle diff
        q1 = R.from_quat(static_target_quat)
        q2 = R.from_quat(dyn_quat)
        diff_q = q1 * q2.inv()
        angle = np.linalg.norm(diff_q.as_rotvec())
        print(f"Quat Diff Angle (rad): {angle:.4f}")
    else:
        print("Dynamic Target Pose not found in obs.")
    
    # Save Side-by-Side
    combined = np.concatenate([static_img, dynamic_img], axis=1)
    Image.fromarray(combined).save("debug_goal_comparison.png")
    print("Saved comparison to debug_goal_comparison.png (Left: Static, Right: Dynamic)")

if __name__ == "__main__":
    main()
