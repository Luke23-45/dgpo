# In utils/oracle_planner.py
import logging
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv
from envs.panda_env import PandaEnv
from utils.scripted_expert import ScriptedExpert, ObjectProfile, ExpertConfig
from utils.ik_solver import IKSolver # <-- IMPORTANT IMPORT

log = logging.getLogger(__name__)

class DeterministicTwinOracle:
    """
    Manages a set of "twin" environments to provide perfect, k-step-ahead subgoals.
    This version correctly uses the IKSolver to generate expert actions.
    """
    def __init__(self, env_fns: list, object_profile: dict, expert_config: dict, k: int, urdf_path: str):
        log.info(f"Initializing DeterministicTwinOracle with k={k} lookahead.")
        self.k = k
        self.n_envs = len(env_fns)
        
        # Create a list of twin environments and experts
        self.twin_envs = [fn() for fn in env_fns]
        
        # Create one IK solver and one expert instance per twin environment
        self.ik_solvers = [IKSolver(urdf_path) for _ in range(self.n_envs)]
        self.experts = [
            ScriptedExpert(ObjectProfile(**object_profile), ExpertConfig(**expert_config))
            for _ in range(self.n_envs)
        ]

        # Cache parameters needed for IK
        # Assumes all envs are the same
        sample_env = self.twin_envs[0]
        N_SUBSTEPS = 20 # Should match your env's setting
        self.effective_dt = sample_env.model.opt.timestep * N_SUBSTEPS
        self.max_dq = sample_env.ACTION_SCALING_FACTOR / self.effective_dt
        self.arm_joint_ids = np.arange(7)

    def sync_and_get_subgoals(self, live_vec_env: VecEnv) -> np.ndarray:
        subgoal_images = []
        for i in range(self.n_envs):
            twin_env = self.twin_envs[i]
            expert = self.experts[i]
            ik_solver = self.ik_solvers[i]

            # 1. Get the full MuJoCo state from the live environment
            live_state = live_vec_env.env_method('get_mj_state', indices=i)[0]
            
            # 2. Set the twin environment to this exact state
            twin_env.set_mj_state(live_state)
            
            # 3. Step the twin forward k times using the expert + IK solver
            for _ in range(self.k):
                expert_obs = twin_env.get_expert_obs()
                target_pose, gripper_action = expert.get_target_pose(expert_obs)
                
                # --- THIS IS THE CORRECT, NON-SIMPLIFIED LOGIC ---
                delta_arm_action = ik_solver.compute_delta_action(
                    target_ee_pose=target_pose,
                    model=twin_env.model,
                    data=twin_env.data,
                    ee_site_id=twin_env.ee_site_id,
                    joint_qpos_indices=self.arm_joint_ids,
                    effective_dt=self.effective_dt,
                    max_dq=self.max_dq
                )
                action = np.concatenate([delta_arm_action, [gripper_action]])
                # --- END OF CORRECT LOGIC ---
                
                _, _, terminated, truncated, _ = twin_env.step(action)
                if expert.is_done() or terminated or truncated:
                    break
            
            # 4. Render the subgoal image from the twin's final state
            subgoal_image = twin_env.render(camera_name="image_primary")
            subgoal_images.append(subgoal_image)

        return np.stack(subgoal_images)

    def reset_experts(self):
        for expert in self.experts:
            expert.reset()