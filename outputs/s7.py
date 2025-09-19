import argparse
import torch
import numpy as np
import sys
from pathlib import Path
import time
import json

# Add project root to path to allow relative imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# NOTE: We copy the setup_environment function here to modify it for the test
# without changing the main project file.
from stable_baselines3.common.env_util import make_vec_env
from envs.panda_env import PandaEnv
from utils.rl_reward_wrapper import RLRewardWrapper
from utils.obs_adapters import OctoToSB3Adapter
from utils.scripted_expert import ScriptedExpert, ObjectProfile
from scripts.arun_experiment import initialize_ppo_agent, load_bc_checkpoint, transfer_bc_weights
from models.bc_policy import BCNet

def setup_environment_for_test(
    xml_path: str,
    seed: int,
    w_guidance_dense: float # <-- Add this to control the reward
):
    """A modified version of the main setup_environment for debugging."""
    object_profile = ObjectProfile(size=np.array([0.04, 0.04, 0.04]), grasp_width_normalized=0.6)
    scripted_expert = ScriptedExpert(object_profile)
    
    def make_env():
        env = PandaEnv(xml_path=xml_path, control_mode='delta')
        env = RLRewardWrapper(env, 
                              scripted_expert=scripted_expert,
                              w_guidance_dense=w_guidance_dense, # Use the passed value
                              w_guidance=0.1) # Keep terminal guidance
        env = OctoToSB3Adapter(env)
        return env

    return make_vec_env(lambda: make_env(), n_envs=1, seed=seed)

def run_test_without_dense_guidance(bc_init_dir: str):
    """
    Runs an instrumented episode but with the dense guidance reward disabled.
    """
    print("="*80)
    print("--- DEBUG SCRIPT 3: RL RUN WITHOUT DENSE GUIDANCE REWARD ---")
    print("This will test if the agent behaves differently without strong per-step shaping.")
    print("="*80)
    
    bc_model_path = str(Path(bc_init_dir) / "checkpoints" / "best_model.pth")
    seed = 42
    
    # 1. Setup the environment with w_guidance_dense=0.0
    env = setup_environment_for_test(
        xml_path="envs/panda_pick_place.xml",
        seed=seed,
        w_guidance_dense=0.0 # <-- THE CRITICAL CHANGE FOR THIS TEST
    )
    print(f"Environment created with w_guidance_dense=0.0")

    run_dir = Path(f"debug_run_{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(exist_ok=True)
    
    agent = initialize_ppo_agent(env, run_dir, seed, "auto")

    # 2. Transfer weights as before
    try:
        print(f"\nAttempting to transfer weights from {bc_model_path}...")
        device = agent.policy.device
        state_dict = load_bc_checkpoint(bc_model_path, device)
        action_dim = env.action_space.shape[0]
        bc_net = BCNet(n_actions=action_dim).to(device)
        bc_net.load_state_dict(state_dict)
        if transfer_bc_weights:
            transfer_bc_weights(bc_net, agent)
            print("Weight transfer successful.")
        else:
            print("[WARNING] transfer_bc_weights utility not found.")
    except Exception as e:
        print(f"\n[ERROR] Failed during BC weight transfer: {e}")
        env.close()
        return

    # --- Run one episode ---
    print("\n--- Starting Instrumented Episode (Dense Guidance OFF) ---")
    obs = env.reset()
    terminated = truncated = False
    total_reward = 0.0
    step = 0

    while not (terminated or truncated):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, infos = env.step(action)
        info = infos[0]
        
        # In this run, R_guidance_dense should be 0.0
        r_reach = info.get('R_reach', 0)
        r_guidance_dense = info.get('R_guidance_dense', 0) # This should be 0
        dist_ee_cube = info.get('dist_ee_to_cube', -1)

        print(f"Step {step:03d} | "
              f"Dist: {dist_ee_cube:.4f} | "
              f"R_total: {reward[0]:+7.3f} | "
              f"R_reach: {r_reach:+7.3f} | "
              f"R_guidance_dense: {r_guidance_dense:+.3f}") # <-- VERIFY THIS IS 0

        total_reward += reward[0]
        step += 1
        
        if terminated or truncated:
            break
            
    print(f"\nEpisode finished. Total Reward: {total_reward:.4f}")
    env.close()
    import shutil
    shutil.rmtree(run_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bc_init_dir", 
        type=str, 
        required=True,
        help="Path to the COMPLETED BC run directory (e.g., artifacts/bc_final_balanced_dropout)"
    )
    args = parser.parse_args()
    run_test_without_dense_guidance(args.bc_init_dir)