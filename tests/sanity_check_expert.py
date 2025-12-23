import logging
import numpy as np
import os
import sys
from omegaconf import OmegaConf

# Setup Path
sys.path.append(os.getcwd())

from envs.panda_env import PandaEnv
from envs.dgpo_env_wrapper import DGPOEnvWrapper

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("EXPERT_CHECK")

def test_expert_performance():
    print("="*60)
    print("🔍 Expert Sanity Check")
    print("="*60)
    
    # 1. Config Simulation
    class SimpleConfig(dict):
        def __getattr__(self, name): return self[name]
    
    # Mimic the structure expected by DGPOEnvWrapper
    cfg = OmegaConf.create({
        'training': {'use_policy_blending': True},
        'environment': {'xml_path': 'envs/panda_pick_place.xml'},
        'expert': {
            'object_size': [0.04, 0.04, 0.04],
            'grasp_width': 0.6,
            'hover_height': 0.15,
            'grasp_offset_z': 0.025
        }
    })
    
    # [OPTIMIZATION] Monkey-patch render to speed up check (Expert doesn't need vision)
    print("[CHECK] Patching PandaEnv.render for speed...")
    def fast_render(self, camera_name=None):
        # Return dummy 84x84 image (or whatever size expected)
        return np.zeros((84, 84, 3), dtype=np.uint8)
    
    PandaEnv.render = fast_render
    
    # 2. Instantiate
    try:
        env_inner = PandaEnv(xml_path=cfg.environment.xml_path, control_mode="delta")
        env = DGPOEnvWrapper(env_inner, cfg)
        print("[CHECK] Environment Initialized.")
    except Exception as e:
        print(f"[FAIL] Initialization: {e}")
        return

    n_episodes = 10
    successes = 0
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        print(f"\n[EPISODE {ep+1}] Started.")
        
        done = False
        step = 0
        ep_success = False
        
        while not done and step < 800:
            env.set_blend_alpha(0.0) # Force Expert
            dummy_action = np.zeros(9)
            
            obs, reward, terminated, truncated, info = env.step(dummy_action)
            
            # Check Success Logic (Manual Check matching Trainer)
            obj_pos = obs['object_pos_world']
            goal_pos = obs['goal_pos_world']
            dist = np.linalg.norm(obj_pos - goal_pos)
            
            if dist < 0.05:
                ep_success = True
            
            step += 1
            if step % 50 == 0:
                print(f"   Step {step}: Dist={dist*100:.1f}cm, Rew={reward:.2f}, Phase={info.get('expert_phase', 'UNK')}")
            
            if terminated or truncated:
                done = True
                
        if ep_success:
            print(f"[EPISODE {ep+1}] ✅ SUCCESS (Steps: {step}, Final Dist: {dist*100:.2f}cm)")
            successes += 1
        else:
            print(f"[EPISODE {ep+1}] ❌ FAILED (Steps: {step}, Final Dist: {dist*100:.2f}cm)")
            # Analyze why
            if step >= 200: print("   Reason: Timeout")
            else: print("   Reason: Terminated (Drop?)")

    rate = (successes / n_episodes) * 100
    print("="*60)
    print(f"RESULTS: {successes}/{n_episodes} Successes ({rate:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    test_expert_performance()
