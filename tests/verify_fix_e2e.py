import logging
import numpy as np
import os
import sys

# Setup Path
sys.path.append(os.getcwd())

from envs.panda_env import PandaEnv
from envs.dgpo_env_wrapper import DGPOEnvWrapper

# Configure minimal logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("E2E_TEST")

def test_full_loop():
    print("[E2E] Starting Verification...")
    
    # 1. Config Simulation (Mimic Hydra DictConfig)
    class SimpleConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    setattr(self, k, SimpleConfig(**v))
                else:
                    setattr(self, k, v)
        
        def get(self, key, default=None):
            return getattr(self, key, default)

    # Mimic the structure expected by DGPOEnvWrapper
    cfg_dict = {
        'training': {'use_policy_blending': True},
        'environment': {'urdf_path': 'urdf/panda_mujoco_kinematics.urdf'},
        'expert': {
            'object_size': [0.04, 0.04, 0.04],
            'grasp_width': 0.6,
            'hover_height': 0.15,
            'grasp_offset_z': 0.025
        }
    }
    cfg = SimpleConfig(**cfg_dict)
    
    # 2. Instantiate
    try:
        env = PandaEnv()
        wrapper = DGPOEnvWrapper(env, cfg)
        print("[E2E] Wrapper Initialized.")
    except Exception as e:
        print(f"[FAIL] Initialization: {e}")
        return

    # 3. Reset
    obs, info = wrapper.reset()
    print("[E2E] Reset Complete.")
    print(f"      Expert Pose: {info['expert_pose'].shape}")
    
    # 4. Simulation Loop (20 steps)
    print("[E2E] Running Control Loop...")
    for t in range(20):
        # Fake Policy Action (zeros + random noise)
        # Action space is 9D in wrapper (7 joint, 1 grip, 1 alpha)
        action = np.zeros(9, dtype=np.float32)
        action[8] = 0.5 # Test 50% blending to trigger IK
        
        try:
            obs, reward, terminated, truncated, info = wrapper.step(action)
            # Check for NaN in qpos which typically means IK exploded
            if np.any(np.isnan(wrapper.env.data.qpos[:7])):
                raise ValueError("NaN Detected in Joint State!")
                
        except Exception as e:
            print(f"[FAIL] Step {t}: {e}")
            return
            
    print("[E2E] Success! 20/20 steps completed without crash or NaN.")
    print("[E2E] The Analytical Smooth IK is stable.")

if __name__ == "__main__":
    test_full_loop()
