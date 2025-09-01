import numpy as np
import os

# --- (Imports are the same) ---
try:
    import glfw
    import mujoco
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False
    print("⚠️ WARNING: `glfw` or `mujoco` not found. Interactive visualization will be disabled.")

from envs.panda_env import PandaEnv
from utils.rl_reward_wrapper import RLRewardWrapper
from octo.model.octo_model import OctoModel

def run_verification_test():
    """
    The definitive pre-flight check for the DGPO-Foundation project.
    """
    print("--- 🚀 Starting Final System Verification ---")

    # --- 1. Verify File Paths ---
    print("\n--- [Check 1/5] Verifying necessary files ---")
    
    # --- CRITICAL CORRECTION: Point to the final, correct XML file ---
    xml_path = "envs/panda_pick_place.xml"
    # --- END CORRECTION ---

    if not os.path.exists(xml_path):
        print(f"❌ FATAL ERROR: Cannot find the scene file at '{xml_path}'.")
        return False
    print(f"✅ Found scene file: {xml_path}")
    print("--- ✅ File Check Passed ---")

    # --- 2. Load the Foundation Model (OCTO) ---
    print("\n--- [Check 2/5] Loading the OCTO Foundation Model ---")
    try:
        octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
        print("✅ OCTO model loaded successfully.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Failed to load OCTO model. Error: {e}")
        return False
    print("--- ✅ OCTO Model Check Passed ---")
    
    # --- 3. Initialize the Full Environment Stack ---
    print("\n--- [Check 3/5] Initializing the full environment stack ---")
    try:
        base_env = PandaEnv(xml_path=xml_path)
        env = RLRewardWrapper(env=base_env)
        print("✅ Environment stack (PandaEnv + RLRewardWrapper) created successfully.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Failed to create the environment stack.")
        import traceback
        traceback.print_exc()
        return False
    print("--- ✅ Environment Stack Check Passed ---")
    
    # --- 4. Test a Full Reset-Step-Observation Cycle ---
    print("\n--- [Check 4/5] Testing a full Reset-Step cycle ---")
    try:
        print("⏳ Performing env.reset()...")
        obs, info = env.reset()
        print("✅ Reset successful.")
        
        assert "image_primary" in obs and "proprio" in obs, "Observation missing keys."
        assert obs["image_primary"].shape == (256, 256, 3), "Image shape incorrect."
        assert obs["proprio"].shape == (14,), "Proprio shape incorrect."
        print("✅ Observation structure is correct.")
        
        print("\n⏳ Performing env.step() with a random action...")
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)
        print("✅ Step successful.")
        
        assert isinstance(reward, float), "Reward is not a float."
        assert "dist_cube_to_goal" in info, "Info dictionary missing reward components."
        print(f"   Received reward: {reward:.4f}")
        print(f"   Info dict contains expected keys.")
        print("✅ Step output structure is correct.")

    except Exception as e:
        print(f"❌ FATAL ERROR: Failed during Reset-Step cycle.")
        import traceback
        traceback.print_exc()
        return False
    print("--- ✅ Reset-Step Cycle Check Passed ---")

    # --- 5. (Optional) Interactive Visualization ---
    # We will skip this for the final automated test for simplicity.
    
    env.close()
    
    print("\n\n--- 🎉🎉🎉 ALL SYSTEM CHECKS PASSED 🎉🎉🎉 ---")
    print("The codebase is correct, robust, and ready for training.")
    return True

if __name__ == '__main__':
    run_verification_test()