#!/usr/bin/env python3
"""
test_sb3_export.py

A small, fast unit test to verify that the Stable-Baselines3 PPO agent
can be successfully instantiated with our custom architecture.

This script isolates the exact logic that was failing in `pretrain_policy.py`
with the `TypeError: unexpected keyword argument 'proprio_dim'`.

- If this script runs and prints "✅ SUCCESS!", the fix is correct.
- If this script fails, there is still an issue in the environment or
  the custom policy definitions.
"""
import logging
import os
import sys

# Add project root to the path to allow imports of our custom modules
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.path.abspath('.')
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("SB3-Export-Test")

def run_test():
    """Performs the PPO agent instantiation test."""
    logger.info("--- Starting SB3 Export Sanity Check ---")

    try:
        from stable_baselines3 import PPO
        from envs.panda_env import PandaEnv
        from models.custom_sb3_extractor import BCFeaturesExtractor
        logger.info("Successfully imported all required modules.")
    except ImportError as e:
        logger.error(f"Failed to import a required module: {e}")
        logger.error("Please ensure all dependencies are installed and the script is run from the project root.")
        return False

    # --- 1. Create the Environment in SB3-Compatible Mode ---
    try:
        xml_path = "envs/panda_pick_place.xml"
        if not os.path.exists(xml_path):
             logger.warning(f"Could not find '{xml_path}'. Using default.")
             xml_path = None
        
        env = PandaEnv(xml_path=xml_path, for_sb3=True)
        logger.info("Successfully created PandaEnv in SB3 mode.")
        logger.info(f"Observation Space: {env.observation_space}")
    except Exception as e:
        logger.error(f"Failed to create PandaEnv: {e}")
        return False

    # --- 2. Define the Policy Configuration (The Critical Part) ---
    # This dictionary must exactly match the one in `pretrain_policy.py`.
    # Crucially, `features_extractor_kwargs` is omitted.
    policy_kwargs = {
        "features_extractor_class": BCFeaturesExtractor,
        "net_arch": {
            "pi": [512, 256],
            "vf": [512, 256],
        }
    }
    logger.info(f"Using policy_kwargs: {policy_kwargs}")

    # --- 3. Attempt to Instantiate the PPO Agent ---
    try:
        logger.info("Attempting to create PPO agent...")
        agent = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
        logger.info("Successfully created PPO agent with custom architecture.")
        
        # Cleanup
        env.close()
        del agent, env
        return True
    except TypeError as e:
        logger.error("❌ TEST FAILED: Caught the exact TypeError we are trying to fix!")
        logger.error(f"Error Details: {e}")
        logger.error("This means the `features_extractor_kwargs` might still be present in `pretrain_policy.py`, or there is another argument mismatch.")
        return False
    except Exception as e:
        logger.error(f"❌ TEST FAILED: An unexpected error occurred during PPO creation: {e}")
        return False

if __name__ == "__main__":
    success = run_test()
    print("\n" + "="*50)
    if success:
        print("✅ SUCCESS: The PPO agent was created successfully with the custom architecture.")
        print("You are now clear to run the full pretrain_policy.py script.")
    else:
        print("❌ FAILURE: The test failed. Please review the error messages above.")
    print("="*50)