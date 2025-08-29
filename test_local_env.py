import jax
import numpy as np
from octo.model.octo_model import OctoModel

# Import our custom environment classes from the 'envs' folder
from envs.panda_env import PandaEnv
from envs.panda_env_wrapper import SafetyGuidedEnvWrapper

def main():
    """
    This script performs the final verification of our entire local environment stack.
    It loads the OCTO model, creates the base MuJoCo environment, wraps it with our
    DGPO-Foundation logic, and runs a full test episode with random actions.
    """
    print("--- Testing the Full Local DGPO-Foundation Environment ---")
    
    # --- 1. Load the Expert Guide (OCTO) ---
    print("⏳ Loading the 'octo-small-1.5' model...")
    # This will download the model weights to ~/.cache/huggingface the first time.
    try:
        octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
        print("✅ OCTO model loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load OCTO model. Error: {e}")
        print("Ensure you have installed the dependencies from the final Colab notebook.")
        return

    # --- 2. Create the Environment Stack ---
    print("\n⏳ Creating the environment stack...")
    try:
        # Create the base MuJoCo environment
        base_env = PandaEnv()
    
        # Wrap it with our custom DGPO-Foundation logic
        env = SafetyGuidedEnvWrapper(
            env=base_env,
            octo_model=octo_model,
            instruction="pick up the red block"
        )
        print("✅ Environment stack created.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to create the MuJoCo environment. Error: {e}")
        print("Ensure 'panda_pick_place.xml' is in the 'envs' directory and all assets are present.")
        return

    # --- 3. Run a Test Episode ---
    print("\n⏳ Running a test episode with random actions...")
    try:
        obs, info = env.reset()
        print(f"Reset successful. Observation keys: {list(obs.keys())}")

        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0

        while not (terminated or truncated):
            random_action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(random_action)
            total_reward += reward
            step_count += 1
            print(f"Step {step_count}...", end='\r')

        print("\n\n--- ✅ Episode Finished Successfully ---")
        print(f"Total steps: {step_count}")
        print(f"Final total reward: {total_reward:.4f}")
        # The crucial output: the info dict should contain our divergence score
        print(f"Final info dict: {info}")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: An error occurred during the episode rollout. Error: {e}")
    
if __name__ == "__main__":
    main()