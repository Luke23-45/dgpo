import time
import imageio
import numpy as np
from stable_baselines3 import PPO

# Import our custom environment classes, exactly as they are in the training script
from envs.panda_env import PandaEnv
from envs.panda_env_wrapper import SafetyGuidedEnvWrapper
from octo.model.octo_model import OctoModel

# --- Configuration ---
# The path to the saved model checkpoint you want to evaluate.
# Ensure this path is correct.
CHECKPOINT_PATH = "training_checkpoints/dgpo_policy_40000_steps.zip"

# The name of the video file that will be saved.
VIDEO_PATH = "evaluation_rollout_40k.mp4"

# How many different episodes to run and record.
NUM_EPISODES = 5

def main():
    """
    Main function to load a trained PPO policy and evaluate its performance
    by running it in the environment and saving a video.
    """
    print("--- Evaluating Trained DGPO Policy ---")

    # --- 1. Load the OCTO model (required for the environment wrapper) ---
    # We don't use its output for guidance here, but the wrapper needs it to initialize.
    print("⏳ Loading OCTO model for environment setup...")
    try:
        octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
        print("✅ OCTO model loaded.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load OCTO model. {e}")
        return

    # --- 2. Create the evaluation environment ---
    print("\n⏳ Creating the evaluation environment...")
    # It is CRITICAL that this environment is identical to the one used for training.
    # We use render_mode="rgb_array" to capture frames for the video.
    base_env = PandaEnv(xml_path="envs/panda_pick_place.xml", render_mode="rgb_array")
    
    # We use the same wrapper to ensure observation and action spaces match perfectly.
    env = SafetyGuidedEnvWrapper(env=base_env, octo_model=octo_model)
    print("✅ Environment created successfully.")

    # --- 3. Load the trained PPO agent ---
    print(f"\n⏳ Loading trained policy from {CHECKPOINT_PATH}...")
    try:
        model = PPO.load(CHECKPOINT_PATH, env=env)
        print("✅ Policy loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load the policy. Ensure the path is correct. {e}")
        return

    # --- 4. Run Evaluation Rollouts ---
    print(f"\n🚀 Running {NUM_EPISODES} evaluation episodes...")
    
    all_episode_rewards = []
    best_episode_frames = []
    best_episode_reward = -np.inf

    for i in range(NUM_EPISODES):
        obs, info = env.reset()
        terminated = False
        truncated = False
        
        frames = []
        total_reward = 0
        
        # Capture the first frame of the episode
        # The base environment has the `renderer` attribute we need.
        frames.append(base_env.renderer.render())

        while not (terminated or truncated):
            # Use the trained model to predict the action.
            # `deterministic=True` makes the agent choose the best action, not a random one.
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Capture the frame after the step
            frames.append(base_env.renderer.render())
        
        print(f"  Episode {i+1}: Total Reward = {total_reward:.2f}")
        all_episode_rewards.append(total_reward)

        # Keep track of the best episode to save as a video
        if total_reward > best_episode_reward:
            best_episode_reward = total_reward
            best_episode_frames = frames

    # --- 5. Report Results and Save Video ---
    print("\n--- Evaluation Summary ---")
    print(f"Average reward over {NUM_EPISODES} episodes: {np.mean(all_episode_rewards):.2f}")
    print(f"Best episode reward: {best_episode_reward:.2f}")

    if best_episode_frames:
        print(f"\n💾 Saving the best episode rollout to {VIDEO_PATH}...")
        imageio.mimsave(VIDEO_PATH, best_episode_frames, fps=30)
        print(f"✅ Video saved successfully to {VIDEO_PATH}")
    else:
        print("⚠️ No frames were recorded, video not saved.")

    env.close()

if __name__ == '__main__':
    main()