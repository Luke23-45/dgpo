import os
import gymnasium as gym
from octo.model.octo_model import OctoModel

# Import our custom environment and wrapper
from envs.panda_env import PandaEnv
from envs.panda_env_wrapper import SafetyGuidedEnvWrapper

# Import the PPO algorithm from Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

# --- Configuration ---
# This dictionary defines all the hyperparameters for our PPO agent.
# We will start with standard values and can tune them later.
PPO_CONFIG = {
    "policy": "MultiInputPolicy",  # Use this policy for dict observations
    "verbose": 1,                  # Print training progress
    "n_steps": 1024,               # Number of steps to run for each environment per update
    "batch_size": 64,              # Minibatch size
    "n_epochs": 10,                # Number of epochs when optimizing the surrogate loss
    "gamma": 0.99,                 # Discount factor
    "gae_lambda": 0.95,            # Factor for trade-off of bias vs variance for GAE
    "clip_range": 0.2,             # Clipping parameter, it can be a function
    "ent_coef": 0.0,               # Entropy coefficient for exploration
    "vf_coef": 0.5,                # Value function coefficient
    "max_grad_norm": 0.5,          # The maximum value for the gradient clipping
    "tensorboard_log": "./tensorboard_logs/dgpo_foundation_run",
}

# --- Main Training Function ---
def main():
    """
    Main function to set up and run the DGPO-Foundation training process.
    """
    print("--- Starting DGPO-Foundation Training ---")

    # --- 1. Load the Expert Guide (OCTO) ---
    print("⏳ Loading the 'octo-small-1.5' model...")
    octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
    print("✅ OCTO model loaded successfully.")

    # --- 2. Create the Custom Environment ---
    print("\n⏳ Creating and wrapping the environment...")
    
    # Define a function that creates an instance of our environment
    def make_env():
        env = PandaEnv()
        # Wrap it with our custom DGPO logic
        env = SafetyGuidedEnvWrapper(
            env=env,
            octo_model=octo_model,
            instruction="pick up the red block"
        )
        return env

    # We use make_vec_env to create a vectorized environment, which can run
    # multiple instances in parallel for faster training, although we'll start with 1.
    env = make_vec_env(make_env, n_envs=1)
    print("✅ Environment created and wrapped successfully.")

    # --- 3. Set up the PPO Agent ---
    print("\n⏳ Setting up the PPO agent...")
    model = PPO(**PPO_CONFIG, env=env)
    print("✅ PPO agent created.")
    print(f"Policy Architecture:\n{model.policy}")

    # --- 4. Set up Callbacks for Saving Models ---
    # This callback will save a checkpoint of the model every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./training_checkpoints/',
        name_prefix='dgpo_policy'
    )

    # --- 5. Start Training ---
    # The total number of timesteps the agent will be trained for.
    # We'll start with a smaller number for the first run to ensure it works.
    TOTAL_TIMESTEPS = 100_000
    
    print(f"\n🚀 Starting training for {TOTAL_TIMESTEPS} timesteps...")
    print("   (This will take a significant amount of time)")
    print("   You can monitor progress in the terminal or with TensorBoard.")
    
    # The `learn` method starts the training process.
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # --- 6. Save the Final Model ---
    final_model_path = "dgpo_final_policy.zip"
    model.save(final_model_path)
    print(f"\n--- ✅ Training Complete ---")
    print(f"Final policy saved to: {final_model_path}")

if __name__ == '__main__':
    main()