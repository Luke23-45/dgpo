# Quick test for the wrapper
from envs.panda_env import PandaEnv
from envs.panda_env_wrapper import SafetyGuidedEnvWrapper
from octo.model.octo_model import OctoModel

# Load OCTO model
octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

# Create env and wrapper
env = SafetyGuidedEnvWrapper(PandaEnv(), octo_model)
obs, info = env.reset()
print("Wrapper reset successful.")
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward}, Info: {info}")
env.close()