
import sys
from unittest.mock import MagicMock

# Mock stable_baselines3 if missing
sys.modules["stable_baselines3"] = MagicMock()
sys.modules["stable_baselines3.common"] = MagicMock()
sys.modules["stable_baselines3.common.vec_env"] = MagicMock()

import os
from pathlib import Path
import torch
import gymnasium as gym

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.train_dgpo import DGPOTrainer
from omegaconf import OmegaConf, DictConfig

# Mock Config
cfg = OmegaConf.create({
    "num_envs": 2,
    "bc_checkpoint": "dummy_ckpt.ckpt", # Added
    "model": {
        "vision_feature_dim": 768,
        "proprio_dim": 22,
        "chunk_size": 10
    },
    "value_net": {
        "hidden_dim": 256
    },
    "ppo": {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "epochs": 2,
        "batch_size": 4,
        "temperature": 1.0, # Added missing key often used in temperature scaling
        "value_coef": 0.5, # Added
        "max_grad_norm": 0.5 # Added
    },
    "reward": {
        "w_dist": 1.0,
        "w_div": 0.1,
        "success_bonus": 10.0,
        "position_weight": 1.0,
        "orientation_weight": 0.1
    },
    "checkpoint": {
        "backup_freq": 5,
        "save_dir": "./checkpoints_test"
    },
    "environment": {
        "xml_path": str(ROOT / "assets" / "franka_emika_panda" / "panda_with_cube.xml") 
    },
    "trainer": {
        "max_iterations": 1,
        "steps_per_iter": 4, 
        "device": "cpu"
    },
    "optimizer": {
        "policy_lr": 1e-4,
        "value_lr": 1e-3
    },
    "logging": {
        "log_dir": "./logs_test",
        "csv_log_name": "test.csv"
    },
    "training": {
        "total_iterations": 1
    }
})

# Mock SemanticPlannerLightningModule
class MockPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.chunk_size = 10
        # Create dummy layers to allow optimizer to work
        self.dummy_param = torch.nn.Parameter(torch.randn(1))
        # Mock backbone for freezing logic
        self.vision_backbone = torch.nn.Linear(10, 10)
    
    def forward(self, batch):
        B = batch['curr_proprio'].shape[0]
        K = 10
        return {
            'pose_chunk': torch.randn(B, K, 7),
            'visual_embedding': torch.randn(B, 768),
            'phase_logits': torch.randn(B, 5) # 5 phases
        }
    
    def train(self): pass
    def eval(self): pass
    def to(self, device): return self

import train.train_dgpo
# Mock the class loader
class MockPLModule:
    model = MockPolicy()
    @classmethod
    def load_from_checkpoint(cls, *args, **kwargs):
        return MockPLModule()

train.train_dgpo.SemanticPlannerLightningModule = MockPLModule

def mock_make_env(cfg):
    # Return a dummy env
    env = gym.make("CartPole-v1") # Dummy
    # Patch observation space to match DGPO
    env.observation_space = gym.spaces.Dict({
        "image_primary": gym.spaces.Box(0, 255, (256, 256, 3), dtype="uint8"),
        "proprio": gym.spaces.Box(-1, 1, (22,), dtype="float32"),
        "object_pos_world": gym.spaces.Box(-10, 10, (3,), dtype="float32"),
        "goal_pos_world": gym.spaces.Box(-10, 10, (3,), dtype="float32"),
        "ee_pose_world": gym.spaces.Box(-10, 10, (7,), dtype="float32"),
    })
    # Patch step to return dict
    original_step = env.step
    def step(action):
        obs, r, term, trunc, info = original_step(0) # Dummy action
        # Mock DGPO Obs
        dgpo_obs = {
            "image_primary": torch.randint(0, 255, (256, 256, 3)).numpy().astype("uint8"),
            "proprio": torch.randn(22).numpy().astype("float32"),
            "object_pos_world": torch.randn(3).numpy().astype("float32"),
            "goal_pos_world": torch.randn(3).numpy().astype("float32"),
            "ee_pose_world": torch.randn(7).numpy().astype("float32")
        }
        # Mock Info
        info['goal_img'] = torch.randint(0, 255, (256, 256, 3)).numpy().astype("uint8")
        info['expert_pose'] = torch.randn(7).numpy().astype("float32")
        info['expert_grip'] = 1.0
        info['expert_phase'] = "MOVE_TO_GOAL"
        info['executed_action'] = torch.randn(8).numpy()
        return dgpo_obs, r, term, trunc, info
    
    original_reset = env.reset
    def reset(seed=None, options=None):
         obs, info = original_reset(seed=seed, options=options) # Capture original
         # Actually CartPole reset returns (obs, info)
         # Mock DGPO Reset Obs
         dgpo_obs = {
            "image_primary": torch.randint(0, 255, (256, 256, 3)).numpy().astype("uint8"),
            "proprio": torch.randn(22).numpy().astype("float32"),
            "object_pos_world": torch.randn(3).numpy().astype("float32"),
            "goal_pos_world": torch.randn(3).numpy().astype("float32"),
            "ee_pose_world": torch.randn(7).numpy().astype("float32"),
         }
         info = {
            'goal_img': torch.randint(0, 255, (256, 256, 3)).numpy().astype("uint8"),
            'expert_pose': torch.randn(7).numpy().astype("float32"),
            'expert_grip': 1.0,
            'expert_phase': "MOVE_TO_GOAL"
         }
         return dgpo_obs, info

    env.step = step
    env.reset = reset
    return env

# Patch make_dgpo_env in train_dgpo
import train.train_dgpo
train.train_dgpo.make_dgpo_env = mock_make_env

# Force AsyncVectorEnv to be Sync for test (avoid multiprocessing in test script)
gym.vector.AsyncVectorEnv = gym.vector.SyncVectorEnv

print("Initializing DGPOTrainer...")
try:
    trainer = DGPOTrainer(cfg)
    print("Trainer Initialized.")
    
    print("Testing collect_rollouts...")
    # Mock Policy to return correct shapes
    # SemanticPlanner is heavy, let's trust it loads if cfg is correct.
    # But it uses 'google/siglip' which needs internet/cache.
    # If cache exists, it works.
    
    # We will try to run 1 iteration
    # trainer.train() # This might be too long.
    
    # Just run collect_rollouts directly?
    metrics = trainer.collect_rollouts(n_steps=4)
    print("Collect Rollouts Success:", metrics)
    
    print("Testing update_policy...")
    # Need some buffer data? collect_rollouts filled it.
    metrics_update = trainer.update_policy()
    print("Update Policy Success:", metrics_update)
    
    print("DGPO v2.0 Verification PASSED.")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
