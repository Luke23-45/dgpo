# scripts/pilot_stable_rl.py
"""
Stable pilot for fine-tuning BC -> PPO.

Implements:
 - VecNormalize (obs + reward normalization)
 - low LR
 - target_kl clipping
 - shorter n_steps
 - freeze feature extractor for initial phase
 - safe BC -> PPO weight transfer (uses run_experiment helpers when available)
"""

import os
import sys
from pathlib import Path
import logging
import time

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import torch

# Try to import your repo modules (defensive)
try:
    from envs.panda_env import PandaEnv
    from utils.rl_reward_wrapper import RLRewardWrapper
    from utils.obs_adapters import OctoToSB3Adapter
    from utils.obs_adapters import VecOctoToSB3Adapter
except Exception as e:
    print("ERROR importing environment modules from repo:", e)
    raise

# Try to import BC transfer helpers if they exist
_have_transfer_helpers = False
try:
    from scripts.run_experiment import load_bc_checkpoint, transfer_bc_weights
    _have_transfer_helpers = True
except Exception:
    # we'll try a fallback later (informative)
    _have_transfer_helpers = False

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Optional: import your BCFeaturesExtractor class if present (so policy_kwargs match)
try:
    from models.custom_sb3_extractor import BCFeaturesExtractor
except Exception:
    BCFeaturesExtractor = None  # fallback: let SB3 choose default extractor

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def make_wrapped_env(seed: int = 0):
    """
    Creates a single environment instance with core logic wrappers.
    NOTE: The observation space is NOT yet adapted for SB3 here.
    """
    def _init():
        # Base environment produces HWC images
        env = PandaEnv(xml_path="envs/panda_pick_place.xml", control_mode="delta")
        
        # Apply the reward wrapper
        env = RLRewardWrapper(env,
                              reach_scale=2.0,
                              place_scale=10.0,
                              grasp_reward=50.0,
                              lift_reward=100.0,
                              success_reward=250.0,
                              w_guidance=0.0,
                              w_guidance_dense=0.0)
        
        # The Monitor wrapper is for single-env logging
        env = Monitor(env)
        return env
    
    set_random_seed(seed)
    return _init



def main():
    seed = 42
    n_envs = 1
    total_timesteps_phase1 = 20_000  # initial frozen features phase
    total_timesteps_phase2 = 10_000  # optional continuation with unfrozen features

    # 1) Create vectorized envs
    log.info("Creating vectorized environments...")
    
    # 1.1: Create the base vectorized environment.
    # It still works with HWC image observations at this point.
    vec_env = DummyVecEnv([make_wrapped_env(seed + i) for i in range(n_envs)])
    
    vec_env = VecMonitor(vec_env)
    
    log.info("Applying observation adapter to vectorized environment...")
    vec_env = VecOctoToSB3Adapter(vec_env) 

    # --- START OF FIX ---
    # The original code normalized all observations, including images, which caused a shape mismatch.
    # The correct approach is to only normalize the continuous vector observations ('proprio')
    # and let the SB3 policy handle the normalization of image pixels internally.
    log.info("Applying observation and reward normalization (only for 'proprio' key)...")
    vec_env = VecNormalize(vec_env,
                           norm_obs=True,
                           norm_reward=True,
                           clip_obs=10.0,
                           norm_obs_keys=["proprio"]) # <-- THIS IS THE FIX
    # --- END OF FIX ---


    # 2) Prepare policy kwargs
    policy_kwargs = dict(
        net_arch={"pi": [512, 256], "vf": [512, 256]},
    )
    if BCFeaturesExtractor is not None:
        policy_kwargs["features_extractor_class"] = BCFeaturesExtractor

    # 3) PPO hyperparams - conservative for stable fine-tuning
    ppo_kwargs = dict(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        n_steps=1024,              # shorter rollouts -> lower variance
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=1e-5,        # small LR to avoid big updates
        ent_coef=0.001,
        clip_range=0.2,
        max_grad_norm=0.5,         # gradient clipping
        vf_coef=0.5,
        tensorboard_log=str(project_root / "trained_models" / "pilot_stable" / "logs"),
        seed=seed,
        policy_kwargs=policy_kwargs,
        target_kl=0.03,            # STOP updates if KL exceeds this
    )

    log.info("Creating PPO agent with conservative hyperparameters...")
    model = PPO(**ppo_kwargs)

    # 4) Load BC checkpoint and transfer weights (best-effort)
    bc_path = Path("artifacts/bc_final_balanced_v1/checkpoints/best_model.pth")
    if not bc_path.exists():
        log.warning(f"BC checkpoint not found at {bc_path}. Skipping transfer.")
    else:
        log.info(f"Loading BC checkpoint from {bc_path} ...")
        if _have_transfer_helpers:
            try:
                # load state dict from repo helper (expected to return dict or state)
                sd = load_bc_checkpoint(str(bc_path))
                # instantiate temporary BCNet? transfer helper may accept path or state
                transfer_bc_weights(sd, model)  # uses your repo transfer util
                log.info("BC -> PPO weight transfer via helper complete.")
            except Exception as e:
                log.exception("BC transfer helper failed; continuing without transfer. Error:")
        else:
            # Fallback: attempt best-effort load of matching keys into policy
            try:
                import torch
                bc_state = torch.load(str(bc_path), map_location="cpu")
                # if saved as dict with 'model_state_dict'
                if isinstance(bc_state, dict) and "model_state_dict" in bc_state:
                    bc_sd = bc_state["model_state_dict"]
                else:
                    bc_sd = bc_state
                # attempt to load matching keys into policy.state_dict()
                policy_sd = model.policy.state_dict()
                # pick keys that match exactly and load them
                matched = {k: v for k, v in bc_sd.items() if k in policy_sd and bc_sd[k].shape == policy_sd[k].shape}
                policy_sd.update(matched)
                model.policy.load_state_dict(policy_sd)
                log.info(f"BC->PPO best-effort transferred {len(matched)} params (fallback path).")
            except Exception as e:
                log.exception("Fallback BC transfer failed. Continuing without BC transfer.")

    # 5) Freeze feature extractor parameters initially
    log.info("Freezing feature extractor params for initial phase...")
    frozen_keys = 0
    for name, p in model.policy.named_parameters():
        if "features_extractor" in name or "cnn" in name or "proprio" in name:
            p.requires_grad = False
            frozen_keys += 1
    log.info(f"Froze {frozen_keys} params (features extractor).")

    # 6) Initial training phase (frozen features)
    log.info(f"Starting initial training phase: {total_timesteps_phase1} timesteps (features frozen).")
    model.learn(total_timesteps=total_timesteps_phase1)

    # Examine KL/clip stats in the logs. If approx_kl is large, STOP and adjust LR/target_kl.

    # 7) Unfreeze and continue a bit (optional)
    log.info("Unfreezing feature extractor and continuing training (short continuation)...")
    for name, p in model.policy.named_parameters():
        if "features_extractor" in name or "cnn" in name or "proprio" in name:
            p.requires_grad = True

    model.learn(total_timesteps=total_timesteps_phase2, reset_num_timesteps=False)

    # 8) Save model and VecNormalize stats
    out_dir = project_root / "trained_models" / f"pilot_stable_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / "pilot_stable_final.zip"))
    try:
        vec_env.save(str(out_dir / "vecnormalize.pkl"))
    except Exception:
        pass
    log.info(f"Pilot complete. Saved to {out_dir}")


if __name__ == "__main__":
    main()