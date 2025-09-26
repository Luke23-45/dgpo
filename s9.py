import torch as th
import numpy as np
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# --- Add project root to path to find local modules ---
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
# ---

from models.bc_policy import BCNet
from models.custom_sb3_extractor import BCFeaturesExtractor
from utils.transfer_bc_to_ppo import transfer_bc_weights
from envs.panda_env import PandaEnv

# --- Basic logger for clean output ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("diagnostic_test")


# ---- CONFIG ----
# Use a valid checkpoint from your successful pretraining run
BC_CKPT_PATH = "artifacts/bc_final_balanced_v1/checkpoints/best_model.pth"
ENV_XML_PATH = "envs/panda_pick_place.xml"
ACTION_DIM = 8 # The PandaEnv has 7 arm joints + 1 gripper = 8


def make_env():
    """Factory function for the environment."""
    env = PandaEnv(
        xml_path=ENV_XML_PATH,
        control_mode="absolute" # Match the BC model's training mode
    )
    return env


def get_policy_predicted_actions(policy, obs_dict):
    """
    Minimal, safe, and self-contained helper to get differentiable actions.
    This is the core logic from SB3's forward pass for the actor.
    """
    # Ensure observations are on the same device as the policy
    obs_tensor = policy.obs_to_tensor(obs_dict)[0]
    
    # Enable gradients for this part of the computation
    with th.set_grad_enabled(True):
        features = policy.extract_features(obs_tensor)
        latent_pi, _ = policy.mlp_extractor(features)
        distribution = policy._get_action_dist_from_latent(latent_pi)
        # Return the mean of the distribution, which is differentiable
        actions = distribution.get_actions(deterministic=True)
    return actions


if __name__ == "__main__":
    # 1. Load the BC model state dictionary correctly
    logger.info(f"Loading BC checkpoint from: {BC_CKPT_PATH}")
    checkpoint = th.load(BC_CKPT_PATH, map_location='cpu')
    # The state_dict is often nested inside the checkpoint dictionary
    bc_state_dict = checkpoint.get("model_state_dict", checkpoint)

    # We need to instantiate a BCNet object to load the state_dict into
    bc_net = BCNet(n_actions=ACTION_DIM)
    bc_net.load_state_dict(bc_state_dict, strict=False)
    logger.info("Successfully loaded BC state_dict into a BCNet instance.")


    # 2. Build the PPO agent
    policy_kwargs = dict(
        features_extractor_class=BCFeaturesExtractor,
        # The BCFeaturesExtractor does not need a `latent_dim` argument
        # net_arch is what defines the MLP layers after the extractor
        net_arch=dict(pi=[512, 256], vf=[512, 256])
    )

    env = make_vec_env(make_env, n_envs=1)
    # Correctly apply VecNormalize for Dict spaces
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,     # Disable reward norm for cleaner debugging
        clip_obs=10.0,
        norm_obs_keys=["proprio"],
    )

    ppo = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        n_steps=128,
        batch_size=64,
        learning_rate=3e-5,
        target_kl=1.0,   # Loosen target_kl for this test to prevent it from stopping early
    )

    # 3. Transfer weights and show a detailed report
    report = transfer_bc_weights(bc_net, ppo, allow_shape_only_fallback=False, verbose=False)
    print("\n" + "="*80)
    print("--- DIAGNOSTIC TEST 1: Weight Transfer Report ---")
    print(f"  - Transfer loaded_ok: {report.get('loaded_ok')}")
    print(f"  - Transferred Pairs Count: {len(report.get('transferred', []))}")
    print(f"  - Skipped (Shape Mismatch): {len(report.get('skipped_shape_mismatch', []))}")
    print(f"  - Not Found in PPO (first 5): {report.get('not_found', [])[:5]}")
    print(f"  - Ambiguous in PPO (first 5): {report.get('ambiguous', [])[:5]}")
    print("="*80)

    # 4. Prepare a sample observation batch
    obs_np = env.reset()
    
    # 5. Check predicted actions differentiability
    # The helper needs a dictionary of tensors
    obs_tensor_dict = {k: th.as_tensor(v) for k, v in obs_np.items()}
    actions_t = get_policy_predicted_actions(ppo.policy, obs_tensor_dict)
    print("\n" + "="*80)
    print("--- DIAGNOSTIC TEST 2: Action Differentiability ---")
    print(f"  - Action `requires_grad`: {getattr(actions_t, 'requires_grad', None)}")
    print("="*80)

    # 6. Inspect action distribution and log_std
    means = actions_t.detach().cpu().numpy()
    print("\n" + "="*80)
    print("--- DIAGNOSTIC TEST 3: Initial Action Distribution ---")
    print(f"  - Action Mean (Sample from Batch): {np.array2string(means.mean(axis=0), precision=4)}")
    if hasattr(ppo.policy, "log_std"):
        action_std = th.exp(ppo.policy.log_std).detach().cpu().numpy()
        print(f"  - Action `log_std` parameter: {np.array2string(ppo.policy.log_std.detach().cpu().numpy(), precision=4)}")
        print(f"  - Inferred Action Std Dev: {np.array2string(action_std, precision=4)}")
    else:
        print("  - `log_std` parameter not found on policy object.")
    print("="*80)

    # 7. Check value predictions
    with th.no_grad():
        # predict_values expects a dictionary of tensors
        values = ppo.policy.predict_values(obs_tensor_dict)
    print("\n" + "="*80)
    print("--- DIAGNOSTIC TEST 4: Critic (Value Head) Stats ---")
    print(f"  - Critic/Value Head Output (Sample): mean={values.mean().item():.4f}, std={values.std().item():.4f}")
    print("="*80)

    # 8. One mini learn iteration to test KL stability
    print("\n" + "="*80)
    print("--- DIAGNOSTIC TEST 5: One-Iteration PPO Learn Test ---")
    try:
        ppo.learn(total_timesteps=256, reset_num_timesteps=False)
        print("\n[SUCCESS] Finished one PPO learn iteration without early stopping.")
    except Exception as e:
        print(f"\n[FAILURE] PPO learn step failed with error: {e}")
    print("="*80)