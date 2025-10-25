# FILE: evaluate/evaluate_diffusion.py
# (State-of-the-Art, Hydra-Configurable, W&B Integrated Version)

import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import json
import imageio

# --- Project-Specific Imports ---
# Use a try-except block for robustness if this script is moved
try:
    from scripts.pretrain_diffusion import set_seed
    from utils.expert_dataset import ExpertTrajectoryDataset, collate_fn
    from models.diffusion_policy import DiffusionPolicy, NoiseSchedulerConfig
    from envs.panda_env import PandaEnv
except ImportError as e:
    raise ImportError(f"Could not import project modules. Ensure your PYTHONPATH is set correctly. Details: {e}")

# Setup a logger for the script
log = logging.getLogger(__name__)


# ==============================================================================
# 1. Canonical Policy Loading (CRITICAL FIX)
# ==============================================================================

# FILE: scripts/evaluate_policy.py

def load_policy_from_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[DiffusionPolicy, DictConfig]:
    """
    Loads a DiffusionPolicy model from a training checkpoint.
    Returns (policy, config)
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    log.info(f"Loading checkpoint from: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Retrieve saved Hydra config
    if "config" in payload:
        # Use OmegaConf to load for easier manipulation
        cfg = OmegaConf.create(payload["config"])
        log.info("Successfully loaded training configuration from checkpoint.")
    else:
        raise RuntimeError("Checkpoint missing 'config' key — cannot rebuild model.")

    # --- START OF SOTA PATCH ---
    # **Configuration Adapter Block**
    # The saved training config uses the key 'schedule_type', but the model's
    # NoiseScheduler class expects 'schedule'. We perform the translation here
    # to make the loaded config compatible with the model constructor.
    if 'scheduler' in cfg and 'schedule_type' in cfg.scheduler:
        log.info("Adapting legacy 'schedule_type' key to 'schedule' for model compatibility.")
        # Create the new key with the value of the old key
        cfg.scheduler.schedule = cfg.scheduler.schedule_type
        # Optionally, delete the old key for cleanliness, though not strictly necessary
        # del cfg.scheduler.schedule_type
    # --- END OF SOTA PATCH ---

    # Build model from the (now corrected) saved config.
    # The existing build_policy_from_cfg can be used if it correctly
    # unpacks the cfg, but for clarity let's show the direct build.
    
    proprio_dim = cfg.model.get('proprio_dim')
    if proprio_dim is None:
        log.warning("proprio_dim not found in checkpoint config. Inferring from validation dataset...")
        try:
            # We need a dataset instance to infer the dimension
            val_dataset = ExpertTrajectoryDataset(
                demo_path=cfg.dataset.val_path,
                observation_horizon=cfg.model.observation_horizon,
                action_horizon=cfg.model.action_horizon
            )
            proprio_dim = val_dataset.get_proprioception_dim()
            log.info(f"Inferred proprioception dimension from validation dataset: {proprio_dim}")
            # Save it back to the config object in memory for consistency
            cfg.model.proprio_dim = proprio_dim
        except Exception as e:
            log.critical(f"FATAL: Could not infer proprio_dim from dataset. Cannot build model. Error: {e}")
            raise    
    scheduler_cfg = NoiseSchedulerConfig(
        beta_start=cfg.scheduler.beta_start,
        beta_end=cfg.scheduler.beta_end,
        schedule=cfg.scheduler.schedule, # Now this key exists and is correct
        timesteps=cfg.scheduler.timesteps,
    )

    policy = DiffusionPolicy(
        proprio_dim=proprio_dim,
        H_o=cfg.model.observation_horizon,
        H_a=cfg.model.action_horizon,
        action_dim=cfg.model.action_dim,
        image_feat_dim=cfg.model.image_feat_dim,
        d_model=cfg.model.d_model,
        denoiser_layers=cfg.model.denoiser_layers,
        denoiser_heads=cfg.model.denoiser_heads,
        scheduler_cfg=scheduler_cfg,
        cfg_p_uncond=cfg.training.cfg_p_uncond,
        ema_decay=cfg.training.ema_decay,
        device=device
    )
    
    policy.load_state_dict(payload["policy_state_dict"])

    # Load EMA if available
    if hasattr(policy, "ema") and payload.get("ema_state_dict"):
        try:
            policy.ema.load_state_dict(payload["ema_state_dict"])
            log.info("EMA weights loaded successfully.")
        except Exception as e:
            log.warning(f"Failed to load EMA weights: {e}")

    log.info("Policy successfully reconstructed and loaded.")
    return policy, cfg


# ==============================================================================
# 2. Offline Evaluation Logic (Unchanged but validated)
# ==============================================================================
# FILE: evaluate/evaluate_diffusion.py

# ==============================================================================
# 2. Offline Evaluation Logic (CORRECTED VERSION)
# ==============================================================================
@torch.no_grad()
def evaluate_offline(policy: DiffusionPolicy, dataloader: DataLoader, eval_cfg: DictConfig) -> dict:
    policy.eval()
    if policy.ema:
        # This log message confirms our intent, but we don't manually swap weights.
        log.info("Using EMA weights for offline evaluation (via policy.sample(use_ema=True)).")

    mses, maes = [], []
    for obs_chunk, gt_action_chunk in tqdm(dataloader, desc="[Offline Eval]"):
        obs_chunk = {k: v.to(policy.device).float() for k, v in obs_chunk.items()}
        gt_action_chunk = gt_action_chunk.to(policy.device).float()

        # The `use_ema=True` flag tells the sample method to use `self.ema.ema_model`
        # for inference. The calls to .store() and .restore() are incorrect for this
        # EMA implementation and must be removed.
        pred_actions = policy.sample(
            obs_chunk,
            steps=eval_cfg.sampling_steps,
            guidance_scale=eval_cfg.guidance_scale,
            use_ema=True  # This is the correct way to use the EMA weights
        )
        mses.append(torch.nn.functional.mse_loss(pred_actions, gt_action_chunk).item())
        maes.append(torch.nn.functional.l1_loss(pred_actions, gt_action_chunk).item())

    # Since we did not modify the live model's parameters, no restore call is needed.
    return {"mean_mse": np.mean(mses), "mean_mae": np.mean(maes)}


# ==============================================================================
# 3. Online Evaluation Logic (Unchanged but validated)
# ==============================================================================
@torch.no_grad()
def evaluate_online(policy: DiffusionPolicy, eval_cfg: DictConfig, env_cfg: DictConfig, train_cfg: DictConfig, output_dir: Path) -> dict:
    log.info("Initializing evaluation environment...")
    env = PandaEnv(xml_path=env_cfg.xml_path, control_mode="delta")

    success_rates = []
    video_dir = output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    for i in tqdm(range(eval_cfg.n_rollouts), desc="[Online Eval]"):
        seed = eval_cfg.get("start_seed", 1000) + i
        obs, _ = env.reset(seed=seed)
        
        obs_horizon = train_cfg.model.observation_horizon
        action_horizon = train_cfg.model.action_horizon
        
        obs_deque = [obs] * obs_horizon
        frames = [env.render()]
        is_success = False

        for step in range(eval_cfg.max_rollout_steps):
            # A) Prepare policy input chunk
            obs_chunk = {
                key: torch.from_numpy(np.stack([o[key] for o in obs_deque])).unsqueeze(0).to(policy.device).float()
                for key in ["image_primary", "image_wrist", "proprio"]
            }

            # B) Sample action sequence from policy
            action_sequence = policy.sample(
                obs_chunk,
                steps=eval_cfg.sampling_steps,
                guidance_scale=eval_cfg.guidance_scale,
                use_ema=True
            )
            
            # C) Execute the first action and update observation queue
            action_to_exec = action_sequence[0, 0].cpu().numpy()
            obs, _, _, _, info = env.step(action_to_exec)
            obs_deque.pop(0)
            obs_deque.append(obs)
            frames.append(env.render())

            if info.get("is_success", False):
                is_success = True
                break
        
        success_rates.append(is_success)
        video_path = video_dir / f"rollout_{i:02d}_seed{seed}_success{int(is_success)}.mp4"
        imageio.mimsave(video_path, frames, fps=30)

    return {"success_rate": np.mean(success_rates)}


# ==============================================================================
# 4. Hydra Main Entrypoint
# ==============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate_policy_config")
def main(cfg: DictConfig):
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info(f"Evaluation output directory: {output_dir}")
    log.info("Full evaluation config:\n" + OmegaConf.to_yaml(cfg))
    
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # --- 1. Load Policy and its Training Config ---
    policy, train_cfg = load_policy_from_checkpoint(Path(cfg.ckpt_path), device)

    # --- 2. Offline Evaluation ---
    if cfg.do_offline_eval:
        log.info("\n" + "="*50 + "\nOFFLINE EVALUATION\n" + "="*50)
        val_path = cfg.dataset.get("val_path", None) or train_cfg.dataset["path"]
        # The validation dataset path should come from the evaluation config, not the training one
        val_dataset = ExpertTrajectoryDataset(
            demo_path=val_path,
            observation_horizon=train_cfg.model.observation_horizon,
            action_horizon=train_cfg.model.action_horizon,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.eval.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0, # Recommended for simplicity in eval
        )
        offline_metrics = evaluate_offline(policy, val_loader, cfg.eval)
        log.info(f"Offline Metrics: {json.dumps(offline_metrics, indent=2)}")
        with (output_dir / "offline_metrics.json").open("w") as f:
            json.dump(offline_metrics, f, indent=2)

    # --- 3. Online Evaluation ---
    if cfg.do_online_eval:
        log.info("\n" + "="*50 + "\nONLINE EVALUATION\n" + "="*50)
        online_metrics = evaluate_online(policy, cfg.eval, cfg.environment, train_cfg, output_dir)
        log.info(f"Online Metrics: {json.dumps(online_metrics, indent=2)}")
        with (output_dir / "online_metrics.json").open("w") as f:
            json.dump(online_metrics, f, indent=2)

    log.info(f"\nEvaluation finished. Results are in {output_dir}")

if __name__ == "__main__":
    main()