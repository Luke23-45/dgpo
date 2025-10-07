import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv
from utils.expert_dataset import ExpertDataset
from models.residual_policy import AdaptiveResidualPolicy

def bootstrap_replay_buffer(
    buffer: RolloutBuffer,
    env: VecEnv,
    policy: AdaptiveResidualPolicy,
    expert_dataset: ExpertDataset,
    num_samples: int,
    batch_size: int = 64
):
    """
    Pre-fills the RolloutBuffer with expert data, calculating values and log-probs.
    """
    print(f"Bootstrapping replay buffer with {num_samples} expert samples...")
    
    data_loader = DataLoader(expert_dataset, batch_size=batch_size, num_workers=1)
    samples_added = 0
    
    # Ensure policy is in eval mode and on the correct device
    policy.eval()
    device = policy.device
    
    pbar = tqdm(total=num_samples, desc="Bootstrapping Buffer")
    
    for expert_obs, expert_actions in data_loader:
        if samples_added >= num_samples:
            break
            
        # Move data to device and ensure correct format
        current_batch_size = expert_actions.shape[0]
        obs_tensor = {k: v.to(device) for k, v in expert_obs.items()}
        expert_actions = expert_actions.to(device)
        
        with torch.no_grad():
            # Get values and log-probs for the expert actions from the policy
            values, log_probs, _ = policy.evaluate_actions(obs_tensor, expert_actions)

        # Format data for buffer (needs to be numpy on CPU)
        obs_cpu = {k: v.cpu().numpy() for k, v in obs_tensor.items()}
        # SB3 expects obs for a VecEnv to have a batch dimension. We need to handle this.
        # We will add samples one by one.
        
        actions_cpu = expert_actions.cpu().numpy()
        values_cpu = values.cpu().numpy().flatten()
        log_probs_cpu = log_probs.cpu().numpy()
        
        for i in range(current_batch_size):
            if samples_added >= num_samples:
                break

            # Add data to the buffer, simulating a single-step trajectory
            buffer.add(
                obs={k: v[i] for k,v in obs_cpu.items()},
                action=actions_cpu[i],
                reward=np.array(0.0), # Reward is irrelevant for bootstrap, will be re-calculated
                episode_start=np.array(True),
                value=values_cpu[i],
                log_prob=log_probs_cpu[i],
            )
            samples_added += 1
            pbar.update(1)

    pbar.close()
    print("Bootstrap complete.")