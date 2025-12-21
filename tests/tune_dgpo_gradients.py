import os
import sys
import torch
import hydra
import logging
import numpy as np
import copy
from typing import Dict, List, Tuple
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.append(os.getcwd())

from train.train_dgpo_robust import DGPOTrainer

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("GRAD_TUNE_FINAL")

class DynamicsTuner:
    """
    SOTA Meta-Optimizer v2.0 (Robust).
    
    Features:
    - Static Gradient Norm Matching (Analytical Baseline)
    - Gradient Conflict Detection (Cosine Similarity)
    - Empirical Dynamics Simulation (Parallel Universe Testing)
    - Robust State Handling (No pickling errors)
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # Ensure determinitic behavior
        torch.manual_seed(42)
        np.random.seed(42)
        
    def _get_gradient_stats(self, trainer: DGPOTrainer) -> Dict[str, float]:
        """Calculates norm and alignment of different loss components."""
        # 1. Collect one batch
        trainer.collect_rollouts(n_steps=trainer.cfg.training.steps_per_iter)
        
        # 2. Re-construct batch tensors
        if len(trainer.buffer) == 0:
            return {"ratio": 1.0, "bc_norm": 0.0, "kl_norm": 0.0, "cosine_sim": 0.0}
            
        indices = np.arange(len(trainer.buffer))
        batch_idx = indices[:16] # Small batch for gradients
        
        prev_imgs = np.stack([trainer.buffer.prev_images[i] for i in batch_idx])
        curr_imgs = np.stack([trainer.buffer.curr_images[i] for i in batch_idx])
        goal_imgs = np.stack([trainer.buffer.goal_images[i] for i in batch_idx])
        proprios = np.stack([trainer.buffer.proprios[i] for i in batch_idx])
        
        batch = trainer._prepare_batch_vectorized(prev_imgs, curr_imgs, goal_imgs, proprios)
        
        # 3. Measure BC Gradient Vector
        trainer.policy.zero_grad()
        policy_out = trainer.policy(batch)
        pred_chunks = policy_out['pose_chunk']
        b_expert_chunks = torch.stack([torch.from_numpy(trainer.buffer.expert_pose_chunks[i]) for i in batch_idx]).to(trainer.device)
        
        loss_bc = torch.nn.functional.mse_loss(pred_chunks[:, 0, :], b_expert_chunks[:, 0, :])
        loss_bc.backward(retain_graph=True)
        
        grad_bc_vec = []
        grad_bc_norm = 0.0
        for p in trainer.policy.parameters():
            if p.grad is not None:
                g_flat = p.grad.view(-1)
                grad_bc_vec.append(g_flat)
                grad_bc_norm += g_flat.norm(2).item() ** 2
        grad_bc_norm = grad_bc_norm ** 0.5
        grad_bc_cat = torch.cat(grad_bc_vec) if grad_bc_vec else torch.tensor([]).to(trainer.device)
        
        # 4. Measure KL Gradient Vector (Simulated max penalty)
        trainer.policy.zero_grad()
        dist = torch.distributions.Normal(pred_chunks, trainer.chk_log_std.exp())
        b_act = torch.stack([torch.from_numpy(trainer.buffer.action_chunks[i]) for i in batch_idx]).to(trainer.device)
        log_prob = dist.log_prob(b_act).sum(dim=[1, 2])
        
        # Simulate: Maximize LogProb (PPO Objective) vs Minimize KL (Penalty)
        # KL Gradient is approx: grad(log_p) * (log_p - log_q).
        # We model the Force of the penalty: "Don't change log probs".
        # Force = Beta * KL_grad.
        target_kl_force = log_prob.mean() * 10.0 * 0.02 
        target_kl_force.backward()
        
        grad_kl_vec = []
        grad_kl_norm = 0.0
        for p in trainer.policy.parameters():
            if p.grad is not None:
                g_flat = p.grad.view(-1)
                grad_kl_vec.append(g_flat)
                grad_kl_norm += g_flat.norm(2).item() ** 2
        grad_kl_norm = grad_kl_norm ** 0.5
        grad_kl_cat = torch.cat(grad_kl_vec) if grad_kl_vec else torch.tensor([]).to(trainer.device)
        
        # 5. Cosine Similarity
        cosine_sim = 0.0
        if len(grad_bc_cat) > 0 and len(grad_kl_cat) > 0:
            cosine_sim = torch.nn.functional.cosine_similarity(grad_bc_cat.unsqueeze(0), grad_kl_cat.unsqueeze(0)).item()
        
        return {
            "bc_norm": grad_bc_norm,
            "kl_norm": grad_kl_norm,
            "ratio": grad_kl_norm / (grad_bc_norm + 1e-9),
            "cosine_sim": cosine_sim
        }

    def simulate_dynamics(self, candidates: List[float], trainer: DGPOTrainer, steps: int = 5) -> Dict[float, Dict]:
        """
        Runs a 'Parallel Universe' simulation for each candidate.
        Fix: Avoids deepcopying the Trainer. Operations are done on Policy state dicts.
        """
        results = {}
        
        print("\n 🌀 STARTING DYNAMICS SIMULATION")
        print(f"    Testing candidates: {candidates}")
        print("-" * 65)
        print(f" {'Coef':<10} | {'BC Loss Δ (Imp)':<15} | {'Ent Δ':<10} | {'Status'}")
        print("-" * 65)
        
        # 1. Extract STATIC batch (The "Territory" Snapshot)
        indices = np.arange(len(trainer.buffer))
        batch_idx = indices[:16] # Small batch
        
        prev_imgs = np.stack([trainer.buffer.prev_images[i] for i in batch_idx])
        curr_imgs = np.stack([trainer.buffer.curr_images[i] for i in batch_idx])
        goal_imgs = np.stack([trainer.buffer.goal_images[i] for i in batch_idx])
        proprios = np.stack([trainer.buffer.proprios[i] for i in batch_idx])
        
        batch = trainer._prepare_batch_vectorized(prev_imgs, curr_imgs, goal_imgs, proprios)
        b_expert = torch.stack([torch.from_numpy(trainer.buffer.expert_pose_chunks[i]) for i in batch_idx]).to(trainer.device)
        
        # 2. Base Policy State (Save to CPU to avoid VRAM issues)
        base_policy_state = {k: v.cpu().clone() for k,v in trainer.policy.state_dict().items()}
        
        for coef in candidates:
            # Fork Universe
            # Restore state to GPU
            trainer.policy.load_state_dict(base_policy_state)
            
            # Reset Optimizer
            optimizer = torch.optim.Adam(
                trainer.policy.parameters(), lr=self.cfg.optimizer.policy_lr
            )
            
            initial_bc = 0.0
            final_bc = 0.0
            initial_ent = 0.0
            final_ent = 0.0
            
            valid_run = True
            
            try:
                # Training Loop
                for step in range(steps):
                    update_stats = self._Mock_update_step(
                        trainer.policy, optimizer, batch, b_expert, 
                        bc_coef=coef, chk_log_std=trainer.chk_log_std
                    )
                    
                    if step == 0:
                        initial_bc = update_stats['bc_loss']
                        initial_ent = update_stats['entropy']
                    final_bc = update_stats['bc_loss']
                    final_ent = update_stats['entropy']
                    
            except Exception as e:
                valid_run = False
                log.error(f"Trial {coef} failed: {e}")
            
            # Analysis
            bc_improv = (initial_bc - final_bc) / (initial_bc + 1e-9) * 100
            ent_drift = final_ent - initial_ent
            
            is_stable = valid_run and abs(ent_drift) < 2.0 
            status = "✅ Stable" if is_stable else "❌ Unstable"
            if bc_improv < 1.0: status = "⚠️ Too Slow"
            
            print(f" {coef:<10.1f} | {bc_improv:<15.2f}% | {ent_drift:<10.4f} | {status}")
            
            results[coef] = {
                "bc_improvement": bc_improv,
                "entropy_drift": ent_drift,
                "stable": is_stable
            }
            
            torch.cuda.empty_cache()
            
        # Restore original state finally
        trainer.policy.load_state_dict(base_policy_state)
        return results

    def _Mock_update_step(self, policy, optimizer, batch, b_expert, bc_coef, chk_log_std):
        """Minimal update loop enforcing the bc_coef."""
        policy.train()
        
        policy_out = policy(batch)
        pred_chunks = policy_out['pose_chunk']
        
        loss_bc = torch.nn.functional.mse_loss(pred_chunks[:, 0, :], b_expert[:, 0, :])
        
        dist = torch.distributions.Normal(pred_chunks, chk_log_std.exp())
        entropy = dist.entropy().mean()
        
        # Total Loss: BC Weighted - Entropy (Standard PPO term)
        loss = bc_coef * loss_bc - 0.01 * entropy
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        
        return {"bc_loss": loss_bc.item(), "entropy": entropy.item()}


@hydra.main(config_path="../configs", config_name="train_dgpo_config", version_base="1.2")
def main(cfg: DictConfig):
    print("\n==================================================================")
    print(" 🧪  SOTA ROBUST GRADIENT TUNER v2.0 (Deep Research Edition)")
    print("==================================================================\n")
    
    # Fast mode
    cfg.num_envs = 2 
    cfg.training.steps_per_iter = 64
    cfg.training.use_amp = False 
    
    # Init
    tuner = DynamicsTuner(cfg)
    base_trainer = DGPOTrainer(cfg) # Initialize once
    
    # Phase 1: Analytical (Map)
    print(" [1/2] Phase 1: Analytical Gradient Analysis & Conflict Detection")
    stats = tuner._get_gradient_stats(base_trainer)
    analytical_ratio = stats['ratio']
    
    print(f"   ► BC Grad Norm: {stats['bc_norm']:.6f}")
    print(f"   ► KL Grad Norm: {stats['kl_norm']:.6f}")
    print(f"   ► Analytical Ratio: {analytical_ratio:.2f}")
    print(f"   ► Cosine Similarity: {stats['cosine_sim']:.4f}")
    
    if stats['cosine_sim'] < 0:
        print("   ⚠️  WARNING: Gradients are conflicting! Consider PCGrad optimizer if tuning fails.")
    
    # Phase 2: Empirical (Territory)
    print("\n [2/2] Phase 2: Empirical Dynamics Simulation (Parallel Universes)")
    
    center = analytical_ratio
    # Widen the search space for robustness
    candidates = [
        normalize_coef(center * 0.1),
        normalize_coef(center * 0.5),
        normalize_coef(center * 1.0),
        normalize_coef(center * 5.0), 
        normalize_coef(center * 20.0) 
    ]
    candidates = sorted(list(set(candidates)))
    
    # Pass trainer explicitly
    results = tuner.simulate_dynamics(candidates, trainer=base_trainer, steps=10)
    
    # Decision Logic
    best_coef = None
    best_score = -999.0
    
    for coef, res in results.items():
        if not res['stable']: continue
        
        # Score = BC_Improvement - Entropy_Drift_Penalty
        score = res['bc_improvement'] - abs(res['entropy_drift']) * 5.0
        
        if score > best_score:
            best_score = score
            best_coef = coef
            
    print("\n" + "="*66)
    print(" 🏆 FINAL VERDICT")
    print("="*66)
    
    if best_coef:
        print(f" ✅ OPTIMAL BC COEFFICIENT: {best_coef:.1f}")
        print(f"    (Analytical Prediction was: {analytical_ratio:.1f})")
        print(f"    This value yields {results[best_coef]['bc_improvement']:.1f}% improvement per 10 steps")
        print(f"    while maintaining policy stability.")
    else:
        print(" ⚠️  No stable coefficient found. Check Learning Rate or Data.")
        print(f"    Fallback to Analytical: {analytical_ratio:.1f}")
        
    print("="*66 + "\n")

    # Clean up
    if hasattr(base_trainer, 'envs'):
        base_trainer.envs.close()

def normalize_coef(val):
    """Rounds to nice human readable numbers."""
    if val <= 0: return 1.0
    order = 10 ** np.floor(np.log10(val))
    norm = val / order
    if norm < 1.75: norm = 1.0
    elif norm < 3.75: norm = 2.5
    elif norm < 6.25: norm = 5.0
    elif norm < 8.75: norm = 7.5
    else: norm = 10.0
    return norm * order

if __name__ == "__main__":
    main()
