This is an in-depth research report and architectural proposal for **DGPO v2.0 (Dense Geodesic Policy Optimization)**. 

Based on the analysis of your codebase and recent literature (Riemannian Motion Policies, Action-Chunked RL), the previous implementation contained fundamental mathematical contradictions that would prevent high-level performance.

Below is the research report detailing the **New Mathematical Definition of Divergence** and the **Corrected Algorithm**.

---

# Research Report: DGPO v2.0 (Semantic-Geodesic Evolution)

## 1. The Core Failure of v1.0: "The Echo Chamber"
The previous implementation defined reward as `Task_Reward - Divergence`. In Reinforcement Learning (RL), this creates a catastrophic feedback loop:
*   **Scenario:** The policy deviates from the expert.
*   **Result:** Reward drops. The advantage ($A$) becomes negative.
*   **Update:** PPO discourages this action.
*   **The Trap:** If the policy *agrees* with the expert, it gets a high reward. If it *disagrees*, it gets a low reward. The policy effectively learns to **never explore**. It becomes a weaker version of Behavior Cloning because it never learns *why* the expert took that action, only that deviating is painful. It cannot discover strategies *better* or *more robust* than the expert.

## 2. The New Solution: Riemannian Semantic Divergence (RSD)

We need a divergence metric that is **geometry-aware** (handling 3D rotation manifolds correctly) and **phase-aware** (knowing that 1mm of error matters during "Grasping" but not during "Approach").

### A. Mathematical Definition
Let the robot state be defined on the manifold $\mathcal{M} = SE(3) \times \mathbb{R}$ (Pose + Gripper).
The standard Euclidean distance $||x - y||^2$ is mathematically invalid for rotations.

We define the **Riemannian Semantic Divergence** $D_{\Sigma}$ between the Policy Chunk $\tau_\pi$ and Expert Chunk $\tau_e$ as:

$$ D_{\Sigma}(\tau_\pi, \tau_e) = \frac{1}{K} \sum_{t=0}^{K} \mathbf{W}_{\phi(s_t)} \cdot \mathcal{E}_{geo}(\mathbf{T}_{\pi, t}, \mathbf{T}_{e, t}) $$

Where:
1.  **$\mathbf{T}$** is the $SE(3)$ transformation matrix of the end-effector.
2.  **$\mathcal{E}_{geo}$** is the **Geodesic Energy** (Shortest path on the manifold):
    $$ \mathcal{E}_{geo}(\mathbf{T}_1, \mathbf{T}_2) = || \log(\mathbf{T}_1^{-1} \mathbf{T}_2) ||^2_{\mathcal{F}} $$
    *Here, $\log$ maps the relative transform to the tangent space $\mathfrak{se}(3)$ (twist coordinates), handling rotation and translation in a unified metric.*
3.  **$\mathbf{W}_{\phi(s_t)}$** is the **Semantic Stiffness Tensor**, conditioned on the current task phase $\phi$:
    *   **Approach Phase:** Low Stiffness (Exploration allowed).
    *   **Grasp Phase:** Infinite Stiffness (Strict imitation required).
    *   **Transport Phase:** Medium Stiffness (Trajectory smoothness).

### B. The New Reward Formulation (DAPG-Style)
Instead of a simple penalty, we use a **Gated Guiding Potential**. The dense reward is only given if the policy makes *task progress*.

$$ r_{dense}(s, a) = r_{task}(s) + \lambda \cdot \mathbb{I}[\text{Success is Possible}] \cdot \exp\left( - \frac{D_{\Sigma}(\pi, e)}{\sigma^2} \right) $$

This prevents the "Echo Chamber": The expert guidance acts as a **magnetic field** that pulls the policy towards the manifold of valid solutions, but the *magnitude* of the update is controlled by the Value function ($V$), which estimates long-term success.

---

## 3. Algorithm: Action-Chunked PPO (AC-PPO)

Standard PPO fails with Action Chunking because it assumes 1 action per 1 step. We must reformulate the PPO loss for **Trajectory Chunks**.

### The "Blind Critic" Fix
The previous Critic only saw joint angles. The new Critic must be **Visual** to understand the goal.
*   **Input:** `[Vision_Features(t), Proprio(t), Goal_Embedding]`
*   **Output:** $V(s_t)$, representing the expected return of the *entire chunk*.

### The Chunked Update Rule
Instead of updating discrete actions, we update the **Macro-Action Distribution**.
The policy outputs a mean $\mu$ and variance $\Sigma$ for the entire trajectory chunk of length $K$.

$$ \mathcal{L}_{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right] $$

*   **Key Change:** The ratio $r_t(\theta)$ is calculated over the **joint probability** of the chunk:
    $$ r_t(\theta) = \frac{\prod_{i=0}^K \pi_\theta(a_{t+i} | s_t)}{\prod_{i=0}^K \pi_{old}(a_{t+i} | s_t)} $$

---

## 4. Implementation (Optimized Code)

Here is the updated, research-grade implementation logic.

### File: `utils/riemannian_diff.py` (New Mathematical Core)

```python
import torch
import numpy as np

def se3_log_map(T_relative: torch.Tensor) -> torch.Tensor:
    """
    Computes the Riemannian Log map on SE(3) (Lie Algebra).
    Maps a relative transformation matrix (4x4) to a twist vector (6D).
    
    Args:
        T_relative: (B, 4, 4) tensor representing T_pred^{-1} @ T_expert
    Returns:
        twist: (B, 6) tensor [v_x, v_y, v_z, w_x, w_y, w_z]
    """
    # Extract rotation R and translation p
    R = T_relative[:, :3, :3]
    p = T_relative[:, :3, 3]
    
    # 1. Rotation Log map (So(3) -> so(3)) using Rodrigues' formula
    # trace(R) = 1 + 2cos(theta)
    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
    theta = torch.acos(torch.clamp(0.5 * (trace - 1), -1 + 1e-6, 1 - 1e-6))
    
    # Handle singularity at theta ~ 0 (use Taylor expansion or limit)
    sin_theta = torch.sin(theta)
    # Avoid div by zero
    scale = theta / (2 * sin_theta + 1e-6)
    scale = torch.where(theta < 1e-4, 0.5 - theta**2/12, scale)
    
    # Skew-symmetric part: (R - R.T)
    w_skew = scale.unsqueeze(-1).unsqueeze(-1) * (R - R.transpose(-2, -1))
    
    # Extract vector w from skew matrix
    w = torch.stack([
        w_skew[:, 2, 1], 
        w_skew[:, 0, 2], 
        w_skew[:, 1, 0]
    ], dim=-1)
    
    # 2. Translation map (approximated for optimization stability as linear error)
    # The exact exponential map coupling is complex; standard robotics practice 
    # uses decoupled linear + angular error for gradients.
    v = p 
    
    return torch.cat([v, w], dim=-1)

def compute_riemannian_divergence(
    pred_pose_chunk: torch.Tensor,  # (B, K, 7) [pos, quat]
    expert_pose_chunk: torch.Tensor, # (B, K, 7)
    phase_scores: torch.Tensor      # (B, Num_Phases) from auxiliary head
) -> torch.Tensor:
    """
    Computes D_sigma: The Phase-Weighted Riemannian Divergence.
    """
    B, K, _ = pred_pose_chunk.shape
    
    # 1. Convert Quat+Pos to SE(3) Matrices
    T_pred = pose7d_to_matrix(pred_pose_chunk.view(-1, 7))      # (B*K, 4, 4)
    T_expert = pose7d_to_matrix(expert_pose_chunk.view(-1, 7))  # (B*K, 4, 4)
    
    # 2. Compute Relative Transform: T_rel = T_pred^{-1} @ T_expert
    T_inv_pred = torch.linalg.inv(T_pred)
    T_rel = T_inv_pred @ T_expert
    
    # 3. Compute Geodesic Twist (The Error in Tangent Space)
    twist_error = se3_log_map(T_rel) # (B*K, 6)
    
    # 4. Define Semantic Stiffness (Weighting Matrix)
    # Phase 0 (Approach): Loose on Rot (0.1), Strict on Pos (1.0)
    # Phase 1 (Grasp): Strict on Rot (5.0), Strict on Pos (5.0)
    # We use a differentiable lookup based on predicted phase
    phase_weights = torch.tensor([
        [1.0, 1.0, 1.0, 0.1, 0.1, 0.1],  # Approach
        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],  # Grasp (Precision)
        [1.0, 1.0, 5.0, 1.0, 1.0, 1.0],  # Lift (Z-axis strict)
    ], device=pred_pose_chunk.device)
    
    # Soft weighted sum based on phase logits
    # phase_scores: (B, N_phases)
    # Expand to (B*K, N_phases) -> simple repeat for chunk
    current_weights = (torch.softmax(phase_scores, dim=1) @ phase_weights) # (B, 6)
    current_weights = current_weights.unsqueeze(1).repeat(1, K, 1).view(B*K, 6)
    
    # 5. Compute Weighted Energy
    # E = (w * twist)^2
    divergence = torch.sum(current_weights * (twist_error ** 2), dim=-1) # (B*K)
    
    return divergence.view(B, K).mean(dim=1) # Mean over chunk
```

### File: `train/train_dgpo.py` (Corrected Architecture)

The critical updates for the Trainer to use this new metric and fix the "Blind Critic":

```python
class VisionCritic(nn.Module):
    """
    SOTA Critic: Sees what the Actor sees.
    """
    def __init__(self, vision_feature_dim, proprio_dim, hidden_dim=256):
        super().__init__()
        # Project frozen visual features
        self.vis_proj = nn.Linear(vision_feature_dim, hidden_dim)
        self.prop_proj = nn.Linear(proprio_dim, hidden_dim)
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1) # Value scalar
        )
        
    def forward(self, visual_emb, proprio):
        # visual_emb: (B, D) from SigLIP (fused)
        v = self.vis_proj(visual_emb)
        p = self.prop_proj(proprio)
        return self.net(torch.cat([v, p], dim=-1))

# In DGPOTrainer.update_policy():

def update_policy(self):
    # ... inside batch loop ...
    
    # 1. Get Action Distribution (Chunk)
    # The policy outputs a chunk of poses. We treat this as a Gaussian.
    # We need the log_prob of the *entire chunk* taking into account correlations?
    # For simplicity/speed, we assume diagonal independence across steps in the chunk 
    # (standard in Action Chunking papers).
    
    policy_out = self.policy(batch) # Returns pose_chunk (B, K, 7)
    
    # 2. Compute Riemannian Semantic Divergence (RSD)
    # Note: We compute this here for the *Loss weighting*, or we used it in the 
    # rollout to shape the Reward.
    # In PPO, we use the Advantage computed from the rollout.
    
    # 3. Action Chunking PPO Loss
    # PPO Ratio must be: prob(chunk_new) / prob(chunk_old)
    # log_prob_chunk = sum(log_prob_steps)
    
    dist_new = torch.distributions.Normal(policy_out['pose_chunk'], self.log_std.exp())
    log_prob_new = dist_new.log_prob(batch_actions).sum(dim=[1, 2]) # Sum over K steps and dims
    
    ratio = torch.exp(log_prob_new - batch_old_log_probs)
    
    # Standard PPO Clip
    surr1 = ratio * batch_advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 4. Semantic Auxiliary Loss (Critical for RSD)
    # Force the visual encoder to understand phases so the stiffness weights are correct
    phase_loss = F.cross_entropy(policy_out['phase_logits'], batch_expert_phases)
    
    total_loss = policy_loss + 0.5 * value_loss + 0.1 * phase_loss
```

## 5. Deployment Strategy
1.  **Replace** `utils/divergence.py` with the Riemannian implementation.
2.  **Upgrade** `ValueNetwork` to `VisionCritic`.
3.  **Refactor** `collect_rollouts` to store and calculate `log_probs` for the *entire chunk*.
4.  **Train** using the new **Phase-Weighted** reward signal.

This upgrade shifts the algorithm from "naive matching" to **"Semantic Manifold Guidance"**, aligning with the highest standards of robotic control theory.