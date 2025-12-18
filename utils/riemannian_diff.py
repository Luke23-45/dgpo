import torch
import torch.nn.functional as F

def pose7d_to_matrix(pose_7d: torch.Tensor) -> torch.Tensor:
    """
    Converts 7D pose [x, y, z, qx, qy, qz, qw] to 4x4 SE(3) transformation matrix.
    
    Args:
        pose_7d: (B, 7) tensor
    Returns:
        matrix: (B, 4, 4) tensor
    """
    batch_size = pose_7d.shape[0]
    
    # Extract translation and quaternion
    pos = pose_7d[:, :3]
    quat = pose_7d[:, 3:]
    
    # Normalize quaternion to ensure valid rotation
    quat = F.normalize(quat, p=2, dim=-1)
    
    # Convert quaternion to rotation matrix
    # Formula: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    r00 = 1 - 2 * (qy*qy + qz*qz)
    r01 = 2 * (qx*qy - qz*qw)
    r02 = 2 * (qx*qz + qy*qw)
    
    r10 = 2 * (qx*qy + qz*qw)
    r11 = 1 - 2 * (qx*qx + qz*qz)
    r12 = 2 * (qy*qz - qx*qw)
    
    r20 = 2 * (qx*qz - qy*qw)
    r21 = 2 * (qy*qz + qx*qw)
    r22 = 1 - 2 * (qx*qx + qy*qy)
    
    # Assemble matrix
    matrix = torch.eye(4, device=pose_7d.device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    matrix[:, 0, 0] = r00
    matrix[:, 0, 1] = r01
    matrix[:, 0, 2] = r02
    
    matrix[:, 1, 0] = r10
    matrix[:, 1, 1] = r11
    matrix[:, 1, 2] = r12
    
    matrix[:, 2, 0] = r20
    matrix[:, 2, 1] = r21
    matrix[:, 2, 2] = r22
    
    matrix[:, :3, 3] = pos
    
    return matrix


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
    # Clamp for numerical stability
    theta = torch.acos(torch.clamp(0.5 * (trace - 1), -1 + 1e-6, 1 - 1e-6))
    
    # Handle singularity at theta ~ 0 (use Taylor expansion or limit)
    sin_theta = torch.sin(theta)
    
    # Avoid div by zero
    # scale = theta / (2 * sin_theta)
    # If theta is small, limit is 0.5
    scale = torch.where(
        torch.abs(sin_theta) < 1e-4,
        0.5 - (theta**2)/12, # Taylor expansion
        theta / (2 * sin_theta + 1e-6)
    )
    
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
    phase_scores: torch.Tensor      # (B, Num_Phases) from auxiliary head (logits)
) -> torch.Tensor:
    """
    Computes D_sigma: The Phase-Weighted Riemannian Divergence.
    """
    B, K, _ = pred_pose_chunk.shape
    
    # 1. Convert Quat+Pos to SE(3) Matrices
    T_pred = pose7d_to_matrix(pred_pose_chunk.reshape(-1, 7))      # (B*K, 4, 4)
    T_expert = pose7d_to_matrix(expert_pose_chunk.reshape(-1, 7))  # (B*K, 4, 4)
    
    # 2. Compute Relative Transform: T_rel = T_pred^{-1} @ T_expert
    T_inv_pred = torch.linalg.inv(T_pred)
    T_rel = T_inv_pred @ T_expert
    
    # 3. Compute Geodesic Twist (The Error in Tangent Space)
    twist_error = se3_log_map(T_rel) # (B*K, 6)
    
    # 4. Define Semantic Stiffness (Weighting Matrix)
    # Phase 0 (Approach): Loose on Rot (0.1), Strict on Pos (1.0)
    # Phase 1 (Grasp): Strict on Rot (5.0), Strict on Pos (5.0)
    # Phase 2 (Lift/Move): Strict on Z/Pos, Smooth on Rot
    # Phase 3 (Place): Strict on everything
    # Phase 4 (Retract): Loose
    
    # 5 Phases defined in SemanticPlannerConfig / Expert
    phase_weights = torch.tensor([
        [1.0, 1.0, 1.0, 0.1, 0.1, 0.1],  # Approach (0)
        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],  # Grasp (1) - Precision
        [1.0, 1.0, 5.0, 1.0, 1.0, 1.0],  # Lift/Move (2)
        [5.0, 5.0, 2.0, 5.0, 5.0, 5.0],  # Place (3)
        [1.0, 1.0, 1.0, 0.1, 0.1, 0.1],  # Retract (4)
    ], device=pred_pose_chunk.device)
    
    # Soft weighted sum based on phase logits
    # phase_scores: (B, N_phases)
    # Expand to (B*K, N_phases) -> simple repeat for chunk
    # (Assuming phase prediction is per-chunk single label, or per-step? 
    # SemanticPlanner outputs ONE phase logit vector per chunk usually, 
    # or per step? The architecture usually predicts 1 phase for the current state)
    
    probs = torch.softmax(phase_scores, dim=1) # (B, N_Phases)
    
    # [FIX] AMP Compatibility: Ensure weights match input dtype (e.g., float16)
    current_weights = (probs @ phase_weights.to(dtype=probs.dtype)) # (B, 6)
    
    # Expand to match B*K
    current_weights = current_weights.unsqueeze(1).repeat(1, K, 1).reshape(B*K, 6)
    
    # 5. Compute Weighted Energy
    # E = sum( (w_i * twist_i)^2 )
    divergence = torch.sum(current_weights * (twist_error ** 2), dim=-1) # (B*K)
    
    return divergence.view(B, K).mean(dim=1) # Mean over chunk -> (B,)
