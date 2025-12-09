
import numpy as np
from typing import Tuple

class RunningMeanStd:
    """
    Welford's online algorithm for computing running mean and std.
    Used for observation normalization (SOTA PPO best practice).
    """
    
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon  # Avoid division by zero
    
    def update(self, x: np.ndarray):
        """Update running statistics with new observation."""
        batch_mean = np.mean(x, axis=0) if x.ndim > 1 else x
        batch_var = np.var(x, axis=0) if x.ndim > 1 else np.zeros_like(x)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)
