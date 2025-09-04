# In file: utils/controls.py
import numpy as np
import time
from typing import Tuple

def gripper_action_to_ctrl(action: float) -> float:
    """
    Simple helper to convert a gripper action from [-1, 1] to a MuJoCo
    control signal, assuming a simple binary open/close mechanism.
    -1 (or less) -> Open (e.g., -1 ctrl)
    +1 (or more) -> Close (e.g., +1 ctrl)
    """
    return -1.0 if action < 0 else 1.0

# ----------------- NEW PID CONTROLLER CLASS -----------------

class PIDController:
    """
    A robust Proportional-Integral-Derivative (PID) controller.

    This class implements a standard PID control loop. It is designed to be
    used for each degree of freedom (e.g., each robot joint) independently.
    It includes an anti-windup mechanism for the integral term to prevent
    runaway behavior when the output is saturated.
    """
    def __init__(self, Kp: float, Ki: float, Kd: float, output_limits: Tuple[float, float] = (-1.0, 1.0)):
        """
        Initializes the PID controller.
        Args:
            Kp: Proportional gain.
            Ki: Integral gain.
            Kd: Derivative gain.
            output_limits: Tuple of (min_output, max_output).
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.output_limits = output_limits
        self.reset()

    def reset(self):
        """Resets the controller's state for a new episode."""
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.time()

    def compute(self, error: float) -> float:
        """
        Computes the control output for a given error.
        Args:
            error: The current error (target - actual).
        Returns:
            The computed control signal, clamped within the output limits.
        """
        current_time = time.time()
        dt = current_time - self._last_time
        
        # Avoid division by zero or stale dt on the first step
        if dt <= 1e-6:
            # On the first step, derivative is zero
            derivative = 0.0
        else:
            derivative = (error - self._last_error) / dt

        # Proportional term
        P = self.Kp * error

        # Integral term (with anti-windup)
        self._integral += error * dt
        # Clamp the integral to prevent it from growing uncontrollably
        self._integral = np.clip(self._integral, self.output_limits[0] / (self.Ki + 1e-6), self.output_limits[1] / (self.Ki + 1e-6))
        I = self.Ki * self._integral

        # Derivative term
        D = self.Kd * derivative

        # Compute total output
        output = P + I + D
        
        # Clamp the final output to ensure it's within bounds
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        # Store state for the next iteration
        self._last_error = error
        self._last_time = current_time
        
        return output