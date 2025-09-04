# In new file: utils/controls.py
import numpy as np

def gripper_action_to_ctrl(
    gripper_action: float,
    actuator_range: tuple = (0.0, 255.0)
) -> float:
    """
    Maps a normalized expert gripper action to an actuator control value.

    This function provides a single source of truth for converting the abstract
    gripper command from the expert into a concrete value for the simulation,
    preventing inconsistencies and "magic numbers" in the control pipeline.

    Args:
        gripper_action: The desired gripper state, where -1.0 is fully open
                        and +1.0 is fully closed.
        actuator_range: A tuple (min, max) of the gripper actuator's control range.
                        For the Panda model used, this is typically (0, 255).

    Returns:
        The corresponding control value for the actuator, clipped to the valid range.
    """
    # Ensure the input is a valid number and clip it to the expected [-1, 1] range.
    grip_normalized = float(np.clip(gripper_action, -1.0, 1.0))
    
    # Get the min and max of the actuator's valid control range.
    lo, hi = actuator_range
    
    # Perform a linear mapping from [-1, 1] to [lo, hi].
    #   - If grip_normalized is -1.0, (grip + 1.0) / 2.0 = 0.0
    #   - If grip_normalized is  0.0, (grip + 1.0) / 2.0 = 0.5
    #   - If grip_normalized is +1.0, (grip + 1.0) / 2.0 = 1.0
    ctrl_value = lo + (hi - lo) * (grip_normalized + 1.0) / 2.0
    
    return ctrl_value