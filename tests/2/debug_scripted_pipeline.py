import numpy as np
import logging
import sys

# Make sure the project root is in the path to import project files
sys.path.append('.') 
from utils.scripted_expert import ScriptedExpert, ExpertConfig

# --- ANSI Color Codes for Pretty Printing ---
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_RESET = "\033[0m"

# --- Test Configuration ---
MAX_SIM_STEPS = 300  # Failsafe to prevent infinite loops
ROBOT_STEP_SIZE = 0.03  # How far the fake robot moves each step (m)

def run_expert_simulation_test():
    """
    Simulates a full pick-and-place task to test the ScriptedExpert's
    state machine logic in isolation, without any physics engine.
    """
    print(f"{C_BLUE}--- Starting Standalone ScriptedExpert Test ---{C_RESET}")
    
    # 1. --- SETUP ---
    print("\n[SETUP] Initializing expert and fake world state...")
    
    # Use the default robust configuration
    config = ExpertConfig() 
    expert = ScriptedExpert(cfg=config)
    expert.reset()
    
    # Define the initial state of our fake world
    initial_ee_pos = np.array([0.4, 0.2, 0.6])
    initial_cube_pos = np.array([0.6, 0.0, 0.42])
    goal_pos = np.array([0.6, 0.2, 0.42])
    
    # Our simulation variables
    current_ee_pos = initial_ee_pos.copy()
    current_cube_pos = initial_cube_pos.copy()
    
    # Fake robot state
    is_grasped = False
    
    print(f"[SETUP] Expert initialized. Start state: '{expert.get_state()}'")
    print(f"[SETUP] Initial EE Pos:      {current_ee_pos}")
    print(f"[SETUP] Initial Cube Pos:    {current_cube_pos}")
    print(f"[SETUP] Goal Pos:          {goal_pos}")
    print(f"[SETUP] Position Tolerance:  {config.pos_tolerance} m")
    
    # 2. --- SIMULATION LOOP ---
    print(f"\n{C_BLUE}--- Running Simulation Loop (max {MAX_SIM_STEPS} steps) ---{C_RESET}")
    
    for t in range(MAX_SIM_STEPS):
        print(f"\n{C_YELLOW}--- SIMULATION STEP {t:03d} ---{C_RESET}")
        
        # We need a full 7D pose for the expert, but only position matters for us
        current_ee_pose_7d = np.concatenate([current_ee_pos, [0, 1, 0, 0]])
        
        # A. Query the expert for its intended action
        print(f"  [QUERY] Current expert state: '{expert.get_state()}'")
        print(f"  [QUERY] Sending EE pos: {np.round(current_ee_pos, 4)}")
        print(f"  [QUERY] Sending Cube pos: {np.round(current_cube_pos, 4)}")
        
        target_pose_world, gripper_action = expert.get_target_pose(
            current_ee_pose_7d, 
            current_cube_pos, 
            goal_pos
        )
        target_pos = target_pose_world[:3]
        
        print(f"  [RESPONSE] Expert wants EE to go to: {np.round(target_pos, 4)}")
        print(f"  [RESPONSE] Expert gripper command: {gripper_action}")
        
        # B. Simulate the fake robot's physics and state changes
        print("  [SIMULATE] Updating fake robot state...")
        
        # Gripper simulation
        if gripper_action > 0 and not is_grasped:
            # Check if we should grasp the cube
            dist_to_cube = np.linalg.norm(current_ee_pos - current_cube_pos)
            if dist_to_cube < 0.05: # Grasp if close enough
                is_grasped = True
                print(f"  [SIMULATE] Gripper CLOSED. Cube is now grasped.")
        elif gripper_action < 0 and is_grasped:
            is_grasped = False
            print(f"  [SIMULATE] Gripper OPENED. Cube is now released.")

        # Arm movement simulation (P-controller)
        direction_to_target = target_pos - current_ee_pos
        dist_to_target = np.linalg.norm(direction_to_target)
        
        if dist_to_target > 1e-6: # Avoid division by zero
            # Move a fixed step size in the direction of the target
            movement = (direction_to_target / dist_to_target) * ROBOT_STEP_SIZE
            
            # Don't overshoot the target
            if dist_to_target < ROBOT_STEP_SIZE:
                current_ee_pos = target_pos
            else:
                current_ee_pos += movement
        
        print(f"  [SIMULATE] New EE pos: {np.round(current_ee_pos, 4)} (moved towards target)")

        # Cube movement simulation
        if is_grasped:
            # If grasped, cube moves with the end-effector
            current_cube_pos = current_ee_pos.copy()
            print(f"  [SIMULATE] Cube is attached, new pos: {np.round(current_cube_pos, 4)}")
        
        # C. Check for completion
        if expert.is_done():
            print(f"\n{C_GREEN}--- TEST PASSED ---{C_RESET}")
            print(f"Expert reached 'DONE' state in {t+1} steps.")
            return

    # 3. --- FAILURE CONDITION ---
    # This code only runs if the loop finishes without the expert being done
    print(f"\n{C_RED}--- TEST FAILED ---{C_RESET}")
    print(f"Simulation reached max steps ({MAX_SIM_STEPS}) but expert was not done.")
    print(f"The expert is likely stuck in the '{expert.get_state()}' state.")
    print("Check the logic and transition conditions for this state.")


if __name__ == "__main__":
    run_expert_simulation_test()