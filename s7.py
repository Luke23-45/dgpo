# In file: debug_scripts/verify_action_translator.py

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from run_experiment import setup_environment
from argparse import Namespace
# This needs to be imported to check for the VecNormalize wrapper
from stable_baselines3.common.vec_env import VecNormalize

# FILE: debug_scripts/verify_action_translator.py
# FILE: debug_scripts/verify_action_translator.py

# FILE: debug_scripts/verify_action_translator.py

def find_proprio_joint_slice_from_obs(obs, expected_joints=7):
    # This function is now corrected for logging purposes.
    # The actual wrapper no longer uses a heuristic.
    proprio = np.asarray(obs["proprio"])
    if proprio.ndim == 1:
        proprio = np.expand_dims(proprio, 0)
    
    # In our environment, the joint positions are always the first 7 elements.
    # This correctly returns slice(0, 7, None).
    return slice(0, expected_joints)


def main():
    print("--- Verification for AbsoluteJointToDeltaJointWrapper ---")
    # Mock arguments
    args = Namespace(
        xml_path="envs/panda_pick_place.xml",
        w_plausibility=0.0, w_guidance=0.0, w_guidance_dense=0.0,
        grasp_reward=50, lift_reward=100, success_reward=250,
        pos_scale=0.05, rot_scale=1.0, div_clip=10.0, guidance_clip=1.0
    )

    print("\n[1/3] Initializing environment stack...")
    # NOTE: Your setup_environment function must now be updated to pass
    # args.pos_scale into the AbsoluteJointToDeltaJointWrapper's `action_scaling` parameter.
    env = setup_environment(
        n_envs=1,
        seed=42,
        xml_path=args.xml_path,
        w_plausibility=args.w_plausibility,
        w_guidance=args.w_guidance,
        w_guidance_dense=args.w_guidance_dense,
        grasp_reward=args.grasp_reward,
        lift_reward=args.lift_reward,
        success_reward=args.success_reward,
        pos_scale=args.pos_scale,
        rot_scale=args.rot_scale,
        div_clip=args.div_clip,
        guidance_clip=args.guidance_clip
    )
    print("✅ Environment created.")

    def _find_vecnormalize_in_stack(env):
        curr = env
        while True:
            if hasattr(curr, "get_original_obs"): # More robust check for VecNormalize
                return True
            if hasattr(curr, "venv"):
                curr = curr.venv
            elif hasattr(curr, "env"):
                curr = curr.env
            else:
                break
        return False
    env.unwrapped.envs[0].unwrapped.debug_instant_move = True
    print("✅ Enabled instant move (teleport) mode for verification.")

    is_normalized = _find_vecnormalize_in_stack(env)
    print(f"Environment is normalized: {is_normalized}")

    obs = env.reset()
    obs_phys = env.get_original_obs() if is_normalized else obs
    expected_joints = env.action_space.shape[-1] - 1
    joint_slice = find_proprio_joint_slice_from_obs(obs_phys, expected_joints=expected_joints)
    print(f"Detected joint_slice = {joint_slice} for expected_joints={expected_joints}")

    # --- Test 1: Stay Still ---
    print("\n[2/3] Testing 'stay still' command for 10 steps...")
    # Get the very first physical observation
    initial_obs_phys = env.get_original_obs() if is_normalized else obs
    initial_ee_pos = initial_obs_phys["proprio"][0][:3]

    for _ in range(10):
        # ALWAYS use the LATEST unnormalized observation to calculate the action
        current_obs_phys = env.get_original_obs() if is_normalized else obs
        current_qpos = current_obs_phys["proprio"][0, joint_slice]
        
        stay_still_action = np.zeros(env.action_space.shape, dtype=np.float32)
        stay_still_action[:expected_joints] = current_qpos # Command to stay at the CURRENT position
        stay_still_action[expected_joints:] = -1.0

        # Take the step, which updates the internal state and returns the NEW observation
        obs, _, _, _ = env.step(np.expand_dims(stay_still_action, 0))

    # After the loop, get the final physical state
    final_obs_phys = env.get_original_obs() if is_normalized else obs
    final_ee_pos = final_obs_phys["proprio"][0][:3]
    drift = np.linalg.norm(final_ee_pos - initial_ee_pos)
    print(f"Total EE drift after 10 steps: {drift:.6f} meters")
    assert drift < 1.5e-2, f"FAIL: Significant drift detected ({drift})"
    print("✅ 'Stay still' OK.")

    # --- Test 2: Maximum Move ---
    print("\n[3/3] Testing maximum possible movement command...")
    obs = env.reset()
    # Get the initial physical state BEFORE the move
    initial_obs_phys_move_test = env.get_original_obs() if is_normalized else obs
    initial_qpos_move_test = initial_obs_phys_move_test["proprio"][0, joint_slice]
    
    # Command a move exactly equal to the action scaling factor.
    max_possible_move = args.pos_scale
    target_qpos = initial_qpos_move_test.copy()
    target_qpos[0] += max_possible_move
    
    move_action = np.zeros(env.action_space.shape, dtype=np.float32)
    move_action[:expected_joints] = target_qpos
    move_action[expected_joints:] = 0.0
    print(f"\n--- DEBUG: VERIFICATION SCRIPT (BEFORE STEP) ---")
    print(f"  - Initial Physical Position [Joint 0]: {initial_qpos_move_test[0]:.6f}")
    print(f"  - Commanded Absolute Target [Joint 0]: {target_qpos[0]:.6f}")
    print(f"--------------------------------------------------\n")
    # Take the step
    obs_after_move, _, _, _ = env.step(np.expand_dims(move_action, 0))
    
    # Get the final physical state AFTER the move
    obs_after_move_phys = env.get_original_obs() if is_normalized else obs_after_move
    final_qpos = obs_after_move_phys["proprio"][0, joint_slice]
    
    # Correctly calculate the actual move
    actual_move = final_qpos[0] - initial_qpos_move_test[0]
    # ^^^^^^ END OF FIX 1 ^^^^^^
    print(f"\n--- DEBUG: VERIFICATION SCRIPT (AFTER STEP) ---")
    print(f"  - Final Physical Position [Joint 0]: {final_qpos[0]:.6f}")
    print(f"  - Calculated Actual Physical Move: {actual_move:.6f}")
    print(f"-------------------------------------------------\n")
    print(f"Commanded +{max_possible_move:.4f} rad on joint 0. Actual move in one step: {actual_move:.6f} rad")
    # Assert that the actual move is very close to the commanded maximum move.
    assert np.allclose(actual_move, max_possible_move, atol=5e-3), "FAIL: Move was not close to the expected maximum."
    print("✅ Maximum movement OK.")

    env.close()
    print("\n--- VERIFICATION COMPLETE ---")

if __name__ == "__main__":
    main()