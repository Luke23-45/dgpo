
import sys
import os
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.dgpo_env_wrapper import DGPOEnvWrapper
from envs.panda_env import PandaEnv

def test_blending_math():
    print("="*60)
    print("🧪 DGPO Blending Logic Verification")
    print("="*60)

    # 1. Setup Mock Config
    cfg = OmegaConf.create({
        "environment": {
            "xml_path": "envs/panda_pick_place.xml",
            "urdf_path": "urdf/panda_mujoco_kinematics.urdf"
        },
        "expert": {
            "object_size": [0.04, 0.04, 0.04],
            "grasp_width": 0.6,
            "hover_height": 0.15,
            "grasp_offset_z": 0.025
        },
        "training": {
            "use_policy_blending": True  # Force Enable
        }
    })

    # 2. Initialize Wrapper (Directly, no AsyncVectorEnv)
    print("\n[1] Initializing Environment...")
    env_inner = PandaEnv(
        xml_path=cfg.environment.xml_path,
        control_mode="delta",
        render_mode="rgb_array"
    )
    env = DGPOEnvWrapper(env_inner, cfg)
    
    # Force blending parameters
    env.use_blending = True
    env.blend_alpha = 0.5 # Default internal alpha
    print(f"    Wrapper Initialized. Use Blending: {env.use_blending}")

    # 3. Reset
    print("\n[2] Resetting...")
    obs, info = env.reset()
    expert_pose = info['expert_pose'] # (7,) x,y,z,qw,qx,qy,qz
    print(f"    Expert Target Pose: {expert_pose[:3]}")

    # 4. Construct Test Inputs
    # We will create a policy pose that is offset by exactly 10cm in X
    offset = np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    policy_pose = expert_pose + offset
    policy_grip = -1.0
    
    # Construct 9D Action: [7 pose + 1 gripper + 1 alpha]
    # We use alpha = 0.5
    TEST_ALPHA = 0.5
    action_9d = np.concatenate([policy_pose, [policy_grip], [TEST_ALPHA]])
    
    print(f"\n[3] Testing Step with Alpha = {TEST_ALPHA}")
    print(f"    Policy Pose Offset: +10cm in X")
    print(f"    Expected Blend:     +5cm in X (0.5 * 0.10)")

    # 5. Execute Step
    # The step() method converts pose -> IK -> Delta Joints
    # To verify blending, we should look at 'blended_pose' in info (if we added it)
    # OR we can inspect the internal method logic if we really want to be sure.
    # But let's check the result from step()
    
    next_obs, reward, term, trunc, info = env.step(action_9d)

    # 6. Verification
    blended_pose = info.get('blended_pose')
    
    if blended_pose is None:
        print("\n❌ FAIL: 'blended_pose' not found in info dict. Did the step() method run the new code?")
        return

    # Check Position Blending
    expert_pos_t = expert_pose[:3]
    policy_pos_t = policy_pose[:3]
    blended_pos_t = blended_pose[:3]
    
    expected_pos = (1 - TEST_ALPHA) * expert_pos_t + (TEST_ALPHA) * policy_pos_t
    diff = np.linalg.norm(blended_pos_t - expected_pos)
    
    print(f"\n[4] Results:")
    print(f"    Expert Pos:   {expert_pos_t}")
    print(f"    Policy Pos:   {policy_pos_t}")
    print(f"    Result Pos:   {blended_pos_t}")
    print(f"    Expected Pos: {expected_pos}")
    print(f"    Difference:   {diff:.6f}")

    if diff < 1e-5:
        print("\n✅ PASS: Pose Blending is Mathematically Correct!")
    else:
        print("\n❌ FAIL: Blending Math mismatch!")

    # 7. Check Alpha Passing
    logged_alpha = info.get('blend_alpha')
    print(f"\n[5] Alpha Verification:")
    print(f"    Input Alpha:  {TEST_ALPHA}")
    print(f"    Logged Alpha: {logged_alpha}")
    
    if abs(logged_alpha - TEST_ALPHA) < 1e-5:
        print("✅ PASS: Alpha passed correctly through 9D action!")
    else:
        print("❌ FAIL: Alpha did not pass through!")

if __name__ == "__main__":
    test_blending_math()
