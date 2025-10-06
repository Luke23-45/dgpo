# FILE: test1_ik_timing.py
# This version is adapted to work directly with the codebase in 9.txt

import numpy as np
import sys
import traceback
from scipy.spatial.transform import Rotation as R

# --- Direct imports from your project structure ---
from envs.panda_env import PandaEnv
from utils.ik_solver import IKSolver
from utils.scripted_expert import ScriptedExpert, ObjectProfile, ExpertConfig

# ====== CONFIG - No need to edit if your file paths are standard ======
# Paths are relative to the project root
URDF_PATH = "urdf/panda_mujoco_kinematics.urdf"
ENV_XML_PATH = "envs/panda_pick_place.xml"

# Test parameters
N_TRIALS = 20
ERROR_THRESHOLD = 0.02  # 2 cm positional error threshold for success
# =====================================================================

def fatal(msg):
    print(f"FATAL: {msg}")
    sys.exit(1)

# --- Test Setup ---
# 1. Environment Factory (creates the env)
try:
    env = PandaEnv(xml_path=ENV_XML_PATH, control_mode="absolute")
    # The environment timestep `dt` is model.opt.timestep * N_SUBSTEPS
    # In PandaEnv, N_SUBSTEPS is 5.
    dt = env.model.opt.timestep * 5
    print(f"INFO: Environment created. Control timestep (dt) = {dt:.4f}s")
except Exception as e:
    fatal(f"Could not create PandaEnv: {e}")

# 2. IK Solver (computes joint targets from a pose)
try:
    ik_solver = IKSolver(urdf_path=URDF_PATH)
    print("INFO: IKSolver initialized.")
except Exception as e:
    fatal(f"Could not create IKSolver: {e}")

# 3. Foundation Model (provides the target pose)
# We use the deterministic ScriptedExpert for this test.
object_profile = ObjectProfile(size=np.array([0.04, 0.04, 0.04]), grasp_width_normalized=0.6)
pi_F = ScriptedExpert(object_profile=object_profile, cfg=ExpertConfig())
print("INFO: Using ScriptedExpert as the Foundation Model.")

# --- Helper Functions (Adapters for your specific codebase) ---
def get_current_state(obs_dict):
    """Extracts necessary states from the environment's observation dict."""
    q_current = obs_dict["proprio"][:7]
    ee_pose_world = obs_dict["ee_pose_world"]
    return q_current, ee_pose_world

def fk(env_instance) -> np.ndarray:
    """Wrapper for the environment's forward kinematics method."""
    return env_instance.get_ee_pose()

def ik(p_target_world, q_current, env_instance, ik_solver_instance):
    """
    Wrapper for your IK solver. It correctly handles the coordinate
    frame transformation from world to the robot's base frame.
    """
    base_pos, base_quat = env_instance.get_base_pose()
    R_world_base = R.from_quat(base_quat)
    R_base_world = R_world_base.inv()
    
    pos_in_base = R_base_world.apply(p_target_world[:3] - base_pos)
    rot_in_base = R_base_world * R.from_quat(p_target_world[3:7])
    target_pose_base = np.concatenate([pos_in_base, rot_in_base.as_quat()])

    # Your IK solver computes a normalized *absolute position* target.
    normalized_q_target = ik_solver_instance.compute_action(target_pose_base, q_current)
    return normalized_q_target

# --- Main Test Loop ---
errors = []
print("\n" + "="*80)
print("--- Starting IK Timing & Alignment Test ---")
for t in range(N_TRIALS):
    try:
        # Get initial state
        obs, _ = env.reset()
        q_t, _ = get_current_state(obs)

        # 1. Foundation Model predicts a target world pose
        p_target, gripper_action = pi_F.get_target_pose(
            obs["ee_pose_world"],
            obs["object_pos_world"],
            obs["object_orn_world"],
            obs["goal_pos_world"],
            obs["is_grasped"][0],
        )

        # 2. IK computes the normalized joint command to reach that pose
        action_arm = ik(p_target, q_t, env, ik_solver)
        action_full = np.append(action_arm, gripper_action)

        # 3. Environment steps using this absolute position command
        obs_next, _, _, _, _ = env.step(action_full)
        
        # 4. We measure the *actually achieved* end-effector pose
        p_achieved = fk(env)
        
        # 5. Calculate the positional error
        # We only care about the 3D position for this test.
        pos_error = np.linalg.norm(p_achieved[:3] - p_target[:3])
        errors.append(pos_error)
        
        status = "PASS" if pos_error <= ERROR_THRESHOLD else "FAIL"
        print(f"Trial {t+1:02d}/{N_TRIALS}: Target State='{pi_F.get_state()}', Positional Error = {pos_error:.6f} m -> {status}")
        
    except Exception as e:
        traceback.print_exc()
        print(f"Trial {t+1} raised an exception: {e}")
        errors.append(float('inf'))

# --- Summary ---
errors = np.array(errors)
pass_rate = np.mean(errors <= ERROR_THRESHOLD)

print("\n" + "="*80)
print("--- Test Summary ---")
print(f"  Trials Run: {len(errors)}")
print(f"  Mean Positional Error: {errors.mean():.6f} m")
print(f"  Median Positional Error: {np.median(errors):.6f} m")
print(f"  Pass Threshold: {ERROR_THRESHOLD:.4f} m")
print(f"  Pass Rate: {pass_rate:.1%}")
print("="*80)

if pass_rate >= 0.8:
    print("\n[TEST 1 OVERALL: PASS]")
    print("The IK -> action -> next_state pipeline is working as expected.")
    print("The target pose from the Foundation Model at step `t` is consistently achieved at step `t+1`.")
else:
    print("\n[TEST 1 OVERALL: FAIL]")
    print("A significant number of trials failed to reach the target pose within the error threshold.")
    print("This indicates a potential bug or mismatch in one of the following:")
    print("  - The IKSolver is producing inaccurate joint targets.")
    print("  - The environment's PD controller is too weak or slow to reach the target in one step.")
    print("  - The coordinate frame transformations (world <-> base) have an error.")