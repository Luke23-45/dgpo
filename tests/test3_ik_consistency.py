"""
test3_ik_consistency.py
-----------------------
Purpose:
  Confirm that IK(p, q_current) returns a consistent, canonical joint solution
  (no randomization or multiple equally-likely solutions). If IK returns
  different results for same inputs, BC labels will be noisy logically.

Behavior:
  For a set of sampled states, call IK(p_target, q_t) multiple times and
  measure the variance across outputs. Reports per-joint std and flags.
"""

import importlib, numpy as np, sys
PROJECT_MODULE = "project_api"
ENV_FACTORY = "make_env"
FOUNDATION_MODEL_LOADER = "load_foundation_model"
IK_FN = "inverse_kinematics"
N_SAMPLES = 50
CALLS_PER_SAMPLE = 8
STD_THRESHOLD = 1e-4   # acceptable jitter threshold in joint-space (radians)

proj = importlib.import_module(PROJECT_MODULE)
env = getattr(proj, ENV_FACTORY)()
pi_F = getattr(proj, FOUNDATION_MODEL_LOADER)()

if hasattr(env, IK_FN): IK = getattr(env, IK_FN)
elif hasattr(proj, IK_FN): IK = getattr(proj, IK_FN)
else:
    raise RuntimeError("IK function not found. Provide env.inverse_kinematics or project_api.inverse_kinematics")

def get_joint_positions(env):
    for name in ("get_joint_positions","get_qpos"):
        if hasattr(env, name): return np.asarray(getattr(env, name)())
    if hasattr(env, "robot") and hasattr(env.robot, "q"):
        return np.asarray(env.robot.q)
    raise RuntimeError("Add joint getter.")

inconsistent_count = 0
for i in range(N_SAMPLES):
    s = env.reset()
    q_t = get_joint_positions(env)
    p_target = np.asarray(pi_F.predict(s) if hasattr(pi_F, "predict") else pi_F(s))
    sols = []
    for k in range(CALLS_PER_SAMPLE):
        sol = np.asarray(IK(p_target, q_t))
        sols.append(sol)
    sols = np.stack(sols, axis=0)  # (calls, joints)
    per_joint_std = sols.std(axis=0)
    max_std = per_joint_std.max()
    if max_std > STD_THRESHOLD:
        inconsistent_count += 1
        print(f"Sample {i}: MAX_STD={max_std:.6e} > {STD_THRESHOLD} -> INCONSISTENT IK")
    else:
        print(f"Sample {i}: MAX_STD={max_std:.6e} OK")
print(f"\nTotal inconsistent samples: {inconsistent_count}/{N_SAMPLES}")
if inconsistent_count > 0:
    print("Recommendation: make IK deterministic / canonicalize solution (e.g., choose closest to q_t).")
else:
    print("IK appears consistent/deterministic for tested samples.")
