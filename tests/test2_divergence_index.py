"""
test2_divergence_index.py
-------------------------
Purpose:
  Determine whether π_F(s_t) corresponds more closely to FK(q_t) (current pose)
  or FK(q_{t+1}) (next pose after applying IK->action). This reveals the
  correct time-indexing for your Divergence term.

Config:
  - Same PROJECT_MODULE, ENV_FACTORY, FOUNDATION_MODEL_LOADER, IK_FN, FK_FN as test1.
  - ROLLOUT_STEPS: number of steps to sample from the expert policy (pi_F).
Usage:
  python test2_divergence_index.py
Output:
  Prints aggregated MSE to FK(q_t) and FK(q_{t+1}) across many samples and tells you which is smaller.
"""
import importlib, sys, traceback
import numpy as np

# ===== CONFIG =====
PROJECT_MODULE = "project_api"
ENV_FACTORY = "make_env"
FOUNDATION_MODEL_LOADER = "load_foundation_model"
IK_FN = "inverse_kinematics"
FK_FN = "forward_kinematics"
ROLLOUT_STEPS = 200
# ==================

def fatal(msg):
    print("FATAL:", msg); sys.exit(1)

proj = importlib.import_module(PROJECT_MODULE)
if not hasattr(proj, ENV_FACTORY): fatal(f"{PROJECT_MODULE} missing {ENV_FACTORY}")
env = getattr(proj, ENV_FACTORY)()
load_pi_F = getattr(proj, FOUNDATION_MODEL_LOADER)
pi_F = load_pi_F()

# helpers
def get_joint_positions(env):
    for n in ("get_joint_positions", "get_qpos", "get_joint_state"):
        if hasattr(env, n):
            return getattr(env,n)()
    if hasattr(env, "robot") and hasattr(env.robot, "q"):
        return np.array(env.robot.q)
    raise RuntimeError("Add get_joint_positions helper.")

if hasattr(env, IK_FN): IK = getattr(env, IK_FN)
elif hasattr(proj, IK_FN): IK = getattr(proj, IK_FN)
else: fatal("IK function not found.")

if hasattr(env, FK_FN): FK = getattr(env, FK_FN)
elif hasattr(proj, FK_FN): FK = getattr(proj, FK_FN)
else: fatal("FK function not found.")

# rollouts
mse_to_current = []
mse_to_next = []
for i in range(ROLLOUT_STEPS):
    try:
        s = env.reset() if i==0 else s_next
        q_t = np.array(get_joint_positions(env))
        # pi_F action / predicted pose
        if hasattr(pi_F, "predict"): p_target = np.asarray(pi_F.predict(s))
        else: p_target = np.asarray(pi_F(s))
        # compute IK-based action and step (same as test1 mapping)
        q_target = IK(p_target, q_t)
        dt = getattr(env, "dt", 0.02)
        a_cmd = np.clip((np.asarray(q_target) - q_t)/dt, -getattr(env, "max_joint_velocity", 1.0), getattr(env, "max_joint_velocity", 1.0))
        out = env.step(a_cmd)
        # read new state
        q_next = np.array(get_joint_positions(env))
        p_current = np.asarray(FK(q_t))
        p_next = np.asarray(FK(q_next))
        # compute MSE
        mse_c = float(((p_target - p_current)**2).mean())
        mse_n = float(((p_target - p_next)**2).mean())
        mse_to_current.append(mse_c)
        mse_to_next.append(mse_n)
        # prepare s_next if env returns observation
        if isinstance(out, tuple):
            s_next = out[0]
        else:
            s_next = env.state if hasattr(env, "state") else None
    except Exception as e:
        traceback.print_exc()
        print("Sample raised:", e)
        break

mse_to_current = np.array(mse_to_current)
mse_to_next = np.array(mse_to_next)
print("MSE to FK(q_t): mean {:.6e}, median {:.6e}".format(mse_to_current.mean(), np.median(mse_to_current)))
print("MSE to FK(q_{t+1}): mean {:.6e}, median {:.6e}".format(mse_to_next.mean(), np.median(mse_to_next)))

if mse_to_next.mean() + 1e-12 < mse_to_current.mean():
    print("Conclusion: π_F(s_t) aligns better with FK(q_{t+1}) -> use FK(q_{t+1}) in Divergence.")
else:
    print("Conclusion: π_F(s_t) aligns better with FK(q_t) (or tie). Check pi_F semantics carefully.")
