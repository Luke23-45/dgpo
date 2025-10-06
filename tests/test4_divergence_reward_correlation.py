"""
test4_divergence_reward_correlation.py
-------------------------------------
Purpose:
  Measure correlation between (task return) and (divergence) on:
    - rollouts under the foundation model π_F (expert)
    - rollouts under a random policy
  If divergence and task reward are negatively correlated (or uncorrelated),
  plausibility reward may conflict with task reward.

Config:
  - N_EPISODES per policy
  - DIVERGENCE metric: use FK(q_{t+1}) or FK(q_t) depending on earlier tests.
Usage:
  python test4_divergence_reward_correlation.py
Output:
  Pearson correlation and scatter summary.
"""
import importlib, numpy as np, sys
from scipy.stats import pearsonr

PROJECT_MODULE = "project_api"
ENV_FACTORY = "make_env"
FOUNDATION_MODEL_LOADER = "load_foundation_model"
IK_FN = "inverse_kinematics"
FK_FN = "forward_kinematics"
N_EPISODES = 50
USE_QNEXT = True   # set True if test2 concluded pi_F aligns with q_{t+1}
MAX_STEPS = 200

proj = importlib.import_module(PROJECT_MODULE)
env = getattr(proj, ENV_FACTORY)()
pi_F = getattr(proj, FOUNDATION_MODEL_LOADER)()
IK = getattr(env, IK_FN) if hasattr(env, IK_FN) else getattr(proj, IK_FN)
FK = getattr(env, FK_FN) if hasattr(env, FK_FN) else getattr(proj, FK_FN)

def get_joint_positions(env):
    for n in ("get_joint_positions","get_qpos"):
        if hasattr(env,n): return np.asarray(getattr(env,n)())
    if hasattr(env,"robot"): return np.asarray(env.robot.q)
    raise RuntimeError

def run_episode(policy_callable, random_policy=False):
    s = env.reset()
    total_reward = 0.0
    all_div = []
    for step in range(MAX_STEPS):
        q_t = get_joint_positions(env)
        if random_policy:
            # sample small random velocity
            a = np.random.normal(scale=0.1, size=q_t.shape)
            out = env.step(a)
            obs, r, done, info = out[0], out[1], out[2], out[3]
        else:
            if hasattr(policy_callable, "predict"):
                a, _ = policy_callable.predict(s, deterministic=True)
                out = env.step(a)
                obs, r, done, info = out[0], out[1], out[2], out[3]
            else:
                # assume policy returns target pose p_target
                p_target = policy_callable(s)
                q_target = IK(p_target, q_t)
                a = (np.asarray(q_target) - q_t) / getattr(env, "dt", 0.02)
                out = env.step(a)
                obs, r, done, info = out[0], out[1], out[2], out[3]
        total_reward += r
        # compute divergence for this step
        if not random_policy:
            p_target = policy_callable.predict(s) if hasattr(policy_callable, "predict") else policy_callable(s)
            if USE_QNEXT:
                q_next = get_joint_positions(env)
                p_achieved = FK(q_next)
            else:
                p_achieved = FK(q_t)
            div = float(((np.asarray(p_achieved) - np.asarray(p_target))**2).mean())
            all_div.append(div)
        s = obs
        if done:
            break
    mean_div = float(np.mean(all_div)) if all_div else float("nan")
    return total_reward, mean_div

# collect stats for expert
expert_rewards = []
expert_divs = []
for _ in range(N_EPISODES):
    r, d = run_episode(pi_F, random_policy=False)
    expert_rewards.append(r)
    expert_divs.append(d)

random_rewards = []
random_divs = []
for _ in range(N_EPISODES):
    r, d = run_episode(None, random_policy=True)
    random_rewards.append(r)
    random_divs.append(d)

# correlation
valid_idx = ~np.isnan(expert_divs)
if valid_idx.sum() > 2:
    rcorr, pval = pearsonr(np.array(expert_rewards)[valid_idx], np.array(expert_divs)[valid_idx])
    print("Expert: Pearson corr between return and divergence: r={:.4f}, p={:.3e}".format(rcorr, pval))
else:
    print("Not enough valid divergence samples for expert.")

print("Expert: avg return {:.4f}, avg divergence {:.6e}".format(np.mean(expert_rewards), np.nanmean(expert_divs)))
print("Random: avg return {:.4f}, avg divergence {:.6e}".format(np.mean(random_rewards), np.nanmean(random_divs)))

print("\nInterpretation tips:")
print("- If expert correlation is strongly negative: divergence reward fights task reward (bad).")
print("- If expert avg divergence is much lower than random avg divergence: plausibility is informative.")
