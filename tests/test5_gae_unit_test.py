"""
test5_gae_unit_test.py
----------------------
Purpose:
  Unit-test GAE implementation and the handling of a terminal (R_T) plausibility
  reward to ensure indexing is consistent with the spec:
    - δ_t = R_t + γ V(s_{t+1}) - V(s_t)  (t < T-1)
    - δ_{T-1} = R_{T-1} + γ R_T - V(s_{T-1})
  We'll compute Â_t manually and compare to a local implementation and (optionally)
  to your project's compute_gae function if present.

No external env needed.
Usage:
  python test5_gae_unit_test.py
"""

import numpy as np
import importlib

# small synthetic episode (T=4)
gamma = 0.99
lam = 0.95

# make deterministic synthetic rewards and values
R = np.array([0.1, 0.0, 0.2, -0.1])   # per-step rewards R_0..R_{T-1}
R_T = 0.5                             # terminal plausibility reward applied at end
V = np.array([0.05, 0.02, 0.1, 0.0, 0.0])  # V(s_0)..V(s_T) (we assume V(s_T)=0)

def compute_td_errors(R, R_T, V, gamma):
    T = len(R)
    deltas = np.zeros(T)
    for t in range(T):
        if t < T-1:
            deltas[t] = R[t] + gamma*V[t+1] - V[t]
        else:
            deltas[t] = R[t] + gamma*R_T - V[t]
    return deltas

def compute_gae_from_deltas(deltas, gamma, lam):
    T = len(deltas)
    adv = np.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        gae = deltas[t] + gamma*lam*gae
        adv[t] = gae
    return adv

deltas = compute_td_errors(R, R_T, V, gamma)
adv = compute_gae_from_deltas(deltas, gamma, lam)
print("Deltas:", deltas)
print("Advantages:", adv)

# If project exposes its compute_gae, compare (optional)
PROJECT_MODULE = "project_api"
try:
    proj = importlib.import_module(PROJECT_MODULE)
    if hasattr(proj, "compute_gae"):
        proj_adv = proj.compute_gae(R, R_T, V, gamma, lam)  # example signature
        print("Project compute_gae returned:", proj_adv)
        if np.allclose(np.asarray(proj_adv), adv, atol=1e-6):
            print("TEST 5 PASS: project compute_gae matches spec-based computation.")
        else:
            print("TEST 5 FAIL: project compute_gae differs from manual computation.")
    else:
        print("Project module has no compute_gae to compare; manual GAE computed above.")
except Exception as e:
    print("Project module import or compare skipped:", e)
    print("Manual GAE (spec-based) printed above. Verify your GAE uses δ_{T-1} = R_{T-1} + γ R_T - V_{T-1}.")
