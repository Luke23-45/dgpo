"""
test6_action_mse_transfer.py
----------------------------
Purpose:
  After weight-transfer (BC -> PPO), verify the transferred PPO policy produces
  actions similar to BCNet on identical observations. This exposes head/head-shape
  or scaling mismatches.

Config:
  - BC_CHECKPOINT: path to BC checkpoint (or set PROJECT_MODULE.load_bc_checkpoint)
  - PPO_CHECKPOINT: path to transferred PPO agent (or use initialize_ppo_agent)
  - ENV_FACTORY, OBS_PREPROCESSOR: helper to convert env observation to model input
  - BATCH_SIZE: number of observations to compare
Usage:
  python test6_action_mse_transfer.py
Output:
  Per-dim MSE and summary PASS/FAIL (thresholds adjustable).
"""

import importlib, numpy as np, sys, traceback
from types import SimpleNamespace

# ===== CONFIG =====
PROJECT_MODULE = "project_api"
ENV_FACTORY = "make_env"
BC_CHECKPOINT = None      # e.g., "checkpoints/bc_best.pth" OR None to use loader
PPO_CHECKPOINT = None     # e.g., "checkpoints/ppo_with_transfer.zip" OR None to use loader
LOAD_BC_FN = "load_bc_checkpoint"   # project function to load BC model or checkpoint
INIT_PPO_FN = "initialize_ppo_agent" # function that returns an SB3-style agent already with transferred weights
OBS_PREPROCESSOR = "prepare_obs_for_policy"  # function to convert env.obs to model input (optional)
BATCH_SIZE = 64
MSE_PASS_THRESHOLD = 1e-4   # per-dim mse threshold to consider very close
# ====================

proj = importlib.import_module(PROJECT_MODULE)
env = getattr(proj, ENV_FACTORY)()

# load/prepare BC model
if BC_CHECKPOINT is None and hasattr(proj, LOAD_BC_FN):
    bc_obj = getattr(proj, LOAD_BC_FN)()
elif BC_CHECKPOINT is not None:
    if hasattr(proj, LOAD_BC_FN):
        bc_obj = getattr(proj, LOAD_BC_FN)(BC_CHECKPOINT)
    else:
        # try to load raw torch checkpoint
        import torch
        ck = torch.load(BC_CHECKPOINT, map_location="cpu")
        bc_obj = ck.get("model_state_dict", ck)
else:
    raise RuntimeError("BC model not found. Set BC_CHECKPOINT or implement load_bc_checkpoint.")

# bc predict helper (try a few shapes)
def bc_predict(obs_batch):
    # try common APIs
    try:
        if hasattr(bc_obj, "predict"):
            return np.asarray(bc_obj.predict(obs_batch))
        elif hasattr(bc_obj, "forward"):
            import torch
            with torch.no_grad():
                x = torch.as_tensor(obs_batch).float()
                out = bc_obj(x)
                if isinstance(out, dict) and "action" in out:
                    return out["action"].cpu().numpy()
                return out.cpu().numpy()
        elif isinstance(bc_obj, dict):
            raise RuntimeError("Loaded raw state_dict; you should load the model class instead.")
        else:
            # assume bc_obj is callable
            return np.asarray(bc_obj(obs_batch))
    except Exception as e:
        raise RuntimeError(f"Could not call BC model: {e}")

# load/prepare PPO agent
ppo_agent = None
if PPO_CHECKPOINT is None and hasattr(proj, INIT_PPO_FN):
    ppo_agent = getattr(proj, INIT_PPO_FN)(env=env)
elif PPO_CHECKPOINT is not None:
    try:
        # try stable-baselines3 load
        from stable_baselines3 import PPO
        ppo_agent = PPO.load(PPO_CHECKPOINT, env=env)
    except Exception:
        # fallback to project loader
        if hasattr(proj, INIT_PPO_FN):
            ppo_agent = getattr(proj, INIT_PPO_FN)(env=env, checkpoint=PPO_CHECKPOINT)
        else:
            raise RuntimeError("Could not load PPO agent from PPO_CHECKPOINT. Implement initializer.")
if ppo_agent is None:
    raise RuntimeError("PPO agent not loaded.")

# obs preprocessor
if hasattr(proj, OBS_PREPROCESSOR):
    preproc = getattr(proj, OBS_PREPROCESSOR)
else:
    preproc = lambda o: o  # assume env.observations are accepted

# collect a batch of observations
obs_list = []
while len(obs_list) < BATCH_SIZE:
    obs = env.reset()
    obs_list.append(obs)
    # do a short random step to get new states
    random_action = np.random.normal(size=getattr(env, "n_actions", 7))
    env.step(random_action)

# prepare input batch for models
input_batch = [preproc(o) for o in obs_list]
# try to convert to numpy arrays for BC
try:
    bc_input = np.stack(input_batch, axis=0)
except Exception:
    # if obs are dicts try to gather 'proprio' key
    if isinstance(input_batch[0], dict) and "proprio" in input_batch[0]:
        bc_input = np.stack([x["proprio"] for x in input_batch], axis=0)
    else:
        raise RuntimeError("Could not stack observations; implement prepare_obs_for_policy to flatten obs.")

# get bc actions
try:
    bc_actions = bc_predict(bc_input)
except Exception as e:
    raise RuntimeError(f"BC predict failed: {e}")

# get ppo actions (try sb3 predict or policy.forward)
try:
    if hasattr(ppo_agent, "predict"):
        ppo_actions, _ = ppo_agent.predict(bc_input, deterministic=True)
    else:
        # try policy.forward (may require torch tensors)
        if hasattr(ppo_agent, "policy") and hasattr(ppo_agent.policy, "forward"):
            import torch
            with torch.no_grad():
                x = torch.as_tensor(bc_input).float()
                out = ppo_agent.policy.forward(x)
                ppo_actions = out.cpu().numpy()
        else:
            raise RuntimeError("Unable to get actions from PPO agent; implement a small wrapper function.")
except Exception as e:
    raise RuntimeError(f"PPO action prediction failed: {e}")

# ensure shapes match
if bc_actions.shape != ppo_actions.shape:
    print("WARNING: action shapes differ:", bc_actions.shape, ppo_actions.shape)
    # try to broadcast/reshape if possible
    min_shape = tuple(min(a,b) for a,b in zip(bc_actions.shape, ppo_actions.shape))
    bc_actions = bc_actions.reshape(bc_actions.shape[0], -1)[:min_shape[0], :min_shape[1]]
    ppo_actions = ppo_actions.reshape(ppo_actions.shape[0], -1)[:min_shape[0], :min_shape[1]]

mse_per_dim = ((bc_actions - ppo_actions)**2).mean(axis=0)
print("Per-dim MSE:", mse_per_dim)
print("Mean MSE:", mse_per_dim.mean())

if float(mse_per_dim.mean()) < MSE_PASS_THRESHOLD:
    print("TEST 6 PASS: Transferred PPO outputs closely match BC outputs on sample batch.")
else:
    print("TEST 6 WARNING / FAIL: High MSE between BC and PPO. Likely causes:")
    print("  - head/name/shape mismatch during transfer (missing_keys/unexpected_keys at load time)")
    print("  - activation/scale mismatch (tanh vs linear) — try rescaling outputs")
    print("  - observation preprocessing mismatch between BC and PPO (normalize differently)")
    print("Recommended actions: inspect transfer logs, run test1 and test2, check observation preprocessing.")
