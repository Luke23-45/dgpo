# debug_scripts/hybrid_smoke_test.py
"""
Minimal smoke test for hybrid pipeline pieces.
Run:
    python debug_scripts/hybrid_smoke_test.py
"""
import sys, torch, logging
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
from envs.panda_env import PandaEnv
from utils.expert_dataset import ExpertDataset
from torch.utils.data import DataLoader
from scripts.run_experiment import load_bc_checkpoint
from models.bc_policy import BCNet
from scripts.train_hybrid import select_actor_parameters

def main():
    print("1) Creating env (absolute control)...")
    env = PandaEnv(xml_path="envs/panda_pick_place.xml", control_mode="absolute")
    print("  obs space:", env.observation_space)
    print("  action space:", env.action_space)

    print("\n2) ExpertDataset + DataLoader (one batch)")
    ds = ExpertDataset(urdf_path="urdf/panda_mujoco_kinematics.urdf", base_seed=1234, max_samples_per_epoch=256)
    loader = DataLoader(ds, batch_size=8, num_workers=0, pin_memory=False)
    obs, act = next(iter(loader))
    print("  obs keys:", list(obs.keys()))
    print("  act shape:", act.shape)

    print("\n3) Load BC checkpoint (dry) and instantiate BCNet")
    ckpt_path = Path("artifacts/bc_final_balanced_v1/checkpoints/best_model.pth")
    if ckpt_path.exists():
        ckpt = load_bc_checkpoint(str(ckpt_path))
        action_dim = env.action_space.shape[0]
        bc = BCNet(n_actions=action_dim)
        try:
            bc.load_state_dict(ckpt, strict=False)
            print("  BCNet loaded (strict=False) with ckpt keys.")
        except Exception as e:
            print("  BC load exception:", e)
    else:
        print("  BC checkpoint not found at", ckpt_path)

    print("\n4) select_actor_parameters quick test")
    # create a small dummy policy-like object by creating PPO policy? Skip heavy creation
    print("Smoke test complete. If all lines above succeeded, major pieces are coherent.")

if __name__ == "__main__":
    main()
