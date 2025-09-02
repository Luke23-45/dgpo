# tests/test_expert_adapter.py
import numpy as np
from utils.expert_dataset import ExpertDataset

# create fake obs and call the adapter via instance to simulate runtime
fake_obs = {
    "image_primary": np.zeros((256,256,3), dtype=np.uint8),
    "proprio": np.zeros((14,), dtype=np.float32),
    "timestep": np.array([0], dtype=np.int32),
    "task_completed": np.array([0.0], dtype=np.float32),
}

ds = ExpertDataset(urdf_path="urdf/panda_mujoco_kinematics.urdf", instruction="pick up the red block", base_seed=0, max_samples_per_epoch=1, warmup=False)
octo = ds._prepare_octo_obs(fake_obs, task_completed_width=4, duplicate_T=2)
print("octo_obs keys:", sorted(octo.keys()))
print("image_primary shape:", octo["image_primary"].shape)
print("proprio shape:", octo["proprio"].shape)
print("timestep_pad_mask shape:", octo.get("timestep_pad_mask", octo.get("timestep")).shape)
