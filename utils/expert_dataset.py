# utils/expert_dataset.py
"""
ExpertDataset (robust, production-ready)

This IterableDataset generates (observation, expert_action) samples on the fly by:
 - resetting a MuJoCo environment (PandaEnv),
 - querying a pre-trained foundation model (OCTO) for an end-effector target,
 - converting that to joint commands via IKSolver.

Design notes:
 - Heavy objects (OctoModel, PandaEnv, IKSolver) are initialized lazily in each worker's iterator
   to avoid pickling / cross-process issues.
 - RNG: a base_seed can be provided for reproducibility. Per-worker seeds are derived deterministically.
 - Works with DataLoader(num_workers=0) and also tolerates num_workers>0 (each worker gets own resources).
 - By default the dataset yields CPU tensors and leaves device placement to the training loop.
   If you pass move_to_device=True and device=..., tensors will be moved inside the dataset.
"""
from __future__ import annotations
from scipy.spatial.transform import Rotation as R

import logging
import time
from typing import Dict, Optional, Tuple, Iterator

import numpy as np
import jax
import torch
from torch.utils.data import IterableDataset, get_worker_info
import mujoco
# Project imports (adjust if your package layout differs)
from octo.model.octo_model import OctoModel
from envs.panda_env import PandaEnv
from utils.ik_solver import IKSolver
from utils.scripted_expert import ScriptedExpert, ExpertConfig
from utils.controls import gripper_action_to_ctrl
from utils.obs_adapters import build_octo_observation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _to_torch_image(img: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 image to CHW float32 in [-1, 1]."""
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    # Ensure HWC
    if img.ndim != 3 or img.shape[2] != 3:
        # Try squeeze/reshape defensively
        img = img.reshape(256, 256, 3)
    # Cast and normalize to [-1, 1]
    img_f = img.astype(np.float32)
    if img_f.max() > 2.0:  # assume 0..255
        img_f = img_f / 127.5 - 1.0
    else:  # already near [-1,1] or [0,1]
        img_f = np.clip(img_f, -1.0, 1.0)
    # HWC -> CHW
    img_chw = np.transpose(img_f, (2, 0, 1))
    return torch.from_numpy(img_chw)


class ExpertDataset(IterableDataset):
    def __init__(
        self,
        urdf_path: str,
        instruction: str = "pick up the red block",
        *,
        octo_model_name: str = "hf://rail-berkeley/octo-small-1.5",
        env_xml_path: Optional[str] = None,
        base_seed: Optional[int] = None,
        max_samples_per_epoch: Optional[int] = None,
        skip_on_error: bool = True,
        warmup: bool = False,
        use_octo: bool = True, # Flag to enable/disable OCTO
        scripted_cfg: ExpertConfig = ExpertConfig(), # Config for our fallback expert
        device: Optional[torch.device] = None,
        move_to_device: bool = False,
        octo_pad_mask: np.ndarray = np.array([[False, True]])
    ) -> None:
        """
        Args:
          urdf_path: path to panda.urdf used by IKSolver.
          instruction: natural-language instruction for OCTO.
          octo_model_name: identifier used by OctoModel.load_pretrained(...)
          env_xml_path: optional MuJoCo xml for PandaEnv (if None, PandaEnv default is used)
          base_seed: optional int seed for deterministic sampling; if None uses non-deterministic seed.
          device: torch.device to place tensors on (only used if move_to_device=True).
          move_to_device: if True, dataset will move each sample to `device` before yielding.
          octo_pad_mask: pad mask passed to OCTO; default [[False, True]] (first frame valid).
          max_samples_per_epoch: if set, each iterator yields at most this many samples then stops.
          skip_on_error: if True, dataset will skip samples that raise exceptions (recommended).
          warmup: if True, calls a small warmup inference after initialization to reduce first-sample latency.
        """
        super().__init__()
        self.urdf_path = urdf_path
        self.instruction = instruction
        self.octo_model_name = octo_model_name
        self.env_xml_path = env_xml_path
        self.base_seed = int(base_seed) if base_seed is not None else None
        self.device = device
        self.move_to_device = bool(move_to_device)
        self.octo_pad_mask = np.asarray(octo_pad_mask)
        self.max_samples_per_epoch = int(max_samples_per_epoch) if max_samples_per_epoch is not None else None
        self.skip_on_error = bool(skip_on_error)
        self.warmup = bool(warmup)

        # Worker-local attributes (initialized in __iter__)
        self._worker_state_initialized = False
        self.use_octo = bool(use_octo)
        self._env = None
        self._ik_solver = None
        self._task = None
        self._jax_key = None
        self._samples_yielded = 0
        self.skipped_samples = 0
        self.scripted_cfg = scripted_cfg
        self._scripted_expert: Optional[ScriptedExpert] = None # Add placeholder

        logger.info("ExpertDataset created (lazy initialization).")

    # ----------------------------
    # Worker-state initialization
    # ----------------------------
    def _init_worker_state(self) -> None:
        """Initialize heavy objects per worker. Safe to call multiple times (idempotent)."""

        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        seed = (self.base_seed if self.base_seed is not None else int(time.time() * 1e6))
        worker_seed = (seed + worker_id) & 0x7FFFFFFF
        
        logger.info(f"[worker {worker_id}] Initializing components with seed {worker_seed}...")
        if self.use_octo:
            self._octo_model = OctoModel.load_pretrained(self.octo_model_name)
            self._task = self._octo_model.create_tasks(texts=[self.instruction])
            self._jax_key = jax.random.PRNGKey(worker_seed)
        
        self._env = PandaEnv(xml_path=self.env_xml_path)
        self._env.reset(seed=worker_seed)
        
        self._ik_solver = IKSolver(urdf_path=self.urdf_path)
        self._scripted_expert = ScriptedExpert(self.scripted_cfg)
        
        self._worker_state_initialized = True
        self._samples_yielded = 0
        logger.info(f"[worker {worker_id}] Worker state initialized.")


    def _align_action_dim(self, action: np.ndarray) -> np.ndarray:
        """Clamp/pad/truncate IK output to match environment action dimension."""
        action = np.asarray(action, dtype=np.float32).ravel()
        desired = int(self._env.action_space.shape[0])
        if action.size < desired:
            pad = np.zeros(desired - action.size, dtype=np.float32)
            action = np.concatenate([action, pad])
        elif action.size > desired:
            action = action[:desired]
        return action

    # ----------------------------
    # Sample generation core
    # ----------------------------

    def _generate_one(self) -> Tuple[Dict, np.ndarray]:
        """
        Generates a single (observation, action) sample.

        It first attempts to get a valid action from the OCTO model. If the OCTO
        prediction is invalid (unreachable, non-finite), it falls back to the
        deterministic ScriptedExpert to guarantee a high-quality sample.
        """
        # 1. Reset env and get a rich observation with ground-truth data
        obs = self._env.get_expert_obs()
        self._scripted_expert.reset() # Reset expert state for each new sample
        obs["task_completed"] = np.array([0.0], dtype=np.float32)
        pose_world = None
        gripper_action = -1.0 # Default to open
        expert_source = "scripted" # Assume scripted unless OCTO succeeds

        # 2. Try to get a pose from OCTO if enabled
        if self.use_octo:
            try:
                # Build the compliant observation for OCTO
                octo_obs = build_octo_observation(obs)
                
                # Query the model
                self._jax_key, key = jax.random.split(self._jax_key)
                raw_action = self._octo_model.sample_actions(octo_obs, self._task, rng=key)
                
                # --- Start Post-Processing and Validation ---
                # This logic is copied from our debug script
                candidate_pose = np.array(raw_action[0, 0, :7], dtype=np.float32)
                
                # Z-flip
                if (candidate_pose[2] < 0.0) and (obs["ee_pose_world"][2] > 0.1):
                    candidate_pose[2] *= -1.0
                
                # Normalize quaternion
                q = candidate_pose[3:7]
                qn = np.linalg.norm(q)
                if qn > 1e-6:
                    candidate_pose[3:7] = q / qn

                # Workspace clamp
                cfg = self.scripted_cfg # Use same workspace config as scripted expert
                for i, ax in enumerate(("x", "y", "z")):
                    lo, hi = cfg.workspace[ax]
                    candidate_pose[i] = np.clip(candidate_pose[i], lo, hi)

                # Reachability check
                base_pos, _ = self._env.get_base_pose()
                dist_from_base = np.linalg.norm(candidate_pose[:3] - base_pos)
                if np.all(np.isfinite(candidate_pose)) and dist_from_base <= 0.85:
                    # SUCCESS! The OCTO pose is valid.
                    pose_world = candidate_pose
                    expert_source = "octo"
                    # Simple gripper heuristic for OCTO
                    near_cube = np.linalg.norm(pose_world[:2] - obs["object_pos_world"][:2]) < 0.04
                    is_low_enough = pose_world[2] < (obs["object_pos_world"][2] + 0.03)
                    gripper_action = 1.0 if (near_cube and is_low_enough) else -1.0
                else:
                    logger.debug(f"OCTO pose rejected (dist: {dist_from_base:.2f}m). Falling back.")

            except Exception as e:
                logger.warning(f"OCTO inference failed: {e}. Falling back to ScriptedExpert.")

        # 3. If OCTO failed or was disabled, use the ScriptedExpert
        if pose_world is None:
            pose_world, gripper_action = self._scripted_expert.get_target_pose(
                obs["ee_pose_world"],
                obs["object_pos_world"],
                obs["goal_pos_world"],
            )
        
        # 4. Convert the final valid world pose to a joint action via IK
        # (This part is the same for both experts)
        base_pos, base_quat = self._env.get_base_pose()
        R_world_base = R.from_quat(base_quat)
        R_base_world = R_world_base.inv()
        pos_in_base = R_base_world.apply(pose_world[:3] - base_pos)
        rot_in_base = R_base_world * R.from_quat(pose_world[3:7])
        target_pose_base = np.concatenate([pos_in_base, rot_in_base.as_quat()]).astype(np.float32)

        current_joints = obs["internal_full_proprio"][:7]
        arm_action_deltas = self._ik_solver.compute_action(target_pose_base, current_joints)

        if np.linalg.norm(arm_action_deltas) < 1e-6:
            logger.debug(
                f"IK from expert '{expert_source}' resulted in a near-zero action. "
                "This is expected if the EE is already at the target pose."
            )

        # 5. Combine arm and gripper actions
        gripper_ctrl = gripper_action_to_ctrl(gripper_action)
        final_action = np.concatenate([arm_action_deltas[:7], [gripper_ctrl]])
        
        # Add metadata to the observation if you want to track the source
        obs["expert_source"] = 1 if expert_source == "octo" else 0
        
        return obs, self._align_action_dim(final_action)
    # ----------------------------
    # IterableDataset API
    # ----------------------------
    def __iter__(self) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        """
        The main iterator. Initializes per-worker state lazily,
        then generates either infinite stream or up to max_samples_per_epoch samples.
        """
        self._init_worker_state()
        samples_this_epoch = 0

        try:
            while True:
                if self.max_samples_per_epoch is not None and samples_this_epoch >= self.max_samples_per_epoch:
                    break

                try:
                    self._env.reset()
                    sample = self._generate_one()
                    samples_this_epoch += 1
                    self._samples_yielded += 1
                    yield sample

                except Exception as exc:
                    if self.skip_on_error:
                        self.skipped_samples += 1
                        if self.skipped_samples % 100 == 1: # Log every 100 skips
                            logger.warning(
                                f"[ExpertDataset] Skipped {self.skipped_samples} samples so far due to errors. "
                                f"Last error: {exc}"
                            )
                        continue
                    else:
                        raise

        finally:
            # Clean up the worker-local env to release mujoco resources
            try:
                if getattr(self, "_env", None) is not None:
                    try:
                        self._env.close()
                    except Exception:
                        pass
            except Exception:
                pass

    # ----------------------------
    # Optional helpers
    # ----------------------------
    def get_stats(self) -> Dict:
        """Return simple stats about the dataset/worker (samples yielded so far)."""
        return {"samples_yielded": int(self._samples_yielded)}

    def __len__(self) -> int:
        """Only return a length if max_samples_per_epoch provided; otherwise raise TypeError (infinite)."""
        if self.max_samples_per_epoch is None:
            raise TypeError("ExpertDataset is an iterable (infinite) dataset; length undefined.")
        return int(self.max_samples_per_epoch)


# ----------------------------
# Quick standalone smoke test
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test ExpertDataset (small batch).")
    parser.add_argument("--samples", type=int, default=4, help="Number of samples to fetch (per worker).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to move data to for the test.")
    parser.add_argument("--move", action="store_true", help="Move tensors to device inside dataset.")
    parser.add_argument("--warmup", action="store_true", help="Warmup OCTO on worker init (reduces first-sample latency).")
    args = parser.parse_args()

    # Configure logging to console
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)

    URDF_PATH = "urdf/panda_mujoco_kinematics.urdf"
    DEVICE = torch.device(args.device)
    instruction = "pick up the red block from the table"
    ds = ExpertDataset(urdf_path=URDF_PATH, instruction=instruction,
                       base_seed=1234, device=DEVICE, move_to_device=args.move,
                       max_samples_per_epoch=args.samples, warmup=args.warmup)

    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)  # recommended: num_workers=0
    it = iter(loader)
    batch_obs, batch_actions = next(it)

    print("Batch image shape (B,C,H,W):", batch_obs["image_primary"].shape)
    print("Batch proprio shape (B,14):", batch_obs["proprio"].shape)
    print("Batch actions shape (B, action_dim):", batch_actions.shape)
    print("Dtype:", batch_obs["image_primary"].dtype, batch_actions.dtype)
    print("Device image:", batch_obs["image_primary"].device)
    print("Sample stats:", ds.get_stats())
    logger.info("ExpertDataset smoke test complete.")


#python -m utils.expert_dataset --samples 2 --warmup
