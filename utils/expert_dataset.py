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

# Project imports (adjust if your package layout differs)
from octo.model.octo_model import OctoModel
from envs.panda_env import PandaEnv
from utils.ik_solver import IKSolver

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
        device: Optional[torch.device] = None,
        move_to_device: bool = False,
        octo_pad_mask: np.ndarray = np.array([[False, True]]),
        max_samples_per_epoch: Optional[int] = None,
        skip_on_error: bool = True,
        warmup: bool = False,
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
        self._octo_model = None
        self._env = None
        self._ik_solver = None
        self._task = None
        self._jax_key = None
        self._samples_yielded = 0

        logger.info("ExpertDataset created (lazy initialization).")

    # ----------------------------
    # Worker-state initialization
    # ----------------------------
    def _init_worker_state(self) -> None:
        """Initialize heavy objects per worker. Safe to call multiple times (idempotent)."""
        if self._worker_state_initialized:
            return

        worker_info = get_worker_info()  # None if single-process DataLoader
        worker_id = 0 if worker_info is None else worker_info.id

        # Deterministic per-worker seeds: base_seed + worker_id
        seed = (self.base_seed if self.base_seed is not None else int(time.time() * 1e6) & 0x7FFFFFFF)
        worker_seed = (seed + worker_id) & 0x7FFFFFFF

        # Make jax key deterministic per worker
        self._jax_key = jax.random.PRNGKey(worker_seed)

        # Initialize OCTO, Env, IK solver
        logger.info(f"[worker {worker_id}] Initializing OctoModel, PandaEnv, IKSolver with seed={worker_seed}...")
        # OctoModel - may perform network IO; calling load_pretrained inside worker isolates it
        self._octo_model = OctoModel.load_pretrained(self.octo_model_name)
        # PandaEnv - allow passing XML path if provided
        if self.env_xml_path:
            self._env = PandaEnv(xml_path=self.env_xml_path)
        else:
            self._env = PandaEnv()
        # IKSolver reads the URDF
        self._ik_solver = IKSolver(urdf_path=self.urdf_path)
        # Create the task object once per worker
        self._task = self._octo_model.create_tasks(texts=[self.instruction])

        # Warmup inference (reduces first-sample latency; optional)
# --- THIS IS THE NEW, CORRECTED BLOCK ---
        # Warmup inference (reduces first-sample latency; optional)
        if self.warmup:
            logger.info(f"[worker {worker_id}] Starting warmup inference...")
            try:
                # Get a single, correctly formatted observation from our new env
                obs, _ = self._env.reset()

                # Use our (soon to be updated) helper to prepare it for OCTO
                octo_obs = self._prepare_octo_obs(obs)

                # Run a single sample action
                self._jax_key, key = jax.random.split(self._jax_key)
                _ = self._octo_model.sample_actions(octo_obs, self._task, rng=key)

                logger.info(f"[worker {worker_id}] Warmup OCTO run complete.")
            except Exception as e:
                # Add more detailed error logging for easier debugging
                logger.error(f"[worker {worker_id}] Warmup FAILED. This is often due to an obs/model mismatch.", exc_info=e)
                # We can choose to raise here to stop a broken run early
                # raise e

        self._worker_state_initialized = True
        self._samples_yielded = 0
        logger.info(f"[worker {worker_id}] Worker state initialized.")

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _prepare_octo_obs(
        self,
        obs: dict,
        *,
        task_completed_width: int = 4,
        duplicate_T: int = 2,
    ) -> dict:
        """
        Prepare a single-step env obs into an OCTO-style batched dict.

        Guarantees:
          - image_primary: (B,T,H,W,C), dtype preserved (uint8/float)
          - image_wrist:   (B,T,h,w,c) present (from obs or safe placeholder)
          - timestep:      (B,T,1) int32
          - task_completed:(B,T,task_completed_width) float32
          - pad masks:     both nested pad_mask_dict[*] and flattened 'pad_mask_dict/*'
          - consistent horizon T across all modalities
          - excludes 'proprio' from OCTO input (checkpoint flagged it as extra)
        """
        import numpy as _np

        # ---- Validate input ----
        if not isinstance(obs, dict):
            raise TypeError(
                "_prepare_octo_obs expected an observation dict (env.reset()[0] style) "
                f"but received {type(obs)}."
            )
        if "image_primary" not in obs:
            raise KeyError("_prepare_octo_obs: missing required key 'image_primary' in obs")

        # ---- Images: ensure HWC ----
        img = _np.asarray(obs["image_primary"])
        if img.ndim != 3:
            raise ValueError(f"_prepare_octo_obs: expected image_primary with 3 dims HWC/CHW, got shape={img.shape}")
        img_hwc = _np.transpose(img, (1, 2, 0)) if img.shape[0] in (1, 3) else img  # CHW->HWC if needed

        # Wrist image is expected by your OCTO checkpoint; provide placeholder if absent
        if "image_wrist" in obs:
            wrist = _np.asarray(obs["image_wrist"])
            if wrist.ndim != 3:
                raise ValueError(f"_prepare_octo_obs: expected image_wrist with 3 dims HWC/CHW, got shape={wrist.shape}")
            wrist_hwc = _np.transpose(wrist, (1, 2, 0)) if wrist.shape[0] in (1, 3) else wrist
            wrist_h, wrist_w, wrist_c = wrist_hwc.shape
        else:
            # default placeholder size (matches your env)
            wrist_h, wrist_w, wrist_c = 128, 128, 3
            wrist_hwc = _np.zeros((wrist_h, wrist_w, wrist_c), dtype=_np.uint8)

        # ---- Control fields ----
        # timestep: accept scalar-like and normalize to python int
        timestep_scalar = int(_np.asarray(obs.get("timestep", _np.int32(0)), dtype=_np.int32).reshape(()))

        # task_completed: ensure desired width (pad/truncate)
        tc_src = _np.asarray(
            obs.get("task_completed", _np.zeros((task_completed_width,), dtype=_np.float32)),
            dtype=_np.float32
        ).reshape(-1)
        if task_completed_width > 0:
            tc_vec = _np.zeros((task_completed_width,), dtype=_np.float32)
            n = min(tc_src.size, task_completed_width)
            if n > 0:
                tc_vec[:n] = tc_src[:n]
        else:
            tc_vec = tc_src  # unusual, but supported

        # ---- Build (B,T,...) ----
        B = 1
        T = int(max(1, duplicate_T))

        # images
        img_step = img_hwc[_np.newaxis, _np.newaxis, ...]                 # (1,1,H,W,C)
        wrist_step = wrist_hwc[_np.newaxis, _np.newaxis, ...]             # (1,1,h,w,c)
        if T > 1:
            img_step = _np.repeat(img_step, T, axis=1)                    # (1,T,H,W,C)
            wrist_step = _np.repeat(wrist_step, T, axis=1)                # (1,T,h,w,c)

        # timestep and task_completed with matching horizon
        # Correct line
        timestep_step = _np.full((B, T), timestep_scalar, dtype=_np.int32)        # Corrected to (1,T)
        task_completed_step = _np.tile(tc_vec.reshape(1, 1, -1), (B, T, 1))          # (1,T,W)

        # ---- Masks (both nested and flattened for compatibility) ----
        timestep_mask = _np.ones((B, T), dtype=bool)                     # (1,T)
        nested_pad = {
            "image_primary":  timestep_mask,
            "image_wrist":    timestep_mask,
            "timestep":       timestep_mask,
            "task_completed": timestep_mask,
        }
        flattened_pad = {
            "pad_mask_dict/image_primary":  timestep_mask,
            "pad_mask_dict/image_wrist":    timestep_mask,
            "pad_mask_dict/timestep":       timestep_mask,
            "pad_mask_dict/task_completed": timestep_mask,
        }

        # ---- Final dict (omit 'proprio' for this OCTO input) ----
        octo_obs = {
            "image_primary":  img_step,
            "image_wrist":    wrist_step,
            "timestep":       timestep_step,
            "task_completed": task_completed_step,
            "pad_mask_dict":  nested_pad,
            "timestep_pad_mask": timestep_mask,  # legacy alias some OCTO utils still read
            **flattened_pad,
        }

        return octo_obs


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
    def _generate_one(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Produce a single sample (obs_dict, action_tensor).
        This method assumes worker-state has been initialized.
        """
        # 1) Reset env (returns obs dict and info)
        obs, _ = self._env.reset()

        # 2) Build OCTO input and query OCTO with deterministic jax key for this sample
        octo_in = self._prepare_octo_obs(obs)
        self._jax_key, key = jax.random.split(self._jax_key)
        expert_action_raw = self._octo_model.sample_actions(octo_in, self._task, rng=key)
        target_pose_world_7d = np.asarray(expert_action_raw[0, 0, :7], dtype=np.float32)

        # --- START OF COORDINATE FRAME FIX ---
        # 3) Transform the WORLD pose from OCTO to the BASE frame for the IK solver.
        # Get the robot's base pose in the world frame from the environment
        base_pos, base_quat_xyzw = self._env.get_base_pose()

        # Create Scipy rotation objects from the quaternions (format is xyzw)
        R_world_base = R.from_quat(base_quat_xyzw)
        R_world_target = R.from_quat(target_pose_world_7d[3:])

        # Find the transformation from the world frame to the base frame
        R_base_world = R_world_base.inv()

        # Transform the target's position and orientation into the base frame
        pos_target_base = R_base_world.apply(target_pose_world_7d[:3] - base_pos)
        R_target_base = R_base_world * R_world_target
        quat_target_base_xyzw = R_target_base.as_quat()

        # This is the final, corrected target pose for the IK solver
        target_pose_base_7d = np.concatenate([pos_target_base, quat_target_base_xyzw])
        # --- END OF COORDINATE FRAME FIX ---

        # 4) Current joint angles from proprio (first 7)
# --- THIS IS THE NEW, CORRECTED BLOCK ---
        # 4) Current joint angles from our DEDICATED internal key
        full_proprio = np.asarray(obs["internal_full_proprio"], dtype=np.float32)
        current_joint_angles = full_proprio[:7] # Use the first 7 values (qpos)

        # 5) Solve IK using the CORRECTED base-frame pose
        expert_action = self._ik_solver.compute_action(target_pose_base_7d, current_joint_angles)
        expert_action = self._align_action_dim(expert_action)

        # 6) Convert obs -> tensors for the training batch
        image_tensor = _to_torch_image(obs["image_primary"])
        # The proprio tensor should be the full 14D state we use for policy training
        proprio_tensor = torch.from_numpy(full_proprio)

        # The final output dictionary for the training loop.
        # We use the key "proprio" here by convention for the downstream consumer.
        obs_t = {"image_primary": image_tensor, "proprio": proprio_tensor}
        action_t = torch.from_numpy(expert_action)

        # Optionally move to device here (controlled by move_to_device)
        if self.move_to_device and (self.device is not None):
            obs_t = {k: v.to(self.device) for k, v in obs_t.items()}
            action_t = action_t.to(self.device)

        return obs_t, action_t

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
                    sample = self._generate_one()
                    samples_this_epoch += 1
                    self._samples_yielded += 1
                    yield sample

                except Exception as exc:
                    if self.skip_on_error:
                        logger.warning(f"[ExpertDataset] Skipping sample due to error: {exc}")
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
