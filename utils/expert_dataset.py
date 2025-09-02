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
        self._octo_model = None
        self._env = None
        self._ik_solver = None
        self._task = None
        self._jax_key = None
        self._samples_yielded = 0
        self.skipped_samples = 0

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

        # (Inside _init_worker_state)
        # (Inside _init_worker_state)
        if self.warmup:
            logger.info(f"[worker {worker_id}] Starting warmup inference...")
            try:
                obs, _ = self._env.reset()
                T = 2
                # Correctly exclude proprio and build observation
                octo_obs = {
                    k: np.stack([v, v], axis=0)[np.newaxis, ...]
                    for k, v in obs.items()
                    if k not in {"internal_full_proprio", "pad_mask_dict", "proprio"}
                }
                # timestep (1, T)
                if "timestep" in obs:
                    t = np.asarray(obs["timestep"], dtype=np.int32)
                    t0 = int(t.reshape(()) if t.ndim == 0 else t.ravel()[0])
                    octo_obs["timestep"] = np.full((1, T), t0, dtype=np.int32)
                else:
                    octo_obs["timestep"] = np.arange(T, dtype=np.int32)[None, :]

                # task_completed (1, T, 4)
                # tc_src = np.asarray(obs.get("task_completed", np.zeros(4, np.float32)), dtype=np.float32).reshape(-1)
                # tc = np.zeros(4, np.float32); n = min(tc_src.size, 4)
                # if n > 0: tc[:n] = tc_src[:n]
                # octo_obs["task_completed"] = np.tile(tc.reshape(1, 1, -1), (1, T, 1))

                # nested pad masks only
                pm = obs.get("pad_mask_dict", {})
                octo_obs["pad_mask_dict"] = {k: np.ones((1, T), dtype=bool) for k in pm if k in octo_obs}

                self._jax_key, key = jax.random.split(self._jax_key)
                _ = self._octo_model.sample_actions(octo_obs, task=self._task, rng=key)
                logger.info(f"[worker {worker_id}] Warmup OCTO run complete.")
            except Exception as e:
                logger.error(f"[worker {worker_id}] Warmup FAILED: {e}", exc_info=True)
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
    def _generate_one(self) -> Tuple[Dict, np.ndarray]:
        """
        Generates a single (observation_dict, action_array) sample as numpy arrays.
        Robust: builds OCTO input inline, normalizes quaternions, transforms
        world->base, tries full-pose IK then position-only fallback.
        """
        import numpy as _np
        from scipy.spatial.transform import Rotation as Rlocal

        # 1) Get a fresh observation (single timestep)
        obs, _ = self._env.reset()

        # 2) Build OCTO-style observation dict (B=1, T=2) - do NOT call any old _prepare_octo_obs
        # 2) Build OCTO-style observation dict (B=1, T=2)
        T = 2
        B = 1
        octo_obs: Dict[str, _np.ndarray] = {}

        # Exclude modalities OCTO doesn't expect
        for k, v in obs.items():
            if k in {"internal_full_proprio", "pad_mask_dict", "proprio"}:
                continue
            arr = _np.asarray(v)
            if arr.ndim == 0:
                arr = arr.reshape(())
            stacked = _np.stack([arr, arr], axis=0)  # (T, ...)
            octo_obs[k] = stacked[_np.newaxis, ...]  # (1, T, ...)

        # Timestep must be (B, T)
        if "timestep" in obs:
            t = _np.asarray(obs["timestep"], dtype=_np.int32)
            t0 = int(t.reshape(()) if t.ndim == 0 else t.ravel()[0])
            octo_obs["timestep"] = _np.full((B, T), t0, dtype=_np.int32)
        else:
            octo_obs["timestep"] = np.arange(T, dtype=np.int32).reshape(B, T)

        # Task completed must be (B, T, W)
        # tc_src = _np.asarray(obs.get("task_completed", _np.zeros(4, dtype=_np.float32)), dtype=_np.float32).reshape(-1)
        # tc_width = 4
        # tc_vec = _np.zeros(tc_width, dtype=_np.float32)
        # n = min(tc_src.size, tc_width)
        # if n > 0:
        #     tc_vec[:n] = tc_src[:n]
        # octo_obs["task_completed"] = _np.tile(tc_vec.reshape(1, 1, -1), (B, T, 1))
            
        # Nested pad masks only, aligned with included keys
        pm = obs.get("pad_mask_dict", {})
        octo_obs["pad_mask_dict"] = {
            k: _np.ones((B, T), dtype=bool) for k in pm if k in octo_obs
        }


        octo_obs["timestep_pad_mask"] = _np.ones((B, T), dtype=bool)  # legacy alias

        # Build octo_input for clarity (also used by diagnostic prints)
        octo_input = {"observations": octo_obs, "task": self._task}

        # 3) Query OCTO (deterministic key split)
        self._jax_key, key = jax.random.split(self._jax_key)
        raw_action = self._octo_model.sample_actions(octo_obs, self._task, rng=key)
        target_pose_world = _np.array(raw_action[0, 0, :7], dtype=_np.float32, copy=True)

        # --- Diagnostics (first few samples) ---
        if self._samples_yielded < 5:
            print("\n" + "=" * 70)
            print(f"--- In-Depth Diagnostic Check for Sample #{self._samples_yielded + 1} ---")
            try:
                print("\n[A. VERIFYING SHAPES SENT TO OCTO]")
                for k, v in octo_input["observations"].items():
                    if isinstance(v, dict):
                        print(f"  - {k}: <dict with {len(v)} keys>")
                    else:
                        print(f"  - {k}: {v.shape}")
            except Exception as e:
                print("  - Diagnostics failed to collect shapes:", e)
            # base / world checks (kept as before)
            try:
                env_base_pos, env_base_quat = self._env.get_base_pose()
                mujoco.mj_forward(self._env.model, self._env.data)
                link0_id = mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_BODY, "link0")
                link0_pos_world = self._env.data.xpos[link0_id].copy()
                print("\n[B. BASE FRAME VERIFICATION]")
                print(f"  - Pose from env.get_base_pose(): pos={_np.round(env_base_pos,3)}")
                print(f"  - Ground Truth Pose of 'link0': pos={_np.round(link0_pos_world,3)}")
            except Exception as e:
                print("  - Base frame diagnostic error:", e)
            try:
                ee_pose_world = self._env.get_ee_pose()
                print("\n[C. WORLD COORDINATE SANITY CHECK]")
                print(f"  - OCTO Predicted Target Z: {target_pose_world[2]:.3f}")
                print(f"  - EE Z (env): {ee_pose_world[2]:.3f}")
                if _np.sign(target_pose_world[2]) != _np.sign(ee_pose_world[2]) and ee_pose_world[2] > 0.1:
                    print("  - ❌ SMOKING GUN: Z-axis mismatch detected!")
                else:
                    print("  - ✅ Z-axis seems consistent")
            except Exception:
                pass
            print("=" * 70 + "\n")

        # 4) Normalize quaternion and basic checks
        q = _np.asarray(target_pose_world[3:7], dtype=_np.float64)
        qn = _np.linalg.norm(q)
        if not _np.isfinite(qn) or qn < 1e-6:
            q = _np.array([0.0, 0.0, 0.0, 1.0], dtype=_np.float64)
        else:
            q = q / qn
        target_pose_world[3:7] = q.astype(_np.float32)

        # 5) Get base and ee poses defensively
        try:
            base_pos, base_quat = self._env.get_base_pose()
        except Exception:
            base_pos = _np.zeros(3, dtype=_np.float32)
            base_quat = _np.array([0.0, 0.0, 0.0, 1.0], dtype=_np.float32)
        try:
            ee_pose_world = self._env.get_ee_pose()
            ee_pos_world = _np.asarray(ee_pose_world[:3], dtype=_np.float32)
            ee_quat_world = _np.asarray(ee_pose_world[3:7], dtype=_np.float32)
        except Exception:
            ee_pos_world = base_pos
            ee_quat_world = base_quat

        # 6) Heuristic Z flip
        if (float(target_pose_world[2]) < 0.0) and (float(ee_pos_world[2]) > 0.1):
            logger.debug("[Coordinate Fix] Flipping Z-axis of OCTO target pose and using EE orientation.")
            target_pose_world[2] *= -1.0
            target_pose_world[3:7] = ee_quat_world

        # 7) Workspace guard
        dist_from_base = _np.linalg.norm(target_pose_world[:3] - base_pos)
        MAX_TARGET_DIST = 1.5
        if dist_from_base > MAX_TARGET_DIST:
            raise RuntimeError(f"Target pose is too far from robot base ({dist_from_base:.2f}m > {MAX_TARGET_DIST}m).")

        # 8) World -> base transform
        R_world_base = Rlocal.from_quat(base_quat)  # base_quat xyzw
        R_base_world = R_world_base.inv()
        pos_target_base = R_base_world.apply(target_pose_world[:3] - base_pos)
        R_target_base = R_base_world * Rlocal.from_quat(target_pose_world[3:7])
        target_pose_base = _np.concatenate([pos_target_base, R_target_base.as_quat()]).astype(_np.float32)

        # 9) Solve IK
        full_proprio = _np.asarray(obs["internal_full_proprio"], dtype=_np.float32)
        current_joints = full_proprio[:7].astype(_np.float32)
        expert_action = self._ik_solver.compute_action(target_pose_base, current_joints)

        def _is_ik_failure(act: _np.ndarray, tol: float = 1e-6) -> bool:
            if act is None:
                return True
            try:
                return float(_np.linalg.norm(_np.asarray(act, dtype=_np.float64))) < tol
            except Exception:
                return True

        if _is_ik_failure(expert_action):
            # fallback: rotate ee_quat into base frame, try position-only
            R_ee_base = R_base_world * Rlocal.from_quat(ee_quat_world)
            ee_quat_base = R_ee_base.as_quat()
            pos_only_target = _np.concatenate([pos_target_base, ee_quat_base]).astype(_np.float32)
            expert_action = self._ik_solver.compute_action(pos_only_target, current_joints)

        if _is_ik_failure(expert_action):
            raise RuntimeError("IK failed for all attempts (full-pose and position-only).")

        # 10) align action dim & dtype
        final_action = self._align_action_dim(_np.asarray(expert_action, dtype=_np.float32))

        # Return original observation (unchanged) and final action vector (numpy)
        return obs, final_action

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
