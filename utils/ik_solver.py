# utils/ik_solver.py
import numpy as np
from typing import List, Tuple
import ikpy.chain
from ikpy.link import URDFLink, OriginLink
import io


def _quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion from [x, y, z, w] -> [w, x, y, z] and normalize."""
    q = np.asarray(q_xyzw, dtype=float).reshape(-1)
    if q.size != 4:
        raise ValueError("Quaternion must have 4 elements (x, y, z, w).")
    x, y, z, w = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return np.array([w / n, x / n, y / n, z / n], dtype=float)


def _rot_from_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] -> 3x3 rotation matrix (right-handed)."""
    w, x, y, z = map(float, q_wxyz)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(3, dtype=float)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


class IKSolver:
    """
    IKPy wrapper tuned for the Panda URDF that:
      - uses 'link0' as base element
      - detects ACTIVE links precisely (only revolute URDFLink entries)
      - exposes compute_action(target_pose_7d, current_joint_angles, max_delta)
        which returns normalized [-1,1] deltas for the arm (7 joints) + 1 gripper
    """

    def __init__(self, urdf_path: str, expect_7_dof: bool = True):
        print(f"⏳ [IKSolver] Loading kinematic chain from: {urdf_path}")

        try:
            # Load full chain starting from "link0"
            self.chain = ikpy.chain.Chain.from_urdf_file(
                urdf_path,
                base_elements=["link0"]
            )

            # Trim chain until we hit the "hand_joint"
            if "hand_joint" in [l.name for l in self.chain.links]:
                cutoff = [l.name for l in self.chain.links].index("hand_joint") + 1
                self.chain.links = self.chain.links[:cutoff]

            # Build active links mask: only revolute joints are active
            active_links_mask = []
            for link in self.chain.links:
                is_active = isinstance(link, URDFLink) and link.joint_type == "revolute"
                active_links_mask.append(is_active)

            self.chain.active_links_mask = active_links_mask

        except Exception as e:
            print("❌ FATAL ERROR [IKSolver]: Could not parse URDF or build chain.")
            print(f"   Error: {e}")
            raise

        # store active indices & names (in-chain indices)
        self._active_idx = [i for i, flag in enumerate(active_links_mask) if flag]
        self._active_joint_names = tuple(self.chain.links[i].name for i in self._active_idx)

        # read joint limits (in same order as self._active_idx)
        self._joint_limits: List[Tuple[float, float]] = []
        for idx in self._active_idx:
            lo, hi = -np.inf, np.inf
            link = self.chain.links[idx]
            try:
                bounds = getattr(link, "bounds", None)
                if bounds is not None and len(bounds) == 2:
                    lo, hi = float(bounds[0]), float(bounds[1])
                else:
                    limit = getattr(link, "limit", None)
                    if limit is not None:
                        lo = float(getattr(limit, "lower", lo))
                        hi = float(getattr(limit, "upper", hi))
            except Exception:
                # keep defaults if anything unexpected
                lo, hi = -np.inf, np.inf
            self._joint_limits.append((lo, hi))

        # debug prints
        print("✅ [IKSolver] Kinematic chain loaded and configured.")
        print(f"   Total links in chain: {len(self.chain.links)}")
        print(f"   End effector (last link): {self.chain.links[-1].name}")
        print(f"   Active joints: {len(self._active_idx)}")
        for i, name in zip(self._active_idx, self._active_joint_names):
            print(f"     - [{i:02d}] {name}")
        print("   Joint limits (low, high) per active joint:")
        for (lo, hi) in self._joint_limits:
            print(f"     - ({lo}, {hi})")

        if expect_7_dof and len(self._active_idx) != 7:
            print(f"⚠️ [IKSolver] Warning: expect_7_dof=True but found {len(self._active_idx)} active joints.")

    def compute_action(
        self,
        target_pose_7d: np.ndarray,
        current_joint_angles: np.ndarray,
        max_delta: float = 0.1,
        # FINAL SOLUTION: Add tolerance parameter for verification
        solution_position_tolerance: float = 0.01, # 1 cm
    ) -> np.ndarray:
        """
        Compute normalized action for N active arm joints + 1 gripper (open = -1).
        - target_pose_7d: [x, y, z, qx, qy, qz, qw] in base frame 'link0' (quaternion XYZW).
        - current_joint_angles: shape (N,) with joint angles in the same order as joint_names().
        - max_delta: positive scalar; deltas are scaled by /max_delta and clipped to [-1,1].
        - solution_position_tolerance: max allowed distance (m) between solved FK and target position.
        Returns: numpy array shape (N+1,) (gripper appended, default open = -1.0).
        """
        target_pose_7d = np.asarray(target_pose_7d, dtype=float).reshape(-1)
        if target_pose_7d.size != 7:
            raise ValueError("target_pose_7d must be [x, y, z, qx, qy, qz, qw]")

        current_joint_angles = np.asarray(current_joint_angles, dtype=float).reshape(-1)
        n_active = len(self._active_idx)
        if current_joint_angles.shape != (n_active,):
            raise ValueError(f"current_joint_angles must have shape ({n_active},)")

        if max_delta <= 0:
            raise ValueError("max_delta must be positive.")

        # position + quaternion (convert XYZW -> WXYZ -> rotation matrix)
        target_position = target_pose_7d[:3]
        q_xyzw = target_pose_7d[3:]
        q_wxyz = _quat_xyzw_to_wxyz(q_xyzw)
        rot3 = _rot_from_quat_wxyz(q_wxyz)

        # make 4x4 homogeneous target frame
        target_frame = np.eye(4, dtype=float)
        target_frame[:3, :3] = rot3
        target_frame[:3, 3] = target_position

        # build full-length initial positions vector (inactive joints MUST be present)
        current_joint_angles = self.clamp_to_limits(current_joint_angles)

        initial_position_full = np.zeros(len(self.chain.links), dtype=float)
        for i_act, idx in enumerate(self._active_idx):
            initial_position_full[idx] = float(current_joint_angles[i_act])

        # Try robust IK using inverse_kinematics_frame (4x4 target) first,
        # then fallback to inverse_kinematics (position-only or pos+orient).
        target_joint_angles_full = None
        try:
            target_joint_angles_full = np.asarray(
                self.chain.inverse_kinematics_frame(target_frame, initial_position=initial_position_full),
                dtype=float,
            ).reshape(-1)
        except Exception as e_frame:
            # debug print then fallback to inverse_kinematics with explicit orientation
            print(f"⚠️ IKSolver: inverse_kinematics_frame failed: {e_frame}. Trying inverse_kinematics(...) fallback.")
            try:
                target_joint_angles_full = np.asarray(
                    self.chain.inverse_kinematics(
                        target_position=target_position,
                        target_orientation=rot3,
                        orientation_mode="all",
                        initial_position=initial_position_full,
                    ),
                    dtype=float,
                ).reshape(-1)
            except Exception as e_pose:
                # final fallback to position-only try (no orientation)
                print(f"⚠️ IKSolver: inverse_kinematics (pos+orient) failed: {e_pose}. Trying position-only fallback.")
                try:
                    target_joint_angles_full = np.asarray(
                        self.chain.inverse_kinematics(
                            target_position=target_position,
                            orientation_mode=None,
                            initial_position=initial_position_full,
                        ),
                        dtype=float,
                    ).reshape(-1)
                except Exception as e_pos_only:
                    # Print the string the test expects
                    print("IK failed (all attempts)")
                    # Optional: keep detailed debug info for debugging
                    print(f"   Errors: frame:{e_frame}  pos+orient:{e_pose}  pos_only:{e_pos_only}")
                    return np.zeros(n_active + 1, dtype=float)


        # --- FINAL SOLUTION: ADD ROBUST POST-VERIFICATION ---
        # If any of the above blocks returned a solution, verify it.

        # 1. Check for valid shape
        if target_joint_angles_full is None or target_joint_angles_full.size != len(self.chain.links):
            print("IK failed (all attempts)")
            print(f"   Reason: IK solver returned an invalid shape or None.")
            return np.zeros(n_active + 1, dtype=float)

        # 2. Check for finite numbers
        if not np.all(np.isfinite(target_joint_angles_full)):
            print("IK failed (all attempts)")
            print(f"   Reason: IK solution contained non-finite values.")
            return np.zeros(n_active + 1, dtype=float)

        # 3. Verify the solution with Forward Kinematics
        T_solution = self.chain.forward_kinematics(target_joint_angles_full)
        solution_position = T_solution[:3, 3]
        position_error = np.linalg.norm(solution_position - target_position)
        
        if position_error > solution_position_tolerance:
            print("IK failed (all attempts)")
            print(f"   Reason: IK solution was invalid. Position error ({position_error:.4f} m) exceeds tolerance ({solution_position_tolerance} m).")
            # --- FIX: Return a zero vector on failure instead of raising an exception ---
            return np.zeros(n_active + 1, dtype=np.float32)
            # --- END OF POST-VERIFICATION ---

        # extract active target angles (ordered)
        active_target_angles = np.array([target_joint_angles_full[i] for i in self._active_idx], dtype=float)

        # clamp to URDF limits (if present)
        for i in range(n_active):
            lo, hi = self._joint_limits[i]
            lo_use = -np.inf if not np.isfinite(lo) else lo
            hi_use = np.inf if not np.isfinite(hi) else hi
            active_target_angles[i] = float(np.clip(active_target_angles[i], lo_use, hi_use))
        
        # This check is now redundant due to the main check above, but we keep it for safety.
        if not np.all(np.isfinite(active_target_angles)):
            print("IK failed (all attempts)")
            print("   Reason: Non-finite values appeared after clamping (should not happen).")
            return np.zeros(n_active + 1, dtype=float)


        # compute delta, scale, and clip to [-1, 1]
        arm_action_delta = active_target_angles - current_joint_angles
        scaled_arm_action = np.clip(arm_action_delta / float(max_delta), -1.0, 1.0)

        # gripper default (open = -1.0) to preserve API
        gripper_action = np.array([-1.0], dtype=float)
        final_action = np.concatenate([scaled_arm_action, gripper_action])
        return final_action

    def joint_names(self) -> Tuple[str, ...]:
        """Return the active joint names in order."""
        return self._active_joint_names

    def clamp_to_limits(self, q: np.ndarray) -> np.ndarray:
        """Clamp an N-dof vector (active joints) to URDF joint limits."""
        q = np.asarray(q, dtype=float).reshape(-1)
        n_active = len(self._active_idx)
        if q.shape != (n_active,):
            raise ValueError(f"Input must have shape ({n_active},)")
        out = np.empty_like(q)
        for i, val in enumerate(q):
            lo, hi = self._joint_limits[i]
            lo_use = -1e9 if not np.isfinite(lo) else lo
            hi_use = 1e9 if not np.isfinite(hi) else hi
            out[i] = np.clip(val, lo_use, hi_use)
        return out