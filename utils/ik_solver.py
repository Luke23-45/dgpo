# utils/ik_solver.py
import numpy as np
from typing import List, Tuple, Optional
import ikpy.chain
from ikpy.link import URDFLink, OriginLink
import io
import logging 

logger = logging.getLogger(__name__) 




class IKSolver:
    """
    IKPy wrapper tuned for the Panda URDF that:
      - uses 'link0' as base element
      - detects ACTIVE links precisely (only revolute URDFLink entries)
      - exposes compute_action(target_pose_7d, current_joint_angles, max_delta)
        which returns normalized [-1,1] deltas for the arm (7 joints) + 1 gripper
    """

    def __init__(self, urdf_path: str, expect_7_dof: bool = True):
        logger.info(f"⏳ [IKSolver] Loading kinematic chain from: {urdf_path}")
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
            logger.critical("Could not parse URDF or build chain.", exc_info=True)
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
        logger.info("Kinematic chain loaded and configured.")
        logger.debug(f"End effector (last link): {self.chain.links[-1].name}")
        logger.debug(f"Found {len(self._active_idx)} active joints: {self._active_joint_names}")

        if expect_7_dof and len(self._active_idx) != 7:
            logger.warning(f"Expected 7 active joints but found {len(self._active_idx)}.")
  
    @staticmethod
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

    @staticmethod
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
    def _get_target_joint_angles(
        self,
        target_pose_7d: np.ndarray,
        current_joint_angles: np.ndarray,
        solution_position_tolerance: float,
    ) -> Optional[np.ndarray]:
        """
        Solves the inverse kinematics problem to find a target joint configuration.

        This method encapsulates the core IK solving, including fallbacks and
        rigorous post-verification. It cleanly separates the kinematic
        problem from any control law.

        Args:
            target_pose_7d: The desired 7D pose of the end-effector [x, y, z, qx, qy, qz, qw].
            current_joint_angles: The current angles of the active joints, used as an initial guess.
            solution_position_tolerance: The maximum allowed Euclidean distance (in meters)
                                         between the solved pose and the target pose.

        Returns:
            A NumPy array of shape (7,) containing the target joint angles if a valid
            solution is found, otherwise None.
        """
        n_active = len(self._active_idx)

        # --- 1. Prepare IK Inputs ---
        target_position = target_pose_7d[:3]
        target_orientation_matrix = self._rot_from_quat_wxyz(self._quat_xyzw_to_wxyz(target_pose_7d[3:]))

        target_frame = np.eye(4, dtype=float)
        target_frame[:3, :3] = target_orientation_matrix
        target_frame[:3, 3] = target_position
        
        initial_position_full = np.zeros(len(self.chain.links), dtype=float)
        for i_act, idx in enumerate(self._active_idx):
            initial_position_full[idx] = float(current_joint_angles[i_act])

        # --- 2. Solve IK with Fallbacks ---
        target_joint_angles_full = None
        try:
            target_joint_angles_full = self.chain.inverse_kinematics_frame(
                target_frame, initial_position=initial_position_full
            )
        except Exception:
            # Fallback to position-only if frame-based IK fails
            try:
                target_joint_angles_full = self.chain.inverse_kinematics(
                    target_position=target_position,
                    orientation_mode=None, # Orientation is ignored for this fallback
                    initial_position=initial_position_full,
                )
            except Exception:
                logger.warning("IK failed on all attempts (frame and position-only).")
                return None

        # --- 3. Rigorous Post-Verification ---
        if target_joint_angles_full is None or target_joint_angles_full.size != len(self.chain.links):
            logger.warning("IK failed: Solver returned an invalid shape or None.")
            return None

        if not np.all(np.isfinite(target_joint_angles_full)):
            logger.warning("IK failed: Solution contained non-finite values.")
            return None

        # Verify the solution with Forward Kinematics
        fk_solution_frame = self.chain.forward_kinematics(target_joint_angles_full)
        position_error = np.linalg.norm(fk_solution_frame[:3, 3] - target_position)
        
        if position_error > solution_position_tolerance:
            logger.warning(
                f"IK failed: Solution error ({position_error:.4f} m) "
                f"exceeds tolerance ({solution_position_tolerance} m)."
            )
            return None

        # --- 4. Extract and Clamp Active Joints ---
        active_target_angles = np.array(
            [target_joint_angles_full[i] for i in self._active_idx], dtype=np.float32
        )
        
        # This is a redundant check due to the one above, but it's good practice
        if not np.all(np.isfinite(active_target_angles)):
            logger.warning("IK failed: Non-finite values appeared after extracting active joints.")
            return None
            
        return self.clamp_to_limits(active_target_angles)

    def compute_action(
        self,
        target_pose_7d: np.ndarray,
        current_joint_angles: np.ndarray,
        max_delta: float = 0.1,
        solution_position_tolerance: float = 0.01,
    ) -> np.ndarray:
        """
        Computes a normalized velocity action to move towards a target pose.

        This method implements a Proportional (P) control law in joint space.
        It first uses the internal IK solver to find the target joint angles
        and then computes a scaled delta to be used as a velocity command.

        Args:
            target_pose_7d: The desired 7D pose [x, y, z, qx, qy, qz, qw].
            current_joint_angles: The current angles of the 7 active joints.
            max_delta: The maximum joint angle change (in radians) that corresponds
                       to a normalized action of 1.0. This is the P-gain.
            solution_position_tolerance: The tolerance for the underlying IK solver.

        Returns:
            A NumPy array of shape (8,) representing the normalized action
            for 7 arm joints and 1 gripper. Returns a zero vector on IK failure.
        """
        if max_delta <= 0:
            raise ValueError("max_delta must be positive.")
        
        n_active = len(self._active_idx)

        # 1. Solve for the target joint configuration using the robust internal solver
        target_joint_angles = self._get_target_joint_angles(
            target_pose_7d,
            current_joint_angles,
            solution_position_tolerance,
        )

        # 2. Handle IK failure
        if target_joint_angles is None:
            # On failure, command a zero-velocity action to hold position
            return np.zeros(n_active, dtype=np.float32)

        # 3. Compute the Proportional control action
        arm_action_delta = target_joint_angles - current_joint_angles
        
        # Scale the delta by the gain (max_delta) and clip to [-1, 1]
        scaled_arm_action = np.clip(arm_action_delta / max_delta, -1.0, 1.0)

        return scaled_arm_action.astype(np.float32)
    
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