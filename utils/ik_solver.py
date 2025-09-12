# utils/ik_solver.py
import numpy as np
from typing import List, Tuple, Optional
import ikpy.chain
from ikpy.link import URDFLink, OriginLink
import io
import logging 
from scipy.spatial.transform import Rotation as R
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
            tcp_virtual_link = URDFLink(
                name="attachment_site_virtual_link",
                origin_translation=[0, 0, 0.1],  # 10cm Z-offset
                origin_orientation=[0, 0, 0],    # No rotation
                joint_type="fixed",
            )

            # 3. Create a new chain by appending the virtual link to the original's list of links.
            #    This is the correct way to modify the chain.
            self.chain = ikpy.chain.Chain(
                self.chain.links + [tcp_virtual_link]
            )

            # Trim chain until we hit the "hand_joint"
            self.chain.active_links_mask = [
                False, # link0 (base) is not active
                True,  # joint1
                True,  # joint2
                True,  # joint3
                True,  # joint4
                True,  # joint5
                True,  # joint6
                True,  # joint7
                False,  # hand_joint (fixed) is not active
                False
            ]
            
            self._active_idx = [i for i, flag in enumerate(self.chain.active_links_mask) if flag]
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

        except Exception as e:
            logger.critical("Could not parse URDF or build chain.", exc_info=True)
            raise


  
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
# FILE: utils/ik_solver.py
#
# REPLACE the entire _get_target_joint_angles method with this new version.

    def _get_target_joint_angles(
        self,
        target_pose_7d: np.ndarray,
        current_joint_angles: np.ndarray,
        solution_position_tolerance: float,
        max_iter: int = 100, # Reduced max_iter for speed
        regularization_strength: float = 1e-4
    ) -> Optional[np.ndarray]:
        """
        Solves the inverse kinematics problem with a robust, multi-attempt fallback strategy.
        """
        current_joint_angles = self.clamp_to_limits(current_joint_angles)
        target_pos = target_pose_7d[:3]
        target_orient_matrix = R.from_quat(target_pose_7d[3:]).as_matrix()

        initial_guess = [0.0] * len(self.chain.links)
        for i, joint_val in enumerate(current_joint_angles):
            initial_guess[self._active_idx[i]] = joint_val

        # --- START OF PATCH: Multi-Attempt IK Solving ---
        
        # Define our attempts, from most to least constrained
        orientation_modes = ["all", "Z", None]
        
        for i, mode in enumerate(orientation_modes):
            solved_joints_full = None # Reset solution for this attempt
            try:
                solved_joints_full = self.chain.inverse_kinematics(
                    target_position=target_pos,
                    target_orientation=target_orient_matrix if mode is not None else None,
                    orientation_mode=mode,
                    initial_position=initial_guess,
                    max_iter=max_iter,
                    regularization_parameter=regularization_strength
                )
            except Exception:
                # This attempt failed entirely, continue to the next one
                continue

            # If a solution was found, verify its quality
            if solved_joints_full is not None and np.all(np.isfinite(solved_joints_full)):
                fk_frame = self.chain.forward_kinematics(solved_joints_full)
                result_pos = fk_frame[:3, 3]
                position_error = np.linalg.norm(result_pos - target_pos)

                # If the solution is accurate enough, we are done!
                if position_error <= solution_position_tolerance:
                    active_solved_joints = [solved_joints_full[i] for i in self._active_idx]
                    return np.array(active_solved_joints, dtype=np.float32)
        
        # If all attempts failed to produce an accurate solution, return None
        logger.warning(f"IK solver failed on all attempts to reach {np.round(target_pos, 2)}.")
        return None
        # --- END OF PATCH ---

    def compute_action(
        self,
        target_pose_7d: np.ndarray,
        current_joint_angles: np.ndarray,
        max_delta: float = 0.1,  # This parameter is no longer used but kept for API consistency
        solution_position_tolerance: float = 0.01,
    ) -> np.ndarray:
        """
        Computes a normalized POSITION action to move towards a target pose.

        This is the definitive, correct version. It solves for the final target joint angles
        and normalizes them to the action space [-1, 1]. The underlying MuJoCo
        simulation's position controller is responsible for executing the motion.

        Returns:
            A NumPy array of shape (7,) representing the normalized target joint positions.
            Returns the current joint positions (a "hold" command) on IK failure.
        """
        n_active = len(self._active_idx)

        # 1. Solve for the final target joint configuration.
        target_joint_angles = self._get_target_joint_angles(
            target_pose_7d,
            current_joint_angles,
            solution_position_tolerance,
        )

        # 2. Handle IK failure: command a "hold position" action.
        if target_joint_angles is None:
            logger.warning("IK solver failed. Commanding a hold action (current joint positions).")
            target_joint_angles = current_joint_angles

        # 3. Normalize the absolute target joint angles to the action space [-1, 1].
        action = np.zeros(n_active, dtype=np.float32)
        for i in range(n_active):
            lo, hi = self._joint_limits[i]
            if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
                action[i] = np.clip(target_joint_angles[i], -1.0, 1.0)
                continue
            
            # Scale to [0, 1]
            scaled_pos = (target_joint_angles[i] - lo) / (hi - lo)
            # Scale to [-1, 1]
            action[i] = 2.0 * scaled_pos - 1.0

        # Clip to ensure it's strictly within the action space bounds.
        final_action = np.clip(action, -1.0, 1.0)

        if not np.all(np.isfinite(final_action)):
            logger.warning("IK produced a non-finite action. Returning zeros to prevent crash.")
            return np.zeros(n_active, dtype=np.float32)

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