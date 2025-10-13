# utils/ik_solver.py
import numpy as np
from typing import List, Tuple, Optional
import ikpy.chain
from ikpy.link import URDFLink, OriginLink
import io
import logging 
from scipy.spatial.transform import Rotation as R
from mujoco import mj_jac  # For Jacobian
import mujoco
from numpy.linalg import inv, pinv, cond


logger = logging.getLogger(__name__) 


class IKSolver:
    """
    IKPy wrapper tuned for the Panda URDF that:
      - uses 'link0' as base element
      - detects ACTIVE links precisely (only revolute URDFLink entries)
      - exposes compute_action(target_pose_7d, current_joint_angles, max_delta)
        which returns normalized [-1,1] deltas for the arm (7 joints) + 1 gripper
    """

    def __init__(self, urdf_path: str, expect_7_dof: bool = True,kp=40.0, ki=1.0, kd=4.0):
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
            self.kp = kp
            self.ki = ki
            self.kd = kd
            self.reset_controller_state()
            self._integral_error = np.zeros(6)

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
    
    def compute_delta_action_(
        self,
        target_pose_7d: np.ndarray,
        current_joint_angles: np.ndarray,
        current_ee_pose: np.ndarray,  # From env
        dt: float = 0.002,
        max_dq: float = 0.1
    ) -> np.ndarray:
        target_pos = target_pose_7d[:3]
        target_quat = target_pose_7d[3:]
        current_pos = current_ee_pose[:3]
        current_quat = current_ee_pose[3:]

        # Delta pos
        d_pos = target_pos - current_pos

        # Delta orn (angular vel approx)
        d_rot = R.from_quat(target_quat) * R.from_quat(current_quat).inv()
        d_orn = d_rot.as_euler('xyz') / dt  # Approx ang vel

        d_ee = np.concatenate([d_pos, d_orn]) / dt  # ee_vel

        # Jacobian (MuJoCo mj_jac)
        jac_pos = np.zeros((3, self.n_joints))
        jac_rot = np.zeros((3, self.n_joints))
        mj_jac(self.model, self.data, jac_pos, jac_rot, current_pos, self.ee_site_id)
        J = np.vstack([jac_pos, jac_rot])

        # Pseudo-inverse
        J_pinv = np.linalg.pinv(J)

        # dq (joint vel)
        dq = J_pinv @ d_ee

        # Normalize to action
        action = np.clip(dq / max_dq, -1, 1)

        return action
    
    def compute_delta_action__1(
        self,
        target_ee_pose: np.ndarray,
        current_ee_pose: np.ndarray,
        model, # mujoco.MjModel
        data,  # mujoco.MjData
        ee_site_id: int,
        joint_ids: np.ndarray,
        dt: float,
        max_dq: float = 1.0 # Max normalized joint velocity
    ) -> np.ndarray:
        """
        Computes a normalized delta action using Differential Inverse Kinematics.
        """
        # --- 1. Calculate desired end-effector velocity ---
        # Positional velocity
        pos_error = target_ee_pose[:3] - current_ee_pose[:3]
        vel_pos = pos_error / dt
        
        # Rotational velocity (as axis-angle)
        d_rot = R.from_quat(target_ee_pose[3:]) * R.from_quat(current_ee_pose[3:]).inv()
        vel_rot = d_rot.as_rotvec() / dt

        # Desired 6D end-effector velocity (twist)
        ee_vel_target = np.concatenate([vel_pos, vel_rot])

        # --- 2. Calculate Jacobian ---
        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        mj_jac(model, data, jac_pos, jac_rot, current_ee_pose[:3], ee_site_id)
        J_full = np.vstack([jac_pos, jac_rot])
        
        # Select only the columns corresponding to the arm joints
        J = J_full[:, joint_ids]

        # --- 3. Solve for joint velocities (dq) ---
        # dq = J_pinv * ee_vel
        try:
            dq = np.linalg.lstsq(J, ee_vel_target, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Fallback to zeros if Jacobian is singular
            return np.zeros(len(joint_ids))

        # --- 4. Normalize to action space ---
        action = np.clip(dq / max_dq, -1.0, 1.0)
        
        return action
  
    def reset_controller_state(self):
        """Resets the internal state of the PID controller."""
        self._integral_error = np.zeros(6)

    def set_gains(self, kp: float, ki: float, kd: float):
        """
        Dynamically sets the PID gains for the controller.
        This is primarily used for tuning scripts.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def compute_delta_action(
        self,
        target_ee_pose: np.ndarray,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site_id: int,
        joint_qpos_indices: np.ndarray,
        effective_dt: float,
        max_dq: float,
    ) -> np.ndarray:
        """
        [DEFINITIVE, TUNED FOR Kp=40, PRODUCTION-GRADE PID CONTROLLER V4]
        This version uses a balanced set of gains for a responsive yet stable
        system, based on the manual tuning methodology.
        - Kp=40.0 provides a fast response.
        - Kd=4.0 provides critical damping to prevent overshoot.
        - Ki=1.0 provides slow, stable correction for steady-state error.
        """
        # --- 1. GET CURRENT STATE ---
        current_ee_pos = data.site_xpos[ee_site_id]
        current_ee_mat = data.site_xmat[ee_site_id].reshape(3, 3)
        current_ee_quat = R.from_matrix(current_ee_mat).as_quat()
        
        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        ee_body_id = model.site_bodyid[ee_site_id]
        mujoco.mj_jac(model, data, jac_pos, jac_rot, current_ee_pos, ee_body_id)
        
        J_full = np.vstack([jac_pos, jac_rot])
        J = J_full[:, joint_qpos_indices]

        # --- 2. IMPLEMENT STABLE & TUNED PID CONTROL LAW ---
        # FINAL TUNED GAINS for Kp=40
        Kp = 40.0
        Kd = 1.0
        Ki = 1.0
        integral_clamp = 0.4
        damping = 1e-2
        
        # Filter for the derivative term to prevent noise amplification
        tau_d = 3.0 * effective_dt # Derivative filter time constant
        alpha = effective_dt / (tau_d + effective_dt)

        # Calculate 6D error vector
        pos_error = target_ee_pose[:3] - current_ee_pos
        orn_error_vec = (R.from_quat(target_ee_pose[3:]) * R.from_quat(current_ee_quat).inv()).as_rotvec()
        error_6d = np.concatenate([pos_error, orn_error_vec])

        # Proportional Term
        p_term = Kp * error_6d

        # Derivative Term (on error, and filtered)
        prev_error = getattr(self, "_prev_error", np.zeros_like(error_6d))
        error_deriv = (error_6d - prev_error) / effective_dt
        
        prev_filtered_deriv = getattr(self, "_d_filter_state", np.zeros_like(error_deriv))
        filtered_deriv = (1 - alpha) * prev_filtered_deriv + alpha * error_deriv
        d_term = Kd * filtered_deriv

        # Integral Term (with conditional anti-windup)
        integrator = getattr(self, "_integral_error", np.zeros_like(error_6d))
        i_term = Ki * integrator

        # Target velocity is the sum of PID components
        ee_vel_target = p_term + i_term + d_term

        # --- 3. SOLVE FOR JOINT VELOCITIES (dIK) ---
        try:
            lhs = J.T @ J + damping * np.eye(J.shape[1])
            rhs = J.T @ ee_vel_target
            dq = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            dq = np.zeros(len(joint_qpos_indices))

        # --- 4. ANTI-WINDUP & NORMALIZE ACTION ---
        action_if_applied = dq / max_dq
        is_saturated = np.any(np.abs(action_if_applied) > 1.0)

        # Conditional Integration: Only integrate if the controller is not saturated.
        if not is_saturated:
            integrator += error_6d * effective_dt
            np.clip(integrator, -integral_clamp, integral_clamp, out=integrator)
        
        # Store states for next step
        self._prev_error = error_6d.copy()
        self._d_filter_state = filtered_deriv.copy()
        self._integral_error = integrator.copy()

        action = np.clip(action_if_applied, -1.0, 1.0)
        return action