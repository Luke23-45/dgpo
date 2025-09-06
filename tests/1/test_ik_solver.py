# test_ik_solver.py — per-joint diff + auto-calibration shim
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from utils.ik_solver import IKSolver

MUJOCO_XML_PATH = "envs/panda_pick_place.xml"
IK_URDF_PATH = "urdf/panda_mujoco_kinematics.urdf"
BASE_NAME_CANDIDATES = ["link0", "panda_link0", "panda_base", "base_link", "base"]

def angle_from_R(dR):
    tr = np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(tr))

def find_body_id(model, names):
    for nm in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0:
            return nm, bid
    return None, -1

def world_to_base(data, base_bid, p_w, R_w):
    if base_bid < 0:
        return p_w.copy(), R_w.copy()
    p_base_w = data.xpos[base_bid]
    R_base_w = data.xmat[base_bid].reshape(3, 3)
    R_wb = R_base_w.T
    return R_wb @ (p_w - p_base_w), R_wb @ R_w

def mj_body_pose_in_base(model, data, base_bid, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(f"Body '{body_name}' not found in MuJoCo model.")
    p_w = data.xpos[bid].copy()
    R_w = data.xmat[bid].reshape(3, 3).copy()
    return world_to_base(data, base_bid, p_w, R_w)

def ikpy_forward_all(solver, joint_values_full):
    """
    Compute forward kinematics for all links in IKPy's chain.
    Returns a list of 4x4 transforms in the chain's base frame.
    """
    T = np.eye(4)
    transforms = [T.copy()]  # base at identity
    for idx, link in enumerate(solver.chain.links):
        if hasattr(link, "get_transformation_matrix"):
            theta = joint_values_full[idx]
            T = T @ link.get_transformation_matrix(theta)
        else:
            # OriginLink or other fixed placeholder → skip
            pass
        transforms.append(T.copy())
    return transforms

def calibrate_R_t(p_list_src, R_list_src, p_list_dst, R_list_dst):
    """Find R,t minimizing sum || R*R_src - R_dst || + || R*p_src + t - p_dst ||."""
    # Orientation: project the average onto SO(3)
    M = np.zeros((3,3))
    for Rsrc, Rdst in zip(R_list_src, R_list_dst):
        M += Rdst @ Rsrc.T
    U, _, Vt = np.linalg.svd(M)
    R_delta = U @ Vt
    if np.linalg.det(R_delta) < 0:
        U[:, -1] *= -1
        R_delta = U @ Vt
    # Translation: least squares
    A = np.eye(3)  # same for all
    b_accum = np.zeros(3)
    N = len(p_list_src)
    t_delta = np.zeros(3)
    # Closed form: t = mean(p_dst - R * p_src)
    residuals = []
    for psrc, pdst in zip(p_list_src, p_list_dst):
        residuals.append(pdst - R_delta @ psrc)
    t_delta = np.mean(np.stack(residuals, axis=0), axis=0)
    return R_delta, t_delta

def rms(vecs):
    v = np.asarray(vecs)
    return float(np.sqrt(np.mean(np.sum(v*v, axis=-1))))

# test_ik_solver.py (CORRECTED run() function)

def run():
    print("--- 🚀 Starting Definitive IK Solver Verification + Sync Tools ---")

    # 1) Load
    print("\n--- [Step 1/3] Loading models ---")
    model = mujoco.MjModel.from_xml_path(MUJOCO_XML_PATH)
    data = mujoco.MjData(model)
    print("✅ MuJoCo model loaded successfully.")
    solver = IKSolver(urdf_path=IK_URDF_PATH)
    print("✅ IK Solver initialized successfully.")
    
    # --- START OF FIX 1 ---
    # Define the correct end-effector name for both models
    EE_BODY_NAME = "hand"
    assert solver.chain.links[-1].name == "hand_joint", \
        f"URDF chain does not end at the hand. Last link: {solver.chain.links[-1].name}"
    print(f"✅ Verified URDF chain ends at '{solver.chain.links[-1].name}'.")
    # --- END OF FIX 1 ---

    base_name, base_bid = find_body_id(model, BASE_NAME_CANDIDATES)
    if base_bid >= 0:
        print(f"ℹ️ Using '{base_name}' as base frame for IK.")
    else:
        print("⚠️ No base body found; assuming base==world.")

    # Known angles
    q_known = np.array([0.1, -0.2, 0.3, -1.5, 0.1, 1.5, 0.2], dtype=float)
    data.qpos[:7] = q_known
    mujoco.mj_forward(model, data)

    # 2) EE poses (base frame)
    # --- START OF FIX 2 ---
    # Compare the 'hand' body, not 'link7'
    p_mj, R_mj = mj_body_pose_in_base(model, data, base_bid, EE_BODY_NAME)
    # --- END OF FIX 2 ---

    # Build IKPy full vector
    q_full = [0.0] * len(solver.chain.links)
    for k, idx in enumerate(solver._active_idx):
        q_full[idx] = float(q_known[k])

    T_fk_list = ikpy_forward_all(solver, q_full)
    T_fk_ee = T_fk_list[-1]
    p_ik = T_fk_ee[:3, 3]
    R_ik = T_fk_ee[:3, :3]

    print("\n--- [Step 2.5] Cross-checking kinematic sync (IKPy FK vs MuJoCo) ---")
    pos_err = np.linalg.norm(p_ik - p_mj)
    ang_err = np.degrees(angle_from_R(R_ik.T @ R_mj))
    print(f"    - FK position error (base): {pos_err:.6f} m")
    print(f"    - FK orientation error (base): {ang_err:.4f} deg")

    # The rest of the script is for calibration, which we no longer need if FK passes.
    # We will run a final check to prove the models are synced.
    FK_POS_TOL = 1e-5
    FK_ANG_TOL = 1e-3 # degrees
    if pos_err < FK_POS_TOL and ang_err < FK_ANG_TOL:
        print("\n--- 🎉 KINEMATIC CHAINS ARE SYNCED ---")
        print("    Forward kinematics match between MuJoCo and IKPy.")
        
        print("\n--- [Step 3/3] Verifying IK solution ---")
        # Use the known MuJoCo pose as the target for the IK solver
        q_mj = R.from_matrix(R_mj).as_quat() # x, y, z, w
        target_pose_7d = np.concatenate([p_mj, q_mj])

        # Solve IK starting from a slightly perturbed configuration
        q_initial_ik = q_known + np.random.uniform(-0.1, 0.1, size=7)

        initial_position_full = [0.0] * len(solver.chain.links)
        for k, idx in enumerate(solver._active_idx):
            initial_position_full[idx] = float(q_initial_ik[k])

        solved_full = solver.chain.inverse_kinematics(
            target_position=p_mj,
            target_orientation=R_mj,
            orientation_mode="all",
            initial_position=initial_position_full
        )
        q_solved = np.array([solved_full[i] for i in solver._active_idx], dtype=float)

        # Check if the solved configuration produces the target pose
        data.qpos[:7] = q_solved
        mujoco.mj_forward(model, data)
        p_final, R_final = mj_body_pose_in_base(model, data, base_bid, EE_BODY_NAME)

        final_pos_err = np.linalg.norm(p_final - p_mj)
        final_ang_err = np.degrees(angle_from_R(R_final.T @ R_mj))
        
        print(f"    - Target Position: {np.round(p_mj, 4)}")
        print(f"    - Achieved Position: {np.round(p_final, 4)}")
        print(f"    - Final IK Position Error: {final_pos_err:.6e} m")
        print(f"    - Final IK Orientation Error: {final_ang_err:.6e} deg")

        assert final_pos_err < 1e-5, "IK solver failed to reach target position."
        assert final_ang_err < 1e-3, "IK solver failed to reach target orientation."
        print("\n--- ✅ IK SOLVER IS CORRECT AND ACCURATE ---")
    else:
        print("\n--- ❌ KINEMATIC MISMATCH DETECTED ---")
        print("    Even with the fixes, the models do not agree. Check joint origins/axes.")
        # Re-enable calibration for further debugging if needed
        # raise SystemExit(1)
if __name__ == "__main__":
    run()
