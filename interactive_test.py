import os
import time
import numpy as np
import mujoco
from mujoco.viewer import launch_passive

# ----------------------------
# Globals
# ----------------------------
KEYS = {
    'W': False, 'S': False, 'A': False, 'D': False,
    'Q': False, 'E': False, 'R': False, 'G': False, 'ESC': False
}
GRIP_TOGGLE_COOLDOWN = 0.2  # seconds
CTRL_MAG = 0.4
PRINT_INTERVAL = 0.25

# ----------------------------
# Key callback for launch_passive
# ----------------------------
def key_callback(key):
    # key is an integer key code
    char = None
    try:
        char = chr(key).upper()
    except Exception:
        return
    if char in KEYS:
        KEYS[char] = not KEYS[char]
        print(f">> Key {char} toggled to {KEYS[char]}")

# ----------------------------
# Reset environment
# ----------------------------
def reset_env(model, data):
    mujoco.mj_resetData(model, data)
    # Randomize object and goal positions
    try:
        # Object: free joint (7 DOF)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
        adr = int(model.jnt_qposadr[jid])
        # Position on table
        tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
        tb_pos = model.body_pos[tb]
        xy_span = 0.2
        x = float(tb_pos[0] + np.random.uniform(-xy_span, xy_span))
        y = float(tb_pos[1] + np.random.uniform(-xy_span, xy_span))
        z = tb_pos[2] + 0.05
        data.qpos[adr:adr+7] = [x, y, z, 1.0, 0.0, 0.0, 0.0]
    except Exception as e:
        print(f"[WARN] Could not reset cube: {e}")

    try:
        goal_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        model.body_pos[goal_b][0:2] = [x + 0.1, y + 0.1]
    except Exception as e:
        print(f"[WARN] Could not reset goal: {e}")

    mujoco.mj_forward(model, data)
    print(f"[RESET] cube @ ({x:.2f}, {y:.2f}), goal @ ({model.body_pos[goal_b][0]:.2f}, {model.body_pos[goal_b][1]:.2f})")

# ----------------------------
# Main
# ----------------------------
def main():
    xml_path = os.path.join(os.path.dirname(__file__), "envs", "panda_pick_place.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Missing XML at {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Actuator IDs
    acts = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}") for i in range(1,9)]
    for i,a in enumerate(acts, 1):
        if a < 0:
            raise ValueError(f"Actuator{i} not found in model")
    arm_acts = acts[:7]
    grip_act = acts[7]

    reset_env(model, data)

    print("""
Controls:
 W/S: actuator1 +/- 
 A/D: actuator2 +/- 
 Q/E: actuator3 +/- 
 R: reset cube/goal
 G: toggle gripper
 ESC: exit
""")

    last_print = time.time()
    grip_state = False
    last_grip_time = 0.0

    with launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running() and not KEYS['ESC']:
            t0 = time.time()
            ctrl = np.zeros(model.nu, dtype=float)

            if KEYS['W']: ctrl[arm_acts[0]] = +CTRL_MAG
            elif KEYS['S']: ctrl[arm_acts[0]] = -CTRL_MAG

            if KEYS['A']: ctrl[arm_acts[1]] = +CTRL_MAG
            elif KEYS['D']: ctrl[arm_acts[1]] = -CTRL_MAG

            if KEYS['Q']: ctrl[arm_acts[2]] = +CTRL_MAG
            elif KEYS['E']: ctrl[arm_acts[2]] = -CTRL_MAG

            if KEYS['G'] and (t0 - last_grip_time) > GRIP_TOGGLE_COOLDOWN:
                grip_state = not grip_state
                last_grip_time = t0
                print(f">> Gripper toggled {'CLOSED' if grip_state else 'OPEN'}")

            ctrl[grip_act] = 255.0 if grip_state else 0.0

            if KEYS['R']:
                reset_env(model, data)
                KEYS['R'] = False

            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            viewer.sync()

            if time.time()-last_print > PRINT_INTERVAL:
                nz = np.where(np.abs(ctrl)>1e-6)[0]
                print(f"[CTRL] Non-zero actuators: {nz}, values: {ctrl[nz]}")
                last_print = time.time()

            dt = model.opt.timestep - (time.time()-t0)
            if dt>0:
                time.sleep(dt)

    print("Exited interactive viewer.")

if __name__ == "__main__":
    main()
