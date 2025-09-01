# utils/mujoco_utils.py
import warnings
from typing import Sequence
import numpy as np
import mujoco

def _body_id_for_name(model: mujoco.MjModel, name: str) -> int:
    """Return a body id or -1 if not found (robust wrapper)."""
    try:
        bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        return bid if bid >= 0 else -1
    except Exception:
        return -1

def _joint_id_for_name(model: mujoco.MjModel, name: str) -> int:
    """Return a joint id or -1 if not found (robust wrapper)."""
    try:
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        return jid if jid >= 0 else -1
    except Exception:
        return -1

def set_joint_qpos_by_name(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, qpos_values: Sequence[float]) -> bool:
    """
    Robustly set qpos for a joint or, as a fallback, set the body position.
    Returns True if something was written, False otherwise.
    """
    # 1) Try joint id -> write into data.qpos using model.jnt_qposadr
    jid = _joint_id_for_name(model, joint_name)
    if jid != -1:
        try:
            adr = int(model.jnt_qposadr[jid])
        except Exception:
            adr = -1

        if adr >= 0:
            qpos_values = np.asarray(qpos_values, dtype=float)
            n_write = min(qpos_values.size, data.qpos.size - adr)
            if n_write <= 0:
                warnings.warn(f"set_joint_qpos_by_name: joint '{joint_name}' qpos adr {adr} too large for qpos array.")
                return False
            data.qpos[adr:adr + n_write] = qpos_values[:n_write]
            try:
                mujoco.mj_forward(model, data)
            except Exception:
                try:
                    mujoco.mj_forward(model, data)
                except Exception:
                    pass
            return True

    # 2) Fallback: try to interpret joint_name as a body name and set model.body_pos
    bid = _body_id_for_name(model, joint_name)
    if bid == -1:
        # sometimes caller passes "object_joint" but wants body "object" -> try stripping suffixes
        alt = joint_name.replace("_joint", "")
        bid = _body_id_for_name(model, alt)

    if bid != -1:
        # if qpos_values contains at least 3 numbers, set the model.body_pos (mutates model)
        q = np.asarray(qpos_values, dtype=float)
        if q.size >= 3:
            try:
                model.body_pos[bid][:3] = q[:3]
                try:
                    mujoco.mj_forward(model, data)
                except Exception:
                    pass
                return True
            except Exception as e:
                warnings.warn(f"set_joint_qpos_by_name: failed to set model.body_pos for '{joint_name}': {e}")
                return False
        else:
            warnings.warn(f"set_joint_qpos_by_name: not enough qpos values to set body pos for '{joint_name}' (need >=3).")
            return False

    warnings.warn(f"set_joint_qpos_by_name: neither joint nor body found for '{joint_name}'.")
    return False


def set_body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, pos: Sequence[float]) -> bool:
    """
    Robustly set the 3D position of a body for initialization.
    Tries to use the joint adr if available; otherwise mutates model.body_pos.
    Returns True on success.
    """
    pos = np.asarray(pos, dtype=float)[:3]
    bid = _body_id_for_name(model, body_name)
    if bid == -1:
        warnings.warn(f"set_body_position: body '{body_name}' not found.")
        return False

    # Try to find any joint adr controlling that body via model.body_jntadr
    try:
        jstart = int(model.body_jntadr[bid])
        if jstart >= 0:
            # jstart is an index into the joint list; write into data.qpos via the joint qpos adr
            jid = jstart
            try:
                adr = int(model.jnt_qposadr[jid])
            except Exception:
                adr = -1
            if adr >= 0:
                n_write = min(3, data.qpos.size - adr)
                if n_write <= 0:
                    warnings.warn(f"set_body_position: qpos adr {adr} out of range.")
                else:
                    data.qpos[adr:adr + n_write] = pos[:n_write]
                    try:
                        mujoco.mj_forward(model, data)
                    except Exception:
                        pass
                    return True
    except Exception:
        pass

    # Last resort: mutate model.body_pos
    try:
        model.body_pos[bid][:3] = pos
        try:
            mujoco.mj_forward(model, data)
        except Exception:
            pass
        return True
    except Exception as e:
        warnings.warn(f"set_body_position: unable to set model.body_pos for '{body_name}': {e}")
        return False
