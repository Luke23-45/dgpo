#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGPO Data Collector — Record-then-Replay workflow (final, robust)

Workflow
--------
Step 1: PATH CREATION ("Ghost" Run, robot frozen)
  - Robot stays at home. You control a red "ghost" target with the mouse.
  - Press [T] to start/stop recording the target's 3D path (and gripper state).
  - After stopping, the path is smoothed (moving average).

Step 2: REPLAY & RECORD ("Puppet" Show)
  - The sim is reset to the EXACT same state as when Step 1 began.
  - The welded mocap target replays your smoothed path frame-by-frame.
  - The robot follows the mocap (via weld). We record states and ctrl each step.
  - On completion: "Save this demonstration? [Y/N]".

Controls
--------
  [T]  start/stop "Ghost" path recording (Step 1)
  [G]  during Step 1: toggle gripper state (0=open / 1=closed) — recorded into path
  [R]  reset randomized scene (only when idle; cancels any in-progress rec/replay)
  [Y]  after Step 2: accept and save
  [N]  after Step 2: discard
  [Q]/[Esc] quit

Mouse
-----
  LMB drag     move GHOST target in camera plane (right/up)
  Alt+Wheel    move GHOST target along camera forward/back (depth)
  RMB drag     rotate camera
  Shift+RMB    pan camera
  Wheel        zoom camera

Requirements
------------
  pip install mujoco glfw lxml numpy
"""

import os
import sys
import json
import time
import math
import shutil
import tempfile
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
import mujoco
import glfw
from lxml import etree


# =============================== Config ==================================== #

# Visual & timing
TARGET_FPS = 60
MAX_SAVE_RETRIES = 1000

# Ghost path creation
MIN_GHOST_STEPS = 32                # minimum path length to proceed
SMOOTH_WINDOW_STEPS = 5             # moving average kernel width (odd integer recommended)
CLAMP_Z_LIMITS = (0.02, 1.5)        # safety clamp for target z during ghost creation

# Mocap motion mapping
MOCAP_MOVE_GAIN_XY = 0.0025         # meters per pixel (camera plane)
MOCAP_MOVE_GAIN_Z  = 0.02           # meters per wheel notch (along camera forward)

# Camera controls
CAM_ROT_GAIN   = 0.25               # deg per pixel (RMB)
CAM_PAN_GAIN   = 0.0008             # meters per pixel (Shift+RMB) in world
CAM_ZOOM_GAIN  = 0.12               # scaled distance per wheel notch
CAM_EL_MIN, CAM_EL_MAX = -89.0, -5.0
CAM_DIST_MIN, CAM_DIST_MAX = 0.35, 4.5

# Randomization knobs (applied on [R])
OBJ_X_RANGE = (0.40, 0.60)
OBJ_Y_RANGE = (-0.20, 0.20)
OBJ_Z_VALUE = 0.45

# Naming in the XML scene (change here if yours differ)
EEF_SITE_NAME = "attachment_site"   # end-effector site
HAND_BODY_NAME = "hand"             # hand body welded to mocap
MOCAP_NAME = "target_mocap"         # welded target for replay
GHOST_NAME = "ghost_target"         # free target for Step 1 path creation

# ============================= Utilities =================================== #

def _now() -> float:
    return time.perf_counter()

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

def _moving_average_path(path_xyz: np.ndarray, win: int) -> np.ndarray:
    """Moving average smoothing for a (T,3) path."""
    if win <= 1 or path_xyz.shape[0] < 3:
        return path_xyz.copy()
    win = max(1, int(win))
    # simple centered moving average; pad reflect to avoid shrinkage
    pad = win // 2
    x = np.pad(path_xyz, ((pad, pad), (0, 0)), mode='edge')
    kernel = np.ones((win, 1), dtype=np.float32) / float(win)
    sm = np.zeros_like(path_xyz, dtype=np.float32)
    for i in range(3):
        sm[:, i] = np.convolve(x[:, i], kernel[:, 0], mode='valid')
    return sm

def _pose_from_free_joint(qpos7):
    pos = np.array(qpos7[:3], dtype=float)
    quat = np.array(qpos7[3:7], dtype=float)
    return pos, quat

@dataclass
class DemoMeta:
    version: str
    xml_source: str
    xml_temp_used: bool
    timestamp_utc: float
    dt: float
    notes: str = ""

# ====================== XML patching (non-destructive) ===================== #

def build_temp_xml_if_needed(xml_path: str) -> Tuple[str, bool]:
    """
    Ensure XML has required mocap/weld/ghost bodies.
    Writes a temporary, auto-patched XML if needed. Crucially, it resolves all
    <include> and <compiler meshdir="..."> paths to be absolute, preventing
    errors when loading from a temporary directory.
    Returns (path_to_use, used_temp_flag).
    """
    xml_path = os.path.abspath(xml_path)
    xml_dir = os.path.dirname(xml_path)

    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(xml_path, parser)
    root = tree.getroot()

    # --- Find key elements ---
    worldbody = root.find("worldbody") or root.find(".//worldbody")
    if worldbody is None:
        raise RuntimeError("XML invalid: <worldbody> not found")
    equality = root.find("equality")
    compiler = root.find("compiler")

    changed = False

    # --- Resolve file paths to be absolute ---
    # This is the core fix.
    if compiler is not None and 'meshdir' in compiler.attrib:
        meshdir = compiler.attrib['meshdir']
        if not os.path.isabs(meshdir):
            compiler.attrib['meshdir'] = os.path.normpath(os.path.join(xml_dir, meshdir))
            changed = True

    for include_node in root.findall(".//include"):
        if 'file' in include_node.attrib:
            filepath = include_node.attrib['file']
            if not os.path.isabs(filepath):
                include_node.attrib['file'] = os.path.normpath(os.path.join(xml_dir, filepath))
                changed = True

    # --- Ensure required bodies and constraints exist ---
    if equality is None:
        equality = etree.Element("equality")
        root.append(equality)
        changed = True

    if root.find(f".//body[@name='{MOCAP_NAME}']") is None:
        mocap = etree.Element('body', name=MOCAP_NAME, mocap='true',
                              pos='0.55 0.0 0.5', quat='1 0 0 0')
        etree.SubElement(mocap, 'geom', type='sphere', size='0.02',
                         rgba='1 0 0 0.6', contype='0', conaffinity='0', group='2')
        worldbody.append(mocap)
        changed = True

    if root.find(f".//body[@name='{GHOST_NAME}']") is None:
            ghost = etree.Element('body', name=GHOST_NAME, pos='0.55 0.0 0.5', quat='1 0 0 0')
            # ADD THIS LINE:
            etree.SubElement(ghost, 'joint', type='free', name=f'{GHOST_NAME}_joint')
            etree.SubElement(ghost, 'geom', type='sphere', size='0.018',
                            rgba='0.9 0 0.9 0.8', contype='0', conaffinity='0', group='2')
            worldbody.append(ghost)
            changed = True

    if root.find(f".//weld[@body1='{MOCAP_NAME}']") is None:
        etree.SubElement(equality, 'weld',
                         body1=MOCAP_NAME, body2=HAND_BODY_NAME,
                         solimp='0.9 0.95 0.001', solref='0.02 1')
        changed = True
    
    # --- If no changes were needed, return original path ---
    if not changed:
        return xml_path, False

    # --- Write a temp copy with absolute paths ---
    tmpdir = tempfile.mkdtemp(prefix="dgpo_xml_")
    temp_xml_path = os.path.join(tmpdir, "scene_patched.xml")
    tree.write(temp_xml_path, pretty_print=True, xml_declaration=True, encoding="utf-8")
    print(f"INFO: Auto-patched XML with absolute paths written to {temp_xml_path}")
    return temp_xml_path, True



# ============================== Main App =================================== #

class DataCollectorApp:
    # ---------------------------- Lifecycle -------------------------------- #
    def __init__(self, xml_scene_path: str, save_dir: str, target_eps: int = 50):
        self.original_xml = os.path.abspath(xml_scene_path)
        self.scene_xml, self.used_temp_xml = build_temp_xml_if_needed(self.original_xml)
        self.dragging = False


        self.save_dir = os.path.abspath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        # GLFW window
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        self.window = glfw.create_window(1280, 768, "DGPO Record-then-Replay Collector", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_input_mode(self.window, glfw.STICKY_MOUSE_BUTTONS, glfw.TRUE)
        glfw.set_input_mode(self.window, glfw.STICKY_KEYS, glfw.TRUE)

        # callbacks
        glfw.set_window_user_pointer(self.window, self)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)

        # MuJoCo
        self.model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.data = mujoco.MjData(self.model)

        self.cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(self.cam)
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=20000)
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self._init_camera()

        # IDs
        self.site_eef = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE_NAME)
        if self.site_eef < 0:
            raise RuntimeError(f"Site '{EEF_SITE_NAME}' not found in model.")
        self.mocap_bodyid = self.model.body(MOCAP_NAME).id
        self.mocap_id = self.model.body_mocapid[self.mocap_bodyid]
        if self.mocap_id < 0:
            raise RuntimeError(f"Body '{MOCAP_NAME}' is not mocap-enabled.")
        self.ghost_bid = self.model.body(GHOST_NAME).id

        # State
        self.episode_idx = self._count_existing_episodes()
        self.target_eps = target_eps
        self.dt = float(self.model.opt.timestep)

        # Modes
        self.mode = "IDLE"            # "IDLE" -> "GHOST_REC" -> "REPLAY_WAIT" -> "REPLAYING" -> "DECIDE"
        self.awaiting_decision = False
        self.ghost_qposadr = self.model.joint(f"{GHOST_NAME}_joint").qposadr[0]

        # Inputs
        self.mouse = {'lmb': False, 'rmb': False, 'mods': 0,
                      'last_x': 0.0, 'last_y': 0.0, 'dx': 0.0, 'dy': 0.0}
        self.scroll_dy = 0.0

        # Gripper (binary state recorded during ghost path)
        self.gripper_state = 0.0  # 0 open, 1 closed

        # Buffers (Step 1: ghost path)
        self.ghost_path_xyz: List[np.ndarray] = []
        self.ghost_gripper: List[float] = []

        # Buffers (Step 2: replay recording)
        self.buf_time = []
        self.buf_eef_pos = []
        self.buf_eef_quat = []
        self.buf_qpos = []
        self.buf_qvel = []
        self.buf_ctrl = []
        self.buf_object_pose = []

        # Snapshot of initial state at start of Step 1 (for exact reset)
        self.snap_qpos = None
        self.snap_qvel = None
        self.snap_mocap_pos = None
        self.snap_mocap_quat = None
        self.snap_ghost_pos = None
        self.snap_ghost_quat = None

        # Replay arrays (filled after smoothing)
        self.replay_xyz = None
        self.replay_grip = None
        self.replay_idx = 0

        # Goal pos (if present)
        self.goal_pos_cached = None

        # Initialize scene
        self.reset_env()
        self._print_help()

    def _init_camera(self):
        self.cam.azimuth = 135
        self.cam.elevation = -30
        self.cam.distance = 1.8
        self.cam.lookat[:] = np.array([0.5, 0.0, 0.35], dtype=float)

    def _count_existing_episodes(self) -> int:
        return sum(1 for f in os.listdir(self.save_dir)
                   if f.startswith("episode_") and f.endswith(".npz"))

    # ------------------------------ Reset ---------------------------------- #
    def reset_env(self):
        # Hard reset physics
        mujoco.mj_resetData(self.model, self.data)

        # Randomize object pose (if free joint exists)
        try:
            j_id = self.model.joint("object_joint").id
            adr = self.model.jnt_qposadr[j_id]
            x = np.random.uniform(*OBJ_X_RANGE)
            y = np.random.uniform(*OBJ_Y_RANGE)
            z = OBJ_Z_VALUE
            quat = np.array([1, 0, 0, 0], dtype=float)
            self.data.qpos[adr:adr+7] = np.array([x, y, z, *quat])
        except Exception:
            pass

        # Randomize goal body (if present)
        try:
            goal_bid = self.model.body("goal").id
            self.model.body_pos[goal_bid, 0] = np.random.uniform(0.40, 0.60)
            self.model.body_pos[goal_bid, 1] = np.random.uniform(0.25, 0.40)
        except Exception:
            pass

        # Forward to update derived quantities
        mujoco.mj_forward(self.model, self.data)

        # Sync both targets to EEF pose at reset
        eef_p = self.data.site_xpos[self.site_eef].copy()
        self.data.mocap_pos[self.mocap_id, :] = eef_p
        self.data.mocap_quat[self.mocap_id, :] = np.array([1, 0, 0, 0], dtype=float)

        self.data.xpos[self.ghost_bid, :] = eef_p
        self.data.xquat[self.ghost_bid, :] = np.array([1, 0, 0, 0], dtype=float)
        mujoco.mj_forward(self.model, self.data)

        # Cache goal world pos if exists
        self.goal_pos_cached = None
        try:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
            if gid >= 0:
                self.goal_pos_cached = np.array(self.data.xpos[gid], dtype=float)
        except Exception:
            pass

        self.mode = "IDLE"
        self.awaiting_decision = False
        self._clear_all_buffers()
        print(f"🔄 Reset. Episodes saved: {self.episode_idx}/{self.target_eps}")

    # ------------------------- Camera math & I/O ---------------------------- #
    def _cam_axes(self):
        az = math.radians(self.cam.azimuth)
        el = math.radians(self.cam.elevation)
        cam_dir = np.array([math.cos(el)*math.sin(az),
                            math.cos(el)*math.cos(az),
                            math.sin(el)], dtype=float)
        cam_fwd = -_unit(cam_dir)
        world_up = np.array([0, 0, 1], dtype=float)
        cam_right = _unit(np.cross(world_up, cam_fwd))
        cam_up = _unit(np.cross(cam_fwd, cam_right))
        return cam_right, cam_up, cam_fwd

    def _on_mouse_button(self, window, button, action, mods):
        self.mouse['mods'] = mods
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse['lmb'] = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse['rmb'] = (action == glfw.PRESS)
        x, y = glfw.get_cursor_pos(window)
        self.mouse['last_x'], self.mouse['last_y'] = x, y

    def _on_cursor(self, window, xpos, ypos):
        dx = xpos - self.mouse['last_x']
        dy = ypos - self.mouse['last_y']
        self.mouse['last_x'], self.mouse['last_y'] = xpos, ypos
        self.mouse['dx'] += dx
        self.mouse['dy'] += dy

        # Camera manip
        if self.mouse['rmb']:
            right, up, _ = self._cam_axes()
            if self.mouse['mods'] & glfw.MOD_SHIFT:
                pan = right * (-dx * CAM_PAN_GAIN) + up * (dy * CAM_PAN_GAIN)
                self.cam.lookat[:] = self.cam.lookat + pan
            else:
                self.cam.azimuth = float(self.cam.azimuth - dx * CAM_ROT_GAIN)
                self.cam.elevation = float(np.clip(self.cam.elevation - dy * CAM_ROT_GAIN,
                                                   CAM_EL_MIN, CAM_EL_MAX))

    def _on_scroll(self, window, xoff, yoff):
        if self.mouse['mods'] & glfw.MOD_ALT:
            # alt+wheel -> ghost depth (handled in _apply_ghost_motion)
            self.scroll_dy += yoff
        else:
            scale = math.exp(-CAM_ZOOM_GAIN * yoff)
            self.cam.distance = float(np.clip(self.cam.distance * scale,
                                              CAM_DIST_MIN, CAM_DIST_MAX))

    def _on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return

        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(self.window, True)
            return

        if key == glfw.KEY_R and self.mode in ("IDLE",):
            self.reset_env()
            return

        if key == glfw.KEY_G and self.mode in ("GHOST_REC", "IDLE"):
            # toggle gripper (recorded only during GHOST_REC)
            self.gripper_state = 1.0 - self.gripper_state
            return

        # Start/stop ghost recording
        if key == glfw.KEY_T:
            if self.mode == "IDLE":
                self._begin_ghost_recording()
                return
            elif self.mode == "GHOST_REC":
                self._end_ghost_recording()
                return

        # Decision after replay
        if key == glfw.KEY_Y and self.mode == "DECIDE":
            self._finalize_and_save()
            return
        if key == glfw.KEY_N and self.mode == "DECIDE":
            self._discard_last()
            return

    # ----------------------- Step 1: Ghost Recording ------------------------ #
    def _begin_ghost_recording(self):
        self.mode = "GHOST_REC"
        self.ghost_path_xyz.clear()
        self.ghost_gripper.clear()

        # Snapshot exact initial state to replay from
        self.snap_qpos = self.data.qpos.copy()
        self.snap_qvel = self.data.qvel.copy()
        self.snap_mocap_pos = self.data.mocap_pos.copy()
        self.snap_mocap_quat = self.data.mocap_quat.copy()
        self.snap_ghost_pos = self.data.xpos[self.ghost_bid].copy()
        self.snap_ghost_quat = self.data.xquat[self.ghost_bid].copy()
        print("🟣 PATH CREATION STARTED — press [T] again to stop.")

    def _end_ghost_recording(self):
        T = len(self.ghost_path_xyz)
        if T < MIN_GHOST_STEPS:
            self.mode = "IDLE"
            self.ghost_path_xyz.clear()
            self.ghost_gripper.clear()
            print(f"⚠️ Path too short ({T} < {MIN_GHOST_STEPS}). Discarded.")
            return

        # Smooth
        raw_xyz = np.asarray(self.ghost_path_xyz, dtype=np.float32)
        sm_xyz = _moving_average_path(raw_xyz, SMOOTH_WINDOW_STEPS)
        # Per-step gripper state (already recorded per frame)
        grip = np.asarray(self.ghost_gripper, dtype=np.float32)
        grip = (grip > 0.5).astype(np.float32)

        self.replay_xyz = sm_xyz
        self.replay_grip = grip
        self.replay_idx = 0

        # Prepare for exact replay: restore snapshot
        self._restore_snapshot()
        self.mode = "REPLAY_WAIT"  # one step to settle, then REPLAYING
        print(f"🟡 PATH CAPTURED (T={T}). Beginning replay & recording...")

    def _apply_ghost_motion(self):
        if self.mode != "GHOST_REC":
            self.mouse['dx'] = 0.0; self.mouse['dy'] = 0.0
            self.scroll_dy = 0.0
            return

        adr = self.ghost_qposadr  # must be cached in __init__

        # LMB drag: move in camera plane
        if self.mouse['lmb']:
            dx, dy = self.mouse['dx'], self.mouse['dy']
            self.mouse['dx'] = 0.0; self.mouse['dy'] = 0.0
            if abs(dx) + abs(dy) > 0:
                cam_right, cam_up, _ = self._cam_axes()
                delta = cam_right * (dx * MOCAP_MOVE_GAIN_XY) + cam_up * (-dy * MOCAP_MOVE_GAIN_XY)
                curr = self.data.qpos[adr:adr+7].copy()
                new_xyz = curr[:3] + delta
                new_xyz[2] = float(np.clip(new_xyz[2], *CLAMP_Z_LIMITS))
                self.data.qpos[adr:adr+3] = new_xyz
                # keep quaternion unchanged (curr[3:7]) or set identity
                self.data.qpos[adr+3:adr+7] = curr[3:7]
                mujoco.mj_forward(self.model, self.data)

        else:
            self.mouse['dx'] = 0.0; self.mouse['dy'] = 0.0

        # Alt+Wheel: move along camera forward/back
        if abs(self.scroll_dy) > 0.0:
            _, _, cam_fwd = self._cam_axes()
            curr = self.data.qpos[adr:adr+7].copy()
            curr[:3] = curr[:3] + cam_fwd * (self.scroll_dy * MOCAP_MOVE_GAIN_Z)
            curr[2] = float(np.clip(curr[2], *CLAMP_Z_LIMITS))
            self.data.qpos[adr:adr+3] = curr[:3]
            self.data.qpos[adr+3:adr+7] = curr[3:7]
            mujoco.mj_forward(self.model, self.data)
            self.scroll_dy = 0.0

        # Record per-step (store the qpos xyz)
        self.ghost_path_xyz.append(self.data.qpos[adr:adr+3].copy())
        self.ghost_gripper.append(float(self.gripper_state))


    def _restore_snapshot(self):
        """Restore the snapshot captured at the start of ghost recording."""
        self.data.qpos[:] = self.snap_qpos
        self.data.qvel[:] = self.snap_qvel
        self.data.mocap_pos[:, :] = self.snap_mocap_pos
        self.data.mocap_quat[:, :] = self.snap_mocap_quat
        self.data.xpos[self.ghost_bid, :] = self.snap_ghost_pos
        self.data.xquat[self.ghost_bid, :] = self.snap_ghost_quat
        mujoco.mj_forward(self.model, self.data)

    # ----------------------- Step 2: Replay & Record ------------------------ #
    def _step_replay(self):
        """
        Move welded mocap along the precomputed path; record robot data.
        Mode transitions:
          REPLAY_WAIT -> REPLAYING (after one sim step)
          REPLAYING -> DECIDE (after final frame)
        """
        if self.mode == "REPLAY_WAIT":
            # give one step for determinism
            self.mode = "REPLAYING"
            self.replay_idx = 0

        if self.mode != "REPLAYING":
            return

        if self.replay_idx >= len(self.replay_xyz):
            # finished
            self.mode = "DECIDE"
            self.awaiting_decision = True
            print("✅ Replay complete. Save this demonstration? [Y/N]")
            return

        # Set mocap to the next replay point
        target_p = self.replay_xyz[self.replay_idx]
        self.data.mocap_pos[self.mocap_id, :] = target_p
        self.data.mocap_quat[self.mocap_id, :] = np.array([1, 0, 0, 0], dtype=float)

        # Apply gripper state (simple mapping via actuator or qpos heuristic)
        self._apply_gripper_state(self.replay_grip[self.replay_idx])

        # Advance physics one step
        mujoco.mj_step(self.model, self.data)

        # Record robot state
        self._record_replay_step()

        self.replay_idx += 1

    def _apply_gripper_state(self, grip: float):
        """Map grip (0/1) -> a target finger open/close position."""
        target_open, target_closed = 0.0, 0.04
        target = target_open * (1 - grip) + target_closed * grip
        try:
            jname = "finger_joint1"; jid = self.model.joint(jname).id
            # Prefer actuator if available
            found = False
            for aid in range(self.model.nu):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
                if "gripper" in name or "finger_joint1" in name or "finger" in name:
                    self.data.ctrl[aid] = target; found = True; break
            if not found:
                adr = self.model.jnt_qposadr[jid]
                current = float(self.data.qpos[adr])
                # gentle pull toward target
                self.data.qpos[adr] = current + 0.2 * (target - current)
        except Exception:
            pass

    def _record_replay_step(self):
        now_ts = _now()
        self.buf_time.append(now_ts)
        self.buf_eef_pos.append(self.data.site_xpos[self.site_eef].copy())
        self.buf_eef_quat.append(self.data.site_xmat[self.site_eef].copy())
        self.buf_qpos.append(self.data.qpos.copy())
        self.buf_qvel.append(self.data.qvel.copy())
        self.buf_ctrl.append(self.data.ctrl.copy())
        # Object pose if free joint exists
        try:
            j_id = self.model.joint("object_joint").id
            adr = self.model.jnt_qposadr[j_id]
            qpos7 = self.data.qpos[adr:adr+7]
            pos, quat = _pose_from_free_joint(qpos7)
            self.buf_object_pose.append(np.concatenate([pos, quat]))
        except Exception:
            self.buf_object_pose.append(np.array([np.nan]*7, dtype=float))

    # --------------------------- Save / Discard ----------------------------- #
    def _finalize_and_save(self):
        self.awaiting_decision = False
        self.mode = "IDLE"

        # Assemble arrays
        eef_pos = np.asarray(self.buf_eef_pos, dtype=np.float32)
        eef_quat = np.asarray(self.buf_eef_quat, dtype=np.float32)
        qpos = np.asarray(self.buf_qpos, dtype=np.float32)
        qvel = np.asarray(self.buf_qvel, dtype=np.float32)
        ctrl = np.asarray(self.buf_ctrl, dtype=np.float32)
        obj_pose = np.asarray(self.buf_object_pose, dtype=np.float32)
        t = np.asarray(self.buf_time, dtype=np.float64)

        ghost_xyz_raw = np.asarray(self.ghost_path_xyz, dtype=np.float32)
        ghost_grip_raw = np.asarray(self.ghost_gripper, dtype=np.float32)
        replay_xyz = np.asarray(self.replay_xyz, dtype=np.float32)
        replay_grip = np.asarray(self.replay_grip, dtype=np.float32)

        meta = DemoMeta(
            version="dgpo_collector_record_then_replay_v1",
            xml_source=self.original_xml,
            xml_temp_used=self.used_temp_xml,
            timestamp_utc=time.time(),
            dt=self.dt,
            notes="Two-stage: ghost path then replay & record."
        )

        # goal (constant per ep if present)
        goal_pos = self.goal_pos_cached if self.goal_pos_cached is not None \
            else np.array([np.nan, np.nan, np.nan], dtype=np.float32)

        # Filename
        base_idx = self.episode_idx
        path = None
        for k in range(MAX_SAVE_RETRIES):
            fname = f"episode_{base_idx + k:03d}.npz"
            candidate = os.path.join(self.save_dir, fname)
            if not os.path.exists(candidate):
                path = candidate
                self.episode_idx = base_idx + k + 1
                break
        if path is None:
            print("❌ Could not allocate filename. Save aborted.")
            self._clear_all_buffers()
            return

        np.savez_compressed(
            path,
            # Step 1 inputs (intent)
            ghost_xyz_raw=ghost_xyz_raw,
            ghost_grip_raw=ghost_grip_raw,
            replay_xyz=replay_xyz,
            replay_grip=replay_grip,
            # Step 2 outputs (execution)
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            object_pose=obj_pose,
            timestamps=t,
            goal_pos=np.asarray(goal_pos, dtype=np.float32),
            meta_json=np.frombuffer(json.dumps(asdict(meta)).encode("utf-8"), dtype=np.uint8),
        )
        print(f"💾 Saved {os.path.basename(path)} ({len(t)} steps).  Progress: {self.episode_idx}/{self.target_eps}")
        self._clear_all_buffers()

    def _discard_last(self):
        print("🗑️ Discarded this demonstration.")
        self.awaiting_decision = False
        self.mode = "IDLE"
        self._clear_all_buffers()

    def _clear_all_buffers(self):
        self.ghost_path_xyz.clear()
        self.ghost_gripper.clear()
        self.buf_time.clear()
        self.buf_eef_pos.clear()
        self.buf_eef_quat.clear()
        self.buf_qpos.clear()
        self.buf_qvel.clear()
        self.buf_ctrl.clear()
        self.buf_object_pose.clear()
        self.replay_xyz = None
        self.replay_grip = None
        self.replay_idx = 0

    # ------------------------------ Main loop ------------------------------ #
    def run(self):
        vis_dt_target = 1.0 / float(max(15, TARGET_FPS))
        last_vis = _now()

        while not glfw.window_should_close(self.window):
            loop_start = _now()

            # Step 1 (ghost)
            self._apply_ghost_motion()

            # Step 2 (replay & record)
            if self.mode in ("REPLAY_WAIT", "REPLAYING"):
                self._step_replay()
            else:
                # advance physics minimally to keep scene responsive
                mujoco.mj_step(self.model, self.data)

            # Render ~TARGET_FPS
            now = _now()
            if (now - last_vis) >= vis_dt_target:
                self._render()
                last_vis = now

            glfw.poll_events()

            # Soft sync to simulate at model timestep
            elapsed = _now() - loop_start
            to_sleep = self.dt - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

        self._cleanup()

    # ------------------------------- Render -------------------------------- #
    def _render(self):
        viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(self.window))
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        mujoco.mjr_render(viewport, self.scn, self.ctx)

        # HUD
        if self.mode == "IDLE":
            status = "⏸️  IDLE — [T] start path, [R] reset, [Q] quit"
        elif self.mode == "GHOST_REC":
            status = "🟣 PATH CREATION — [T] stop, [G] toggle gripper, LMB drag / Alt+Wheel"
        elif self.mode in ("REPLAY_WAIT", "REPLAYING"):
            status = "🟠 REPLAY & RECORDING…"
        elif self.mode == "DECIDE":
            status = "🟢 DONE — Save? [Y/N]"
        else:
            status = self.mode

        right = f"Episodes: {self.episode_idx}/{self.target_eps}"
        bottom = "Mouse: LMB move-ghost | Alt+Wheel depth | RMB rotate | Shift+RMB pan | Wheel zoom"

        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           viewport, status, "", self.ctx)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                           viewport, bottom, "", self.ctx)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                           viewport, right, "", self.ctx)
        glfw.swap_buffers(self.window)

    # ------------------------------ Cleanup -------------------------------- #
    def _cleanup(self):
        print("Closing viewer.")
        glfw.terminate()
        if self.used_temp_xml:
            try:
                shutil.rmtree(os.path.dirname(self.scene_xml), ignore_errors=True)
            except Exception:
                pass

    # ------------------------------ Help ----------------------------------- #
    def _print_help(self):
        print("\n" + "="*64)
        print("DGPO Record-then-Replay Data Collector — FINAL")
        print("="*64)
        print("Step 1 (Ghost):")
        print("  [T]  start/stop path recording (robot frozen)")
        print("  [G]  toggle gripper state (recorded)")
        print("  LMB drag  move ghost in camera plane | Alt+Wheel depth")
        print("Step 2 (Replay):")
        print("  runs automatically after Step 1; robot follows welded mocap")
        print("  then choose: [Y] save  |  [N] discard")
        print("Global:")
        print("  [R] reset scene when IDLE | [Q]/[Esc] quit")
        print("="*64 + "\n")


# ================================= Main ==================================== #

def main():
    # Resolve defaults relative to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "..", "envs", "panda_pick_place.xml")
    save_dir = os.path.join(script_dir, "..", "data")

    # CLI overrides
    if len(sys.argv) >= 2:
        xml_path = sys.argv[1]
    if len(sys.argv) >= 3:
        save_dir = sys.argv[2]

    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML scene not found at {xml_path}")
    os.makedirs(save_dir, exist_ok=True)

    app = DataCollectorApp(xml_scene_path=xml_path, save_dir=save_dir, target_eps=50)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app._cleanup()


if __name__ == "__main__":
    main()
