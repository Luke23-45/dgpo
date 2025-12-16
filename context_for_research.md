# Project Context: RedHot (Unified Diffusion Planner for Panda Robot)

## 1. High-Level Objective
We are training a **Unified Diffusion Planner** (Diffusion Policy) to control a 7-DoF Franka Emika Panda robot to performing **Pick-and-Place** tasks.
The system uses **Visuomotor Control**:
- **Input**: Current Image (RGB), Proprioception (Joints/EE Pose), Goal Image (RGB).
- **Output**: Action Chunk (Target Delta Poses for next K steps).
- **Execution**: Closed-loop Inverse Kinematics (IK) with PID control.

## 2. System Architecture

### A. Model: UnifiedDiffusionPlanner
- **Type**: Conditional Diffusion Probabilistic Model (DDPM/DDIM).
- **Backbone**: ResNet/CNN for Vision -> Transformer/U-Net for Diffusion.
- **Conditioning**:
    - `image_primary`: 224x224 RGB (ResNet encoding).
    - `goal_image`: 224x224 RGB (ResNet encoding).
    - `proprio`: EE Pose (7D: 3 pos + 4 quat).
- **Output**: `action_chunk` (Batch, K, Datadim).
    - We typically use `action_chunk_size=8`, but only execute the first step (`k=0`) for closed-loop control (Receding Horizon Control).

### B. Control Stack (Critical)
The model does *not* output motor torques directly. It outputs **End-Effector Delta Poses**.
1.  **Model Output**: $\Delta P_{model} = (\Delta x, \Delta y, \Delta z, \Delta rot...)$
2.  **Scaling**: $\Delta P_{real} = \Delta P_{model} \times \text{ActionScale}$
3.  **Target**: $P_{target} = P_{current} + \Delta P_{real}$
4.  **Inverse Kinematics (IK)**:
    - Calculates $\Delta Q$ (Joint Velocities) to reach $P_{target}$.
    - Uses `utils/ik_solver.py` (Analytical/Numerical damped least squares).
    - Tuned via PID Gains (`Kp`, `Ki`, `Kd`).
5.  **Environment (`PandaEnv`)**:
    - MuJoCo Physics.
    - Control Mode: `delta` (Input is delta joint positions).
    - Sim Substeps: 20 per control step.

## 3. Current Issues & Symptoms

### A. The "Unified Planner Failure" (Epoch 44)
The latest checkpoint (Epoch 44) fails to grasp the object.
- **Symptom 1 (Offset)**: The robot moves *near* the object but lands consistently to the Left or Right (e.g., 5cm offset). It "hovers" or misses the grasp.
- **Symptom 2 (Slow)**: Initially, the robot barely moved.
    - **Fix Found**: We discovered `action_scale` was set to `1.0` in evaluation, but likely needs to be `50.0` to match the model's output magnitude.
    - **Current State**: Even with `action_scale=50.0`, the "Offset/Bias" persists.

### B. PID vs Model Bias
We are currently distinguishing between two failure modes:
1.  **Control Error**: The Model predicts the *correct* target, but the PID controller is too soft/damped to reach it.
    - *Debugged via*: Tuning PID (Grid Search / Bayesian Opt).
2.  **Prediction Bias**: The Model (Vision System) predicts the *wrong* target (e.g., "The cube is 5cm left of where it really is").
    - *Debugged via*: HUD Overlay showing `Model Target` vs `Ground Truth`.
    - *Status*: Strong suspicion of Prediction Bias.

### C. CRITICAL FINDING: Coordinate Frame Mismatch (X-Axis Inverted)
Forensic analysis of the logs revealed a fundamental contradiction:
- **Policy Output**: `policy_dx` is POSITIVE (trying to move +X towards the object).
- **Robot Motion**: `ee_x` moves NEGATIVE (moving away from the object).
### C. CRITICAL FINDING: Reference Frame Mismatch (CONFIRMED)
Forensic analysis confirmed the "X-Flip" patch failed, proving the issue is a **Local vs World Frame** mismatch.
- **Diagnosis**: The Diffusion Policy predicts actions in the **End-Effector (Gripper) Frame**.
- **Issue**: The Evaluation script was applying these deltas in the **World Frame**.
- **Result**: When gripper is rotated (pointing down), "Forward" in Gripper Frame becomes "Down" or "Back" in World Frame, causing the robot to retreat or dive throughout the episode.
- **Fix Applied**: We implemented a rotation transformation in `evaluate_unified_planner.py`:
  $$ \Delta P_{world} = R_{current} \times \Delta P_{local} $$
  This aligns the policy's intent with the physics engine.

### A. Diagnostic Tools
1.  **`evaluate_unified_planner.py`**:
    - Production evaluator. Logs `MdlBias` (Model Prediction Error) and `ControlErr` (Servo Error).
2.  **`grid_search_unified.py`**:
    - Brute-force tests combos of `Kp`, `Kd`, `ActionScale`.
    - Revealed that low `ActionScale` (=1.0) paralyzed the robot.
3.  **`optimize_pid_bayesian.py`** (New):
    - Uses Optuna (Bayesian Optimization) to automatically find the best PID gains for a *given* model.
4.  **`tune_pid.py`** (New):
    - Physics-only tuner. Ignores the AI model. Finds PIDs that accurately track a strict mathematical trajectory.

### B. Handoff Analysis
We implemented a "Handoff" mode:
- **Phase 0-1 (Approach/Grasp)**: Expert checks if Model fails here.
- **Phase 2 (Lift)**: Expert hands off to Model after grasp.
- *Result*: Model struggles even after handoff, suggesting basic control/vision issues.

## 5. Key Hypotheses for Research
If asking another AI, focus on these:

1.  **Normalization Mismatch**:
    - Evaluation uses `transforms.Normalize(mean=[0.5], std=[0.5])` (Results in `[-1, 1]`).
    - Does Training do the same? If Training uses `[0, 1]` or ImageNet stats, the model will have a massive spatial offset bias.
2.  **Camera Extrinsics/Intrinsics**:
    - If `goal_image` generation (teleporting object) places the object slightly differently than the "real" object, the model learns a bias.
3.  **Action Space Mismatch**:
    - The Expert generates `delta_pose`. If the Expert's delta is calculated in `Base Frame` but Model predicts in `End-Effector Frame` (or vice versa), the robot will move in wrong directions (e.g., "Left" becomes "Forward").

## 6. Files Reference
- `evaluate/evaluate_unified_planner.py`: Main Eval Logic.
- `evaluate/optimize_pid_bayesian.py`: Auto-Tuner.
- `utils/ik_solver.py`: The bridge between Model and Env.
- `envs/panda_env.py`: The Physics World.
