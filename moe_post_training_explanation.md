# Deep Dive: MoE Post-Training Workflow

This document details exactly **HOW** we convert the "Generalist" (Monolithic Semantic Planner) into a "Specialist Team" (MoE) using Post-Training.

---

## The Scenario: "The Coffee Mug"
Imagine the robot task is to **Pick up a Coffee Mug and Place it on a Coaster**.
Total Horizon: 400 Steps.
*   **Steps 0-100**: Approaching the mug (Fast motion).
*   **Steps 100-150**: Grasping the handle (Precise, slow).
*   **Steps 150-300**: Carrying it (Steady).
*   **Steps 300-400**: Placing on coaster (Precise).

---

## Step 1: The Setup (Loading & Freezing)
We start with `train_semantic_planner.py` having finished. We have a file: `best_model.ckpt`.
This model is **Good but not Perfect**. It gets 80% success. It sometimes shakes the mug because it's trying to be fast (like the Approach phase) while holding it.

**Action:**
1.  Load `best_model.ckpt`.
2.  **Freeze the "Eyes" and "Brain"**: We lock the weights of the Vision Encoder (SigLIP) and the Transformer. They will **NOT change** anymore.
3.  **The "Router" is born**: This frozen model is now designated as the **Router**. It will decide *who* acts.

---

## Step 2: The "Hiring" (Expert Initialization)
We need to create the experts. We don't start them as "stupid" random networks. We use **Cloning**.

**Concept**:
*   The Monolithic model had a "Trajectory Head" (The output layer).
*   We make **5 Copies** of this Head.
    *   `Expert_0 (Approach)`: Copy of original head.
    *   `Expert_1 (Grasp)`: Copy of original head.
    *   ...etc.

**Status**: Right now, all 5 experts are identical clones of the original Generalist.

---

## Step 3: The Data Slicing (The Virtual Routing)
We rely on the **Ground Truth Phase** labels in our dataset (`DGPOExpert` generates these: 0=Approach, 1=Grasp, etc.).

We look at our "Coffee Mug" dataset:
*   **Frame 50 (Approach)**: Phase Label = 0.
*   **Frame 120 (Grasp)**: Phase Label = 1.

We **split** the dataset virtually:
*   `Batch_0`: Contains ONLY Approach frames.
*   `Batch_1`: Contains ONLY Grasp frames.

---

## Step 4: The Training Loop (Specialization)
We run the training loop. This is where the magic happens.

**Scenario: Training Step on Frame 120 (Grasp Phase)**

1.  **Router Pass (Frozen)**:
    *   Input: `Image_120`.
    *   Router says: *"I see a hand near a mug handle. My internal state says Phase 1."*
    *   Output: `Visual_Embedding` (The compressed info).

2.  **Expert Selection**:
    *   Since the Label is **Phase 1**, we activate **ONLY Expert_1 (The Grasp Specialist)**.
    *   Experts 0, 2, 3, 4 are **Asleep** (Gradients = 0).

3.  **Expert Forward Pass**:
    *   `Expert_1` takes the `Visual_Embedding`.
    *   It tries to predict the action.
    *   *Initial Behavior:* It predicts a "Generalist" action (mediocre).

4.  **Gradient Descent (Correction)**:
    *   We compare `Expert_1`'s prediction to the Ground Truth (Perfect Grasp).
    *   We update **ONLY Expert_1's weights**.
    *   **Crucial Detail**: We **delete** the gradients for the Router. The Router is not allowed to change.

**Result after 1000 steps**:
*   `Expert_1` has seen *only* Grasping examples. It forgets how to Approach. It becomes **Hyper-Specialized** in Grasping.
*   `Expert_0` has seen *only* Approach examples. It forgets how to Grasp. It becomes **Fast and Aggressive** (perfect for approach).

---

## Step 5: Inference (The Final Exam)
Now we deploy the robot.

1.  **Robot turns on**. It sees the mug far away.
2.  **Router (Frozen)** sees "Far away". Predicts **Phase 0**.
3.  **Router** calls **Expert_0**.
4.  **Expert_0** (The Aggressive Mover) takes control. Robot swoops in fast.
5.  **Router** sees "Hand is near handle". Switches to **Phase 1**.
6.  **Router** calls **Expert_1**.
7.  **Expert_1** (The Surgeon) takes control. It is steady, precise. It executes the grasp perfectly.

---

## Summary of Benefits
1.  **No Interference**: The "Fast Approach" gradients never touch the "Precise Grasp" expert. No more shaky grasps!
2.  **Perfect Routing**: We typically struggle to train routers. But here, we **reused** the Semantic Planner, which we *know* understands the phases (95%+ accuracy).
3.  **Stability**: Because the Router is frozen, it won't suddenly decide to switch experts randomly. It forces the robot to stick to the plan.
