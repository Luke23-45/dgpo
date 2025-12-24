
System RAM
11.3 / 12.7 GB
 
GPU RAM
5.3 / 15.0 GB
 
Disk
46.5 / 112.6 GB
# Technical Proposal: Bootstrapped Phase-Locked Mixture-of-Experts (APEX-MoE)

**To:** Research Supervisor
**Date:** December 22, 2025
**Subject:** Solving Long-Horizon Manipulation via Monolithic Bootstrapping & Phase-Locked Routing
**Status:** DRAFT 2.0 (Bootstrapping Focus)

---

## 1. Executive Summary

We propose a novel **"Bootstrapped Mixture-of-Experts" (MoE)** architecture for long-horizon robotic manipulation. Unlike standard MoE approaches that attempt to train specialized experts from scratch (often leading to "Mode Collapse" or instability), our strategy leverages a **Two-Stage Training Pipeline**:

1.  **Stage 1 (Foundation):** Train a robust **Monolithic `SemanticPlanner`** end-to-end. This model learns the global task structure, phase transitions, and visual representations.
2.  **Stage 2 (Specialization):** "Split" the Monolithic Policy into a **Phase-Locked Router** (the pre-trained Planner) and specialized **Parametric Experts**.

This strategy provides the best of both worlds: the **Global Coherence** of a monolithic policy and the **Local Precision** of experts. It specifically addresses two critical failure modes in robotics: **Perceptual Aliasing** (confusing similar phases) and **Control Chattering** (high-frequency expert switching).

---

## 2. The Core Innovation: "Bootstrapping" from a Monolithic Seed

The user's key insight is that training an MoE from random initialization is dangerous. The Router doesn't know *what* to route, and Experts don't know *what* to specialize in.

### 2.1 The "General Practitioner" Analogy
*   **Stage 1 (Monolithic Semantic Planner):** Think of this as training a **General Practitioner (GP)** doctor. The GP knows a little bit about everything (Grasping, Lifting, Placing). They understand the *entire lifecycle* of the patient (the task).
*   **Stage 2 (Mixture of Experts):** We don't fire the GP and hire random people. We **promote** the GP to be the **Hospital Administrator (Router)**.
    *   Because the GP (Planner) already knows "This looks like a heart problem," they can perfectly route the patient to the **Heart Surgeon (Grasp Expert)**.
    *   The Surgeon (Expert) doesn't need to know about brains or feet. They only train on heart surgery.

This guarantees that the Router is **intelligent from Step 0 of Phase 2**.

---

## 3. Architecture Phase 1: The Foundation (End-to-End)

We train the `SemanticPlanner` exactly as we do today (`train_semantic_planner.py`), but with **Phase Self-Supervision**.

*   **Input:** Multi-view images, Proprioception.
*   **Outputs:** 
    1.  `Action Chunk` (Standard Policy)
    2.  `Phase Logits` (Auxiliary Task: "What am I doing?")
*   **Loss:** $L_{Total} = L_{BC} + \lambda L_{Phase}$

**Result:** A strong, generalist policy that understands "I am approaching" vs "I am grasping," even if its physical execution is "imprecise" (the jack-of-all-trades problem).

---

## 4. Architecture Phase 2: The MoE Split (Post-Training)

This is where the new idea comes in. We perform a surgical architectural change to create **APEX-MoE**.

### 4.1 The Router (The Upgraded Planner)
*   **Backbone:** We take the **FROZEN** `SemanticPlanner` from Stage 1.
*   **Role:** It no longer outputs actions. It strictly outputs:
    1.  **Phase ID ($k_t$):** "Switch to Expert 1."
    2.  **Context Vector ($\alpha_t$):** "The object looks heavy/slippery." (Modulation).
    3.  **Visual Embedding ($z_t$):** The compressed "vision" for the experts.

### 4.2 The Phase-Locked Experts
We instantiate $K$ separate, smaller policy networks (The Experts).
*   **Input:** Visual Embedding ($z_t$) from Router, Proprioception.
*   **Modulation:** They also take $\alpha_t$ (Context) as input.
*   **Training:**
    *   **Expert 1 (Grasp)** is ONLY trained on data where `Phase == GRASP`.
    *   **Expert 2 (Place)** is ONLY trained on data where `Phase == PLACE`.

**Why this fixes "Control Chattering":**
Because the Router is the *Pre-Trained* Semantic Planner, its internal state is stable. It doesn't flicker randomly. It has "Temporal Inertia" learned from the 400-step horizon of Stage 1. It only switches when the semantic visual cues change effectively.

---

## 5. Mathematical Formulation: Parametric Skill Modulation

How do we handle variations (heavy vs light objects) without training 100 experts?

Let $E(s, \alpha)$ be a **Generalist Expert** (e.g., the Grasp Expert).
*   $s$: State (Position, Image).
*   $\alpha$: **Context Vector** predicted by the Router.

$$ a_t = E_{k_t}(s_t, \alpha_t) $$

where $k_t = \text{Router}(s_t)$.

**Training Objective (Phase 2):**
$$ L = \underbrace{|| a_t - E_{k_t}(s_t, \alpha_t) ||^2}_{\text{Expert Cloning}} + \underbrace{\lambda_{\text{sparse}} || \alpha_t ||^2}_{\text{Regularization}} $$

The Router learns to tweak $\alpha_t$ to help the Expert minimize error.
*   If the Expert is failing to grasp a heavy object, the gradients tell the Router: *"Modify $\alpha$ to increase grip force!"*

---

## 6. Implementation Plan: The "Bridge" Strategy

### Step 1: Verify the Foundation
*   Run `train_semantic_planner.py`.
*   **Metric:** Ensure `Phase Accuracy > 95%`. If the Planner can't identify the phase, it can't be a Router.

### Step 2: The Fork (Post-Training Script)
Create `train_apex_moe_post.py`.
1.  **Load** `checkpoints/semantic_planner_best.ckpt`.
2.  **Freeze** the Vision Encoder and Transformer.
3.  **Instantiate** the Expert Array (List of MLPs).
4.  **Dataset Split:**
    *   `Data_Grasp = All frames where GT_Phase == GRASP`
    *   `Data_Place = All frames where GT_Phase == PLACE`
5.  **Train Experts:** Train each Expert on its specific slice of data.

### Step 3: End-to-End Fine-Tuning (Optional)
Unfreeze everything and train with RL (DGPO/RLIF) to allow the Router to "invent" new strategies (e.g., switching to 'Grasp' earlier than the human demonstrator did).

---

## 7. Conclusion

By bootstrapping from a monolithic policy, we solve the cold-start problem of Mixture-of-Experts. **The Semantic Planner is not useless; it is the seed of intelligence.** It provides the visual understanding and temporal stability required to orchestrate a team of high-performance specialists.

This approach transforms the "Black Box" of Neural Networks into a **Structured, Hierarchical Control System** capable of solving 400+ step tasks reliability.
