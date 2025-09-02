Pipeline A (Modular) = ExpertDataset + BCTrainer + BCNet + transfer_bc_weights.

Pipeline B (Legacy Integrated) = pretrain_policy.py with its own dataset generator + its own BC loop inside an SB3 agent.


### Overview of Training Processes

There are three primary training strategies to evaluate, which correspond to your main method and two essential baselines.

1.  **DGPO-Foundation (Full Method):** Pre-train with Behavioral Cloning (BC), then fine-tune with Reinforcement Learning (RL) *and* the OCTO divergence reward.
2.  **BC + RL (Baseline 1):** Pre-train with BC, then fine-tune with standard RL *without* the divergence reward.
3.  **RL from Scratch (Baseline 2):** Train with standard RL from a randomly initialized policy, with no pre-training.

---

### **Stage 1: Pre-training (Run Once for Methods 1 & 2)**

This is the required first step for any method involving pre-training. It generates the expert dataset and creates the initial "smart" policy.

**Goal:** Create the `policy_pretrained_bc.zip` file.

**Command:**
```bash
python training/pretrain.py --num_samples 20000 --epochs 20 --pretrained_out trained_models/policy_pretrained_bc.zip
```

---

### **Stage 2: Fine-Tuning Experiments**

After completing Stage 1, you can run the following training processes in any order. Each is a separate experiment.

#### 1. Full DGPO-Foundation (Your Main Method)

**Concept:** Starts with the smart BC policy and uses RL to master the task while the OCTO divergence reward prevents it from "forgetting" plausible behaviors.

**Key Flags:**
*   `--bc_model_path`: Points to your pre-trained model.
*   `--w_plausibility`: **Set to a value greater than 0** to enable the divergence reward.

**Command:**
```bash
python run_experiment.py --bc_model_path trained_models/policy_pretrained_bc.zip --run_name DGPO_Full --w_plausibility 0.1
```

---

#### 2. Baseline: BC + RL (No Divergence Reward)

**Concept:** Starts with the smart BC policy but uses standard RL fine-tuning. This experiment measures the specific benefit of the divergence reward itself.

**Key Flags:**
*   `--bc_model_path`: Points to your pre-trained model.
*   `--w_plausibility`: **Set to 0** to disable the divergence reward.

**Command:**
```bash
python run_experiment.py --bc_model_path trained_models/policy_pretrained_bc.zip --run_name Baseline_BC_plus_RL --w_plausibility 0.0
```

---

#### 3. Baseline: RL from Scratch

**Concept:** Standard RL with a "blank slate" agent. This measures the total benefit of your entire pre-training pipeline (both the data generation and the BC).

**Key Flags:**
*   `--bc_model_path`: **Set to `None`** to start from a random policy.
*   `--w_plausibility`: Set to `0` as the divergence reward is not applicable.

**Command:**
```bash
python run_experiment.py --bc_model_path None --run_name Baseline_RL_from_Scratch --w_plausibility 0.0
```

