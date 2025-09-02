# debug_octo_inference.py
"""
A minimal, standalone script to debug the core OCTO model inference step.

This script isolates the OctoModel from the rest of the training pipeline
to verify that, given a single observation from the environment, it produces
a sane action.

It performs the following steps:
1. Initializes the PandaEnv and OctoModel.
2. Resets the environment to get a single observation.
3. Saves the camera images ('image_primary' and 'image_wrist') to disk
   for manual verification.
4. Constructs the OCTO-compliant observation dictionary using the
   *exact* same logic as the ExpertDataset.
5. Constructs the language task.
6. Prints a detailed report of all data being sent to the model.
7. Calls `model.sample_actions` and prints the raw output pose.
8. Performs a final sanity check on the output.
"""
import os
import logging
import numpy as np
import jax
from PIL import Image

# --- Project Imports ---
# Make sure your project structure allows these imports from the root
from envs.panda_env import PandaEnv
from octo.model.octo_model import OctoModel

# --- Configuration ---
# Match these to your training setup
OCTO_MODEL_NAME = "hf://rail-berkeley/octo-small-1.5"
ENV_XML_PATH = "envs/panda_pick_place.xml"  # Use the same scene XML
INSTRUCTION = "pick up the red block"
SEED = 1234  # Use a fixed seed for reproducibility

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("OCTO_DEBUG")

def build_octo_observation_for_debug(env_obs: dict) -> dict:
    """
    A standalone copy of the exact observation building logic from ExpertDataset.
    This ensures we are testing the exact same data transformation.
    """
    T, B = 2, 1
    octo_obs = {}

    # image_primary (ensure HWC)
    img = np.asarray(env_obs["image_primary"])
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    octo_obs["image_primary"] = np.repeat(img[np.newaxis, np.newaxis, ...], T, axis=1)

    # image_wrist (ensure HWC)
    if "image_wrist" in env_obs:
        wrist = np.asarray(env_obs["image_wrist"])
        if wrist.ndim == 3 and wrist.shape[0] in (1, 3):
            wrist = np.transpose(wrist, (1, 2, 0))
        octo_obs["image_wrist"] = np.repeat(wrist[np.newaxis, np.newaxis, ...], T, axis=1)

    # timestep (B, T)
    t0 = int(np.asarray(env_obs.get("timestep", 0), dtype=np.int32).reshape(()))
    octo_obs["timestep"] = np.full((B, T), t0, dtype=np.int32)

    # pad masks ONLY for keys present in the final dict
    pad = np.ones((B, T), dtype=bool)
    nested_pad = {k: pad for k in octo_obs if k != "pad_mask_dict"}
    octo_obs["pad_mask_dict"] = nested_pad

    # Add the required legacy key back to fix the KeyError
    octo_obs["timestep_pad_mask"] = pad
    # Add the required legacy key
    octo_obs["timestep_pad_mask"] = pad

    return octo_obs

def main():
    logger.info("--- Starting OCTO Inference Debug Test ---")

    # 1. Initialize Environment
    logger.info(f"Loading environment from: {ENV_XML_PATH}")
    env = PandaEnv(xml_path=ENV_XML_PATH)

    # 2. Get a single observation
    logger.info(f"Resetting environment with seed: {SEED}")
    obs, _ = env.reset(seed=SEED)

    # 3. Save images for verification
    output_dir = "debug_output"
    os.makedirs(output_dir, exist_ok=True)
    
    primary_img_path = os.path.join(output_dir, "debug_primary_image.png")
    wrist_img_path = os.path.join(output_dir, "debug_wrist_image.png")
    
    Image.fromarray(obs["image_primary"]).save(primary_img_path)
    Image.fromarray(obs["image_wrist"]).save(wrist_img_path)
    logger.info(f"✅ Saved primary camera view to: {primary_img_path}")
    logger.info(f"✅ Saved wrist camera view to: {wrist_img_path}")
    
    # 4. Initialize OCTO Model
    logger.info(f"Loading OCTO model: {OCTO_MODEL_NAME}")
    model = OctoModel.load_pretrained(OCTO_MODEL_NAME)
    
    # 5. Construct Inputs
    logger.info("Constructing OCTO-compliant observation and task...")
    octo_obs = build_octo_observation_for_debug(obs)
    task = model.create_tasks(texts=[INSTRUCTION])
    
    # 6. Print Detailed Report of Inputs
    print("\n" + "="*80)
    print("🔬 DATA SENT TO OCTO MODEL 🔬")
    print("="*80)
    print(f"\n[TASK - Language Instruction]")
    print(f"  - Text: '{INSTRUCTION}'")
    
    print("\n[OBSERVATIONS - Shapes and Dtypes]")
    for key, value in octo_obs.items():
        if isinstance(value, dict):
            print(f"  - {key}: <dict with {len(value)} keys>")
            for sub_key, sub_value in value.items():
                print(f"    - {sub_key}: {sub_value.shape}, dtype={sub_value.dtype}")
        else:
            print(f"  - {key}: {value.shape}, dtype={value.dtype}")
    print("="*80 + "\n")

    # 7. Call sample_actions
    logger.info("Calling model.sample_actions(...)")
    key = jax.random.PRNGKey(SEED)
    raw_action = model.sample_actions(octo_obs, task, rng=key)
    
    # 8. Print and Analyze Output
    target_pose_world = np.array(raw_action[0, 0, :7], dtype=np.float32)
    
    print("\n" + "="*80)
    print("🔬 RAW OUTPUT FROM OCTO MODEL 🔬")
    print("="*80)
    print(f"  - Full action shape: {raw_action.shape}")
    print(f"  - Extracted 7D Pose (x, y, z, qx, qy, qz, qw):")
    print(f"    {np.round(target_pose_world, 4)}")
    print("="*80 + "\n")

    # 9. Final Sanity Check
    logger.info("--- Final Sanity Check ---")
    ee_pose_env = env.get_ee_pose()
    predicted_z = target_pose_world[2]
    env_z = ee_pose_env[2]
    
    logger.info(f"  - Predicted Target Z: {predicted_z:.4f}")
    logger.info(f"  - Environment EE Z:   {env_z:.4f}")
    
    if np.sign(predicted_z) != np.sign(env_z) and env_z > 0.1:
        logger.error("❌ FAILURE: Z-axis mismatch detected! The problem is reproduced.")
    elif not np.isfinite(predicted_z):
        logger.error("❌ FAILURE: Predicted Z is not a finite number (NaN or Inf).")
    else:
        logger.info("✅ SUCCESS: Predicted Z-axis has the correct sign.")
        
    env.close()

if __name__ == "__main__":
    main()