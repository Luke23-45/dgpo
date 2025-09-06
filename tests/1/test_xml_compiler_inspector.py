import mujoco
import os

# --- Configuration ---
# The path to the scene file we want to inspect.
# Ensure this is our final, correct scene file.
SCENE_XML_PATH = "envs/panda_pick_place.xml"

# The path where we will save the fully compiled model.
COMPILED_XML_SAVE_PATH = "compiled_model.xml"

def inspect_compiled_model():
    """
    Loads a MuJoCo scene, compiles it, and saves the final,
    fully-resolved model to a new XML file for inspection.
    """
    print("--- MuJoCo Compiler Inspector ---")

    # --- 1. Load the Model ---
    # This step performs the in-memory compilation. It reads final_scene.xml,
    # finds the <include> tag, loads panda.xml, and merges them into one
    # complete model in memory.
    print(f"⏳ Loading and compiling scene from: {SCENE_XML_PATH}...")
    try:
        model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
        print("✅ Model loaded and compiled in memory successfully.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not load the model. This is the source of the problem.")
        print(f"   Please check your XML files and asset paths.")
        print(f"   Error: {e}")
        return

    # --- 2. Save the Compiled Model to a File ---
    # This is the key step. We take the fully-resolved model from memory
    # and write it back to disk as a new, standalone XML file.
    print(f"\n⏳ Saving the compiled model to: {COMPILED_XML_SAVE_PATH}...")
    try:
        mujoco.mj_saveLastXML(COMPILED_XML_SAVE_PATH, model)
        print(f"✅ Compiled model saved successfully.")
    except Exception as e:
        print(f"❌ ERROR: Could not save the compiled model. Error: {e}")
        return

    # --- 3. Print a summary of key components for quick verification ---
    print("\n--- Summary of Compiled Model ---")
    print(f"  Number of bodies (nbody): {model.nbody}")
    print(f"  Number of joints (njnt): {model.njnt}")
    print(f"  Number of geoms (ngeom): {model.ngeom}")
    print(f"  Number of sites (nsite): {model.nsite}")
    print(f"  Number of actuators (nu): {model.nu}")
    print("\n--- ✅ Inspection Complete ---")
    print(f"Please open '{COMPILED_XML_SAVE_PATH}' in a text editor to view the full model.")

if __name__ == "__main__":
    inspect_compiled_model()