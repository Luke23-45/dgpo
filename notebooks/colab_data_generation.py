# =============================================================================
# COLAB DATA GENERATION NOTEBOOK
# =============================================================================
# Run these cells in ORDER. Each cell must complete before running the next.
# =============================================================================

# =============================================================================
# CELL 1: GPU RENDERING SETUP (RUN FIRST - BEFORE ANY OTHER IMPORTS)
# =============================================================================
"""
This MUST be the first code cell in your notebook.
It configures EGL rendering before mujoco is imported.
"""

import os
import subprocess
import sys

# 1. Set MUJOCO_GL to EGL BEFORE any mujoco import
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# 2. Install EGL dependencies
print("📦 Installing EGL dependencies...")
subprocess.run(['apt-get', 'update', '-qq'], check=False, capture_output=True)
subprocess.run(['apt-get', 'install', '-y', '-qq', 'libegl1', 'libegl1-mesa', 'libgl1-mesa-glx', 'patchelf'], 
               check=False, capture_output=True)

# 3. Find NVIDIA EGL library and configure vendor
print("🔍 Configuring NVIDIA EGL...")
try:
    result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True, timeout=10)
    nvidia_lib = None
    for line in result.stdout.split('\n'):
        if 'libEGL_nvidia.so.0' in line and '=>' in line:
            nvidia_lib = line.split('=>')[1].strip()
            break
    
    if nvidia_lib:
        icd_path = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
        icd_content = f'{{"file_format_version": "1.0.0", "ICD": {{"library_path": "{nvidia_lib}"}}}}'
        os.makedirs(os.path.dirname(icd_path), exist_ok=True)
        with open(icd_path, 'w') as f:
            f.write(icd_content)
        print(f"✅ NVIDIA EGL configured: {nvidia_lib}")
    else:
        print("⚠️ NVIDIA EGL not found via ldconfig, trying find...")
        result = subprocess.run("find /usr/ -name 'libEGL_nvidia.so.0' 2>/dev/null | head -1",
                               shell=True, capture_output=True, text=True, timeout=30)
        nvidia_lib = result.stdout.strip()
        if nvidia_lib:
            icd_path = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
            icd_content = f'{{"file_format_version": "1.0.0", "ICD": {{"library_path": "{nvidia_lib}"}}}}'
            os.makedirs(os.path.dirname(icd_path), exist_ok=True)
            with open(icd_path, 'w') as f:
                f.write(icd_content)
            print(f"✅ NVIDIA EGL configured via find: {nvidia_lib}")
        else:
            print("❌ NVIDIA EGL not found. Rendering will be slow or fail.")
            del os.environ['MUJOCO_GL']
except Exception as e:
    print(f"⚠️ EGL setup error: {e}")

# 4. Verify
print("\n🧪 Verifying MuJoCo rendering...")
try:
    import mujoco
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light name="l"/><body><geom size="0.1"/></body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=64, width=64)
    renderer.update_scene(data)
    pixels = renderer.render()
    print(f"✅ EGL VERIFIED! Rendered {pixels.shape} image. Hardware acceleration ACTIVE.")
    del renderer, data, model
except Exception as e:
    print(f"❌ Render verification failed: {e}")
    print("   Data generation may produce blank images.")

print("\n" + "="*60)
print("CELL 1 COMPLETE. Now run Cell 2 to clone the repo.")
print("="*60)


# =============================================================================
# CELL 2: CLONE REPOSITORY
# =============================================================================
"""
Clone your repository and install dependencies.
"""

# Clone repo (uncomment and modify for your repo)
# !git clone https://github.com/YOUR_USERNAME/redhot.git dgpo
# %cd dgpo

# If already cloned, just cd
# %cd dgpo

# Install dependencies
# !pip install -q mujoco dm_control transformers pytorch-lightning omegaconf lmdb tqdm pyyaml

print("✅ Repository ready. Now run Cell 3 for data generation.")


# =============================================================================
# CELL 3: RUN DATA GENERATION
# =============================================================================
"""
Generate training data using bootstrap script.
"""

# Configure generation parameters
NUM_EPISODES = 400
NUM_WORKERS = 2
OUTPUT_DIR = "/content/fresh_data/"

# Run bootstrap (includes generation + advantage calculation)
!python scripts/bootstrap_training_data.py \
    --episodes {NUM_EPISODES} \
    --workers {NUM_WORKERS} \
    --output_dir {OUTPUT_DIR}

print("\n" + "="*60)
print("DATA GENERATION COMPLETE!")
print(f"Dataset location: {OUTPUT_DIR}/final_training_set/training_set.lmdb")
print("="*60)
