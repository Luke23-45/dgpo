# FILE: scripts/colab_setup.py
# SOTA Google Colab GPU Rendering Setup for MuJoCo
# 
# This script configures hardware-accelerated EGL rendering in Google Colab.
# Run this ONCE at the start of your notebook, BEFORE importing mujoco.
#
# Features:
# - Fast NVIDIA EGL library lookup via ldconfig cache (<0.1s vs 10s for find)
# - Automatic fallback to software rendering if GPU unavailable
# - Minimal apt-get calls for speed
# - Verification via dummy render test

import os
import subprocess
import sys
import logging

# Configure Logger
log = logging.getLogger("GPU_Setup")
log.setLevel(logging.INFO)
if not log.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(handler)


def setup_colab_rendering(force_egl: bool = True, verify: bool = True) -> bool:
    """
    Configure MuJoCo for hardware-accelerated EGL rendering in Google Colab.
    
    This MUST be called BEFORE importing mujoco.
    
    Args:
        force_egl: If True, set MUJOCO_GL=egl. If False, auto-detect.
        verify: If True, run a lightweight render test to verify setup.
        
    Returns:
        True if hardware rendering is enabled, False if falling back to software.
    """
    # Check if we're in Colab
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False
        
    if not in_colab:
        log.info("ℹ️ Not in Colab. Skipping EGL setup.")
        return False
    
    log.info("🚀 Google Colab detected. Configuring hardware-accelerated rendering...")
    
    # =========================================================================
    # STEP 1: Set MUJOCO_GL environment variable BEFORE any mujoco import
    # =========================================================================
    if force_egl:
        os.environ['MUJOCO_GL'] = 'egl'
        log.info("✅ MUJOCO_GL=egl (Hardware Rendering Mode)")
    
    # =========================================================================
    # STEP 2: Fast NVIDIA EGL library lookup via ldconfig cache
    # This is ~100x faster than 'find /usr/ -name ...'
    # =========================================================================
    log.info("🔍 Locating NVIDIA EGL driver...")
    nvidia_lib_path = None
    
    try:
        # ldconfig -p returns cached library info
        result = subprocess.run(
            ["ldconfig", "-p"], 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=5
        )
        
        # Parse output for NVIDIA EGL library
        for line in result.stdout.split('\n'):
            if "libEGL_nvidia.so.0" in line and "=>" in line:
                nvidia_lib_path = line.split("=>")[1].strip()
                break
                
    except Exception as e:
        log.warning(f"⚠️ ldconfig failed: {e}. Trying fallback...")
    
    # Fallback: disk scan (slower, but works)
    if not nvidia_lib_path:
        try:
            result = subprocess.run(
                "find /usr/ -name 'libEGL_nvidia.so.0' 2>/dev/null | head -1",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            nvidia_lib_path = result.stdout.strip()
        except Exception:
            pass
    
    # =========================================================================
    # STEP 3: Write NVIDIA EGL Vendor Configuration
    # =========================================================================
    if nvidia_lib_path:
        log.info(f"✅ Found NVIDIA EGL: {nvidia_lib_path}")
        
        icd_path = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
        icd_content = f'''{{"file_format_version": "1.0.0", "ICD": {{"library_path": "{nvidia_lib_path}"}}}}'''
        
        try:
            os.makedirs(os.path.dirname(icd_path), exist_ok=True)
            with open(icd_path, 'w') as f:
                f.write(icd_content)
            log.info(f"✅ EGL vendor config written: {icd_path}")
        except PermissionError:
            log.warning("⚠️ Could not write EGL config (permission denied). Trying sudo...")
            try:
                subprocess.run(
                    f'echo \'{icd_content}\' | sudo tee {icd_path}',
                    shell=True,
                    check=True,
                    capture_output=True
                )
                log.info(f"✅ EGL vendor config written via sudo")
            except Exception as e:
                log.error(f"❌ Failed to write EGL config: {e}")
                return False
    else:
        log.error("❌ NVIDIA EGL library not found. GPU rendering unavailable.")
        log.warning("   Falling back to software rendering (SLOW).")
        if 'MUJOCO_GL' in os.environ:
            del os.environ['MUJOCO_GL']
        return False
    
    # =========================================================================
    # STEP 4: Install minimal dependencies (usually pre-installed in Colab)
    # =========================================================================
    try:
        subprocess.run(
            ['apt-get', 'install', '-y', '-qq', 'libegl1', 'patchelf'],
            capture_output=True,
            timeout=60
        )
    except Exception:
        pass  # Usually already installed
    
    # =========================================================================
    # STEP 5: Verification (optional but recommended)
    # =========================================================================
    if verify:
        log.info("🧪 Verifying EGL rendering...")
        try:
            import mujoco
            model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light/></worldbody></mujoco>')
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height=64, width=64)
            renderer.update_scene(data)
            renderer.render()
            log.info("✅ EGL Verification PASSED. Hardware rendering active.")
            return True
        except Exception as e:
            log.error(f"❌ EGL Verification FAILED: {e}")
            log.warning("   Rendering may fall back to software mode.")
            return False
    
    return True


# =============================================================================
# AUTO-EXECUTE WHEN IMPORTED
# =============================================================================
# If this module is imported in Colab, automatically run setup
# This allows: `from scripts.colab_setup import *` to work seamlessly

_SETUP_COMPLETE = False

def _auto_setup():
    global _SETUP_COMPLETE
    if not _SETUP_COMPLETE:
        try:
            import google.colab
            setup_colab_rendering(force_egl=True, verify=False)
            _SETUP_COMPLETE = True
        except ImportError:
            pass

# Run on import (verify=False to avoid importing mujoco during setup)
_auto_setup()


# =============================================================================
# CLI USAGE
# =============================================================================
if __name__ == "__main__":
    success = setup_colab_rendering(force_egl=True, verify=True)
    sys.exit(0 if success else 1)
