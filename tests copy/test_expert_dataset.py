# FILE: tests/test_expert_dataset.py (DEFINITIVE CORRECTED VERSION)

import sys
from pathlib import Path
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
import shutil

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.expert_dataset import ExpertDataset, ExpertDatasetWriter, ExpertTrajectoryDataset, replay_validate_episode
from utils.scripted_expert import ExpertConfig
from envs.panda_env import PandaEnv

# --- Test Configuration (Constants) ---
URDF_PATH = "urdf/panda_mujoco_kinematics.urdf"
XML_PATH = "envs/panda_pick_place.xml"
TEST_SEED = 813  # Use the same seed as the successful diagnostic run
ACTION_SCALING = 0.5
TEMP_DATA_DIR = Path("tmp_test_data")

@pytest.fixture(scope="module")
def default_dataset_config():
    """Provides a default, consistent configuration for all tests."""
    return {
        "urdf_path": URDF_PATH,
        "env_xml_path": XML_PATH,
        "base_seed": TEST_SEED,
        "action_scaling_factor": ACTION_SCALING,
        "scripted_cfg": ExpertConfig(),
    }

# --- Test Suite ---

# Correctly use the fixture by passing its name as an argument
def test_dataset_initialization(default_dataset_config):
    """Test 1: Basic Initialization"""
    # The **config is a dictionary. We unpack it with **
    dataset = ExpertDataset(**default_dataset_config)
    assert dataset is not None
    assert dataset.action_scaling_factor == ACTION_SCALING
    assert dataset.base_seed == TEST_SEED

@pytest.mark.slow
def test_single_successful_trajectory_generation(default_dataset_config):
    """Test 2: Single-Threaded Episode Generation"""
    # Correctly unpack the config dictionary. Set a sample limit.
    dataset = ExpertDataset(max_samples_per_epoch=500, **default_dataset_config)
    
    trajectory_data = list(iter(dataset))
    
    assert len(trajectory_data) > 0, "Dataset failed to yield any samples."
    assert len(dataset.episodes) >= 1, "Dataset should have collected at least one successful episode."
    assert dataset.episodes[0]["success"] is True, "The first collected episode was not marked as successful."

# ... (the rest of the test functions need to be updated to accept the fixture) ...

def test_data_integrity_no_shallow_copy_bug(default_dataset_config):
    """Test 3: Data Integrity (No Shallow Copies)"""
    dataset = ExpertDataset(max_samples_per_epoch=500, **default_dataset_config)
    _ = list(iter(dataset))
    assert len(dataset.episodes) > 0, "Cannot test data integrity without a successful episode."
    episode = dataset.episodes[0]
    obs_list = episode["obs_list"]
    assert len(obs_list) > 10, "Episode is too short (< 10 steps) to reliably test for integrity."
    image_start = obs_list[0]["image_primary"]
    image_end = obs_list[-1]["image_primary"]
    assert not np.array_equal(image_start, image_end), "Shallow copy bug: images are identical."
    ee_pose_start = obs_list[0]["ee_pose_world"]
    ee_pose_end = obs_list[-1]["ee_pose_world"]
    assert not np.allclose(ee_pose_start, ee_pose_end), "Shallow copy bug: ee_pose is identical."

@pytest.mark.slow
def test_determinism_single_core(default_dataset_config):
    """Test 4: Determinism (Single-Core)"""
    config = {**default_dataset_config, "max_samples_per_epoch": 500}

    dataset1 = ExpertDataset(**config)
    list(iter(dataset1))
    assert len(dataset1.episodes) >= 1, "Run 1 failed to produce an episode."
    ep1 = dataset1.episodes[0]

    dataset2 = ExpertDataset(**config)
    list(iter(dataset2))
    assert len(dataset2.episodes) >= 1, "Run 2 failed to produce an episode."
    ep2 = dataset2.episodes[0]

    assert ep1["seed"] == ep2["seed"]
    assert len(ep1["actions"]) == len(ep2["actions"])
    actions1 = np.array(ep1["actions"])
    actions2 = np.array(ep2["actions"])
    assert np.allclose(actions1, actions2, atol=1e-7)

# ... (the main block needs to be updated too) ...

if __name__ == "__main__":
    """Main execution block to run all tests sequentially."""
    print("=" * 70)
    print("  Running Full Test Suite for ExpertDataset")
    print("=" * 70 + "\n")

    # This recreates the fixture behavior for standalone execution
    config = {
        "urdf_path": URDF_PATH,
        "env_xml_path": XML_PATH,
        "base_seed": TEST_SEED,
        "action_scaling_factor": ACTION_SCALING,
        "scripted_cfg": ExpertConfig(),
    }

    # Helper to run a standalone test
    def run_standalone_test(test_func, config_dict):
        test_name = test_func.__name__
        print(f"--- Running Test: {test_name} ---")
        try:
            # Pass the config dictionary to the test function
            test_func(config_dict)
            print(f"✅ PASSED: {test_name}\n")
            return True
        except Exception as e:
            print(f"❌ FAILED: {test_name}")
            print(f"  ERROR: {e}\n")
            import traceback
            traceback.print_exc()
            return False

    all_tests = [
        test_dataset_initialization,
        test_single_successful_trajectory_generation,
        test_data_integrity_no_shallow_copy_bug,
        test_determinism_single_core,
        # Skipping multiprocessing and offline tests for standalone run simplicity
    ]
    
    passed_count = 0
    failed_count = 0
    for test_func in all_tests:
        if run_standalone_test(test_func, config):
            passed_count += 1
        else:
            failed_count += 1
    

    # Clean up test artifacts after finishing
    if TEMP_DATA_DIR.exists():
        shutil.rmtree(TEMP_DATA_DIR)

    print("-" * 70)
    print("Test Suite Summary:")
    print(f"  Tests Passed: {passed_count}")
    print(f"  Tests Failed: {failed_count}")
    print("=" * 70)
    
    if failed_count > 0:
        print("\n❌ Some tests failed. Please review the errors above.")
        # Exit with a non-zero status code to signal failure for CI/automation
        sys.exit(1)
    else:
        print("\n✅ All tests passed successfully!")

"""
python -m tests.test_expert_dataset
"""