"""
A simple, powerful tool to inspect and compare the kinematic chains
defined in a MuJoCo XML file and a URDF file.

This script prints a side-by-side view of the link transforms, making
any discrepancies in translation (xyz) or rotation (rpy) immediately obvious.
"""
import mujoco
import numpy as np
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R
from typing import List
import traceback
from utils.ik_solver import IKSolver

try:
    from ikpy.link import URDFLink
except ImportError:
    # In ikpy 3.4.2 URDFLink might not be exposed. 
    # We'll do a fallback: just allow any object with .name and .joint_type etc.
    URDFLink = None

# --- Configuration: Point these to your files ---
XML_PATH = "envs/panda_pick_place.xml"
URDF_PATH = "urdf/panda_mujoco_kinematics.urdf"
BASE_LINK_NAME = "link0"

def inspect_mujoco_chain(model: mujoco.MjModel):
    """Prints the kinematic chain from a loaded MuJoCo model."""
    print("---  inspecting MuJoCo XML Kinematics (Ground Truth) ---")
    print(f"{'Parent':<15} -> {'Child':<15} | {'Joint Name':<20} | {'Translation (xyz)':<25} | {'Rotation (rpy deg)':<25}")
    print("-" * 110)

    try:
        base_body_id = model.body(BASE_LINK_NAME).id
        print(f"[DEBUG] Base body id for '{BASE_LINK_NAME}': {base_body_id}")
    except Exception:
        print(f"❌ ERROR: Base link '{BASE_LINK_NAME}' not found in MuJoCo model bodies.")
        traceback.print_exc()
        return

    # Use a queue to traverse the tree iteratively
    bodies_to_visit = []
    for i in range(model.nbody):
        try:
            if model.body(i).parentid == base_body_id:
                bodies_to_visit.append(i)
        except Exception as e:
            print(f"[WARNING] Failed accessing body {i}: {e}")

    while bodies_to_visit:
        body_id = bodies_to_visit.pop(0)
        try:
            body = model.body(body_id)
            parent_name = model.body(body.parentid).name
            
            # Extract translation and rotation
            pos = body.pos
            quat_wxyz = body.quat
            
            # Convert quat to rpy degrees for human-readable comparison
            rpy_deg = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_euler('xyz', degrees=True)
            
            # Find the associated joint, if any
            joint_name = "(fixed)"
            if body.jntnum > 0:
                joint_id = body.jntadr[0]
                joint_name = model.jnt(joint_id).name

            # Format and print the information
            pos_str = f"[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]"
            rpy_str = f"[{rpy_deg[0]:.4f}, {rpy_deg[1]:.4f}, {rpy_deg[2]:.4f}]"
            print(f"{parent_name:<15} -> {body.name:<15} | {joint_name:<20} | {pos_str:<25} | {rpy_str:<25}")

            # Add children of the current body to the queue
            for i in range(model.nbody):
                try:
                    if model.body(i).parentid == body_id:
                        bodies_to_visit.append(i)
                except Exception as e:
                    print(f"[WARNING] Failed accessing child body {i}: {e}")

        except Exception:
            print(f"❌ ERROR processing body id {body_id}")
            traceback.print_exc()

def inspect_urdf_chain(urdf_path: str):
    """Prints the kinematic chain from a loaded URDF file."""
    print("\n--- inspecting URDF Kinematics ---")
    print(f"{'Parent':<15} -> {'Child':<15} | {'Joint Name':<20} | {'Translation (xyz)':<25} | {'Rotation (rpy deg)':<25}")
    print("-" * 110)

    try:
        # Load chain exactly like in IKSolver (robust, defensive parsing)
        temp_chain = Chain.from_urdf_file(
            urdf_path,
            base_elements=[BASE_LINK_NAME],
        )
        print(f"[DEBUG] Loaded chain from URDF file: {urdf_path}")
    except Exception:
        print(f"❌ ERROR loading URDF file '{urdf_path}'")
        traceback.print_exc()
        return

    try:
        # Generate active mask (same as IKSolver)
        active_links_mask = []
        for i, link in enumerate(temp_chain.links):
            # Defensive check for attributes
            joint_type = getattr(link, 'joint_type', None)
            is_active = (i != 0 and i != len(temp_chain.links) - 1 and joint_type == 'revolute')
            active_links_mask.append(is_active)
        print(f"[DEBUG] Computed active_links_mask: {active_links_mask}")

        # Final chain (keeps consistent indexing)
        chain = Chain.from_urdf_file(
            urdf_path,
            base_elements=[BASE_LINK_NAME],
            active_links_mask=active_links_mask,
        )
        print("\n[DEBUG] Joint info after loading URDF chain:")
        for idx, link in enumerate(chain.links):
            try:
                # Ignore OriginLink (first link usually)
                if link.__class__.__name__ == "OriginLink":
                    print(f"  Link {idx}: {link.name} (OriginLink) - skipping joint info")
                    continue

                joint_name = getattr(link, 'name', '(no name)')
                joint_type = getattr(link, 'joint_type', '(no joint_type)')
                origin_xyz = getattr(link, 'origin_translation', [0, 0, 0])
                origin_rpy = getattr(link, 'origin_orientation', [0, 0, 0])

                print(f"  Link {idx}: {joint_name} | Joint Type: {joint_type}")
                print(f"    origin_xyz: {origin_xyz}")
                print(f"    origin_rpy: {origin_rpy}")

            except Exception as e:
                print(f"  [ERROR] Reading joint info for link index {idx}: {e}")

    except Exception:
        print("❌ ERROR re-loading chain with active_links_mask")
        traceback.print_exc()
        return



if __name__ == "__main__":
    import sys
    import time

    print("="*50)
    print(" Kinematic Chain Inspector")
    print("="*50)

    # Load MuJoCo model once
    try:
        start = time.time()
        mujoco_model = mujoco.MjModel.from_xml_path(XML_PATH)
        print(f"[DEBUG] Loaded MuJoCo model from {XML_PATH} in {time.time()-start:.3f}s")
        inspect_mujoco_chain(mujoco_model)
    except Exception as e:
        print(f"❌ ERROR loading or inspecting MuJoCo XML: {e}")
        traceback.print_exc()

    print()

    # Inspect the URDF
    try:
        start = time.time()
        inspect_urdf_chain(URDF_PATH)
        print(f"[DEBUG] Finished URDF inspection in {time.time()-start:.3f}s")
    except Exception as e:
        print(f"❌ ERROR loading or inspecting URDF: {e}")
        traceback.print_exc()

    print("\n\n--- Comparison Guide ---")
    print("Carefully compare the 'Translation (xyz)' and 'Rotation (rpy deg)' columns.")
    print("Every value in a row should be identical (within a tiny margin of error).")
    print("If you find a mismatch, the URDF is incorrect and needs to be fixed.")
