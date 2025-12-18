
import sys
import logging

# Configure logging to see output
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Verification")

def check_import(module_name):
    try:
        log.info(f"Checking import: {module_name}...")
        __import__(module_name)
        log.info(f"SUCCESS: {module_name} imported correctly.")
    except Exception as e:
        log.error(f"FAILURE: Could not import {module_name}. Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_import("evaluate.evaluate_dgpo")
    check_import("evaluate.evaluate_semantic_planner_auto")
    check_import("evaluate.evaluate_unified_planner_auto")
    log.info("All modules verified successfully.")
