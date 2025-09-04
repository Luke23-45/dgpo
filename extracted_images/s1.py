import os
import shutil

# --- CONFIGURATION ---
# The script is being run from this directory, which contains the 'page_...' folders.
# We use an 'r' before the string to make it a "raw string", which handles backslashes correctly on Windows.
BASE_DIR = r"C:\Users\Hellx\Documents\Programming\python\Project\dgpo_project\extracted_images"

# Let's save the new images to your Desktop for easy access.
# os.path.expanduser("~") gets your user home directory (e.g., C:\Users\Hellx)
DEST_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "Renamed_Lab_Images")

# This mapping remains the same.
FILE_MAP = {
    "page_1/img_1.png":      "university_logo.png",
    "page_7/img_1.png":      "waterfall_model.png",
    "page_8/img_1.png":      "incremental_model.png",
    "page_9/img_1.png":      "prototyping_model.png",
    "page_19/img_1.png":     "use_case_diagram.png",
    "page_21/img_1.png":     "context_diagram_banking.png",
    "page_21/img_2.png":     "use_case_diagram_banking.png",
    "page_22/img_1.jpeg":    "sequence_diagram_banking.jpeg",
    "page_22/img_2.png":     "activity_diagram_banking.png",
    "page_25/img_1.png":     "layered_architecture_ecommerce.png",
    "page_26/img_1.png":     "client_server_architecture_ecommerce.png",
    "page_29/img_1.png":     "class_diagram_hospital.png",
    "page_29/img_2.jpeg":    "object_diagram_hospital.jpeg",
}

def main():
    """
    Main function to orchestrate the copying and renaming of image files.
    """
    print("--- Starting Image Organization Script ---")

    # Check if the BASE_DIR actually exists
    if not os.path.isdir(BASE_DIR):
        print(f"FATAL ERROR: The source directory does not exist: '{BASE_DIR}'")
        print("Please make sure the BASE_DIR variable is set correctly.")
        return

    # 1. Create the destination directory if it doesn't exist.
    try:
        os.makedirs(DEST_DIR, exist_ok=True)
        print(f"Destination folder is ready: '{DEST_DIR}'")
    except OSError as e:
        print(f"FATAL ERROR: Could not create destination folder. Reason: {e}")
        return

    success_count = 0
    error_count = 0

    # 2. Iterate through the map and process each file.
    print("\n--- Processing Files ---")
    for old_relative_path, new_filename in FILE_MAP.items():
        # Correctly handle path separators for Windows
        old_relative_path_windows = old_relative_path.replace('/', os.sep)
        source_path = os.path.join(BASE_DIR, old_relative_path_windows)
        dest_path = os.path.join(DEST_DIR, new_filename)

        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, dest_path)
                print(f"  [SUCCESS] Copied '{source_path}' to '{dest_path}'")
                success_count += 1
            except Exception as e:
                print(f"  [ERROR] Could not copy '{source_path}'. Reason: {e}")
                error_count += 1
        else:
            print(f"  [WARNING] Source file not found, skipping: '{source_path}'")
            error_count += 1

    # 4. Print a final summary report.
    print("\n--- Script Finished ---")
    print(f"Summary: {success_count} files successfully copied.")
    print(f"         {error_count} files could not be found or copied.")
    if error_count == 0:
        print("All files processed successfully!")
    else:
        print("Please review the warnings/errors above.")
    print(f"Your renamed files are located in: '{DEST_DIR}'")


if __name__ == "__main__":
    main()