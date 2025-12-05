import os

# --- Configuration ---
FOLDER_PATH = 'challenge2/clean_train_data'  # Folder containing your images and masks
DRY_RUN = False              # Set to True to print only. Set to False to delete files.

def remove_orphan_masks(folder, dry_run=True):
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        return

    print(f"Scanning '{folder}' for orphan masks...")
    
    files = os.listdir(folder)
    deleted_count = 0
    
    for filename in files:
        # Check if the file is a mask
        if filename.startswith("mask_") and filename.endswith(".png"):
            # Construct expected image filename: mask_1234.png -> img_1234.png
            img_filename = filename.replace("mask_", "img_")
            
            img_path = os.path.join(folder, img_filename)
            mask_path = os.path.join(folder, filename)
            
            # If the corresponding image does NOT exist, remove the mask
            if not os.path.exists(img_path):
                if dry_run:
                    print(f"[DRY RUN] Would delete: {filename} (Missing {img_filename})")
                else:
                    try:
                        os.remove(mask_path)
                        print(f"[DELETED] {filename}")
                    except OSError as e:
                        print(f"Error deleting {filename}: {e}")
                
                deleted_count += 1

    if dry_run:
        print(f"\nDry run complete. Found {deleted_count} orphan masks.")
        print("Set 'DRY_RUN = False' in the script to actually delete them.")
    else:
        print(f"\nCleanup complete. Deleted {deleted_count} orphan masks.")

if __name__ == "__main__":
    remove_orphan_masks(FOLDER_PATH, DRY_RUN)