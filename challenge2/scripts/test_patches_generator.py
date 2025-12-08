import os
import cv2
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Hyperparameters (Must match Training)
# -----------------------------------------------------------------------------
INPUT_SIZE = 128            # Patch size
STRIDE = 64                 # Overlap
MASK_OVERLAP_THRESH = 0.70  # Strict: Patch must overlap the mask by at least 70%
# Tissue Detector Params
TISSUE_FRACTION_THRESH = 0.70 
VARIANCE_THRESH = 5.0

# --- PARAMETER ---
# How many distinct "highest distribution" areas to find in Fallback
NUM_PATCHES = 4 
# -----------------

TEST_FOLDER = "challenge2/test_data"
TEST_PATCHES_DIR = "challenge2/test_patches"
TEST_PATCH_CSV = "challenge2/test_patches.csv"

os.makedirs(TEST_PATCHES_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Helpers (Same as Training)
# -----------------------------------------------------------------------------
def patch_has_tissue(patch_rgb, frac_color_thresh=0.30, var_thresh=20.0):
    patch_hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
    Hc, Sc, Vc = cv2.split(patch_hsv)
    sat_thresh = 25    
    val_max = 240
    color_mask = (Sc > sat_thresh) & (Vc < val_max)
    frac_color = color_mask.mean()
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    var_gray = gray.var()
    return (frac_color >= frac_color_thresh) and (var_gray >= var_thresh)

def patch_overlaps_mask(patch_mask, thresh=0.5):
    binary = (patch_mask > 0).astype(float)
    return binary.mean() >= thresh

def save_patch_test(img_id, y, x, img_rgb, records_list):
    """
    Saves RGB patch only (masks usually aren't saved for test, but can be if needed).
    """
    # 1. Crop patch
    patch_rgb = img_rgb[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
    
    # Convert RGB patch to BGR for OpenCV saving
    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)

    # 2. Define Filename
    filename_img = f"img_{img_id}_y{y}_x{x}.png"
    path_img = os.path.join(TEST_PATCHES_DIR, filename_img)
    
    # 3. Save File
    cv2.imwrite(path_img, patch_bgr)
    
    # 4. Record
    records_list.append((filename_img, img_id))

# -----------------------------------------------------------------------------
# 2. Main Loop
# -----------------------------------------------------------------------------
patch_records = []

test_files = [f for f in os.listdir(TEST_FOLDER) 
               if f.startswith("img_") and f.endswith(".png")]
test_ids = sorted([f.split("_")[1].split(".")[0] for f in test_files], key=int)

print(f"Found {len(test_ids)} test images to process.")

for img_id in test_ids:
    # --- A. Load Data (High Precision) ---
    img_path = os.path.join(TEST_FOLDER, f"img_{img_id}.png")
    mask_path = os.path.join(TEST_FOLDER, f"mask_{img_id}.png")

    # 1. Load Image UNCHANGED (High Fidelity)
    img_bgr_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_bgr_raw is None: continue

    # 2. Handle Channels (Sanitize)
    if img_bgr_raw.ndim == 3 and img_bgr_raw.shape[2] == 4:
        img_bgr_raw = img_bgr_raw[:, :, :3]
    
    if img_bgr_raw.ndim == 2:
        img_rgb = cv2.cvtColor(img_bgr_raw, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr_raw, cv2.COLOR_BGR2RGB)

    H, W = img_rgb.shape[:2]

    # 3. Load Mask (If provided in test set)
    # If mask is missing, we might want to skip or use full tissue scan
    if not os.path.exists(mask_path): 
        print(f"Mask missing for {img_id}, skipping.")
        continue
    
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_gray is None: continue

    # --- B. Analyze Mask (Initial Scan) ---
    points = cv2.findNonZero(mask_gray)
    if points is None: continue 

    bx, by, bw, bh = cv2.boundingRect(points)
    
    # --- C. Attempt Standard Extraction ---
    patches_generated_for_this_img = 0

    start_y = max(0, by - STRIDE)
    end_y = min(H - INPUT_SIZE, by + bh + STRIDE)
    start_x = max(0, bx - STRIDE)
    end_x = min(W - INPUT_SIZE, bx + bw + STRIDE)

    for y in range(start_y, end_y + 1, STRIDE):
        for x in range(start_x, end_x + 1, STRIDE):
            
            mask_patch = mask_gray[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
            
            # Check Overlap
            if not patch_overlaps_mask(mask_patch, thresh=MASK_OVERLAP_THRESH):
                continue

            # Check Tissue
            patch_rgb = img_rgb[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
            if not patch_has_tissue(patch_rgb, 
                                    frac_color_thresh=TISSUE_FRACTION_THRESH, 
                                    var_thresh=VARIANCE_THRESH):
                continue

            # SAVE PATCH
            save_patch_test(img_id, y, x, img_rgb, patch_records)
            patches_generated_for_this_img += 1

    # --- D. FALLBACK: Largest Blob Logic ---
    if patches_generated_for_this_img == 0:
        
        # 1. Morphological Dilation (Grouping)
        kernel = np.ones((15, 15), np.uint8) 
        dilated_mask = cv2.dilate(mask_gray, kernel, iterations=2)
        
        # 2. Connected Components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_mask, connectivity=8)

        if num_labels > 1:
            # Get areas (ignoring background at index 0)
            object_areas = stats[1:, cv2.CC_STAT_AREA]
            
            # Sort indices by area in descending order (Largest -> Smallest)
            sorted_indices = np.argsort(object_areas)[::-1]
            
            # Select the top N indices (or fewer if fewer blobs exist)
            top_n_indices = sorted_indices[:NUM_PATCHES]
            
            for i, idx in enumerate(top_n_indices):
                label_idx = idx + 1
                
                lx = stats[label_idx, cv2.CC_STAT_LEFT]
                ly = stats[label_idx, cv2.CC_STAT_TOP]
                lw = stats[label_idx, cv2.CC_STAT_WIDTH]
                lh = stats[label_idx, cv2.CC_STAT_HEIGHT]

                center_y = ly + (lh // 2)
                center_x = lx + (lw // 2)

                fallback_y = center_y - (INPUT_SIZE // 2)
                fallback_x = center_x - (INPUT_SIZE // 2)

                fallback_y = max(0, min(H - INPUT_SIZE, fallback_y))
                fallback_x = max(0, min(W - INPUT_SIZE, fallback_x))

                # SAVE PATCH
                save_patch_test(img_id, fallback_y, fallback_x, img_rgb, patch_records)

print(f"Total kept test patches: {len(patch_records)}")

# -----------------------------------------------------------------------------
# 3. Save CSV
# -----------------------------------------------------------------------------
df_patches = pd.DataFrame(patch_records, columns=["patch_filename", "image_id"])
df_patches.to_csv(TEST_PATCH_CSV, index=False)
print(f"Saved test patch labels to {TEST_PATCH_CSV}")

if not df_patches.empty:
    print(f"Unique images with patches: {df_patches['image_id'].nunique()}")
else:
    print("No patches generated.")