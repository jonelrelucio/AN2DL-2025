import os
import cv2
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
INPUT_SIZE = 128            # Patch size
STRIDE = 64                 # Overlap
MASK_OVERLAP_THRESH = 0.70  # Strict: Patch must overlap the mask by at least 70%
# Tissue Detector Params
TISSUE_FRACTION_THRESH = 0.70 
VARIANCE_THRESH = 5.0

# Overlay settings
OVERLAY_ALPHA = 0.4         # Opacity of the green mask layer (0.0 - 1.0)

# --- PARAMETER ---
# How many distinct "highest distribution" areas to find
NUM_PATCHES = 4 
# -----------------

TRAIN_FOLDER = "challenge2/train_data"
TRAIN_LABEL_CSV = "challenge2/train_labels.csv"
WHITELIST_PATH = "challenge2/whitelist/clean_train_data.txt"
PATCHES_DIR = "challenge2/patches"
PATCH_LABEL_CSV = "challenge2/label_patches.csv"

os.makedirs(PATCHES_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 0. Load whitelist (if provided)
# -----------------------------------------------------------------------------
whitelist_ids = None
if os.path.exists(WHITELIST_PATH):
    with open(WHITELIST_PATH, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    whitelist_ids = set(
        name.split("_")[1].split(".")[0]
        for name in names
        if name.startswith("img_")
    )
    print(f"Loaded whitelist with {len(whitelist_ids)} image IDs.")
else:
    print("No whitelist found; using all images.")

# -----------------------------------------------------------------------------
# 1. Load image-level labels
# -----------------------------------------------------------------------------
df_labels = pd.read_csv(TRAIN_LABEL_CSV)
id_col = df_labels.columns[0]
label_col = df_labels.columns[1]

df_labels["clean_id"] = df_labels[id_col].astype(str).apply(
    lambda x: x.split("_")[1].split(".")[0] if "_" in x else x
)
id_to_label = dict(zip(df_labels["clean_id"], df_labels[label_col]))

# -----------------------------------------------------------------------------
# 2. Helpers
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

def create_green_overlay(patch_bgr, patch_mask_gray, alpha=0.4):
    """
    Creates an image with a semi-transparent green overlay based on the mask.
    patch_bgr: BGR image (H, W, 3)
    patch_mask_gray: Grayscale mask (H, W)
    alpha: transparency (0.0-1.0)
    """
    # Create a copy to draw the overlay on
    mixed_img = patch_bgr.copy()

    # Find boolean indices where the mask is present (pixel value > 0)
    mask_indices = patch_mask_gray > 0

    # Apply manual alpha blending using numpy vectorization
    # Formula: final = original * (1-alpha) + overlay_color * alpha
    
    # B channel (index 0): Blend original B with 0 (no blue in green layer)
    mixed_img[mask_indices, 0] = (patch_bgr[mask_indices, 0] * (1 - alpha)).astype(np.uint8)
    
    # G channel (index 1): Blend original G with 255 (full green layer)
    mixed_img[mask_indices, 1] = (patch_bgr[mask_indices, 1] * (1 - alpha) + 255 * alpha).astype(np.uint8)
    
    # R channel (index 2): Blend original R with 0 (no red in green layer)
    mixed_img[mask_indices, 2] = (patch_bgr[mask_indices, 2] * (1 - alpha)).astype(np.uint8)
    
    return mixed_img

def save_patch_triplet(img_id, y, x, img_rgb, mask_gray, label, records_list):
    """
    Saves RGB patch, Grayscale Mask patch, and Mixed overlay patch.
    """
    # 1. Crop patches
    patch_rgb = img_rgb[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
    patch_mask = mask_gray[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
    
    # Convert RGB patch to BGR for OpenCV saving and blending
    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)

    # 2. Create Mixed Overlay Image
    patch_mixed = create_green_overlay(patch_bgr, patch_mask, alpha=OVERLAY_ALPHA)

    # 3. Define Filenames
    filename_img = f"img_{img_id}_y{y}_x{x}.png"
    filename_mask = f"mask_{img_id}_y{y}_x{x}.png"
    filename_mixed = f"mixed_img_{img_id}_y{y}_x{x}.png"
    
    # 4. Define Paths
    path_img = os.path.join(PATCHES_DIR, filename_img)
    path_mask = os.path.join(PATCHES_DIR, filename_mask)
    path_mixed = os.path.join(PATCHES_DIR, filename_mixed)
    
    # 5. Save Files (using BGR versions for cv2.imwrite)
    cv2.imwrite(path_img, patch_bgr)
    cv2.imwrite(path_mask, patch_mask)
    cv2.imwrite(path_mixed, patch_mixed)
    
    # 6. Record
    records_list.append((filename_img, label))

# -----------------------------------------------------------------------------
# 3. Main Loop
# -----------------------------------------------------------------------------
patch_records = []

train_files = [f for f in os.listdir(TRAIN_FOLDER) 
               if f.startswith("img_") and f.endswith(".png")]
train_ids_all = [f.split("_")[1].split(".")[0] for f in train_files]

if whitelist_ids is not None:
    train_ids = sorted([i for i in train_ids_all if i in whitelist_ids], key=int)
else:
    train_ids = sorted(train_ids_all, key=int)

print(f"Found {len(train_ids)} training images to process.")

for img_id in train_ids:
    # --- A. Load Data (High Precision) ---
    img_path = os.path.join(TRAIN_FOLDER, f"img_{img_id}.png")
    mask_path = os.path.join(TRAIN_FOLDER, f"mask_{img_id}.png")

    # 1. Load Image UNCHANGED
    # This preserves 16-bit depth (0-65535) if the image has it.
    img_bgr_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    if img_bgr_raw is None: continue

    # 2. Handle Channels (Sanitize)
    # If image is RGBA (4 channels), drop the Alpha channel to get standard BGR
    if img_bgr_raw.ndim == 3 and img_bgr_raw.shape[2] == 4:
        img_bgr_raw = img_bgr_raw[:, :, :3]
    
    # If image is Grayscale (2 dims), convert to 3-channel RGB
    if img_bgr_raw.ndim == 2:
        img_rgb = cv2.cvtColor(img_bgr_raw, cv2.COLOR_GRAY2RGB)
    else:
        # Convert BGR to RGB
        # (OpenCV automatically handles uint16 here without losing precision)
        img_rgb = cv2.cvtColor(img_bgr_raw, cv2.COLOR_BGR2RGB)

    H, W = img_rgb.shape[:2]

    # 3. Load Mask
    if not os.path.exists(mask_path): continue
    
    # Load mask as grayscale (standard for binary masks)
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_gray is None: continue

    label = id_to_label.get(img_id, None)
    if label is None: continue

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

            # SAVE TRIPLET (Img, Mask, Mixed)
            save_patch_triplet(img_id, y, x, img_rgb, mask_gray, label, patch_records)
            patches_generated_for_this_img += 1

    # --- D. FALLBACK: Largest Blob Logic ---
    if patches_generated_for_this_img == 0:
        


        # 1. Morphological Dilation (Grouping)
        # 
        kernel = np.ones((15, 15), np.uint8) 
        dilated_mask = cv2.dilate(mask_gray, kernel, iterations=2)
        
        # 2. Connected Components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_mask, connectivity=8)

        if num_labels > 1:
            # Get areas (ignoring background at index 0)
            # stats shape is [num_labels, 5]
            # We slice [1:] to skip the background
            object_areas = stats[1:, cv2.CC_STAT_AREA]
            
            # Sort indices by area in descending order (Largest -> Smallest)
            sorted_indices = np.argsort(object_areas)[::-1]
            
            # Select the top N indices (or fewer if fewer blobs exist)
            top_n_indices = sorted_indices[:NUM_PATCHES]
            
            for i, idx in enumerate(top_n_indices):
                # Map the index back to the original stats array
                # (idx 0 in object_areas corresponds to label 1 in stats)
                label_idx = idx + 1
                
                # Retrieve bounding box of this specific blob
                lx = stats[label_idx, cv2.CC_STAT_LEFT]
                ly = stats[label_idx, cv2.CC_STAT_TOP]
                lw = stats[label_idx, cv2.CC_STAT_WIDTH]
                lh = stats[label_idx, cv2.CC_STAT_HEIGHT]

                # Center patch on blob
                center_y = ly + (lh // 2)
                center_x = lx + (lw // 2)

                fallback_y = center_y - (INPUT_SIZE // 2)
                fallback_x = center_x - (INPUT_SIZE // 2)

                # Clamp to image boundaries
                fallback_y = max(0, min(H - INPUT_SIZE, fallback_y))
                fallback_x = max(0, min(W - INPUT_SIZE, fallback_x))

                # SAVE TRIPLET (for each of the top N distributions)
                # You might want to pass 'i' or a unique ID if your saver requires it
                save_patch_triplet(img_id, fallback_y, fallback_x, img_rgb, mask_gray, label, patch_records)

print(f"Total kept patches: {len(patch_records)}")

# -----------------------------------------------------------------------------
# 4. Save CSV and Print Distribution
# -----------------------------------------------------------------------------
df_patches = pd.DataFrame(patch_records, columns=["patch_filename", "label"])
df_patches.to_csv(PATCH_LABEL_CSV, index=False)
print(f"Saved patch labels to {PATCH_LABEL_CSV}")

print("\n" + "="*40)
print("PATCH LABEL DISTRIBUTION")
print("="*40)

if not df_patches.empty:
    print("Counts:")
    print(df_patches["label"].value_counts())
    print("\nPercentages:")
    print((df_patches["label"].value_counts(normalize=True) * 100).map('{:.2f}%'.format))
else:
    print("No patches generated.")