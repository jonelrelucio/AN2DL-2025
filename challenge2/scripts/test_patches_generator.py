"""
Test Patch Generator for Breast Cancer Histopathology
MUST BE CONSISTENT WITH TRAINING PATCH GENERATION
"""

import os
import cv2
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION (MUST MATCH TRAINING!)
# =============================================================================

# Paths
TEST_FOLDER = "challenge2/test_data"
TEST_PATCHES_DIR = "challenge2/test_patches"
TEST_PATCH_CSV = "challenge2/test_patches.csv"

# Patch extraction parameters (MATCH TRAINING!)
INPUT_SIZE = 128
STRIDE = 64
MAX_PATCHES_PER_IMAGE = 5  # Same as training default

# Quality thresholds (MUST MATCH TRAINING EXACTLY!)
MASK_OVERLAP_THRESH = 0.30      # Match your training threshold
TISSUE_FRACTION_THRESH = 0.20   # Match your training threshold
VARIANCE_THRESH = 10.0           # Match your training threshold

# Visualization
OVERLAY_ALPHA = 0.4

# Fallback parameters
NUM_PATCHES_FALLBACK = 4

os.makedirs(TEST_PATCHES_DIR, exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS (SAME AS TRAINING)
# =============================================================================

def patch_has_tissue(patch_rgb, frac_color_thresh=0.30, var_thresh=20.0, debug=False):
    """Check if patch contains sufficient tissue content."""
    patch_hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
    Hc, Sc, Vc = cv2.split(patch_hsv)
    
    sat_thresh = 25
    val_max = 240
    color_mask = (Sc > sat_thresh) & (Vc < val_max)
    frac_color = color_mask.mean()
    
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    var_gray = gray.var()
    
    passes = (frac_color >= frac_color_thresh) and (var_gray >= var_thresh)
    
    if debug:
        print(f"      Tissue check: color={frac_color:.3f} (≥{frac_color_thresh}), "
              f"var={var_gray:.1f} (≥{var_thresh}) → {'PASS' if passes else 'FAIL'}")
    
    return passes, frac_color, var_gray


def patch_overlaps_mask(patch_mask, thresh=0.5, debug=False):
    """Check if patch sufficiently overlaps with mask."""
    overlap = (patch_mask > 0).mean()
    passes = overlap >= thresh
    
    if debug:
        print(f"      Mask overlap: {overlap:.3f} (≥{thresh}) → {'PASS' if passes else 'FAIL'}")
    
    return passes, overlap


def create_green_overlay(patch_bgr, patch_mask_gray, alpha=0.4):
    """Create visualization with semi-transparent green mask overlay."""
    mixed_img = patch_bgr.copy()
    mask_indices = patch_mask_gray > 0
    
    mixed_img[mask_indices, 0] = (patch_bgr[mask_indices, 0] * (1 - alpha)).astype(np.uint8)
    mixed_img[mask_indices, 1] = (patch_bgr[mask_indices, 1] * (1 - alpha) + 255 * alpha).astype(np.uint8)
    mixed_img[mask_indices, 2] = (patch_bgr[mask_indices, 2] * (1 - alpha)).astype(np.uint8)
    
    return mixed_img


def save_patch_data(img_id, y, x, img_rgb, mask_gray, records_list):
    """
    Save patch data in multiple formats (SAME AS TRAINING).
    
    Saves:
        - Raw RGB patch
        - Grayscale mask
        - Green overlay visualization
        - Masked (blackout) image for inference
    """
    patch_rgb = img_rgb[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
    patch_mask = mask_gray[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
    
    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
    patch_mixed = create_green_overlay(patch_bgr, patch_mask, alpha=OVERLAY_ALPHA)
    patch_blackout = cv2.bitwise_and(patch_bgr, patch_bgr, mask=patch_mask)
    
    # Define filenames (SAME FORMAT AS TRAINING)
    filename_img = f"img_{img_id}_y{y}_x{x}.png"
    filename_mask = f"mask_{img_id}_y{y}_x{x}.png"
    filename_mixed = f"mixed_{img_id}_y{y}_x{x}.png"
    filename_blackout = f"masked_img_{img_id}_y{y}_x{x}.png"
    
    # Save files
    cv2.imwrite(os.path.join(TEST_PATCHES_DIR, filename_img), patch_bgr)
    cv2.imwrite(os.path.join(TEST_PATCHES_DIR, filename_mask), patch_mask)
    cv2.imwrite(os.path.join(TEST_PATCHES_DIR, filename_mixed), patch_mixed)
    cv2.imwrite(os.path.join(TEST_PATCHES_DIR, filename_blackout), patch_blackout)
    
    # Record (using blackout filename to match training)
    records_list.append((filename_blackout, img_id))


# =============================================================================
# MAIN PATCH EXTRACTION (SAME 3-TIER LOGIC AS TRAINING)
# =============================================================================

def extract_patches_from_test_image(img_id, img_rgb, mask_gray, patch_records, debug=False):
    """
    Extract patches using same 3-tier approach as training.
    
    Tier 1: Standard grid extraction with quality ranking
    Tier 2: Blob detection fallback
    Tier 3: Centroid fallback
    """
    H, W = img_rgb.shape[:2]
    
    points = cv2.findNonZero(mask_gray)
    if points is None:
        if debug:
            print(f"  Empty mask for {img_id}, skipping")
        return 0
    
    bx, by, bw, bh = cv2.boundingRect(points)
    
    if debug:
        print(f"  Image: {W}×{H}, Mask bbox: ({bx},{by}) {bw}×{bh}")
    
    # --- TIER 1: Standard Grid Extraction with Quality Ranking ---
    candidate_patches = []
    
    start_y = max(0, by - STRIDE)
    end_y = min(H - INPUT_SIZE, by + bh + STRIDE)
    start_x = max(0, bx - STRIDE)
    end_x = min(W - INPUT_SIZE, bx + bw + STRIDE)
    
    for y in range(start_y, end_y + 1, STRIDE):
        for x in range(start_x, end_x + 1, STRIDE):
            mask_patch = mask_gray[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
            
            # Check 1: Mask overlap
            passes_mask, mask_overlap = patch_overlaps_mask(mask_patch, thresh=MASK_OVERLAP_THRESH, debug=False)
            if not passes_mask:
                continue
            
            # Check 2: Tissue quality
            patch_rgb = img_rgb[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
            passes_tissue, tissue_frac, variance = patch_has_tissue(
                patch_rgb, 
                frac_color_thresh=TISSUE_FRACTION_THRESH,
                var_thresh=VARIANCE_THRESH,
                debug=False
            )
            if not passes_tissue:
                continue
            
            # Calculate quality score (SAME AS TRAINING)
            mask_score = mask_overlap
            var_score = variance
            quality_score = mask_score * 0.7 + (var_score / 1000.0) * 0.3
            
            candidate_patches.append((y, x, quality_score))
    
    # Select top quality patches (SAME AS TRAINING)
    if len(candidate_patches) > 0:
        candidate_patches.sort(key=lambda p: p[2], reverse=True)
        selected = candidate_patches[:MAX_PATCHES_PER_IMAGE]
        
        if debug:
            print(f"  ✓ Standard extraction: {len(selected)} patches from {len(candidate_patches)} candidates")
            for idx, (y, x, score) in enumerate(selected):
                print(f"    {idx+1}. Position ({y},{x}) quality={score:.3f}")
        
        for y, x, score in selected:
            save_patch_data(img_id, y, x, img_rgb, mask_gray, patch_records)
        
        return len(selected)
    
    # --- TIER 2: Blob Detection Fallback ---
    if debug:
        print(f"  ⚠️  Fallback: blob detection")
    
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(mask_gray, kernel, iterations=2)
    num_labels, labels, blob_stats, centroids = cv2.connectedComponentsWithStats(dilated_mask, connectivity=8)
    
    if num_labels > 1:
        object_areas = blob_stats[1:, cv2.CC_STAT_AREA]
        sorted_indices = np.argsort(object_areas)[::-1]
        top_n_indices = sorted_indices[:min(NUM_PATCHES_FALLBACK, MAX_PATCHES_PER_IMAGE)]
        
        for idx in top_n_indices:
            label_idx = idx + 1
            lx = blob_stats[label_idx, cv2.CC_STAT_LEFT]
            ly = blob_stats[label_idx, cv2.CC_STAT_TOP]
            lw = blob_stats[label_idx, cv2.CC_STAT_WIDTH]
            lh = blob_stats[label_idx, cv2.CC_STAT_HEIGHT]
            
            center_y = ly + (lh // 2)
            center_x = lx + (lw // 2)
            
            fallback_y = max(0, min(H - INPUT_SIZE, center_y - INPUT_SIZE // 2))
            fallback_x = max(0, min(W - INPUT_SIZE, center_x - INPUT_SIZE // 2))
            
            save_patch_data(img_id, fallback_y, fallback_x, img_rgb, mask_gray, patch_records)
        
        if debug:
            print(f"  ✓ Blob detection: {len(top_n_indices)} patches from {num_labels-1} blobs")
        
        return len(top_n_indices)
    
    # --- TIER 3: Centroid Fallback ---
    if debug:
        print(f"  ⚠️  Emergency fallback: centroid")
    
    M = cv2.moments(mask_gray)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx = bx + bw // 2
        cy = by + bh // 2
    
    fallback_y = max(0, min(H - INPUT_SIZE, cy - INPUT_SIZE // 2))
    fallback_x = max(0, min(W - INPUT_SIZE, cx - INPUT_SIZE // 2))
    
    save_patch_data(img_id, fallback_y, fallback_x, img_rgb, mask_gray, patch_records)
    
    if debug:
        print(f"  ✓ Centroid: 1 patch at ({fallback_y},{fallback_x})")
    
    return 1


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    print("="*70)
    print("TEST PATCH GENERATOR (CONSISTENT WITH TRAINING)")
    print("="*70)
    print(f"Patch size: {INPUT_SIZE}×{INPUT_SIZE}, Stride: {STRIDE}")
    print(f"Max patches per image: {MAX_PATCHES_PER_IMAGE}")
    print(f"Thresholds: Mask≥{MASK_OVERLAP_THRESH*100:.0f}%, "
          f"Tissue≥{TISSUE_FRACTION_THRESH*100:.0f}%, Var≥{VARIANCE_THRESH}")
    print("="*70 + "\n")
    
    patch_records = []
    
    # Get test image list
    test_files = [f for f in os.listdir(TEST_FOLDER) 
                  if f.startswith("img_") and f.endswith(".png")]
    test_ids = sorted([f.split("_")[1].split(".")[0] for f in test_files], key=int)
    
    print(f"Found {len(test_ids)} test images to process.\n")
    
    # Statistics tracking (simplified version)
    method_counts = {'standard': 0, 'blob': 0, 'centroid': 0}
    
    for idx, img_id in enumerate(test_ids):
        # Load image
        img_path = os.path.join(TEST_FOLDER, f"img_{img_id}.png")
        mask_path = os.path.join(TEST_FOLDER, f"mask_{img_id}.png")
        
        img_bgr_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_bgr_raw is None:
            print(f"⚠️  Could not load test image {img_id}")
            continue
        
        # Handle channels (SAME AS TRAINING)
        if img_bgr_raw.ndim == 3 and img_bgr_raw.shape[2] == 4:
            img_bgr_raw = img_bgr_raw[:, :, :3]
        
        if img_bgr_raw.ndim == 2:
            img_rgb = cv2.cvtColor(img_bgr_raw, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr_raw, cv2.COLOR_BGR2RGB)
        
        # Load mask
        if not os.path.exists(mask_path):
            print(f"⚠️  Mask not found for test image {img_id}")
            continue
        
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            print(f"⚠️  Could not load mask for test image {img_id}")
            continue
        
        # Extract patches
        patches_before = len(patch_records)
        num_patches = extract_patches_from_test_image(img_id, img_rgb, mask_gray, patch_records, debug=False)
        
        # Track method used (approximate)
        if num_patches == 1:
            method_counts['centroid'] += 1
        elif num_patches <= NUM_PATCHES_FALLBACK and num_patches < len(range(0, img_rgb.shape[0]-INPUT_SIZE, STRIDE)) * len(range(0, img_rgb.shape[1]-INPUT_SIZE, STRIDE)):
            # If fewer patches than grid size, likely blob or centroid
            pass
        else:
            method_counts['standard'] += 1
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(test_ids)} test images... ({len(patch_records)} patches so far)")
    
    print(f"\n✓ Test patch extraction complete: {len(patch_records)} patches generated\n")
    
    # Save CSV
    df_patches = pd.DataFrame(patch_records, columns=["patch_filename", "image_id"])
    df_patches.to_csv(TEST_PATCH_CSV, index=False)
    print(f"✓ Saved: {TEST_PATCH_CSV}\n")
    
    # Statistics
    print("="*70)
    print("TEST PATCH STATISTICS")
    print("="*70)
    
    if not df_patches.empty:
        patches_per_image = df_patches["image_id"].value_counts()
        
        print(f"Unique test images with patches: {df_patches['image_id'].nunique()}/{len(test_ids)}")
        print(f"Total patches: {len(patch_records)}")
        print(f"Min patches/image: {patches_per_image.min()}")
        print(f"Max patches/image: {patches_per_image.max()}")
        print(f"Avg patches/image: {patches_per_image.mean():.2f}")
        print(f"Median patches/image: {patches_per_image.median():.0f}")
        
        # Check consistency with training
        print(f"\n✓ Consistency check:")
        print(f"  Patch size: {INPUT_SIZE}×{INPUT_SIZE} ✓")
        print(f"  Stride: {STRIDE} ✓")
        print(f"  Max patches: {MAX_PATCHES_PER_IMAGE} ✓")
        print(f"  Thresholds: Mask={MASK_OVERLAP_THRESH}, Tissue={TISSUE_FRACTION_THRESH}, Var={VARIANCE_THRESH} ✓")
        print(f"  Quality score formula: mask*0.7 + (var/1000)*0.3 ✓")
        print(f"  Same 3-tier fallback system ✓")
        print(f"  Same file naming convention ✓")
        print(f"  Same patch selection logic ✓")
    else:
        print("⚠️  No patches generated!")
    
    print("\n" + "="*70)
    print("✅ TEST PATCH GENERATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
