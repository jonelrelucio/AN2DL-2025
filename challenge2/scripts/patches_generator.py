"""
Breast Cancer Histopathology Patch Generator
Extracts patches from images with quality filtering and multi-tier fallback system.
WITH CLASS BALANCING
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from collections import defaultdict, Counter

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
TRAIN_FOLDER = "challenge2/train_data"
TRAIN_LABEL_CSV = "challenge2/train_labels.csv"
WHITELIST_PATH = "challenge2/whitelist/clean_train_data.txt"
PATCHES_DIR = "challenge2/patches"
PATCH_LABEL_CSV = "challenge2/label_patches.csv"

# Patch extraction parameters
INPUT_SIZE = 128
STRIDE = 64
MAX_PATCHES_PER_IMAGE = 5  # Default max (will be adjusted per class for balancing)
MIN_PATCHES_PER_IMAGE = 1

# Quality thresholds
MASK_OVERLAP_THRESH = 0.30
TISSUE_FRACTION_THRESH = 0.20
VARIANCE_THRESH = 10.0

# Visualization
OVERLAY_ALPHA = 0.4

# Fallback parameters
NUM_PATCHES_FALLBACK = 4

# Debug mode
DEBUG_MODE = True
DEBUG_IMAGE_ID = "0002"
SAVE_DEBUG_VISUALIZATIONS = True

# CLASS BALANCING PARAMETERS
BALANCE_CLASSES = True  # Enable/disable class balancing
BALANCE_STRATEGY = "hybrid"  # "oversample", "undersample", or "hybrid"
REALISTIC_BALANCING = True  # Adjust targets based on achievable patches
TARGET_PATCHES_PER_CLASS = None  # Auto-calculate if None

# =============================================================================
# GLOBAL STATISTICS TRACKER
# =============================================================================

class ExtractionStats:
    """Track detailed statistics during patch extraction."""
    
    def __init__(self):
        self.total_positions_checked = 0
        self.mask_failures = 0
        self.tissue_failures = 0
        self.candidates_found = 0
        self.patches_saved = 0
        self.method_counts = {'standard': 0, 'blob': 0, 'centroid': 0}
        self.per_class_stats = defaultdict(lambda: {'images': 0, 'patches': 0})
        self.quality_scores = []
        self.mask_overlaps = []
        self.tissue_fractions = []
        self.variances = []
        self.processing_times = []
        self.patches_per_image_list = []
        
        # Edge case tracking
        self.barely_passed = []
        self.barely_failed = []
        
        # Class balancing tracking
        self.class_targets = {}
        self.class_max_patches = {}
        
    def add_candidate(self, img_id, y, x, mask_overlap, tissue_frac, variance, quality_score):
        """Record a candidate patch."""
        self.candidates_found += 1
        self.quality_scores.append(quality_score)
        self.mask_overlaps.append(mask_overlap)
        self.tissue_fractions.append(tissue_frac)
        self.variances.append(variance)
        
        if (MASK_OVERLAP_THRESH <= mask_overlap < MASK_OVERLAP_THRESH + 0.05 or
            TISSUE_FRACTION_THRESH <= tissue_frac < TISSUE_FRACTION_THRESH + 0.05 or
            VARIANCE_THRESH <= variance < VARIANCE_THRESH + 5):
            self.barely_passed.append({
                'img_id': img_id, 'y': y, 'x': x,
                'mask_overlap': mask_overlap,
                'tissue_frac': tissue_frac,
                'variance': variance,
                'quality': quality_score
            })
    
    def add_rejection(self, img_id, y, x, reason, mask_overlap=None, tissue_frac=None, variance=None):
        """Record a rejected patch."""
        if reason == 'mask':
            self.mask_failures += 1
            if mask_overlap and MASK_OVERLAP_THRESH - 0.05 <= mask_overlap < MASK_OVERLAP_THRESH:
                self.barely_failed.append({
                    'img_id': img_id, 'y': y, 'x': x,
                    'reason': 'mask',
                    'mask_overlap': mask_overlap
                })
        elif reason == 'tissue':
            self.tissue_failures += 1
            if tissue_frac and variance:
                if (TISSUE_FRACTION_THRESH - 0.05 <= tissue_frac < TISSUE_FRACTION_THRESH or
                    VARIANCE_THRESH - 5 <= variance < VARIANCE_THRESH):
                    self.barely_failed.append({
                        'img_id': img_id, 'y': y, 'x': x,
                        'reason': 'tissue',
                        'tissue_frac': tissue_frac,
                        'variance': variance
                    })
    
    def print_summary(self):
        """Print comprehensive statistics."""
        print("\n" + "="*70)
        print("DETAILED EXTRACTION STATISTICS")
        print("="*70)
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"   Total positions checked: {self.total_positions_checked:,}")
        print(f"   Candidates found: {self.candidates_found:,} ({self.candidates_found/max(self.total_positions_checked,1)*100:.2f}%)")
        print(f"   Patches saved: {self.patches_saved:,}")
        print(f"   Rejection rate: {(self.mask_failures + self.tissue_failures)/max(self.total_positions_checked,1)*100:.1f}%")
        
        print(f"\n🚫 REJECTION BREAKDOWN:")
        print(f"   Mask overlap failures: {self.mask_failures:,} ({self.mask_failures/max(self.total_positions_checked,1)*100:.1f}%)")
        print(f"   Tissue quality failures: {self.tissue_failures:,} ({self.tissue_failures/max(self.total_positions_checked,1)*100:.1f}%)")
        
        print(f"\n🔧 EXTRACTION METHODS:")
        for method, count in self.method_counts.items():
            print(f"   {method.capitalize()}: {count} images")
        
        if self.quality_scores:
            print(f"\n📈 QUALITY METRICS:")
            print(f"   Quality score: min={min(self.quality_scores):.3f}, "
                  f"max={max(self.quality_scores):.3f}, "
                  f"mean={np.mean(self.quality_scores):.3f}, "
                  f"median={np.median(self.quality_scores):.3f}")
            
            print(f"   Mask overlap: min={min(self.mask_overlaps):.3f}, "
                  f"max={max(self.mask_overlaps):.3f}, "
                  f"mean={np.mean(self.mask_overlaps):.3f}")
            
            print(f"   Tissue fraction: min={min(self.tissue_fractions):.3f}, "
                  f"max={max(self.tissue_fractions):.3f}, "
                  f"mean={np.mean(self.tissue_fractions):.3f}")
            
            print(f"   Variance: min={min(self.variances):.1f}, "
                  f"max={max(self.variances):.1f}, "
                  f"mean={np.mean(self.variances):.1f}")
        
        if self.processing_times:
            print(f"\n⏱️  PERFORMANCE:")
            print(f"   Avg time per image: {np.mean(self.processing_times):.3f}s")
            print(f"   Total processing time: {sum(self.processing_times):.1f}s")
        
        if self.patches_per_image_list:
            print(f"\n📦 PATCHES PER IMAGE:")
            print(f"   Min: {min(self.patches_per_image_list)}")
            print(f"   Max: {max(self.patches_per_image_list)}")
            print(f"   Mean: {np.mean(self.patches_per_image_list):.2f}")
            print(f"   Median: {np.median(self.patches_per_image_list):.0f}")
        
        print(f"\n🎯 EDGE CASES:")
        print(f"   Barely passed: {len(self.barely_passed)} patches")
        print(f"   Barely failed: {len(self.barely_failed)} patches")
        
        if self.per_class_stats:
            print(f"\n📋 PER-CLASS STATISTICS:")
            for cls, stats in sorted(self.per_class_stats.items()):
                avg_patches = stats['patches'] / max(stats['images'], 1)
                target = self.class_targets.get(cls, 'N/A')
                max_patches = self.class_max_patches.get(cls, 'N/A')
                print(f"   {cls}:")
                print(f"      Images: {stats['images']}")
                print(f"      Patches: {stats['patches']} (target: {target})")
                print(f"      Avg patches/image: {avg_patches:.2f} (max: {max_patches})")

# Global stats object
stats = ExtractionStats()

# =============================================================================
# HELPER FUNCTIONS
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

def save_patch_data(img_id, y, x, img_rgb, mask_gray, label, records_list):
    """Save patch data in multiple formats."""
    patch_rgb = img_rgb[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
    patch_mask = mask_gray[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
    
    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
    patch_mixed = create_green_overlay(patch_bgr, patch_mask, alpha=OVERLAY_ALPHA)
    patch_blackout = cv2.bitwise_and(patch_bgr, patch_bgr, mask=patch_mask)
    
    filename_img = f"img_{img_id}_y{y}_x{x}.png"
    filename_mask = f"mask_{img_id}_y{y}_x{x}.png"
    filename_mixed = f"mixed_{img_id}_y{y}_x{x}.png"
    filename_blackout = f"masked_img_{img_id}_y{y}_x{x}.png"
    
    cv2.imwrite(os.path.join(PATCHES_DIR, filename_img), patch_bgr)
    cv2.imwrite(os.path.join(PATCHES_DIR, filename_mask), patch_mask)
    cv2.imwrite(os.path.join(PATCHES_DIR, filename_mixed), patch_mixed)
    cv2.imwrite(os.path.join(PATCHES_DIR, filename_blackout), patch_blackout)
    
    records_list.append((filename_blackout, label))
    stats.patches_saved += 1

def load_whitelist(whitelist_path):
    """Load image IDs from whitelist file."""
    if not os.path.exists(whitelist_path):
        return None
    
    with open(whitelist_path, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    
    whitelist_ids = set(
        name.split("_")[1].split(".")[0]
        for name in names
        if name.startswith("img_")
    )
    
    return whitelist_ids

def load_labels(csv_path):
    """Load image-level labels from CSV."""
    df = pd.read_csv(csv_path)
    id_col = df.columns[0]
    label_col = df.columns[1]
    
    df["clean_id"] = df[id_col].astype(str).apply(
        lambda x: x.split("_")[1].split(".")[0] if "_" in x else x
    )
    
    return dict(zip(df["clean_id"], df[label_col]))


def estimate_achievable_patches(train_ids, id_to_label):
    """
    Do a quick pre-scan to estimate how many patches each class can realistically generate.
    """
    print("\n🔍 Pre-scanning to estimate achievable patches...")
    
    class_potential = defaultdict(list)
    
    for img_id in train_ids[:50]:  # Sample first 50 images
        if img_id not in id_to_label:
            continue
            
        label = id_to_label[img_id]
        img_path = os.path.join(TRAIN_FOLDER, f"img_{img_id}.png")
        mask_path = os.path.join(TRAIN_FOLDER, f"mask_{img_id}.png")
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask_gray is None:
            continue
        
        # Quick candidate count
        points = cv2.findNonZero(mask_gray)
        if points is None:
            class_potential[label].append(1)
            continue
        
        bx, by, bw, bh = cv2.boundingRect(points)
        H, W = img_rgb.shape[:2]
        
        start_y = max(0, by - STRIDE)
        end_y = min(H - INPUT_SIZE, by + bh + STRIDE)
        start_x = max(0, bx - STRIDE)
        end_x = min(W - INPUT_SIZE, bx + bw + STRIDE)
        
        candidates = 0
        for y in range(start_y, end_y + 1, STRIDE):
            for x in range(start_x, end_x + 1, STRIDE):
                mask_patch = mask_gray[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
                if (mask_patch > 0).mean() >= MASK_OVERLAP_THRESH:
                    patch_rgb = img_rgb[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
                    passes, _, _ = patch_has_tissue(patch_rgb, TISSUE_FRACTION_THRESH, VARIANCE_THRESH)
                    if passes:
                        candidates += 1
        
        class_potential[label].append(min(candidates, 10))
    
    # Estimate for all images
    class_estimates = {}
    class_counts = Counter([id_to_label[img_id] for img_id in train_ids if img_id in id_to_label])
    
    for cls in class_counts:
        if cls in class_potential and class_potential[cls]:
            avg_per_img = np.mean(class_potential[cls])
            estimated_total = int(class_counts[cls] * avg_per_img)
            class_estimates[cls] = estimated_total
            print(f"   {cls}: ~{avg_per_img:.1f} patches/img → ~{estimated_total} total achievable")
        else:
            class_estimates[cls] = class_counts[cls] * 3  # Conservative estimate
    
    return class_estimates

def calculate_class_balancing_params(train_ids, id_to_label):
    """Calculate balanced patch extraction parameters."""
    class_counts = Counter([id_to_label[img_id] for img_id in train_ids if img_id in id_to_label])
    
    print("\n" + "="*70)
    print("CLASS BALANCING ANALYSIS")
    print("="*70)
    print(f"\nImage distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"   {cls}: {count} images")
    
    # Get realistic estimates
    if REALISTIC_BALANCING:
        achievable_patches = estimate_achievable_patches(train_ids, id_to_label)
        # Target: maximum achievable from minority class
        target_patches_global = max(achievable_patches.values())
    else:
        # Original logic
        if BALANCE_STRATEGY == "oversample":
            max_class_count = max(class_counts.values())
            target_patches_global = TARGET_PATCHES_PER_CLASS or (max_class_count * MAX_PATCHES_PER_IMAGE)
        elif BALANCE_STRATEGY == "undersample":
            min_class_count = min(class_counts.values())
            target_patches_global = TARGET_PATCHES_PER_CLASS or (min_class_count * MAX_PATCHES_PER_IMAGE)
        else:
            avg_class_count = sum(class_counts.values()) / len(class_counts)
            target_patches_global = TARGET_PATCHES_PER_CLASS or int(avg_class_count * MAX_PATCHES_PER_IMAGE)
    
    class_max_patches = {}
    for cls, img_count in class_counts.items():
        patches_per_img = target_patches_global / img_count
        patches_per_img = max(MIN_PATCHES_PER_IMAGE, min(10, int(np.ceil(patches_per_img))))
        class_max_patches[cls] = patches_per_img
    
    print(f"\nBalancing strategy: {BALANCE_STRATEGY.upper()}")
    print(f"Target patches per class: {target_patches_global}")
    print(f"\nMax patches per image (by class):")
    for cls, max_patches in sorted(class_max_patches.items()):
        expected_total = class_counts[cls] * max_patches
        achievable = achievable_patches.get(cls, '?') if REALISTIC_BALANCING else '?'
        print(f"   {cls}: {max_patches} patches/image → ~{expected_total} target (~{achievable} achievable)")
    
    stats.class_targets = {cls: target_patches_global for cls in class_counts}
    stats.class_max_patches = class_max_patches
    
    return class_max_patches


# =============================================================================
# VISUALIZATION FUNCTIONS (keeping existing ones)
# =============================================================================

def visualize_saved_patches(img_id, img_rgb, mask_gray, saved_patches, save_path=None):
    """Display all saved patches from one image in a grid."""
    if not saved_patches:
        return
    
    n_patches = len(saved_patches)
    cols = min(5, n_patches)
    rows = (n_patches + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    if n_patches == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (y, x) in enumerate(saved_patches):
        patch = img_rgb[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
        patch_mask = mask_gray[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
        
        mask_overlap = (patch_mask > 0).mean()
        _, tissue_frac, variance = patch_has_tissue(patch, TISSUE_FRACTION_THRESH, VARIANCE_THRESH)
        
        axes[idx].imshow(patch)
        axes[idx].set_title(f"Patch {idx+1}\n({y},{x})\nMask:{mask_overlap:.2f} Var:{variance:.0f}",
                           fontsize=10)
        axes[idx].axis('off')
    
    for idx in range(n_patches, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"All Saved Patches from Image {img_id}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved patches visualization: {save_path}")
    
    plt.show()

def visualize_quality_distributions(save_path=None):
    """Visualize distributions of quality metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(stats.quality_scores, bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(stats.quality_scores), color='red', linestyle='--', label=f'Mean: {np.mean(stats.quality_scores):.3f}')
    axes[0, 0].set_title('Quality Score Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Quality Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(stats.mask_overlaps, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(MASK_OVERLAP_THRESH, color='red', linestyle='--', label=f'Threshold: {MASK_OVERLAP_THRESH}')
    axes[0, 1].axvline(np.mean(stats.mask_overlaps), color='orange', linestyle='--', label=f'Mean: {np.mean(stats.mask_overlaps):.3f}')
    axes[0, 1].set_title('Mask Overlap Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Mask Overlap Fraction')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(stats.tissue_fractions, bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(TISSUE_FRACTION_THRESH, color='red', linestyle='--', label=f'Threshold: {TISSUE_FRACTION_THRESH}')
    axes[1, 0].axvline(np.mean(stats.tissue_fractions), color='orange', linestyle='--', label=f'Mean: {np.mean(stats.tissue_fractions):.3f}')
    axes[1, 0].set_title('Tissue Fraction Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Tissue Fraction')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(stats.variances, bins=30, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(VARIANCE_THRESH, color='red', linestyle='--', label=f'Threshold: {VARIANCE_THRESH}')
    axes[1, 1].axvline(np.mean(stats.variances), color='blue', linestyle='--', label=f'Mean: {np.mean(stats.variances):.1f}')
    axes[1, 1].set_title('Texture Variance Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Variance')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Quality Metrics Distributions for All Candidate Patches', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Quality distributions saved: {save_path}")
    
    plt.show()

def visualize_edge_cases(img_rgb_dict, mask_gray_dict, save_path=None):
    """Visualize patches that barely passed or barely failed thresholds."""
    if not stats.barely_passed and not stats.barely_failed:
        print("No edge cases to visualize")
        return
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for idx in range(5):
        if idx < len(stats.barely_passed):
            case = stats.barely_passed[idx]
            img_id = case['img_id']
            if img_id in img_rgb_dict:
                img_rgb = img_rgb_dict[img_id]
                y, x = case['y'], case['x']
                patch = img_rgb[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
                
                axes[0, idx].imshow(patch)
                axes[0, idx].set_title(f"✅ PASSED\nImg {img_id}\nMask:{case['mask_overlap']:.2f}\nVar:{case['variance']:.0f}",
                                      fontsize=9, color='green', fontweight='bold')
                axes[0, idx].axis('off')
        else:
            axes[0, idx].axis('off')
    
    for idx in range(5):
        if idx < len(stats.barely_failed):
            case = stats.barely_failed[idx]
            img_id = case['img_id']
            if img_id in img_rgb_dict:
                img_rgb = img_rgb_dict[img_id]
                y, x = case['y'], case['x']
                patch = img_rgb[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
                
                axes[1, idx].imshow(patch)
                reason = case['reason']
                if reason == 'mask':
                    title = f"❌ FAILED (Mask)\nImg {img_id}\nMask:{case['mask_overlap']:.2f}"
                else:
                    title = f"❌ FAILED (Tissue)\nImg {img_id}\nVar:{case.get('variance', 0):.0f}"
                axes[1, idx].set_title(title, fontsize=9, color='red', fontweight='bold')
                axes[1, idx].axis('off')
        else:
            axes[1, idx].axis('off')
    
    plt.suptitle('Edge Cases: Barely Passed (Top) vs Barely Failed (Bottom)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Edge cases visualization saved: {save_path}")
    
    plt.show()

def visualize_mask_density(img_id, img_rgb, mask_gray, saved_patches=None, save_path=None):
    """Visualize mask density and show which patches were actually saved."""
    H, W = mask_gray.shape
    
    points = cv2.findNonZero(mask_gray)
    if points is None:
        print(f"Empty mask for {img_id}")
        return
    
    bx, by, bw, bh = cv2.boundingRect(points)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title(f"Image {img_id}", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_gray, cmap='gray')
    rect = Rectangle((bx, by), bw, bh, linewidth=2, edgecolor='red', facecolor='none')
    axes[0, 1].add_patch(rect)
    axes[0, 1].set_title("Mask (Red = Bounding Box)", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    grid_size = 32
    density_h = H // grid_size + 1
    density_w = W // grid_size + 1
    density_map = np.zeros((density_h, density_w))
    
    for i in range(density_h):
        for j in range(density_w):
            y_start = i * grid_size
            y_end = min((i + 1) * grid_size, H)
            x_start = j * grid_size
            x_end = min((j + 1) * grid_size, W)
            
            cell = mask_gray[y_start:y_end, x_start:x_end]
            density_map[i, j] = (cell > 0).mean()
    
    im = axes[1, 0].imshow(density_map, cmap='hot', interpolation='nearest', aspect='auto')
    axes[1, 0].set_title("Mask Density Heatmap", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[1, 0], label='Tissue Density')
    axes[1, 0].set_xlabel('X (32-pixel blocks)')
    axes[1, 0].set_ylabel('Y (32-pixel blocks)')
    
    img_with_grid = img_rgb.copy()
    
    start_y = max(0, by - STRIDE)
    end_y = min(H - INPUT_SIZE, by + bh + STRIDE)
    start_x = max(0, bx - STRIDE)
    end_x = min(W - INPUT_SIZE, bx + bw + STRIDE)
    
    cv2.rectangle(img_with_grid, (bx, by), (bx + bw, by + bh), (255, 0, 0), 3)
    
    grid_count = 0
    for idx_y, y in enumerate(range(start_y, end_y + 1, STRIDE)):
        for idx_x, x in enumerate(range(start_x, end_x + 1, STRIDE)):
            if (idx_y + idx_x) % 2 == 0:
                cv2.rectangle(img_with_grid, (x, y), (x + INPUT_SIZE, y + INPUT_SIZE),
                            (0, 255, 255), 1)
            grid_count += 1
    
    num_saved = 0
    if saved_patches:
        for y, x in saved_patches:
            cv2.rectangle(img_with_grid, (x, y), (x + INPUT_SIZE, y + INPUT_SIZE),
                        (0, 255, 0), 4)
            num_saved += 1
    
    axes[1, 1].imshow(img_with_grid)
    axes[1, 1].set_title(
        f"Scanning Grid\nRed=BBox | Cyan=Checked (n={grid_count}) | Green=Saved (n={num_saved})",
        fontsize=14, fontweight='bold'
    )
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Mask density visualization saved: {save_path}")
    
    plt.show()
    
    mask_pixels = np.sum(mask_gray > 0)
    bbox_area = bw * bh
    mask_density = mask_pixels / bbox_area if bbox_area > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"MASK ANALYSIS: Image {img_id}")
    print(f"{'='*50}")
    print(f"Image size: {W} × {H}")
    print(f"Bounding box: ({bx}, {by}) → {bw} × {bh}")
    print(f"Bbox area: {bbox_area:,} pixels")
    print(f"Actual mask: {mask_pixels:,} pixels")
    print(f"Density: {mask_density*100:.1f}%")
    print(f"Grid positions checked: {grid_count}")
    print(f"Patches saved: {num_saved}")
    print(f"{'='*50}\n")

# =============================================================================
# MAIN PATCH EXTRACTION LOGIC
# =============================================================================

def extract_patches_from_image(img_id, img_rgb, mask_gray, label, patch_records, max_patches_for_class, debug=False):
    """Extract patches from a single image using multi-tier approach."""
    start_time = time.time()
    H, W = img_rgb.shape[:2]
    
    points = cv2.findNonZero(mask_gray)
    if points is None:
        if debug:
            print("  Empty mask, skipping")
        return 0
    
    bx, by, bw, bh = cv2.boundingRect(points)
    
    if debug:
        print(f"  Image: {W}×{H}, Mask bbox: ({bx},{by}) {bw}×{bh}")
        print(f"  Max patches for this class: {max_patches_for_class}")
    
    # --- TIER 1: Standard Grid Extraction ---
    candidate_patches = []
    
    start_y = max(0, by - STRIDE)
    end_y = min(H - INPUT_SIZE, by + bh + STRIDE)
    start_x = max(0, bx - STRIDE)
    end_x = min(W - INPUT_SIZE, bx + bw + STRIDE)
    
    num_positions = len(range(start_y, end_y + 1, STRIDE)) * len(range(start_x, end_x + 1, STRIDE))
    stats.total_positions_checked += num_positions
    
    if debug:
        print(f"  Scanning {num_positions} grid positions...")
    
    for y in range(start_y, end_y + 1, STRIDE):
        for x in range(start_x, end_x + 1, STRIDE):
            mask_patch = mask_gray[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
            
            passes_mask, mask_overlap = patch_overlaps_mask(mask_patch, thresh=MASK_OVERLAP_THRESH)
            if not passes_mask:
                stats.add_rejection(img_id, y, x, 'mask', mask_overlap=mask_overlap)
                continue
            
            patch_rgb = img_rgb[y:y+INPUT_SIZE, x:x+INPUT_SIZE]
            passes_tissue, tissue_frac, variance = patch_has_tissue(
                patch_rgb, frac_color_thresh=TISSUE_FRACTION_THRESH, var_thresh=VARIANCE_THRESH
            )
            if not passes_tissue:
                stats.add_rejection(img_id, y, x, 'tissue', tissue_frac=tissue_frac, variance=variance)
                continue
            
            mask_score = mask_overlap
            var_score = variance
            quality_score = mask_score * 0.7 + (var_score / 1000.0) * 0.3
            
            stats.add_candidate(img_id, y, x, mask_overlap, tissue_frac, variance, quality_score)
            candidate_patches.append((y, x, quality_score))
    
    # Select top quality patches (using class-specific max)
    if len(candidate_patches) > 0:
        candidate_patches.sort(key=lambda p: p[2], reverse=True)
        selected = candidate_patches[:max_patches_for_class]
        
        if debug:
            print(f"  ✓ Standard extraction: {len(selected)} patches from {len(candidate_patches)} candidates")
            for idx, (y, x, score) in enumerate(selected):
                print(f"    {idx+1}. Position ({y},{x}) quality={score:.3f}")
        
        for y, x, score in selected:
            save_patch_data(img_id, y, x, img_rgb, mask_gray, label, patch_records)
        
        stats.method_counts['standard'] += 1
        elapsed = time.time() - start_time
        stats.processing_times.append(elapsed)
        stats.patches_per_image_list.append(len(selected))
        stats.per_class_stats[label]['images'] += 1
        stats.per_class_stats[label]['patches'] += len(selected)
        
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
        top_n_indices = sorted_indices[:min(NUM_PATCHES_FALLBACK, max_patches_for_class)]
        
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
            
            save_patch_data(img_id, fallback_y, fallback_x, img_rgb, mask_gray, label, patch_records)
        
        if debug:
            print(f"  ✓ Blob detection: {len(top_n_indices)} patches from {num_labels-1} blobs")
        
        stats.method_counts['blob'] += 1
        elapsed = time.time() - start_time
        stats.processing_times.append(elapsed)
        stats.patches_per_image_list.append(len(top_n_indices))
        stats.per_class_stats[label]['images'] += 1
        stats.per_class_stats[label]['patches'] += len(top_n_indices)
        
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
    
    save_patch_data(img_id, fallback_y, fallback_x, img_rgb, mask_gray, label, patch_records)
    
    if debug:
        print(f"  ✓ Centroid: 1 patch at ({fallback_y},{fallback_x})")
    
    stats.method_counts['centroid'] += 1
    elapsed = time.time() - start_time
    stats.processing_times.append(elapsed)
    stats.patches_per_image_list.append(1)
    stats.per_class_stats[label]['images'] += 1
    stats.per_class_stats[label]['patches'] += 1
    
    return 1

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    os.makedirs(PATCHES_DIR, exist_ok=True)
    
    print("="*70)
    print("BREAST CANCER PATCH GENERATOR (CLASS-BALANCED)")
    print("="*70)
    print(f"Patch size: {INPUT_SIZE}×{INPUT_SIZE}, Stride: {STRIDE}")
    print(f"Patches per image: {MIN_PATCHES_PER_IMAGE}-{MAX_PATCHES_PER_IMAGE} (default)")
    print(f"Thresholds: Mask≥{MASK_OVERLAP_THRESH*100:.0f}%, "
          f"Tissue≥{TISSUE_FRACTION_THRESH*100:.0f}%, Var≥{VARIANCE_THRESH}")
    print(f"Class balancing: {'ENABLED' if BALANCE_CLASSES else 'DISABLED'}")
    print("="*70 + "\n")
    
    # Load data
    whitelist_ids = load_whitelist(WHITELIST_PATH)
    if whitelist_ids:
        print(f"✓ Loaded whitelist: {len(whitelist_ids)} images")
    
    id_to_label = load_labels(TRAIN_LABEL_CSV)
    print(f"✓ Loaded labels: {len(id_to_label)} images\n")
    
    # Get image list
    train_files = [f for f in os.listdir(TRAIN_FOLDER)
                   if f.startswith("img_") and f.endswith(".png")]
    train_ids_all = [f.split("_")[1].split(".")[0] for f in train_files]
    
    if whitelist_ids:
        train_ids = sorted([i for i in train_ids_all if i in whitelist_ids], key=int)
    else:
        train_ids = sorted(train_ids_all, key=int)
    
    # Calculate class balancing parameters
    if BALANCE_CLASSES:
        class_max_patches = calculate_class_balancing_params(train_ids, id_to_label)
    else:
        # Use default max for all classes
        unique_classes = set(id_to_label.values())
        class_max_patches = {cls: MAX_PATCHES_PER_IMAGE for cls in unique_classes}
    
    print(f"\nProcessing {len(train_ids)} images...\n")
    
    # Extract patches
    patch_records = []
    img_rgb_cache = {}
    mask_gray_cache = {}
    
    for idx, img_id in enumerate(train_ids):
        debug_this = (DEBUG_MODE and img_id == DEBUG_IMAGE_ID)
        
        if debug_this:
            print(f"\n{'='*70}")
            print(f"DEBUGGING IMAGE {img_id}")
            print(f"{'='*70}")
        
        # Load image
        img_path = os.path.join(TRAIN_FOLDER, f"img_{img_id}.png")
        mask_path = os.path.join(TRAIN_FOLDER, f"mask_{img_id}.png")
        
        img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            print(f"⚠️  Could not load {img_id}")
            continue
        
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            img_bgr = img_bgr[:, :, :3]
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            print(f"⚠️  Could not load mask for {img_id}")
            continue
        
        label = id_to_label.get(img_id)
        if label is None:
            print(f"⚠️  No label for {img_id}")
            continue
        
        # Cache for visualizations
        if len(img_rgb_cache) < 20:
            img_rgb_cache[img_id] = img_rgb
            mask_gray_cache[img_id] = mask_gray
        
        # Get max patches for this image's class
        max_patches_for_this_image = class_max_patches.get(label, MAX_PATCHES_PER_IMAGE)
        
        # Extract patches
        num_patches = extract_patches_from_image(
            img_id, img_rgb, mask_gray, label, patch_records, max_patches_for_this_image, debug=debug_this
        )
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(train_ids)} images... ({stats.patches_saved} patches so far)")
    
    print(f"\n✓ Extraction complete: {len(patch_records)} patches generated\n")
    
    # Save results
    df_patches = pd.DataFrame(patch_records, columns=["patch_filename", "label"])
    df_patches.to_csv(PATCH_LABEL_CSV, index=False)
    print(f"✓ Saved: {PATCH_LABEL_CSV}\n")
    
    # Statistics
    print("="*70)
    print("PATCH DISTRIBUTION")
    print("="*70)
    print(df_patches["label"].value_counts())
    print("\nPercentages:")
    print((df_patches["label"].value_counts(normalize=True) * 100).map('{:.1f}%'.format))
    
    # Verification
    df_patches["img_id"] = df_patches["patch_filename"].str.extract(r"(\d+)_y")[0]
    patches_per_image = df_patches["img_id"].value_counts()
    
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    print(f"Images with patches: {len(patches_per_image)}/{len(train_ids)}")
    print(f"Min patches/image: {patches_per_image.min()}")
    print(f"Max patches/image: {patches_per_image.max()}")
    print(f"Avg patches/image: {patches_per_image.mean():.2f}")
    
    if len(patches_per_image) == len(train_ids):
        print("✓ SUCCESS: All images have patches!")
    
    # Print detailed statistics
    stats.print_summary()
    
    # Visualizations
    if SAVE_DEBUG_VISUALIZATIONS:
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        if stats.quality_scores:
            visualize_quality_distributions(save_path="challenge2/quality_distributions.png")
        
        debug_img_path = os.path.join(TRAIN_FOLDER, f"img_{DEBUG_IMAGE_ID}.png")
        debug_mask_path = os.path.join(TRAIN_FOLDER, f"mask_{DEBUG_IMAGE_ID}.png")
        
        if os.path.exists(debug_img_path):
            img_bgr = cv2.imread(debug_img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mask_gray = cv2.imread(debug_mask_path, cv2.IMREAD_GRAYSCALE)
            
            saved_patches_for_img = []
            for filename, _ in patch_records:
                if f"_{DEBUG_IMAGE_ID}_" in filename:
                    parts = filename.split('_')
                    for i, part in enumerate(parts):
                        if part.startswith('y') and i+1 < len(parts):
                            y = int(part[1:])
                            x_part = parts[i+1]
                            x = int(x_part[1:].split('.')[0])
                            saved_patches_for_img.append((y, x))
                            break
            
            print(f"Found {len(saved_patches_for_img)} saved patches for image {DEBUG_IMAGE_ID}")
            
            visualize_mask_density(
                DEBUG_IMAGE_ID, img_rgb, mask_gray,
                saved_patches=saved_patches_for_img,
                save_path=f"challenge2/mask_density_{DEBUG_IMAGE_ID}.png"
            )
            
            visualize_saved_patches(
                DEBUG_IMAGE_ID, img_rgb, mask_gray,
                saved_patches=saved_patches_for_img,
                save_path=f"challenge2/saved_patches_{DEBUG_IMAGE_ID}.png"
            )
        
        if stats.barely_passed or stats.barely_failed:
            visualize_edge_cases(
                img_rgb_cache, mask_gray_cache,
                save_path="challenge2/edge_cases.png"
            )
    
    print("\n" + "="*70)
    print("✅ PATCH GENERATION COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
