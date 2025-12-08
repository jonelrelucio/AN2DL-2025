import os
import cv2
import numpy as np
import pandas as pd
from skimage.color import separate_stains, hed_from_rgb
from skimage.exposure import rescale_intensity

# ---- CONFIG ----
PATCHES_DIR = "challenge2/patches"                 # source patches (RGB)
OUT_ROOT    = "challenge2/patches_hed_separate"    # root for separate stains
PATCH_LABEL_CSV = "challenge2/label_patches.csv"   # original patch labels

# Subfolders for H, E, D
H_DIR = os.path.join(OUT_ROOT, "H")
E_DIR = os.path.join(OUT_ROOT, "E")
D_DIR = os.path.join(OUT_ROOT, "D")
os.makedirs(H_DIR, exist_ok=True)
os.makedirs(E_DIR, exist_ok=True)
os.makedirs(D_DIR, exist_ok=True)

# Load patch list + labels
df_patches = pd.read_csv(PATCH_LABEL_CSV)
print(f"Converting {len(df_patches)} patches from RGB to separate H/E/D stains...")

records = []  # (base_name, H_filename, E_filename, D_filename, label)

def to_uint8(ch):
    """Rescale stain channel to [0,255] uint8 for saving."""
    ch_rescaled = rescale_intensity(ch, in_range="image", out_range=(0, 255))
    return ch_rescaled.astype(np.uint8)

for _, row in df_patches.iterrows():
    fname = row["patch_filename"]
    label = row["label"]

    src_path = os.path.join(PATCHES_DIR, fname)
    img_bgr = cv2.imread(src_path)
    if img_bgr is None:
        print(f"Warning: could not read {src_path}, skipping.")
        continue

    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ---- RGB -> HED using separate_stains ----
    # hed_from_rgb is the standard color deconvolution matrix for H&E+DAB. [web:302][web:303]
    hed = separate_stains(img_rgb, hed_from_rgb)   # (H, W, 3) float
    H_ch = hed[:, :, 0]
    E_ch = hed[:, :, 1]
    D_ch = hed[:, :, 2]

    H_u8 = to_uint8(H_ch)
    E_u8 = to_uint8(E_ch)
    D_u8 = to_uint8(D_ch)

    base, ext = os.path.splitext(fname)
    h_name = base + "_H.png"
    e_name = base + "_E.png"
    d_name = base + "_D.png"

    h_path = os.path.join(H_DIR, h_name)
    e_path = os.path.join(E_DIR, e_name)
    d_path = os.path.join(D_DIR, d_name)

    cv2.imwrite(h_path, H_u8)
    cv2.imwrite(e_path, E_u8)
    cv2.imwrite(d_path, D_u8)

    records.append((base, h_name, e_name, d_name, label))

print(f"Saved separate H/E/D patches under {OUT_ROOT}")

# CSV describing all three stain images per original patch
df_hed = pd.DataFrame(
    records,
    columns=["base_name", "H_filename", "E_filename", "D_filename", "label"]
)
csv_out = os.path.join(OUT_ROOT, "label_patches_hed_separate.csv")
df_hed.to_csv(csv_out, index=False)
print(f"Saved {csv_out}")
