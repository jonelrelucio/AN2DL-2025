import cv2
import os
import csv
import numpy as np

FOLDER = "challenge2/train_data"
CSV_PATH = "challenge2/train_labels.csv"
OUTPUT = "challenge2/pairs.csv"  # <--- Changed to CSV
PROGRESS_FILE = "challenge2/progress.txt" 
THRESHOLD = 80

# -------------------- LOAD LABELS --------------------
def load_labels(path):
    labels = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    labels[row[0]] = row[1]
    return labels

# -------------------- FAST FEATURE EXTRACTION --------------------
def load_features(folder):
    print("Step 1: Loading images and computing features (Memory Optimized)...")
    orb = cv2.ORB_create(2000)
    data = [] 
    
    filenames = sorted([f for f in os.listdir(folder) if f.startswith("img_") and f.lower().endswith(".png")])
    
    for i, f in enumerate(filenames):
        if i % 100 == 0: print(f"Encoded {i}/{len(filenames)}")
        
        path = os.path.join(folder, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            data.append((f, None))
            continue
            
        _, des = orb.detectAndCompute(img, None)
        data.append((f, des))
        
    return data

# -------------------- SETUP --------------------
labels = load_labels(CSV_PATH)
dataset = load_features(FOLDER) 

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

start_i = 0

# -------------------- RESUME LOGIC --------------------
if os.path.exists(PROGRESS_FILE):
    try:
        with open(PROGRESS_FILE, "r") as f:
            content = f.read().strip()
            if content.isdigit():
                start_i = int(content) + 1 
                print(f"Crash detected! Resuming from index {start_i}...")
            else:
                print("Progress file found but empty/invalid. Starting from 0.")
    except Exception as e:
        print(f"Error reading progress file: {e}. Starting from 0.")
else:
    print("No progress file found. Starting from beginning.")

# -------------------- MAIN LOOP --------------------

# Check if file exists to decide whether to write headers
file_exists = os.path.exists(OUTPUT)

# Open in append mode
with open(OUTPUT, "a", newline="") as f_out:
    writer = csv.writer(f_out)
    
    # Write Header if new file
    if not file_exists:
        writer.writerow(["Image1", "Label1", "Image2", "Label2", "Score"])
    
    for i in range(start_i, len(dataset)):
        name1, d1 = dataset[i]
        
        if d1 is None:
            with open(PROGRESS_FILE, "w") as f_prog:
                f_prog.write(str(i))
            continue

        if i % 10 == 0: print(f"Processing {name1} ({i}/{len(dataset)})...")

        for j in range(i + 1, len(dataset)):
            name2, d2 = dataset[j]
            
            if d2 is None:
                continue

            matches = bf.match(d1, d2)
            score = sum(1 for m in matches if m.distance < 40)
            
            if score > THRESHOLD:
                print(f"FOUND: {name1} <-> {name2} (Score: {score})")
                
                l1 = labels.get(name1, 'NA')
                l2 = labels.get(name2, 'NA')
                
                # Write CSV Row
                writer.writerow([name1, l1, name2, l2, score])
                
                # Flush to ensure data is saved immediately
                f_out.flush() 

        # --- SAVE PROGRESS AFTER FINISHING IMAGE i ---
        with open(PROGRESS_FILE, "w") as f_prog:
            f_prog.write(str(i))