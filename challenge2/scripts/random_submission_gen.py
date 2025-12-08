import csv
import random

def generate_random_predictions():
    # Configuration
    start_id = 0
    end_id = 953
    filename = "random_prediction.csv"
    
    # The 4 labels
    labels = [
        "Luminal A", 
        "Luminal B", 
        "HER2(+)", 
        "Triple negative"
    ]

    print(f"Generating {filename}...")

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(["sample_index", "label"])
        
        # Loop from 0 to 953
        for i in range(start_id, end_id + 1):
            # Format filename with 4-digit zero padding (e.g., img_0005.png)
            img_name = f"img_{i:04d}.png"
            
            # Pick a random label (random.choice gives equal probability ~0.25)
            selected_label = random.choice(labels)
            
            # Write row
            writer.writerow([img_name, selected_label])

    print(f"Done! Generated {end_id + 1} rows in '{filename}'.")

if __name__ == "__main__":
    generate_random_predictions()