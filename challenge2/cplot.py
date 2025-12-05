import numpy as np
import matplotlib.pyplot as plt

def visualize_image_components(raw_image, raw_mask, train_labels, filenames=None, num_samples=6):
    
    # Safety check: ensure we don't sample more than available
    n_samples = min(num_samples, len(raw_image))
    
    # 1. Pick random indices
    random_indices = np.random.choice(len(raw_image), size=n_samples, replace=False)

    # 2. Setup Plot
    fig, axes = plt.subplots(2, n_samples, figsize=(18, 6)) 

    for i, idx in enumerate(random_indices):
        
        # Grab data at this index
        img = np.clip(raw_image[idx], 0, 1) 
        mask = np.clip(raw_mask[idx], 0, 1)        
        label = train_labels[idx]
        
        # --- KEY FIX: Resolve the Name ---
        # If a filename list is provided, lookup the name at this index.
        # Otherwise, fall back to the numeric index.
        if filenames is not None:
            image_id = filenames[idx] 
        else:
            image_id = f"Index {idx}"

        # Calculate masked image for display
        masked_img = img * mask 
        
        # --- Row 0: Original Image ---
        ax_orig = axes[0, i] if n_samples > 1 else axes[0]
        ax_orig.imshow(img) 
        ax_orig.set_title(f"{image_id}\nLabel: {label}", fontsize=9) 
        ax_orig.axis('off')
        
        # --- Row 1: Masked Input ---
        ax_masked = axes[1, i] if n_samples > 1 else axes[1]
        ax_masked.imshow(masked_img)
        ax_masked.set_title("Masked Input", fontsize=9)
        ax_masked.axis('off')

    plt.suptitle(f"Visualizing {n_samples} Samples", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_training_history(history, title="Training History"):

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
    
    # --- Plot 1: Loss ---
    ax1.plot(history['train_loss'], label='Training Loss', alpha=0.5, color='#ff7f0e', linestyle='--')
    ax1.plot(history['val_loss'], label='Validation Loss', alpha=1.0, color='#ff7f0e')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Categorical Crossentropy')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- Plot 2: F1 Score ---
    # Using a different color (e.g., Blue) for the metric to visually separate it from loss
    ax2.plot(history['train_f1'], label='Training F1', alpha=0.5, color='#1f77b4', linestyle='--')
    ax2.plot(history['val_f1'], label='Validation F1', alpha=1.0, color='#1f77b4')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score')
    ax2.set_title('F1 Score')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Final Layout Adjustments
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9) # Make space for suptitle
    plt.show()

