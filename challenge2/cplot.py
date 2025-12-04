import numpy as np
import matplotlib.pyplot as plt

def visualize_image_components(train_images, train_masks, train_labels, num_samples=10):

    # Select random indices for the samples
    random_indices = np.random.choice(len(train_images), size=num_samples, replace=False)

    # 5 rows: Image, R, G, B, Mask
    fig, axes = plt.subplots(5, num_samples, figsize=(20, 10))

    for i, idx in enumerate(random_indices):
        # Clip image and mask data once for plotting
        img = np.clip(train_images[idx], 0, 1)
        mask = np.clip(train_masks[idx], 0, 1)
        label = train_labels[idx] # Get the label
        
        # Row 0: Image (RGB)
        ax_img = axes[0, i]
        ax_img.imshow(img) 
        ax_img.set_title(f"Image {idx}\n({label})", fontsize=8) 
        ax_img.axis('off')

        # Row 1: Red Channel (Grayscale representation)
        ax_r = axes[1, i]
        ax_r.imshow(img[..., 0], cmap='gray')
        ax_r.set_title("R Channel", fontsize=8)
        ax_r.axis('off')

        # Row 2: Green Channel (Grayscale representation)
        ax_g = axes[2, i]
        ax_g.imshow(img[..., 1], cmap='gray')
        ax_g.set_title("G Channel", fontsize=8)
        ax_g.axis('off')

        # Row 3: Blue Channel (Grayscale representation)
        ax_b = axes[3, i]
        ax_b.imshow(img[..., 2], cmap='gray')
        ax_b.set_title("B Channel", fontsize=8)
        ax_b.axis('off')

        # Row 4: Corresponding Mask
        ax_mask = axes[4, i]
        ax_mask.imshow(mask)
        ax_mask.set_title("Mask", fontsize=8)
        ax_mask.axis('off')

    plt.suptitle(f"Random Sample of {num_samples} Image/Mask Pairs with Grayscale RGB Channels", y=1.02)
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

