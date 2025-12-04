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