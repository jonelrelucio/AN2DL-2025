import numpy as np
import matplotlib.pyplot as plt

def visualize_image_components(raw_image, raw_mask, train_labels, filenames=None, num_samples=6):
    
    # Safety check
    n_samples = min(num_samples, len(raw_image))
    
    # 1. Pick random indices from the FILTERED list
    random_indices = np.random.choice(len(raw_image), size=n_samples, replace=False)

    fig, axes = plt.subplots(2, n_samples, figsize=(18, 7)) 

    for i, idx in enumerate(random_indices):
        
        # Grab data
        img = raw_image[idx]
        mask = raw_mask[idx]
        label = train_labels[idx]
        
        # --- Auto-convert for Plotting ---
        # If data is 0-255 (Integers), convert to 0.0-1.0 (Floats)
        if img.max() > 1:
            img = img.astype(np.float32) / 255.0
        if mask.max() > 1:
            mask = mask.astype(np.float32) / 255.0
            
        img = np.clip(img, 0, 1) 
        mask = np.clip(mask, 0, 1)        
        
        # --- Resolve Name ---
        if filenames is not None:
            fname = filenames[idx]
        else:
            fname = "Unknown"
            
        # Title shows Internal Index AND Real Filename to avoid confusion
        title_text = f"{fname}\nLabel: {label}"

        # Masked Image
        masked_img = img * mask 
        
        # Plotting
        ax_orig = axes[0, i] if n_samples > 1 else axes[0]
        ax_orig.imshow(img) 
        ax_orig.set_title(title_text, fontsize=9) 
        ax_orig.axis('off')
        
        ax_masked = axes[1, i] if n_samples > 1 else axes[1]
        ax_masked.imshow(masked_img)
        ax_masked.set_title("Masked & Cropped", fontsize=9)
        ax_masked.axis('off')

    plt.suptitle(f"Visualizing {n_samples} Random Samples", y=1.02, fontsize=14)
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


def plot_comprehensive_results(model, loader, device, label_encoder, history=None):
    """
    Plots Training History, Confusion Matrix, and Prediction Distribution.
    """
    print("Generating comprehensive evaluation plots...")
    
    # --- 1. Get Predictions ---
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            # Handle tuple (inputs, labels)
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            
            # Forward pass
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            
    class_names = label_encoder.classes_
    
    # --- 2. Setup Figure Grid ---
    fig = plt.figure(figsize=(20, 12))
    # 2 rows, 2 columns
    gs = fig.add_gridspec(2, 2) 

    # --- 3. Plot History (Row 1) ---
    if history:
        # Loss Plot (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(history['train_loss'], label='Train Loss', color='tab:blue', linewidth=2)
        ax1.plot(history['val_loss'], label='Val Loss', color='tab:orange', linewidth=2)
        ax1.set_title("Loss Curves", fontsize=14)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # F1 Score Plot (Top Right)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(history['train_f1'], label='Train F1', color='tab:green', linewidth=2)
        ax2.plot(history['val_f1'], label='Val F1', color='tab:red', linewidth=2)
        ax2.set_title("F1 Score Curves", fontsize=14)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("F1 Score")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        # Placeholders if no history provided
        ax1 = fig.add_subplot(gs[0, 0]); ax1.text(0.5, 0.5, "No History Provided", ha='center')
        ax2 = fig.add_subplot(gs[0, 1]); ax2.text(0.5, 0.5, "No History Provided", ha='center')

    # --- 4. Confusion Matrix (Bottom Left) ---
    ax3 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(all_targets, all_preds)
    
    # Use Seaborn for a nice heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax3, cbar=False)
    
    ax3.set_title("Confusion Matrix", fontsize=14)
    ax3.set_xlabel("Predicted Label")
    ax3.set_ylabel("True Label")
    ax3.tick_params(axis='x', rotation=45)
    ax3.tick_params(axis='y', rotation=0)

    # --- 5. Prediction Distribution (Bottom Right) ---
    # This helps check for Class Imbalance bias
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Count occurrences
    unique_true, true_counts = np.unique(all_targets, return_counts=True)
    
    # Ensure pred counts matches size of true counts (fill missing classes with 0)
    pred_counts_full = np.zeros(len(class_names))
    unique_pred, pred_counts = np.unique(all_preds, return_counts=True)
    for cls, count in zip(unique_pred, pred_counts):
        pred_counts_full[cls] = count

    # Bar Chart
    x = np.arange(len(class_names))
    width = 0.35
    
    ax4.bar(x - width/2, true_counts, width, label='True (Ground Truth)', color='gray', alpha=0.6)
    ax4.bar(x + width/2, pred_counts_full, width, label='Predicted by Model', color='tab:purple', alpha=0.9)
    
    ax4.set_title("Class Distribution: Truth vs Predictions", fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(class_names, rotation=45)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


