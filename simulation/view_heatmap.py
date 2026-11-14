import numpy as np
import matplotlib.pyplot as plt

def view_heatmap():
    print("Loading the 3D database, please wait...")
    # 1. Load the 5D matrix you have just generated
    # Dimensions: [X index (11), Y index (11), Z index (7), Rx(125), Ry(125)]
    database = np.load('paris_pg_database_3d.npy')
    print(f"Database loaded successfully. Shape: {database.shape}")

    # 2. Pick an AP position you are interested in to "peek" at
    # For example, we pick the point at the very center of the map (6th in X, 6th in Y)
    x_idx = 5  
    y_idx = 5  
    
    # Look at the difference between the highest height (index 6, i.e. 35 m) and the lowest height (index 0, i.e. 5 m)
    z_idx_high = 6  
    z_idx_low = 0   

    # 3. Extract these two 2D heatmaps
    heatmap_high = database[x_idx, y_idx, z_idx_high, :, :]
    heatmap_low = database[x_idx, y_idx, z_idx_low, :, :]

    # Physical display correction:
    # The database stores very small "linear gain" values, so plotting them directly may give an almost entirely black image.
    # To make it visible to the eye, we temporarily convert it back to dB when plotting.
    # 1e-15 is added to avoid an error when taking the logarithm of 0.
    heatmap_high_db = 10 * np.log10(heatmap_high + 1e-15)
    heatmap_low_db = 10 * np.log10(heatmap_low + 1e-15)

    # 4. Plot the comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the heatmap for the low position (5 m)
    im1 = axes[0].imshow(heatmap_low_db, cmap='viridis', origin='lower')
    axes[0].set_title(f"AP Height: 5m (Index: {z_idx_low})")
    fig.colorbar(im1, ax=axes[0], label='Path Gain (dB)')

    # Plot the heatmap for the high position (35 m)
    im2 = axes[1].imshow(heatmap_high_db, cmap='viridis', origin='lower')
    axes[1].set_title(f"AP Height: 35m (Index: {z_idx_high})")
    fig.colorbar(im2, ax=axes[1], label='Path Gain (dB)')

    plt.suptitle("3D Cell-Free Database Verification", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    view_heatmap()