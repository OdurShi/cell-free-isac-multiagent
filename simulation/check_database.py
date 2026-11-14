import numpy as np
import matplotlib.pyplot as plt

def fast_diagnostic(db_path='paris_pg_database_3d.npy', pos_path='candidate_positions_3d.npy'):
    print(f"Loading the database: {db_path} ...")
    db = np.load(db_path)
    positions = np.load(pos_path)
    
    print(f"Database shape: {db.shape}")
    print(f"Number of candidate positions: {positions.shape[0]}")
    
    # Flatten the 5D matrix by merging the first three dimensions, to check the AP positions one by one
    # the shape becomes (847, 125, 125)
    num_candidates = db.shape[0] * db.shape[1] * db.shape[2]
    db_flat = db.reshape(num_candidates, 125, 125)
    
    # ---------------------------------------------------------
    # Diagnostic 1: check for "dead zones" (AP positions with all-zero or extremely weak signal)
    # ---------------------------------------------------------
    print("\n--- Diagnostic 1: blind-zone and dead-point detection ---")
    # Compute the total energy within the coverage area of each AP (if the signal is too weak, the AP may be located inside a building)
    total_energy_per_tx = np.sum(db_flat, axis=(1, 2))
    
    # Find the abnormal points whose energy is 0 or extremely low
    dead_zones = np.where(total_energy_per_tx < 1e-15)[0] 
    print(f"[WARNING] Found {len(dead_zones)} AP positions with extremely low total coverage energy (possibly invalid positions / stuck inside a wall)")
    
    if len(dead_zones) > 0:
        print("   First 5 problematic coordinates:")
        for idx in dead_zones[:5]:
            print(f"   - Index {idx}: coordinates {positions[idx]}")

    # ---------------------------------------------------------
    # Diagnostic 2: check for "speckle" and an insufficient number of rays (sparsity check)
    # ---------------------------------------------------------
    print("\n--- Diagnostic 2: ray-coverage sparsity check ---")
    # Compute, for each heatmap (125x125), the fraction of pixels that received no signal at all
    zero_pixel_ratio = np.sum(db_flat == 0, axis=(1, 2)) / (125 * 125)
    
    # If more than 80% of a heatmap is 0, it is either heavily blocked or the number of rays is too small to fill it
    sparse_maps = np.where(zero_pixel_ratio > 0.80)[0]
    print(f"[WARNING] Found {len(sparse_maps)} heatmaps whose invalid coverage area exceeds 80% (samples_per_tx may need to be increased)")
    
    # ---------------------------------------------------------
    # Diagnostic 3: global numerical statistics and extreme-value check
    # ---------------------------------------------------------
    print("\n--- Diagnostic 3: global numerical health ---")
    print(f"   Maximum channel gain (Linear): {np.max(db)}")
    print(f"   Minimum channel gain (Linear): {np.min(db)}")
    print(f"   Mean channel gain (Linear): {np.mean(db)}")
    
    # Check for NaN or infinity (this would make the subsequent SAC neural network crash outright)
    if np.isnan(db).any() or np.isinf(db).any():
        print("Severe warning: the database contains abnormal NaN or Inf values! They must be cleaned!")
    else:
        print("Numerical cleanliness check passed (no NaN/Inf).")

    # ---------------------------------------------------------
    # Diagnostic 4: visualization of random samples (very important!)
    # ---------------------------------------------------------
    print("\n--- Diagnostic 4: visualize 3 randomly drawn normal samples ---")
    # Skip the dead_zones found above and randomly pick a few normal ones to see what they look like
    valid_indices = np.setdiff1d(np.arange(num_candidates), dead_zones)
    
    if len(valid_indices) >= 3:
        sample_idx = np.random.choice(valid_indices, 3, replace=False)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, idx in enumerate(sample_idx):
            # Convert to the log domain (dB) to make large-scale fading easier to see; add 1e-15 to avoid log10(0)
            heatmap_db = 10 * np.log10(db_flat[idx] + 1e-15) 
            
            im = axes[i].imshow(heatmap_db, cmap='jet', vmin=-150, vmax=-50)
            axes[i].set_title(f"TX {idx} (Z={positions[idx][2]}m)\nAvg: {np.mean(heatmap_db):.1f} dB")
            fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
        plt.tight_layout()
        plt.show()
    else:
        print("[WARNING] Not enough valid heatmaps available for visualization.")

if __name__ == "__main__":
    fast_diagnostic()