import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Loading your physical simulation database...")
    database = np.load('paris_pg_database_3d.npy')
    
    print("Aggregating the signals of 121 APs to reverse-extract the building outlines...")
    # Take the data at a height of 35 m (Z-axis index 6)
    # Take the maximum over the first two dimensions (X, Y coordinates, 11x11=121 APs in total)
    # This is equivalent to: "at every location in the city, pick the strongest signal that can be received"
    max_coverage_linear = np.max(database[:, :, 6, :, :], axis=(0, 1))
    
    # Convert to dB for display
    max_coverage_db = 10 * np.log10(max_coverage_linear + 1e-15)
    
    # Start plotting
    plt.figure(figsize=(10, 10))
    # Use the magma colormap: streets (with signal) appear in bright yellow/orange/pink, the interior of solid buildings (no signal) is pure black
    plt.imshow(max_coverage_db, cmap='magma', origin='lower')
    plt.title("Paris Etoile True Layout (Reconstructed from Ray Tracing)", fontsize=16)
    plt.axis('off')
    
    save_path = "reconstructed_map.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Done. The map reverse-extracted from your own data has been saved as: {save_path}")

if __name__ == "__main__":
    main()