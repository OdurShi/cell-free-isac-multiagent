import os
# Disable the GPU and compute by brute force on the CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import sionna
from sionna.rt import load_scene, Transmitter, PlanarArray, RadioMapSolver
import matplotlib.pyplot as plt

def main():
    print("Loading the Paris 3D city model...")
    scene = load_scene(sionna.rt.scene.etoile)

    print("Registering the transmit and receive antenna arrays globally...")
    scene.tx_array = PlanarArray(num_rows=8, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

    print("Setting up the 5G base station...")
    tx = Transmitter(name="BaseStation_1", position=[0, 0, 30], orientation=[0, 0, 0])
    scene.add(tx)

    print("Starting the solver for ray tracing...")
    solver = RadioMapSolver()
    
    radio_map = solver(
        scene=scene, 
        cell_size=[4, 4], 
        center=[0, 0, 1.5], 
        orientation=[0, 0, 0], 
        size=[500, 500],
        samples_per_tx=int(1e6)
    )

    print("Computation finished. Extracting and plotting the heatmap...")
    
    # Key fix: read the path_gain attribute of the latest version
    pg = radio_map.path_gain
    
    # Take the data of the first base station and drop the extra dimensions so that imshow does not raise an error
    pg_np = np.squeeze(pg[0].numpy())
    
    # Convert Path Gain (linear value) to Path Loss (dB)
    # Add 1e-10 to avoid a log10(0) error in blind spots with no signal at all
    path_loss_db = -10.0 * np.log10(pg_np + 1e-10)

    plt.figure(figsize=(10, 8))
    plt.imshow(path_loss_db, cmap="viridis_r", origin="lower")
    plt.colorbar(label="Path Loss (dB)")
    plt.title("28GHz Radio Map (Paris Etoile)")
    plt.savefig("coverage_map_result.png", dpi=300, bbox_inches='tight')
    
    print("-" * 40)
    print("Done. The image has been generated successfully.")
    print("Result saved: coverage_map_result.png")
    print("-" * 40)

if __name__ == "__main__":
    main()