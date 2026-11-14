import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import sionna
from sionna.rt import load_scene, Transmitter, PlanarArray, RadioMapSolver
import time

def main():
    # 1. Load the scene and configure the core physical parameters
    scene = load_scene(sionna.rt.scene.etoile)
    scene.frequency = 28e9 
    
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

    # 2. Define the 3D candidate region
    x_coords = np.linspace(-250, 250, 11)
    y_coords = np.linspace(-250, 250, 11)
    
    # Key change 1: add layered height sampling along the Z axis (e.g. from 5 m to 35 m in steps of 5 m)
    z_coords = np.linspace(5, 35, 7) 
    
    solver = RadioMapSolver()
    solver.max_depth = 4 # the default is usually 3; add one more reflection to the computation
    # Key change 2: pre-allocate a 5D NumPy array so that the SAC algorithm can later look values up directly by 3D index
    # Meaning of the dimensions: [X coordinate index, Y coordinate index, Z coordinate index, number of receiver X grid cells (125), number of receiver Y grid cells (125)]
    database_3d = np.zeros((len(x_coords), len(y_coords), len(z_coords), 125, 125), dtype=np.float32)
    positions = []

    total_points = len(x_coords) * len(y_coords) * len(z_coords)
    print(f"Starting generation of the 3D offline fingerprint database, {total_points} candidate points in total...")
    start_time = time.time()

    count = 0
    # Nested loops: iterate over every candidate AP position in 3D space
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            for k, z in enumerate(z_coords):
                count += 1
                pos = [float(x), float(y), float(z)]
                positions.append(pos)
                
                # Add a temporary omnidirectional AP
                tx = Transmitter(name="temp_tx", position=pos, orientation=[0, 0, 0])
                scene.add(tx)
                
                # Ray-tracing computation
                # Note: the coverage plane of the receivers (users) is always fixed at a ground-level height of z=1.5 m
                radio_map = solver(scene, 
                                   cell_size=[4, 4], 
                                   center=[0, 0, 1.5], 
                                   size=[500, 500],
                                   orientation=[0, 0, 0],
                                   samples_per_tx=int(5e6)) 
                
                # Get the linear channel gain (Linear Path Gain)
                pg = radio_map.path_gain.numpy().astype(np.float32)
                
                # Store this 2D heatmap directly in the corresponding 3D slot of the 5D matrix
                database_3d[i, j, k, :, :] = np.squeeze(pg)
                
                # Remove the current AP to make room for the next point
                scene.remove("temp_tx")
                
                # Print progress (the computation is heavier now, so the print frequency can be adjusted as appropriate)
                if count % 20 == 0:
                    elapsed = time.time() - start_time
                    print(f"Completed {count}/{total_points} | Elapsed: {elapsed:.1f}s")

    # 3. Save the database
    np.save('paris_pg_database_3d.npy', database_3d)
    np.save('candidate_positions_3d.npy', np.array(positions))
    
    print("-" * 50)
    print(f"3D database generation finished. Total time: {(time.time()-start_time)/60:.1f} minutes")
    print(f"Final shape of the database matrix: {database_3d.shape}")
    print("paris_pg_database_3d.npy (linear channel gain matrix)")
    print("candidate_positions_3d.npy (corresponding 3D coordinates)")
    print("-" * 50)

if __name__ == "__main__":
    main()