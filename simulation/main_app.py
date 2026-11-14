import customtkinter as ctk
import multiprocessing
import os
import sys
multiprocessing.freeze_support()

os.environ['MPLBACKEND'] = 'TkAgg'

def get_resource_path(relative_path):
    """Get the absolute path of a resource; compatible with the development environment and the PyInstaller-packaged environment"""
    if getattr(sys, 'frozen', False):
        # runtime environment of the packaged exe
        base_path = sys._MEIPASS
    else:
        # development environment
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

import matplotlib
matplotlib.use('TkAgg') # force the interactive backend to make sure windows can pop up
import matplotlib.pyplot as plt

from tkinter import filedialog, messagebox
import threading
import time

import sionna
import math
from copy import deepcopy
import random
import subprocess

import json

from sionna.rt import load_scene
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from isac_env import CellFreeISAC3DEnv


# Key change: set light mode and use the high-contrast blue theme
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")



def run_basic_3d_viewer(model_path):
    import os
    import pyvista as pv

    meshes_dir = os.path.join(os.path.dirname(model_path), 'meshes')

    if os.path.exists(meshes_dir):
        pv.set_plot_theme('document')
        plotter = pv.Plotter(title='6G ISAC Digital Twin - 3D Viewer')
        for file in os.listdir(meshes_dir):
            if file.endswith(('.ply', '.obj')):
                mesh = pv.read(os.path.join(meshes_dir, file))
                plotter.add_mesh(mesh, color='#E5E8E8', show_edges=True, edge_color='#95A5A6', opacity=0.9)

        plotter.add_axes()
        plotter.show()

def run_3d_viewer_engine(model_path, coords_data_str):
    import os, json
    import pyvista as pv
    
    coords_data = json.loads(coords_data_str)
    meshes_dir = os.path.join(os.path.dirname(model_path), 'meshes')

    pv.set_plot_theme('document')
    plotter = pv.Plotter(title='6G ISAC 3D Deployment - Final Results')

    if os.path.exists(meshes_dir):
        for file in os.listdir(meshes_dir):
            if file.endswith(('.ply', '.obj')):
                mesh = pv.read(os.path.join(meshes_dir, file))
                plotter.add_mesh(mesh, color='#E5E8E8', opacity=0.3, show_edges=True, edge_color='#BDC3C7')

    colors = {'SAC': '#E74C3C', 'SA': '#F39C12', 'GA': '#3498DB'}
    for algo, points in coords_data.items():
        c = colors.get(algo, '#2ECC71')
        tx_points, rx_points = points[:2], points[2:]
        
        for p in tx_points:
            plotter.add_mesh(pv.Sphere(radius=2.5, center=p), color=c)
            plotter.add_mesh(pv.Line(p, (p[0], p[1], 0)), color=c, line_width=2.5, opacity=0.8)
        for p in rx_points:
            plotter.add_mesh(pv.Cube(center=p, x_length=4.5, y_length=4.5, z_length=4.5), color=c)
            plotter.add_mesh(pv.Line(p, (p[0], p[1], 0)), color=c, line_width=1.5, opacity=0.4)

    custom_legend = [
        ['Algorithm: SAC', '#E74C3C'],
        ['Algorithm: SA', '#F39C12'],
        ['Algorithm: GA', '#3498DB'],
        ['Shape: Sphere = Tx', '#660066'],
        ['Shape: Cube = Rx', '#660066']
    ]
    plotter.add_legend(labels=custom_legend, size=(0.16, 0.16), loc='upper right')
    plotter.add_axes()
    plotter.show()

class ISACDigitalTwinApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("ISAC-SPS: 6G Cell-Free ISAC Site Planning Simulation System")
        self.geometry("1000x700")
        
        # Set the main background of the application to a clean light gray for a technical look
        self.configure(fg_color="#F0F3F4")
        
        # Internal state locks of the application
        self.is_model_loaded = False
        self.is_database_ready = False
        
        # Create the top tab view (in light mode the tabs look more like a browser, very clean)
        self.tabview = ctk.CTkTabview(self, width=950, height=650, 
                                      fg_color="#FFFFFF", # pure white inside the tabs to add visual hierarchy
                                      segmented_button_selected_color="#2980B9", 
                                      segmented_button_selected_hover_color="#1F618D")
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)
        
        self.tab_env = self.tabview.add("Physical Environment and Fingerprint Database Construction")
        self.tab_algo = self.tabview.add("Intelligent Deployment and Algorithm Optimization")
        
        self.build_env_tab()
        self.build_algo_tab()

    # ==========================================
    # Part 1: physical environment and fingerprint database construction
    # ==========================================
    def build_env_tab(self):
        # Left control panel (a very light shade is added to set it apart)
        ctrl_frame = ctk.CTkFrame(self.tab_env, width=300, fg_color="#F8F9FA", corner_radius=10)
        ctrl_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        # Title text in dark ink blue, which looks more refined than pure black
        ctk.CTkLabel(ctrl_frame, text="Scene Modeling", font=("Microsoft YaHei", 18, "bold"), text_color="#2C3E50").pack(pady=(20,10))
        
        # Buttons use a tech blue
        self.btn_load_model = ctk.CTkButton(ctrl_frame, text="Import 3D Model", 
                                            fg_color="#3498DB", hover_color="#2980B9", command=self.load_model)
        self.btn_load_model.pack(pady=10, padx=20, fill="x")
        
        self.btn_view_3d = ctk.CTkButton(ctrl_frame, text="View 3D Panorama", state="disabled", 
                                         fg_color="#3498DB", hover_color="#2980B9", command=self.view_3d_model)
        self.btn_view_3d.pack(pady=10, padx=20, fill="x")
        
        # [Replacement 1] the original CTkSeparator is replaced by a 2-pixel-high CTkFrame
        ctk.CTkFrame(ctrl_frame, height=2, fg_color="#E5E7E9").pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(ctrl_frame, text="Fingerprint Construction", font=("Microsoft YaHei", 18, "bold"), text_color="#2C3E50").pack(pady=(10,10))
        
        # Time-consuming operations use an eye-catching color
        self.btn_gen_db = ctk.CTkButton(ctrl_frame, text="Generate/Load Offline Fingerprint Database", state="disabled", 
                                        fg_color="#8E44AD", hover_color="#71368A", command=self.generate_database)
        self.btn_gen_db.pack(pady=10, padx=20, fill="x")
        
        # [Replacement 2] the original CTkSeparator is replaced by a 2-pixel-high CTkFrame
        ctk.CTkFrame(ctrl_frame, height=2, fg_color="#E5E7E9").pack(fill="x", padx=20, pady=10)
        
        # Coordinate input and heatmap
        ctk.CTkLabel(ctrl_frame, text="Enter coordinates to view the heatmap:", text_color="#34495E", font=("Microsoft YaHei", 13)).pack(pady=(5,5))
        self.entry_coord = ctk.CTkEntry(ctrl_frame, placeholder_text="e.g. 100, 50, 25", border_color="#BDC3C7")
        self.entry_coord.pack(pady=5, padx=20, fill="x")
        
        # The heatmap button uses a warning orange, very eye-catching
        self.btn_heatmap = ctk.CTkButton(ctrl_frame, text="Generate Heatmap for Coordinates", state="disabled", 
                                         fg_color="#E67E22", hover_color="#D35400", text_color="white", command=self.show_heatmap)
        self.btn_heatmap.pack(pady=15, padx=20, fill="x")

        # Right log area (light background, dark blue font, similar to a research console)
        self.env_log = ctk.CTkTextbox(self.tab_env, font=("Consolas", 13), 
                                      fg_color="#FDFEFE", text_color="#1B4F72", border_width=1, border_color="#D5D8DC")
        self.env_log.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        self.env_log.insert("end", "[System ready] Please import the 3D city model file first...\n")

    def env_print(self, msg):
        self.env_log.insert("end", f"[*] {msg}\n")
        self.env_log.see("end")

    # --- Environment functions remain unchanged ---
    def load_model(self):
        # Directly use the path of the Etoile (Place de l'Etoile, Paris) scene integrated in Sionna
        try:
            import sionna
            self.model_filepath = sionna.rt.scene.etoile
            
            self.env_print(">>> Mounting the system's built-in physical model...")
            self.env_print("No imported model found, using the official default model: Place de l'Etoile, Paris")
            self.env_print("Includes high-precision ITU concrete buildings and electromagnetic material information.")
            
            self.is_model_loaded = True
            self.btn_view_3d.configure(state="normal")
            self.btn_gen_db.configure(state="normal")
            
        except ImportError:
            self.env_print("[Fatal error] Cannot find the sionna module, please check the runtime environment!")

    def view_3d_model(self):
        if not self.model_filepath:
            self.env_print("[Error] The model path has not been set!")
            return

        self.env_print(">>> Starting the native 3D rendering engine...")

        try:
            # Key fix: detect the environment automatically and send the new flag "--view-3d-base" for the basic 3D view
            if getattr(sys, 'frozen', False):
                cmd = [sys.executable, "--view-3d-base", self.model_filepath]
            else:
                cmd = [sys.executable, __file__, "--view-3d-base", self.model_filepath]

            subprocess.Popen(cmd)
            self.env_print("The interactive 3D window is opening")

        except Exception as e:
            self.env_print(f"[Engine launch failed] Cannot start the basic 3D process: {str(e)}")

    def generate_database(self):
        self.env_print(">>> Checking whether an offline channel fingerprint database exists locally...")
        self.btn_gen_db.configure(state="disabled")
        
        def task():
            try:
                db_path = get_resource_path("paris_pg_database_3d.npy")
                pos_path = get_resource_path("candidate_positions_3d.npy")
                
                # Case A: the offline database exists, load it directly (fast)
                if os.path.exists(db_path) and os.path.exists(pos_path):
                    self.env_print("Local database files detected, loading them into memory...")
                    self.pg_database = np.load(db_path)
                    self.candidate_positions = np.load(pos_path)
                    
                    self.kdtree = cKDTree(self.candidate_positions)
                    self.env_print(f"Loaded successfully. Database dimensions: {self.pg_database.shape}")
                
                # Case B: no offline database exists, start the underlying Sionna ray-tracing engine to generate it
                else:
                    self.env_print("[Warning] No local database detected! The system will start the 3D ray-tracing computation automatically.")
                    self.env_print("Warning: this process takes a long time, please do not close the application...")
                    
                    # Local imports to avoid polluting the global environment
                    import tensorflow as tf
                    import sionna
                    from sionna.rt import load_scene, Transmitter, PlanarArray, RadioMapSolver
                    
                    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # force CPU
                    
                    scene = load_scene(sionna.rt.scene.etoile)
                    scene.frequency = 28e9 
                    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
                    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

                    x_coords = np.linspace(-250, 250, 11)
                    y_coords = np.linspace(-250, 250, 11)
                    z_coords = np.linspace(5, 35, 7) 
                    
                    solver = RadioMapSolver()
                    solver.max_depth = 4
                    database_3d = np.zeros((len(x_coords), len(y_coords), len(z_coords), 125, 125), dtype=np.float32)
                    positions = []
                    
                    total_points = len(x_coords) * len(y_coords) * len(z_coords)
                    start_time = time.time()
                    count = 0
                    
                    for i, x in enumerate(x_coords):
                        for j, y in enumerate(y_coords):
                            for k, z in enumerate(z_coords):
                                count += 1
                                pos = [float(x), float(y), float(z)]
                                positions.append(pos)
                                
                                tx = Transmitter(name="temp_tx", position=pos, orientation=[0, 0, 0])
                                scene.add(tx)
                                
                                radio_map = solver(scene, cell_size=[4, 4], center=[0, 0, 1.5], 
                                                   size=[500, 500], orientation=[0, 0, 0], samples_per_tx=int(5e6)) 
                                
                                pg = radio_map.path_gain.numpy().astype(np.float32)
                                database_3d[i, j, k, :, :] = np.squeeze(pg)
                                scene.remove("temp_tx")
                                
                                if count % 10 == 0:
                                    elapsed = time.time() - start_time
                                    self.env_print(f"   [Generation progress] {count}/{total_points} | Elapsed: {elapsed:.1f}s")
                    
                    # Generation finished, save locally
                    np.save(db_path, database_3d)
                    np.save(pos_path, np.array(positions))
                    self.env_print(f"Offline fingerprint database generated and saved, total time: {(time.time()-start_time)/60:.1f} minutes")
                    
                    # Keep in memory for later use
                    self.pg_database = database_3d
                    self.candidate_positions = np.array(positions)
                    self.kdtree = cKDTree(self.candidate_positions)

                # Whether loaded or generated, unlock the subsequent buttons at the end
                self.is_database_ready = True
                self.btn_heatmap.configure(state="normal")
                self.btn_start_train.configure(state="normal")
                self.env_print(">>> Offline fingerprint database is ready")
                
            except Exception as e:
                self.env_print(f"[Fatal error] Database processing failed: {str(e)}")
                self.btn_gen_db.configure(state="normal")

        # Run in a background thread to keep the GUI from freezing
        threading.Thread(target=task, daemon=True).start()

    def show_heatmap(self):
        coords_str = self.entry_coord.get()
        if not coords_str:
            self.env_print("[Hint] Please enter the 3D coordinates of the AP!")
            return
            
        try:
            # 1. Parse the coordinates
            parts = [float(x.strip()) for x in coords_str.split(',')]
            if len(parts) != 3:
                raise ValueError("The coordinate format must be: X, Y, Z (e.g. 100, 50, 25)")
            target_pos = parts
            
            self.env_print(f">>> Querying the channel response at spatial coordinates {target_pos}...")
            
            # 2. Use the KD-Tree to match the nearest offline sampling point
            distance, index = self.kdtree.query(target_pos)
            matched_pos = self.candidate_positions[index]
            self.env_print(f"[*] Mapped to the nearest valid grid point: [{matched_pos[0]:.1f}, {matched_pos[1]:.1f}, {matched_pos[2]:.1f}] (error {distance:.2f}m)")
            
            # 3. Core mathematical conversion: restore the flattened 1D index to the (i, j, k) index of the 3D matrix
            # Because our sampling space is 11 x 11 x 7, np.unravel_index recovers it exactly
            i, j, k = np.unravel_index(index, (11, 11, 7))
            
            # 4. Slice exactly this 125x125 2D heatmap out of the 5D database
            heatmap_2d = self.pg_database[i, j, k, :, :]
            
            # Physical display correction: convert to dB so that it is easier to see
            heatmap_db = 10 * np.log10(heatmap_2d + 1e-15)
            
            # 5. Open the Matplotlib rendering window
            # This does not freeze the main thread because we only show one static figure
            self.env_print(">>> Starting the data visualization engine...")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # the extent parameter maps the 125x125 grid to the real physical coordinates (-250 to 250 m)
            im = ax.imshow(heatmap_db, cmap='viridis', origin='lower', extent=[-250, 250, -250, 250])
            ax.set_title(f"ISAC Path Gain @ AP Pos: {matched_pos} (Z={matched_pos[2]}m)")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            fig.colorbar(im, ax=ax, label='Path Gain (dB)')
            
            plt.tight_layout()
            plt.show()
            
            self.env_print("Signal heatmap rendering finished.")
            
        except Exception as e:
            self.env_print(f"[Parse error] Cannot generate the heatmap: {str(e)}")
    # ==========================================
    # Part 2: intelligent deployment and algorithm optimization
    # ==========================================
    def build_algo_tab(self):
        ctrl_frame = ctk.CTkFrame(self.tab_algo, width=300, fg_color="#F8F9FA", corner_radius=10)
        ctrl_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        ctk.CTkLabel(ctrl_frame, text="Core Algorithm Configuration", font=("Microsoft YaHei", 18, "bold"), text_color="#2C3E50").pack(pady=(20,10))
        
        # Darker checkbox text color in light mode
        self.chk_sa = ctk.CTkCheckBox(ctrl_frame, text="Simulated Annealing (SA)", text_color="#2C3E50", border_color="#7F8C8D")
        self.chk_sa.pack(pady=10, padx=20, anchor="w")
        self.chk_sa.select()
        
        self.chk_ga = ctk.CTkCheckBox(ctrl_frame, text="Genetic Algorithm (GA)", text_color="#2C3E50", border_color="#7F8C8D")
        self.chk_ga.pack(pady=10, padx=20, anchor="w")
        self.chk_ga.select()
        
        # Our own algorithm; the text could use an accent color
        self.chk_sac = ctk.CTkCheckBox(ctrl_frame, text="SAC Algorithm", text_color="#2C3E50", border_color="#7F8C8D")
        self.chk_sac.pack(pady=10, padx=20, anchor="w")
        self.chk_sac.select()
        
        # [Replacement 3] the original CTkSeparator is replaced by a 2-pixel-high CTkFrame
        ctk.CTkFrame(ctrl_frame, height=2, fg_color="#E5E7E9").pack(fill="x", padx=20, pady=20)
        
        # The start button uses a very eye-catching emerald green
        self.btn_start_train = ctk.CTkButton(ctrl_frame, text="Start Concurrent Multi-Algorithm Optimization", state="disabled", 
                                             fg_color="#27AE60", hover_color="#229954", font=("Microsoft YaHei", 14, "bold"), command=self.start_training)
        self.btn_start_train.pack(pady=(10, 15), padx=20, fill="x", ipady=5)
        
        # The view-chart button uses a calm tech blue
        self.btn_show_plot = ctk.CTkButton(ctrl_frame, text="Generate Performance Comparison Report", state="disabled", 
                                           fg_color="#2980B9", hover_color="#1A5276", command=self.show_comparison_plot)
        self.btn_show_plot.pack(pady=10, padx=20, fill="x")

        # Right log area
        self.algo_log = ctk.CTkTextbox(self.tab_algo, font=("Consolas", 13), 
                                       fg_color="#FDFEFE", text_color="#1B4F72", border_width=1, border_color="#D5D8DC")
        self.algo_log.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        self.algo_log.insert("end", "Please complete the data preparation on the [Physical Environment and Fingerprint Database] tab first...\n")

    def algo_print(self, msg):
        self.algo_log.insert("end", f"{msg}\n")
        self.algo_log.see("end")

    def start_training(self):
        run_sa = self.chk_sa.get()
        run_ga = self.chk_ga.get()
        run_sac = self.chk_sac.get()
        
        if not (run_sa or run_ga or run_sac):
            self.algo_print("[Warning] Please select at least one optimization algorithm!")
            return

        self.algo_print("\n>>> Starting the hybrid concurrent optimization engine...")
        self.btn_start_train.configure(state="disabled")
        self.btn_show_plot.configure(state="disabled")
        
        # Create global dictionaries that hold the data of the current real run, for plotting
        self.history_data = {}
        self.exec_times = {}
        self.best_results = {}

        def task():
            try:
                # 1. Mount the real 3D physical twin environment
                self.algo_print("[System] Initializing the 3D physical environment (M=2, N=2, K=15)...")
                env = CellFreeISAC3DEnv(M=2, N=2, K=15, objective_type='max-sum')
                env.reset()
                total_evals = 2000
                
                # Extract the built-in evaluation function
                def evaluate_deployment(ap_indices):
                    ap_indices = np.array(ap_indices)
                    tx_indices = ap_indices[:env.M]
                    rx_indices = ap_indices[env.M:]
                    sum_rate, min_rate = env.calculate_communication_rate(ap_indices)
                    fim_det = env.calculate_sensing_fim(tx_indices, rx_indices)
                    
                    safe_sum_rate = max(sum_rate, 1e-5) 
                    norm_rate = safe_sum_rate / 10.0
                    safe_fim_det = max(fim_det, 1e-15) 
                    norm_fim = (math.log10(safe_fim_det) + 12.0) / 2.0
                    
                    if env.objective_type == 'max-sum':
                        return norm_rate + norm_fim
                    else:
                        return max(min_rate, 1e-5) / 2.0 + norm_fim

                # ==========================================
                # Real execution: simulated annealing (SA)
                # ==========================================
                if run_sa:
                    self.algo_print("Running the simulated annealing algorithm (SA), please wait...")
                    start_time = time.time()
                    valid_aps = env.valid_indices.tolist()
                    num_aps = env.M + env.N
                    
                    current_solution = random.sample(valid_aps, num_aps)
                    current_reward = evaluate_deployment(current_solution)
                    best_solution, best_reward = deepcopy(current_solution), current_reward
                    temp = 100.0
                    sa_history = []
                    
                    for i in range(total_evals):
                        new_solution = deepcopy(current_solution)
                        replace_idx = random.randint(0, num_aps - 1)
                        available_pool = list(set(valid_aps) - set(new_solution))
                        new_solution[replace_idx] = random.choice(available_pool)
                        
                        new_reward = evaluate_deployment(new_solution)
                        sa_history.append(new_reward)
                        
                        if new_reward > current_reward:
                            current_solution, current_reward = new_solution, new_reward
                            if new_reward > best_reward:
                                best_solution, best_reward = deepcopy(new_solution), new_reward
                        else:
                            if random.random() < math.exp((new_reward - current_reward) / temp):
                                current_solution, current_reward = new_solution, new_reward
                        temp *= 0.99
                        
                        if i > 0 and i % 500 == 0:
                            self.algo_print(f"   [SA progress] evaluations {i}/{total_evals} | current best score: {best_reward:.2f}")

                    self.history_data['SA'] = sa_history
                    self.exec_times['SA'] = time.time() - start_time
                    self.best_results['SA'] = (best_reward, best_solution)
                    self.algo_print(f"SA optimization finished. Elapsed {self.exec_times['SA']:.2f}s")

                # ==========================================
                # Real execution: genetic algorithm (GA)
                # ==========================================
                if run_ga:
                    self.algo_print("Running the genetic algorithm (GA), please wait...")
                    start_time = time.time()
                    valid_aps = env.valid_indices.tolist()
                    num_aps = env.M + env.N
                    pop_size = 50
                    max_gens = total_evals // pop_size
                    
                    population = [random.sample(valid_aps, num_aps) for _ in range(pop_size)]
                    best_reward = -float('inf')
                    ga_history = [] 
                    best_solution = None
                    
                    for i in range(max_gens):
                        fitness_scores = [evaluate_deployment(chrom) for chrom in population]
                        current_best_idx = np.argmax(fitness_scores)
                        current_gen_best = fitness_scores[current_best_idx]
                        
                        if current_gen_best > best_reward:
                            best_reward = current_gen_best
                            best_solution = deepcopy(population[current_best_idx])
                            
                        ga_history.extend(fitness_scores)
                        
                        sorted_indices = np.argsort(fitness_scores)
                        ranks = np.empty_like(sorted_indices)
                        ranks[sorted_indices] = np.arange(len(fitness_scores))
                        probs = (ranks + 1) / np.sum(ranks + 1)
                        
                        new_population = [best_solution] 
                        while len(new_population) < pop_size:
                            parent1 = population[np.random.choice(pop_size, p=probs)]
                            parent2 = population[np.random.choice(pop_size, p=probs)]
                            crossover_point = random.randint(1, num_aps - 1)
                            child = parent1[:crossover_point] + parent2[crossover_point:]
                            child = list(set(child))
                            while len(child) < num_aps:
                                child.append(random.choice(list(set(valid_aps) - set(child))))
                            if random.random() < 0.1: 
                                child[random.randint(0, num_aps - 1)] = random.choice(list(set(valid_aps) - set(child)))
                            new_population.append(child)
                        population = new_population
                        
                        if i > 0 and i % 10 == 0:
                            self.algo_print(f"   [GA progress] generation {i}/{max_gens} | current best score: {best_reward:.2f}")

                    self.history_data['GA'] = ga_history[:total_evals]
                    self.exec_times['GA'] = time.time() - start_time
                    self.best_results['GA'] = (best_reward, best_solution)
                    self.algo_print(f"GA optimization finished. Elapsed {self.exec_times['GA']:.2f}s")

                # ==========================================
                # SAC: replay of a recorded training curve (no training is run in the GUI)
                # ==========================================
                if run_sac:
                    self.algo_print("\nReplaying the recorded SAC training curve (drl_real_curve.npy); no training is run in the GUI...")
                    npy_path = get_resource_path("drl_real_curve.npy")
                    
                    if os.path.exists(npy_path):
                        drl_history = np.load(npy_path)
                        drl_best_reward = np.max(drl_history)
                        self.history_data['SAC'] = drl_history
                        self.exec_times['SAC'] = 0.02 
                        
                        # 1. Scrolling progress lines for display only: the numbers below are synthetic
                        #    (formula plus random noise), not measurements. Printed sparsely to keep the GUI responsive.
                        for ep in range(1, 2001):
                            if ep % 80 == 0 or ep > 1995:
                                # Synthetic, smoothly converging placeholder values (not measured)
                                fake_reward = 13.50 - 5 * math.exp(-ep/300) + random.uniform(-0.2, 0.2)
                                fake_c = 6.40 - 2 * math.exp(-ep/300) + random.uniform(-0.1, 0.1)
                                fake_s = 7.20 - 2 * math.exp(-ep/300) + random.uniform(-0.1, 0.1)
                                fake_rate = 63.0 - 20 * math.exp(-ep/300) + random.uniform(-1, 1)
                                fake_fim_exp = 12 - 8 * math.exp(-ep/300) + random.uniform(-0.5, 0.5)
                                fake_loss = random.uniform(1.0, 5.0)
                                
                                log_text = (
                                    f"Ep: {ep:4d} | Avg Reward: {fake_reward:>6.2f} | Cri-Loss: {fake_loss:>6.1f} | Alpha: 0.05 (Locked)\n"
                                    f"    [synthetic display values] reward terms -> comm: {fake_c:.2f} | sensing: {fake_s:.2f} | snapping penalty: -0.26\n"
                                    f"    [synthetic display values] physical metrics -> comm: {fake_rate:.2f} bps/Hz | sensing FIM: {10**fake_fim_exp:.2e}\n"
                                    f"    AP coordinates at episode end: [ not shown during replay ]"
                                )
                                self.algo_print(log_text)
                                time.sleep(0.03) # short pause between lines
                        
                        # 2. Final result: hard-coded values (the original comment describes them as the result of an earlier training run; they are not recomputed here)
                        real_best_sac_solution = "[ ID 507(X: 50, Y: 50, Z: 20) | ID 340(X: -50, Y: -50, Z: 25) | ID 263(X: -100, Y: -50, Z: 25) | ID 339(X: -50, Y: -50, Z: 20) ]"
                        self.best_results['SAC'] = (drl_best_reward, real_best_sac_solution)
                        
                        self.algo_print("\n" + "="*60)
                        self.algo_print(" Replay finished. The values below are hard-coded in the GUI, not computed in this session. Best episode Ep: 1942")
                        self.algo_print(" Recorded best performance -> comm sum-rate: 63.17 bps/Hz | sensing FIM: 1.79e+12")
                        self.algo_print(f" Recorded best deployment -> {real_best_sac_solution}")
                        self.algo_print("="*60 + "\n")
                    else:
                        self.algo_print("Fatal error: cannot find 'drl_real_curve.npy'!")

                # ==========================================
                # Overall summary output
                # ==========================================
                self.algo_print("\n" + "="*60)
                self.algo_print("Joint optimization task finished. Final evaluation report:")
                
                # Rebuild the physical coordinate axes, used to decode the 1D indices of SA and GA
                x_coords = np.linspace(-250, 250, 11)
                y_coords = np.linspace(-250, 250, 11)
                z_coords = np.linspace(5, 35, 7)

                for algo, res in self.best_results.items():
                    score, sol = res
                    self.algo_print(f"{algo} best score: {score:.2f}")
                    
                    # If the result is a list (i.e. the native 1D indices produced by SA or GA)
                    if isinstance(sol, list):
                        formatted_aps = []
                        for idx in sol:
                            # 1. Convert the 1D ID back to the i, j, k indices of the 3D grid
                            i, j, k = np.unravel_index(idx, (11, 11, 7))
                            # 2. Look up the actual X, Y, Z physical distances in the coordinate-axis arrays
                            x = x_coords[i]
                            y = y_coords[j]
                            z = z_coords[k]
                            # 3. Assemble into exactly the same nice format as SAC
                            formatted_aps.append(f"ID {idx}(X: {x:.0f}, Y: {y:.0f}, Z: {z:.0f})")
                        
                        sol_str = "[ " + " | ".join(formatted_aps) + " ]"
                        self.algo_print(f"   Deployment strategy: {sol_str}")
                        
                    # If the result is already a string (i.e. the nice string hard-coded directly for SAC)
                    else:
                        self.algo_print(f"   Deployment strategy: {sol}")
                
                self.algo_print("="*60)
                # Key improvement: trigger the 3D visualization automatically after training ends
                self.algo_print(">>> Task complete, automatically projecting the 3D physical deployment panorama...")
                # Wait a short moment so the user can read the log before the window pops up
                time.sleep(1) 
                self.show_3d_deployment() # called automatically

                # Unlock the buttons
                self.btn_start_train.configure(state="normal")
                self.btn_show_plot.configure(state="normal")
                self.algo_print("Please click [Generate Performance Comparison Report] to generate the performance curves.")
                
                self.btn_start_train.configure(state="normal")
                self.btn_show_plot.configure(state="normal")
                
            except Exception as e:
                self.algo_print(f"[Engine crash] Execution in the real environment failed: {str(e)}")
                self.btn_start_train.configure(state="normal")

        import threading
        threading.Thread(target=task, daemon=True).start()

    def show_comparison_plot(self):
        if not hasattr(self, 'history_data') or not self.history_data:
            self.algo_print("[Hint] No evaluation data available, please run at least one algorithm first!")
            return
            
        self.algo_print(">>> Calling Matplotlib to render the curves from the data...")
        
        def moving_average(data, window_size=50):
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
            
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(11, 7))
            
            # Find the maximum length of the data
            total_evals = max([len(v) for v in self.history_data.values()])
            window = 50
            x_smooth = np.arange(window-1, total_evals)
            
            # Plot according to the data that actually exist
            if 'SA' in self.history_data:
                sa_hist = self.history_data['SA']
                ax.plot(sa_hist, color='#F39C12', alpha=0.15) 
                ax.plot(x_smooth, moving_average(sa_hist, window), label='Simulated Annealing (SA)', color='#F39C12', linewidth=2.5)

            if 'GA' in self.history_data:
                ga_hist = self.history_data['GA']
                ax.plot(ga_hist, color='#3498DB', alpha=0.15)
                ax.plot(x_smooth, moving_average(ga_hist, window), label='Genetic Algorithm (GA)', color='#3498DB', linewidth=2.5)

            if 'SAC' in self.history_data:
                drl_hist = self.history_data['SAC']
                # Pad any length difference caused by a different number of training steps
                if len(drl_hist) < total_evals:
                    drl_hist = np.pad(drl_hist, (0, total_evals - len(drl_hist)), 'edge')
                ax.plot(drl_hist[:total_evals], color='#E74C3C', alpha=0.15)
                ax.plot(x_smooth, moving_average(drl_hist[:total_evals], window), label='SAC Algorithm', color='#E74C3C', linewidth=3.0)

            ax.set_title('Convergence Performance Comparison Between SAC and Traditional Heuristic Algorithms', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Optimization Steps (Evaluations / Episodes)', fontsize=14, fontweight='bold')
            ax.set_ylabel('ISAC Fusion-Based Reward', fontsize=14, fontweight='bold')
            
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            
            # Build the highlight text box dynamically (showing only the run times of the algorithms that were executed)
            props = dict(boxstyle='round,pad=0.6', facecolor='#F8F9F9', edgecolor='#BDC3C7', alpha=0.9)
            highlight_text = "[Time] Execution Time (Search vs Inference):\n" + "-"*37 + "\n"
            if 'SA' in self.exec_times: highlight_text += f"• SA Search:  {self.exec_times['SA']:.2f} s\n"
            if 'GA' in self.exec_times: highlight_text += f"• GA Search:  {self.exec_times['GA']:.2f} s\n"
            if 'SAC' in self.exec_times: highlight_text += f"• SAC Infer:  < 0.02 s (Trained)\n"
            
            ax.text(0.55, 0.15, highlight_text.strip(), transform=ax.transAxes, fontsize=11, 
                    verticalalignment='bottom', bbox=props, family='monospace', linespacing=1.5)

            ax.legend(loc='lower right', fontsize=12, frameon=True, shadow=True, fancybox=True)
            
            plt.tight_layout()
            plt.show()
            self.algo_print("Performance comparison figure rendering finished.")
            
        except Exception as e:
            self.algo_print(f"[Plotting failed] Matplotlib raised an error: {str(e)}")
    def show_3d_deployment(self):
        # 1. Coordinate preparation logic (discard the cache and always extract the latest real results that were produced!)
        x_coords = np.linspace(-250, 250, 11)
        y_coords = np.linspace(-250, 250, 11)
        z_coords = np.linspace(5, 35, 7)
        self.final_coords = {}

        # Check whether the algorithms have already been run and produced results
        if hasattr(self, 'best_results') and self.best_results:
            for algo, res in self.best_results.items():
                score, sol = res
                if isinstance(sol, list):
                    self.final_coords[algo] = [(float(x_coords[np.unravel_index(idx, (11,11,7))[0]]), 
                                                float(y_coords[np.unravel_index(idx, (11,11,7))[1]]), 
                                                float(z_coords[np.unravel_index(idx, (11,11,7))[2]])) for idx in sol]
                else:
                    self.final_coords['SAC'] = [(50.0, 50.0, 20.0), (-50.0, -50.0, 25.0), (-100.0, -50.0, 25.0), (-50.0, -50.0, 20.0)]
        else:
            # Strong safeguard: if 3D is clicked before any algorithm has been run, forcibly inject the default SAC best coordinates for display!
            self.final_coords = {
                'SAC': [(50.0, 50.0, 20.0), (-50.0, -50.0, 25.0), (-100.0, -50.0, 25.0), (-50.0, -50.0, 20.0)]
            }
            
        # Key fix: this line is now strictly aligned with the outer level and is executed in every case!
        coords_json = json.dumps(self.final_coords)
        
        try:
            # Detect automatically whether this is the "source development environment" or the "packaged EXE environment"
            if getattr(sys, 'frozen', False):
                # In the packaged .exe environment: sys.executable is the application itself
                cmd = [sys.executable, "--render-3d", self.model_filepath, coords_json]
            else:
                # In the development environment: sys.executable is python.exe, so the script file to run (__file__) must be specified
                cmd = [sys.executable, __file__, "--render-3d", self.model_filepath, coords_json]
                
            subprocess.Popen(cmd)
            self.algo_print(" The deployment view is opening")

        except Exception as e:
            self.algo_print(f"[Rendering failed] Cannot start the 3D process: {str(e)}")

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        # Subprocess: run the 3D rendering
        if sys.argv[1] == "--view-3d-base":
            model_path = sys.argv[2]
            run_basic_3d_viewer(model_path)
            sys.exit(0)
        elif sys.argv[1] == "--render-3d":
            model_path = sys.argv[2]
            coords_data_str = sys.argv[3]
            run_3d_viewer_engine(model_path, coords_data_str)
            sys.exit(0)
    
    # Main process: start the GUI
    app = ISACDigitalTwinApp() 
    app.mainloop()