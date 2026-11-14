import os
import sionna
import open3d as o3d

def view_model(ply_path):
    """Generic function for loading and rendering a 3D model"""
    print(f"Loading the mesh data with Open3D...\nFile path: {ply_path}")
    mesh = o3d.io.read_triangle_mesh(ply_path)
    
    if mesh.is_empty():
        print("[ERROR] Failed to load the model, or the model is empty!")
        return
        
    # Compute the vertex normals so that the buildings show realistic 3D shading under lighting
    mesh.compute_vertex_normals() 
    
    print("Opening the interactive 3D viewer...")
    print("-" * 30)
    print("[Operation guide]")
    print("   Left mouse button drag: rotate the view freely")
    print("   Shift + left mouse button drag: pan the map")
    print("   Mouse wheel: zoom in/out")
    print("   Press the 'Q' key or click the red close button: exit the viewer")
    print("-" * 30)
    
    # Launch the local GUI for rendering
    o3d.visualization.draw_geometries([mesh], 
                                      window_name="Sionna 3D City Viewer",
                                      width=1280, height=720,
                                      left=100, top=100)

def main():
    print("Locating the 3D model of Place de l'Etoile, Paris, built into Sionna...")
    
    # Get the absolute path of the XML file of Sionna's underlying Paris scene
    paris_xml_path = sionna.rt.scene.etoile
    
    # Get the path of the directory that contains this XML file
    scene_dir = os.path.dirname(paris_xml_path)
    print(f"Searching the directory for the actual 3D files: {scene_dir}")
    
    # Smart search mechanism: walk this directory and all its subdirectories looking for .ply or .obj files
    target_mesh = None
    for root, dirs, files in os.walk(scene_dir):
        for file in files:
            # Take the first file found in a supported 3D format
            if file.endswith('.ply') or file.endswith('.obj'):
                target_mesh = os.path.join(root, file)
                break
        if target_mesh:
            break
            
    if target_mesh:
        print(f"Found the hidden model file: {target_mesh}")
        view_model(target_mesh)
    else:
        print("[ERROR] No 3D source file was found anywhere in this directory!")

if __name__ == "__main__":
    main()