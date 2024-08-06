import open3d as o3d

def load_and_view_ply(file_path):
    """
    Load a PLY file and visualize it using Open3D.

    Parameters:
    file_path (str): The file path to the PLY file to be loaded and viewed.
    """
    # Load the PLY file as a point cloud object.
    pcd = o3d.io.read_point_cloud(file_path)

    # Check if the point cloud is not empty (i.e., the file was loaded successfully).
    if not pcd.is_empty():
        print("File loaded successfully.")
        
        # Visualize the loaded point cloud using Open3D's built-in viewer.
        o3d.visualization.draw_geometries([pcd])
    else:
        # Print an error message if the file failed to load.
        print("Failed to load the file.")

# Replace "path_to_your_file.ply" with the path to your actual PLY file.
load_and_view_ply("/home/s5639776/MasterProject/test/ImmersiveDepth/example_data/kitchen/image_ref_geometry.ply")
