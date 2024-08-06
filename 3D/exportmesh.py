import trimesh
import trimesh.exchange.export

def convert_ply_to_fbx(input_path, output_path):
    """
    Convert a 3D model file from PLY format to FBX format.

    Parameters:
    input_path (str): The file path to the input PLY file.
    output_path (str): The file path where the output FBX file will be saved.
    """
    # Load the PLY file into a Trimesh object.
    mesh = trimesh.load(input_path)
    
    # Open the specified output path as a writable binary file and export the mesh as an FBX file.
    with open(output_path, 'wb') as f:
        # Export the loaded mesh as an FBX file and write it to the output file.
        f.write(trimesh.exchange.export.export_mesh(mesh, file_type='fbx'))


# Specify the input and output file paths
input_path = "/home/s5639776/MasterProject/test/ImmersiveDepth/example_data/t3/t3_ref_geometry.ply"  # Path to the input .ply file
output_path = "/home/s5639776/MasterProject/test/ImmersiveDepth/example_data/t3/output.fbx"         # Path to save the output .fbx file

# Call the conversion function to convert the .ply file to an .fbx file
convert_ply_to_fbx(input_path, output_path)
