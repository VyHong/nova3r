import argparse
import os
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

# Adjust the import based on your real module path
from training.data.datasets.replica_utils.igibson_utils import ReplicaPanoScene

def crop_with_open3d(pcd, trimesh_mesh, padding=(0.07, 0.07, 0.07, 0.07, 0.07, 0.07)):
    """
    Crops a 3D point cloud keeping points inside the mesh, plus an asymmetric 
    padding buffer outside across the 6 cardinal directions.
    
    padding format: (pad_x_neg, pad_x_pos, pad_y_neg, pad_y_pos, pad_z_neg, pad_z_pos)
    """
    # 1. Convert trimesh to Open3D Tensor mesh
    vertices = o3d.core.Tensor(np.array(trimesh_mesh.vertices, dtype=np.float32))
    triangles = o3d.core.Tensor(np.array(trimesh_mesh.faces, dtype=np.int32))
    o3d_mesh = o3d.t.geometry.TriangleMesh(vertices, triangles)

    # 2. Build Raycasting Scene
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d_mesh)

    # 3. Compute signed distance and closest surface points
    points_tensor = o3d.core.Tensor(np.asarray(pcd.points, dtype=np.float32))

    # Open3D convention: Negative is inside, Positive is outside
    signed_distances = scene.compute_signed_distance(points_tensor).numpy()

    # Get the closest point on the mesh for each query point
    closest_points_info = scene.compute_closest_points(points_tensor)
    closest_points = closest_points_info['points'].numpy()
    points_np = np.asarray(pcd.points, dtype=np.float32)

    # Calculate vector from the closest surface point TO the query point (outward direction)
    v = points_np - closest_points
    dx, dy, dz = v[:, 0], v[:, 1], v[:, 2]

    px_neg, px_pos, py_neg, py_pos, pz_neg, pz_pos = padding

    # 4. Create filtering masks
    # Create an absolute bounding box from the mesh bounds and padding to robustly handle negative padding (shrinking)
    min_mesh = o3d_mesh.get_min_bound().numpy()
    max_mesh = o3d_mesh.get_max_bound().numpy()
    
    bbox_mask = (points_np[:, 0] >= min_mesh[0] - px_neg) & (points_np[:, 0] <= max_mesh[0] + px_pos) & \
                (points_np[:, 1] >= min_mesh[1] - py_neg) & (points_np[:, 1] <= max_mesh[1] + py_pos) & \
                (points_np[:, 2] >= min_mesh[2] - pz_neg) & (points_np[:, 2] <= max_mesh[2] + pz_pos)

    # Keep everything that is strictly inside the mesh
    inside_mask = signed_distances <= 0

    # For points outside, verify their outward vector respects the 6-way boundaries.
    # To prevent precision issues or cross-axis dropouts when padding is negative, we limit it to a tiny epsilon.
    # Actual negative cropping is safely and fully enforced by the bbox_mask.
    eps = 1e-4
    px_nr, px_pr = max(eps, px_neg), max(eps, px_pos)
    py_nr, py_pr = max(eps, py_neg), max(eps, py_pos)
    pz_nr, pz_pr = max(eps, pz_neg), max(eps, pz_pos)

    outside_mask = (signed_distances > 0) & \
                   (dx >= -px_nr) & (dx <= px_pr) & \
                   (dy >= -py_nr) & (dy <= py_pr) & \
                   (dz >= -pz_nr) & (dz <= pz_pr)

    mask = (inside_mask | outside_mask) & bbox_mask
    
    # 5. Filter the point cloud
    indices = np.nonzero(mask)[0]
    cropped_pcd = pcd.select_by_index(indices)
    
    return cropped_pcd

def main():
    parser = argparse.ArgumentParser(description="Crop rooms from a specific apartment scene.")
    
    parser.add_argument("--scene_dir", type=str, required=True, 
                        help="Path to a specific scene directory (e.g. /tmp/datasets/replica_pano/large_apartment_1)")
    
    # New argument for output directory
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Optional. Path to save the cropped point clouds. If not provided, saves in the respective room folders.")
    
    # 6-Directional padding arguments
    parser.add_argument("--pad_x_neg", type=float, default=0.07, help="Padding in the -X direction")
    parser.add_argument("--pad_x_pos", type=float, default=0.07, help="Padding in the +X direction")
    parser.add_argument("--pad_y_neg", type=float, default=0.07, help="Padding in the -Y direction")
    parser.add_argument("--pad_y_pos", type=float, default=0.07, help="Padding in the +Y direction")
    parser.add_argument("--pad_z_neg", type=float, default=0.07, help="Padding in the -Z direction")
    parser.add_argument("--pad_z_pos", type=float, default=0.07, help="Padding in the +Z direction")
    
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)

    if not scene_dir.exists():
        print(f"Directory {scene_dir} does not exist!")
        return

    # Handle output directory creation
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory set to: {output_dir}")

    scene_name = scene_dir.name
    
    # Deduce base name for the aligned point cloud
    if scene_name.startswith("large_apartment_") or scene_name.startswith("hotel_"):
        base_name = scene_name[:-3] # e.g. "large_apartment_1" -> "large_apartment_"
    else:
        base_name = f"{scene_name}_"
        
    world_pcd_path = scene_dir / f"{base_name}aligned.ply"
    
    if not world_pcd_path.exists():
        print(f"Error: Full point cloud {world_pcd_path} not found.")
        return
        
    print(f"\nProcessing {scene_name}...")
    scene_pcd = o3d.io.read_point_cloud(str(world_pcd_path))
    print(f"Loaded full point cloud from {world_pcd_path} with {len(scene_pcd.points)} points.")

    scene_info_dir = scene_dir / "Scene_Info"
    if not scene_info_dir.exists():
        print(f"No Scene_Info found for {scene_name}.")
        return

    # Tuple grouping the 6 directions
    padding_tuple = (args.pad_x_neg, args.pad_x_pos, 
                     args.pad_y_neg, args.pad_y_pos, 
                     args.pad_z_neg, args.pad_z_pos)

    print(f"Using 6-directional padding bounds:\n"
          f"X: [ -{padding_tuple[0]} , +{padding_tuple[1]} ]\n"
          f"Y: [ -{padding_tuple[2]} , +{padding_tuple[3]} ]\n"
          f"Z: [ -{padding_tuple[4]} , +{padding_tuple[5]} ]")

    # Look for each sequence entry folder (i.e. rooms)
    i = 0
    for room_folder in tqdm(list(scene_info_dir.iterdir()), desc=f"Rooms in {scene_name}"):
        if i == 1:
            if not room_folder.is_dir():
                continue
                
            pkl_path = room_folder / "data.pkl"
            if not pkl_path.exists():
                continue

            try:
                # Load metadata & layout
                replica_scene = ReplicaPanoScene.from_pickle(str(pkl_path))
                layout_mesh = replica_scene.save_layout_mesh(to_world_space=True)

                # Perform quick crop
                cropped_pcd = crop_with_open3d(scene_pcd, layout_mesh, padding=padding_tuple)

                # Determine where to save the point cloud
                file_name = f"{scene_name[:-4]}_cropped.ply"
                if output_dir:
                    dest_pcd_path = output_dir / file_name
                else:
                    dest_pcd_path = room_folder / file_name
                    
                o3d.io.write_point_cloud(str(dest_pcd_path), cropped_pcd)
               

            except Exception as e:
                print(f"Failed to process {room_folder.name}: {e}")
        i += 1

    print(f"\nCropping finished for {scene_name}!")

if __name__ == "__main__":
    main()