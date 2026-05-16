import argparse
import os
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

# Adjust the import based on your real module path
from training.data.datasets.replica_utils.igibson_utils import ReplicaPanoScene

def crop_with_open3d(pcd, trimesh_mesh, padding=0.06):
    """
    Crops a 3D point cloud to keep points inside a 3D mesh, plus a padding buffer outside.
    This method is significantly quicker than trimesh's signed_distance, 
    since it utilizes Open3D's C++ RaycastingScene (Embree accelerated).
    """
    # 1. Convert trimesh to Open3D Tensor mesh
    vertices = o3d.core.Tensor(np.array(trimesh_mesh.vertices, dtype=np.float32))
    triangles = o3d.core.Tensor(np.array(trimesh_mesh.faces, dtype=np.int32))
    o3d_mesh = o3d.t.geometry.TriangleMesh(vertices, triangles)

    # 2. Build Raycasting Scene
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d_mesh)

    # 3. Compute signed distance 
    # Open3D convention: Negative is inside, Positive is outside
    points_tensor = o3d.core.Tensor(np.asarray(pcd.points, dtype=np.float32))
    distances = scene.compute_signed_distance(points_tensor).numpy()

    # inside ( < 0 ) or within padding distance outside ( <= padding )
    mask = distances <= padding
    
    # 4. Filter the point cloud
    indices = np.nonzero(mask)[0]
    cropped_pcd = pcd.select_by_index(indices)
    
    return cropped_pcd

def main():
    parser = argparse.ArgumentParser(description="Offline script to crop rooms from large apartments.")
    parser.add_argument("--data_root", type=str, default="/tmp/datasets/replica_pano/", help="Path to ReplicaPano dataset root")
    parser.add_argument("--padding", type=float, default=0.06, help="Padding distance buffer")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if not data_root.exists():
        print(f"Directory {data_root} does not exist!")
        return

    # Find all large_apartment directories
    apartment_folders = [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("large_apartment")]

    if not apartment_folders:
        print("No large_apartment directories found.")
        return

    for apt_folder in apartment_folders:
        scene_name = apt_folder.name
        
        # Original aligned scene point cloud path
        base_name = scene_name[:-3] # e.g. "large_apartment_1" -> "large_apartment_"
        world_pcd_path = apt_folder / scene_name / f"{base_name}aligned.ply"
        
        if not world_pcd_path.exists():
            print(f"Skipping {scene_name}, {world_pcd_path} not found.")
            continue
            
        print(f"\nProcessing {scene_name}...")
        scene_pcd = o3d.io.read_point_cloud(str(world_pcd_path))
        print(f"Loaded full point cloud from {world_pcd_path} with {len(scene_pcd.points)} points.")

        scene_info_dir = apt_folder / scene_name / "Scene_Info"
        if not scene_info_dir.exists():
            print(f"No Scene_Info found for {scene_name}.")
            continue

        # Look for each sequence entry folder (i.e. rooms)
        for room_folder in tqdm(list(scene_info_dir.iterdir()), desc=f"Rooms in {scene_name}"):
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
                cropped_pcd = crop_with_open3d(scene_pcd, layout_mesh, padding=args.padding)

                # Save the cropped point cloud directly in the room folder
                #dest_pcd_path = room_folder / "cropped_points.ply"
                dest_pcd_path = room_folder / f"{scene_name}_{room_folder.name}_cropped.ply"
                o3d.io.write_point_cloud(str(dest_pcd_path), cropped_pcd)

            except Exception as e:
                print(f"Failed to process {room_folder.name}: {e}")

    print("\nCropping finished globally!")

if __name__ == "__main__":
    main()
