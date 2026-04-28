import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from eval.mv_recon.metric import SSI3DScore_Scene_Multi

import torch

# Path to your .ply file
gt_ply_path = "/mnt/d/scannet/data/08bbbdcc3d/scans/mesh_aligned_0.05.ply"
ply_path = "/home/vy/project/panorama/code/nova3r_output_6/08bbbdcc3d/1_cube_120/0000/pointcloud.ply"

with open("../subset.json", "r") as f:
    subset = json.load(f)
results = {}
for scene in subset["scenes"]:
    gt_ply_path = os.path.join(scene, "scans/mesh_aligned_0.05.ply")
    pcd_gt = o3d.io.read_point_cloud(gt_ply_path)
    print("Ground Truth Point Cloud:")
    print(pcd_gt)
    scene_results = []
    for pcd_folders in os.listdir(
        os.path.join("/home/vy/project/panorama/code/nova3r_output_6", os.path.basename(scene))
    ):
        ply_path = os.path.join(
            "/home/vy/project/panorama/code/nova3r_output_6",
            os.path.basename(scene),
            pcd_folders,
            "0000/pointcloud.ply",
        )
        # Load point cloud
        pcd_pred = o3d.io.read_point_cloud(ply_path)

        print("\nPredicted Point Cloud:")
        print(pcd_pred)

        pts_gt = torch.from_numpy(np.asarray(pcd_gt.points)).float().unsqueeze(0).cuda()
        pts_pred = torch.from_numpy(np.asarray(pcd_pred.points)).float().unsqueeze(0).cuda()

        data = [{"pcd_eval": pts_gt, "pcd_eval_visible": pts_gt, "camera_pose": torch.eye(4).unsqueeze(0).cuda()}]
        pred = {"pts3d_xyz": pts_pred}

        test_criterion = SSI3DScore_Scene_Multi(
            num_eval_pts=8192,
            fs_thres=[0.1, 0.05, 0.02],
            pts_sampling_mode="uniform",
            alignment="none",
            use_cd_align=True,
        ).to("cuda")

        try:
            data, score = test_criterion(data, pred)
            print("SSI3D Score:")
            print(score)
        except Exception as e:
            import traceback

            traceback.print_exc()
        scene_results.append(score)
    results[scene] = scene_results

with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)
