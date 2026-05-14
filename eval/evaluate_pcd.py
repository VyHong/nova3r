import argparse
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from eval.mv_recon.metric import SSI3DScore_Scene_Multi
import torch

def main(args):
    try:
        pcd_gt = o3d.io.read_point_cloud(args.gt_ply)
    except Exception as e:
        print(f"Error loading GT {args.gt_ply}: {e}")
        return

    try:
        pcd_pred = o3d.io.read_point_cloud(args.pred_ply)
    except Exception as e:
        print(f"Error loading Pred {args.pred_ply}: {e}")
        return

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
        print("SSI3D Score:", score)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return

    # Append to JSON if output file is specified
    if args.output_file:
        if os.path.exists(args.output_file):
            with open(args.output_file, "r") as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError:
                    results = {}
        else:
            results = {}

        if args.scene_id not in results:
            results[args.scene_id] = []
        results[args.scene_id].append({k: v for k, v in score.items()})

        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_ply", type=str, required=True, help="Path to ground truth point cloud (.ply)")
    parser.add_argument("--pred_ply", type=str, required=True, help="Path to predicted point cloud (.ply)")
    parser.add_argument("--output_file", type=str, default="", help="Output json file to append results")
    parser.add_argument("--scene_id", type=str, default="unknown_scene", help="Scene identifier for JSON structuring")
    args = parser.parse_args()
    main(args)