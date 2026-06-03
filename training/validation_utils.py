import sys
import os

from demo_nova3r import render_360_video, save_pointcloud
from eval.mv_recon.metric import scale_shift_alignment_chamfer
from nova3r.inference import get_all_pts3d, inference_nova3r

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import time
import numpy as np
from omegaconf import OmegaConf
from training.training_utils import get_all_pts3d, normalize_input
from training.training_utils import save_points_ply


def normalize_pointclouds(cfg, batch):
    if "query_source" in cfg.model.params.cfg.pts3d_head.params:
        query_src = cfg.model.params.cfg.pts3d_head.params.query_source
    else:
        query_src = "src_complete"

    if "target_source" in cfg.model.params.cfg.pts3d_head.params:
        target_src = cfg.model.params.cfg.pts3d_head.params.target_source
    else:
        target_src = "src_complete"

    if "down_resolution" in cfg.model.params.cfg.pts3d_head.params:
        down_resolution = cfg.model.params.cfg.pts3d_head.params.down_resolution
    else:
        down_resolution = 224

    if "norm_mode" in cfg.model.params.cfg.pts3d_head.params:
        norm_mode = cfg.model.params.cfg.pts3d_head.params.norm_mode
    else:
        norm_mode = "none"

    pts3d_src, valid_src = get_all_pts3d(batch, mode=query_src, down_resolution=down_resolution)
    pts3d_trg, valid_trg = get_all_pts3d(batch, mode=target_src, down_resolution=down_resolution)

    if "hunyuan" in query_src:
        pts3d_src_norm, normals_src = pts3d_src[..., :3], pts3d_src[..., 3:]
        pts3d_trg_norm, normals_trg = pts3d_trg[..., :3], pts3d_trg[..., 3:]
    else:
        pts3d_src_norm, pts3d_trg_norm = normalize_input(pts3d_src, valid_src, pts3d_trg, valid_trg, mode=norm_mode)

    # save_points_ply(pts3d_trg_norm[0], f"debug_points/points_trg.ply")

    return (
        pts3d_src_norm,
        valid_src,
        pts3d_trg_norm,
        valid_trg,
        normals_src if "hunyuan" in query_src else None,
        normals_trg if "hunyuan" in target_src else None,
    )


def generate_pointcloud(cfg, model, batch, num_queries, device, stage="val"):
    # Set inference defaults if not in the saved config
    OmegaConf.set_struct(cfg, False)
    if "fm_step_size" not in cfg:
        cfg.fm_step_size = 0.04
    if "fm_sampling" not in cfg:
        cfg.fm_sampling = "euler"
    OmegaConf.set_struct(cfg, True)

    images = batch["images"]
    pairs = [tuple(images)]
    batch_size = images[0].shape[0]

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if stage == "val":
        pointmaps = batch["pts3d_src_norm"] if "pts3d_src_norm" in batch else None
        if batch["normals_src"] is not None:
            pointmaps = torch.cat([pointmaps, batch["normals_src"]], dim=-1)

    if stage == "test":
        pointmaps = batch["cam_points"]
        if len(pointmaps) > num_queries:
            indices = np.random.choice(len(pointmaps), num_queries, replace=False)
            pointmaps = pointmaps[indices]

    output = inference_nova3r(
        cfg,
        pairs,
        model,
        device,
        batch=batch,
        batch_size=batch_size,
        num_queries=num_queries,
        n_views=len(images),
        method=cfg.get("fm_sampling", "euler"),
        pointmaps=pointmaps,
    )

    elapsed = time.time() - start_time
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Inference: {elapsed:.2f}s | Peak memory: {peak_mem:.0f} MB")
    else:
        print(f"Inference: {elapsed:.2f}s")
    pts3d = output["pred"]["pts3d_xyz"]
    return pts3d


def generate_example(batch, pts3d, log_dir, current_epoch):
    batch_size = pts3d.shape[0]
    save_dir = os.path.join(log_dir, "val_points")
    os.makedirs(save_dir, exist_ok=True)
    scene_dir = os.path.join(save_dir, f"epoch_{current_epoch}")
    os.makedirs(scene_dir, exist_ok=True)
    for i in range(batch_size):
        sample_dir = os.path.join(scene_dir, f"{batch['seq_name'][i]}_{batch['id'][i]}")
        os.makedirs(sample_dir, exist_ok=True)
        ply_path = save_pointcloud(pts3d[i], sample_dir)
        render_360_video(ply_path, sample_dir)
    print("Done!")


def run_validation_loss(gt_pts3d, gt_valid, pts3d, criterion, device):

    pts3d = pts3d.to(device=device)
    pts3d_pred_eval, align_shift, align_scale = scale_shift_alignment_chamfer(pts3d, gt_pts3d, max_iterations=200, lr=0.01, return_transform=True)

    gt_list = {"target_pts3d": gt_pts3d, "target_valid": gt_valid}
    pred = {"pts3d_xyz": pts3d_pred_eval}

    loss, details = criterion(gt_list, pred)
    return loss, details


def run_test_score(batch, pts3d, criterion, device):

    B = batch["cam_points"].shape[0]
    eval_points = batch["test_points"] if "test_points" in batch else batch["cam_points"]
    gt_list = [{"pcd_eval": eval_points, "camera_pose": torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)}]

    pts3d = pts3d.to(device=device)
    pred = {"pts3d_xyz": pts3d}

    data, details = criterion(gt_list, pred)
    return data, details


def save_test_scores_to_csv(batch, details, log_dir):
    import pandas as pd
    import os
    import torch

    res = {}
    for k, v in details.items():
        if isinstance(v, torch.Tensor):
            res[k] = v.item()
        else:
            res[k] = v

    if "seq_name" in batch:
        res["seq_name"] = batch["seq_name"][0]
    if "id" in batch:
        res["id"] = str(batch["id"][0]) if isinstance(batch["id"][0], torch.Tensor) else batch["id"][0]

    save_path = f"{log_dir}/test_scores.csv" if log_dir else "test_scores.csv"
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    df = pd.DataFrame([res])
    if not os.path.exists(save_path):
        df.to_csv(save_path, index=False)
    else:
        df.to_csv(save_path, mode="a", header=False, index=False)

    return res
