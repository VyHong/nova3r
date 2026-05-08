import sys
import os

from demo_nova3r import render_360_video, save_pointcloud

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import numpy as np
import open3d as o3d
import time

import PIL.Image
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from nova3r.models.nova3r_img_cond import Nova3rImgCond
from nova3r.models.nova3r_pts_cond import Nova3rPtsCond  # noqa: F401 — needed by load_model's eval()
from nova3r.inference import inference_nova3r

def generate_example(cfg,model,images,num_queries,log_dir,device,current_epoch): 
    # Set inference defaults if not in the saved config
    OmegaConf.set_struct(cfg, False)
    if "fm_step_size" not in cfg:
        cfg.fm_step_size = 0.04
    if "fm_sampling" not in cfg:
        cfg.fm_sampling = "euler"
    OmegaConf.set_struct(cfg, True)

    # For 2-view input, don't symmetrize — run one pass with both views
    pairs = [tuple(images)]

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    output = inference_nova3r(
        cfg,
        pairs,
        model,
        device,
        batch_size=1,
        num_queries=num_queries,
        n_views=len(images),
        method=cfg.get("fm_sampling", "euler"),
    )

    elapsed = time.time() - start_time
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Inference: {elapsed:.2f}s | Peak memory: {peak_mem:.0f} MB")
    else:
        print(f"Inference: {elapsed:.2f}s")

    pts3d = output["pred"]["pts3d_xyz"]

    save_dir = os.path.join(log_dir, "val_points")
    os.makedirs(save_dir, exist_ok=True)
    scene_dir = os.path.join(save_dir, f"epoch_{current_epoch}")
    os.makedirs(scene_dir, exist_ok=True)

    ply_path = save_pointcloud(pts3d, scene_dir)
    render_360_video(ply_path, scene_dir)

    # if len(args.images) == 2:

    print("Done!")