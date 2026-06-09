import os
import time
from collections import defaultdict

import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_zeros
from dust3r.utils.geometry import geotrf, inv

# flow_matching
from nova3r.flow_matching.path.scheduler import CosineScheduler
from nova3r.flow_matching.path import AffineProbPath
from nova3r.utils.sampling import sampling_train_gen_target
from einops import rearrange

from nova3r.heads.hunyuan_model.surface_loaders_cuda import SharpEdgeSurfaceLoader

path = AffineProbPath(scheduler=CosineScheduler())
hunyuan_loader = SharpEdgeSurfaceLoader(
    num_sharp_points=5120,
    num_uniform_points=5120,
)


def save_points_ply(points, filename):
    pts = points.reshape(-1, 3).detach().cpu().numpy()
    header = f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    with open(filename, "w") as f:
        f.write(header)
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def save_batch_images(images, filename):
    try:
        from torchvision.utils import save_image

        # images is expected to be [B, V, C, H, W], we take the first batch item
        v_images = images[0]  # [V, C, H, W]
        save_image(v_images, filename, normalize=True)
    except Exception as e:
        print("Could not save images:", e)


def visualize_dx_t_v_pred_distribution(dx_t, v_pred, t, save_path=None, bins=100, density=True, show=False):
    """Visualize the value distribution for the first batch element of dx_t and v_pred.

    Parameters
    ----------
    dx_t : torch.Tensor
        Flow target tensor with shape [B, N, 3] or compatible.
    v_pred : torch.Tensor
        Predicted flow tensor with shape [B, N, 3] or compatible.
    save_path : str or None
        If provided, save the figure to this path.
    bins : int
        Histogram bin count.
    density : bool
        Plot normalized histograms when True.
    show : bool
        If True, display the figure. Defaults to False so plots are not opened automatically.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
    """
    dx_t_first = dx_t[0].detach().float().flatten().cpu().numpy()
    v_pred_first = v_pred[0].detach().float().flatten().cpu().numpy()
    t_first = t[0].detach().float().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(dx_t_first, bins=bins, density=density, alpha=0.55, label="dx_t")
    ax.hist(v_pred_first, bins=bins, density=density, alpha=0.55, label="v_pred")
    ax.set_title(f"Value distribution for first batch element at time {t_first:.2f}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if density else "Count")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if not show:
        plt.close(fig)

    return fig


def replace_with_sphere(B, point_num, device, radius=1.0):
    import math

    B_trg, N_trg, C_trg = B, point_num, 3
    g = torch.Generator(device=device)
    g.manual_seed(42)
    phi = torch.acos(1 - 2 * torch.rand(1, N_trg, generator=g, device=device))
    theta = 2 * math.pi * torch.rand(1, N_trg, generator=g, device=device)
    dummy_x = torch.sin(phi) * torch.cos(theta)
    dummy_y = torch.sin(phi) * torch.sin(theta)
    dummy_z = torch.cos(phi)
    simple_shape = torch.stack([dummy_x, dummy_y, dummy_z], dim=-1)  # [1, N, 3]
    pts3d_trg_norm = simple_shape.expand(B_trg, N_trg, C_trg).contiguous()
    return pts3d_trg_norm


def get_all_pts3d(gt_list, mode=None, down_resolution=112):
    """Extract and optionally downsample/FPS-sample ground truth 3D points from a batch."""
    if mode == "cube":
        pts_xyz = gt_list[0]["global_center_xyz"]

        valid = torch.ones_like(pts_xyz[..., 0]).bool()  # B, N
        in_camera1 = inv(gt_list[0]["camera_pose"])
        gt_pts = geotrf(in_camera1, pts_xyz)

    elif mode == "src_complete":
        gt_pts, valid = get_complete_pts3d(gt_list)

    elif "src_complete_fps" in mode:
        batch_size = int(mode.split("_")[-1])
        gt_pts, valid = get_complete_pts3d(gt_list)
        # run fps_fast
        gt_pts, valid = sampling_train_gen_target(gt_pts, valid, None, target_sampling="fps_fast", batch_size=batch_size)

    elif "src_complete_fps_edge" in mode:
        batch_size = int(mode.split("_")[-1])
        gt_pts, valid = get_complete_pts3d(gt_list)
        # run fps_fast
        gt_pts, valid = sampling_train_gen_target(gt_pts, valid, None, target_sampling="fps_edge_fast", batch_size=batch_size)

    elif "src_complete_hunyuan" in mode:
        gt_pts = hunyuan_loader(gt_list)
        valid = torch.ones_like(gt_pts[..., 0]).bool()  # B, N

        return gt_pts, valid

    elif mode == "cube_global":
        pts_xyz = gt_list[0]["global_center_xyz"]
        valid = torch.ones_like(pts_xyz[..., 0]).bool()  # B, N
        gt_pts = pts_xyz

    elif mode == "src_view":
        gt_pts_list = [gt["pts3d"] for gt in gt_list]

        in_camera1 = inv(gt_list[0]["camera_pose"])
        gt_pts_list = [geotrf(in_camera1, gt["pts3d"]) for gt in gt_list]

        gt_pts = torch.stack(gt_pts_list, dim=1)
        B, H, W, C = gt_pts_list[0].shape
        gt_pts = rearrange(gt_pts, "b s h w c -> (b s) c h w")
        gt_pts = F.interpolate(gt_pts, size=down_resolution, mode="nearest")

        gt_pts = rearrange(gt_pts, "(b s) c h w -> b (s h w) c", b=B)

        valid_list = [gt["valid_mask"].clone() for gt in gt_list]
        valid = torch.stack(valid_list, dim=1).float()
        valid = rearrange(valid, "b s h w -> (b s) 1 h w")  # Add channel dimension
        valid = F.interpolate(valid, size=down_resolution, mode="nearest")
        valid = rearrange(valid, "(b s) 1 h w -> b (s h w)", b=B).bool()

    elif "src_view_fps" in mode:
        batch_size = int(mode.split("_")[-1])
        gt_pts_list = [gt["pts3d"] for gt in gt_list]

        in_camera1 = inv(gt_list[0]["camera_pose"])
        gt_pts_list = [geotrf(in_camera1, gt["pts3d"]) for gt in gt_list]

        gt_pts = torch.stack(gt_pts_list, dim=1)
        B, H, W, C = gt_pts_list[0].shape
        gt_pts = rearrange(gt_pts, "b s h w c -> (b s) c h w")
        gt_pts = F.interpolate(gt_pts, size=down_resolution, mode="nearest")

        gt_pts = rearrange(gt_pts, "(b s) c h w -> b (s h w) c", b=B)

        valid_list = [gt["valid_mask"].clone() for gt in gt_list]
        valid = torch.stack(valid_list, dim=1).float()
        valid = rearrange(valid, "b s h w -> (b s) 1 h w")  # Add channel dimension
        valid = F.interpolate(valid, size=down_resolution, mode="nearest")
        valid = rearrange(valid, "(b s) 1 h w -> b (s h w)", b=B).bool()

        gt_pts, valid = sampling_train_gen_target(gt_pts, valid, None, target_sampling="fps_fast", batch_size=batch_size)

    else:
        raise NotImplementedError
    return gt_pts, valid


def get_complete_pts3d(gt_list, valid_front=False, format="pointcloud"):
    if format == "mesh":
        return gt_list["cam_points"], gt_list["point_masks"], gt_list["cam_faces"], gt_list["face_masks"]
    return gt_list["cam_points"], gt_list["point_masks"]


def normalize_input(pts3d_src, valid_src, pts3d_trg, valid_trg, mode="none"):
    """Normalize the input points"""
    if mode == "none":
        return pts3d_src, pts3d_trg

    elif "median" in mode:
        if mode == "median":
            target_median = 1.0
        else:
            target_median = float(mode.split("_")[-1])

        pts3d_src_new = []
        pts3d_trg_new = []

        for b in range(pts3d_src.shape[0]):
            src_xyz = pts3d_src[b]
            trg_xyz = pts3d_trg[b]
            src_valid = valid_src[b]
            trg_valid = valid_trg[b]

            nan_pts, nnz = invalid_to_zeros(trg_xyz, trg_valid, ndim=3)

            all_dis = nan_pts.norm(dim=-1)

            mean_factor = all_dis.sum() / (nnz.sum() + 1e-8)

            valid_dis = all_dis[trg_valid]
            norm_factor = valid_dis.median() if valid_dis.numel() > 0 else torch.tensor(1.0, device=all_dis.device)

            norm_factor = norm_factor.clip(min=0.01, max=100.0)

            src_xyz_norm = src_xyz / norm_factor * target_median
            trg_xyz_norm = trg_xyz / norm_factor * target_median

            src_xyz_norm = torch.clamp(src_xyz_norm, min=-1000.0, max=1000.0)
            trg_xyz_norm = torch.clamp(trg_xyz_norm, min=-1000.0, max=1000.0)

            pts3d_src_new.append(src_xyz_norm)
            pts3d_trg_new.append(trg_xyz_norm)

        pts3d_src_new = torch.stack(pts3d_src_new, dim=0)  # B, N, 3
        pts3d_trg_new = torch.stack(pts3d_trg_new, dim=0)  # B, N, 3
        return pts3d_src_new, pts3d_trg_new

    elif "cube" in mode:
        if mode == "cube":
            target_scale = 1.0
        else:
            target_scale = float(mode.split("_")[-1])

        pts3d_src_new = []
        pts3d_trg_new = []

        for b in range(pts3d_src.shape[0]):
            src_xyz = pts3d_src[b]
            trg_xyz = pts3d_trg[b]
            src_valid = valid_src[b]
            trg_valid = valid_trg[b]

            center_trg = trg_xyz[trg_valid].mean(dim=0)

            src_xyz_centered = src_xyz - center_trg
            trg_xyz_centered = trg_xyz - center_trg

            dist_trg = torch.norm(trg_xyz_centered[trg_valid], dim=1)
            max_dist_trg = torch.quantile(dist_trg, 0.9)

            src_xyz_norm = src_xyz_centered / max_dist_trg * target_scale
            trg_xyz_norm = trg_xyz_centered / max_dist_trg * target_scale

            pts3d_src_new.append(src_xyz_norm)
            pts3d_trg_new.append(trg_xyz_norm)

        pts3d_src = torch.stack(pts3d_src_new, dim=0)
        pts3d_trg = torch.stack(pts3d_trg_new, dim=0)

        return pts3d_src, pts3d_trg


def _predict_vector_field(model, batch, images, query_points, timestep, token_mask, pointmaps, cfg_scale=1.0):
    """Predict velocity field v_theta(x_t, t | cond) in a single forward pass."""

    encoder_data = model._encode(images=images, batch=batch, pointmaps=pointmaps, test=False, cfg_scale=cfg_scale)
    out = model._decode(
        tokens=encoder_data["tokens"],
        images=images,
        token_mask=token_mask,
        query_points=query_points,
        timestep=timestep,
    )
    return out["pts3d_xyz"]


def loss_of_one_batch_train(
    args,
    batch,
    model,
    criterion,
    device,
    ret=None,
    **kwargs,
):
    """Compute predictions and training loss for one batch."""

    images = torch.stack(batch["images"], dim=1)
    # images = torch.zeros_like(images) # Replace with black images

    token_mask = None

    if "query_source" in args.model.params.cfg.pts3d_head.params:
        query_src = args.model.params.cfg.pts3d_head.params.query_source
    else:
        query_src = "src_complete"

    if "target_source" in args.model.params.cfg.pts3d_head.params:
        target_src = args.model.params.cfg.pts3d_head.params.target_source
    else:
        target_src = "src_complete"

    if "down_resolution" in args.model.params.cfg.pts3d_head.params:
        down_resolution = args.model.params.cfg.pts3d_head.params.down_resolution
    else:
        down_resolution = 224

    if "norm_mode" in args.model.params.cfg.pts3d_head.params:
        norm_mode = args.model.params.cfg.pts3d_head.params.norm_mode
    else:
        norm_mode = "none"

    pts3d_src, valid_src = get_all_pts3d(batch, mode=query_src, down_resolution=down_resolution)
    pts3d_trg, valid_trg = get_all_pts3d(batch, mode=target_src, down_resolution=down_resolution)
    if "hunyuan" in query_src:
        pts3d_src_norm, normals_src = pts3d_src[..., :3], pts3d_src[..., 3:]
        pts3d_trg_norm, normals_trg = pts3d_trg[..., :3], pts3d_trg[..., 3:]
    else:
        pts3d_src_norm, pts3d_trg_norm = normalize_input(pts3d_src, valid_src, pts3d_trg, valid_trg, mode=norm_mode)

    B = images.shape[0]
    # save_points_ply(pts3d_trg_norm[0], f"debug_points/points_trg.ply")
    # save_points_ply(pts3d_src_norm[0], f"debug_points/points_src.ply")
    # save_batch_images(images, f"debug_points/{batch['seq_name'][0]}_images.png")

    # Use uniform noise [-1, 1]^3 to match inference.py prior instead of Gaussian
    x_0 = torch.rand_like(pts3d_trg_norm, device=device) * 2 - 1
    t = torch.rand(B, device=device)

    fm_path = path
    path_sample = fm_path.sample(x_0=x_0, x_1=pts3d_trg_norm, t=t)
    x_t = path_sample.x_t
    dx_t = path_sample.dx_t

    t_query = t[:, None].expand(B, x_t.shape[1])
    cfg_scale = args.cfg_scale if "cfg_scale" in args else 1.0

    if "hunyuan" in query_src:
        pts3d_src_norm = torch.cat([pts3d_src_norm, normals_src], dim=-1)
    v_pred = _predict_vector_field(
        model=model,
        batch=batch,
        images=images,
        query_points=x_t,
        timestep=t_query,
        token_mask=token_mask,
        pointmaps=pts3d_src_norm,
        cfg_scale=cfg_scale,
    )
    # visualize_dx_t_v_pred_distribution(dx_t, v_pred, t, save_path=f"debug_points/value_distribution.png")

    gt_list = {
        "velocity_trg": dx_t,
        "valid_trg": valid_trg,
        "pts3d_target": pts3d_trg_norm,
    }
    pred_dict = {
        "velocity_pred": v_pred,
        "x_t": x_t,
        "t": t,
    }
    loss, details = criterion(gt_list=gt_list, pred_dict=pred_dict)

    return loss, details
