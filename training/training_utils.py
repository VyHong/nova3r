import os
import time
from collections import defaultdict

import tqdm
import torch
import torch.nn.functional as F

from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_zeros
from dust3r.utils.geometry import geotrf, inv

# flow_matching
from nova3r.flow_matching.path.scheduler import CosineScheduler
from nova3r.flow_matching.path import AffineProbPath
from nova3r.utils.sampling import sampling_train_gen_target
from einops import rearrange

path = AffineProbPath(scheduler=CosineScheduler())

amp_dtype_mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "tf32": torch.float32}

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
        gt_pts, valid = sampling_train_gen_target(
            gt_pts, valid, None, target_sampling="fps_fast", batch_size=batch_size
        )

    elif "src_complete_fps_edge" in mode:
        batch_size = int(mode.split("_")[-1])
        gt_pts, valid = get_complete_pts3d(gt_list)
        # run fps_fast
        gt_pts, valid = sampling_train_gen_target(
            gt_pts, valid, None, target_sampling="fps_edge_fast", batch_size=batch_size
        )

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

        gt_pts, valid = sampling_train_gen_target(
            gt_pts, valid, None, target_sampling="fps_fast", batch_size=batch_size
        )

    else:
        raise NotImplementedError
    return gt_pts, valid


def get_complete_pts3d(gt_list, valid_front=False):
    """Get complete (amodal) 3D point clouds from all views, transformed to camera 1 coordinates."""

    return gt_list["cam_points"], gt_list["point_masks"]

    pts_xyz = [gt["pts3d_complete"] for gt in gt_list]
    in_camera1 = inv(gt_list[0]["camera_pose"])
    pts_xyz = [geotrf(in_camera1, pts) for pts in pts_xyz]  # B, N, 3
    gt_pts = torch.stack(pts_xyz, dim=1)  # B, S, N, 3

    valid_num_list = [gt["pts3d_complete_valid_num"] for gt in gt_list]  # B, S
    valid = torch.zeros_like(gt_pts[..., 0]).bool()  # B, S, N
    for i in range(len(gt_list)):
        for j in range(valid_num_list[i].shape[0]):
            valid[j, i, : valid_num_list[i][j]] = True

    gt_pts = rearrange(gt_pts, "b s n c -> b (s n) c")  # B, S*N, 3
    valid = rearrange(valid, "b s n -> b (s n)")  # B, S*N

    if valid_front:
        reordered_pts = []
        reordered_valid = []
        valid_counts = []

        for b in range(gt_pts.shape[0]):
            valid_mask = valid[b]
            valid_indices = torch.where(valid_mask)[0]
            invalid_indices = torch.where(~valid_mask)[0]

            reorder_indices = torch.cat([valid_indices, invalid_indices])

            reordered_pts.append(gt_pts[b][reorder_indices])
            reordered_valid.append(valid[b][reorder_indices])
            valid_counts.append(len(valid_indices))

        gt_pts = torch.stack(reordered_pts, dim=0)
        valid = torch.stack(reordered_valid, dim=0)
        valid_counts = torch.tensor(valid_counts, device=gt_pts.device)

        return gt_pts, valid, valid_counts
    else:
        return gt_pts, valid


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

def _predict_vector_field(model, images, query_points, timestep, token_mask, pointmaps, cfg_scale=1.0):
    """Predict velocity field v_theta(x_t, t | cond) in a single forward pass."""
    if hasattr(model, "module"):
        encoder_data = model.module._encode(images=images, pointmaps=pointmaps, test=False, cfg_scale=cfg_scale)
        out = model.module._decode(
            tokens=encoder_data["tokens"],
            images=images,
            token_mask=token_mask,
            query_points=query_points,
            timestep=timestep,
        )
    else:
        encoder_data = model._encode(images=images, pointmaps=pointmaps, test=False, cfg_scale=cfg_scale)
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
    ignore_keys = set(["dataset", "label", "instance", "idx", "true_shape", "rng", "view_label"])


    images = torch.stack(batch["images"], dim=1)
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
    pts3d_src_norm, pts3d_trg_norm = normalize_input(pts3d_src, valid_src, pts3d_trg, valid_trg, mode=norm_mode)

    pts3d_trg_norm = pts3d_trg_norm.to(device)
    B = images.shape[0]
    x_0 = torch.randn_like(pts3d_trg_norm,device=device)
    t = torch.rand(B, device=device)

    fm_path = AffineProbPath(scheduler=CosineScheduler())
    path_sample = fm_path.sample(x_0=x_0, x_1=pts3d_trg_norm, t=t)
    x_t = path_sample.x_t
    dx_t = path_sample.dx_t

    t_query = t[:, None].expand(B, x_t.shape[1])
    cfg_scale = args.cfg_scale if "cfg_scale" in args else 1.0

    v_pred = _predict_vector_field(
        model=model,
        images=images,
        query_points=x_t,
        timestep=t_query,
        token_mask=token_mask,
        pointmaps=pts3d_src_norm,
        cfg_scale=cfg_scale,
    )

    point_loss = F.mse_loss(v_pred, dx_t, reduction="none").mean(dim=-1)  # B, N
    if valid_trg is not None:
        valid_mask = valid_trg.bool()
        denom = torch.clamp(valid_mask.sum(), min=1)
        fm_loss = point_loss[valid_mask].sum() / denom
    else:
        fm_loss = point_loss.mean()

    pred_dict = {
        "pts3d_xyz": v_pred,
        "images": images,
        "input_pts3d": pts3d_src,
        "input_valid": valid_src,
        "target_pts3d": pts3d_trg,
        "target_valid": valid_trg,
        "flow_x_t": x_t,
        "flow_t": t,
        "flow_dx_t": dx_t,
        "flow_v_pred": v_pred,
    }

    # Standard flow-matching objective: direct velocity regression.
    loss_dict = {
        "loss": fm_loss,
        "fm_loss": fm_loss,
    }

    if criterion is not None and getattr(args, "add_aux_criterion", False):
        pts3d_data, aux_loss = criterion(batch, pred_dict)
        if isinstance(aux_loss, dict) and "loss" in aux_loss:
            total_aux = aux_loss["loss"]
        elif torch.is_tensor(aux_loss):
            total_aux = aux_loss
        else:
            total_aux = None

        if total_aux is not None:
            aux_weight = float(getattr(args, "aux_criterion_weight", 1.0))
            loss_dict["aux_loss"] = total_aux
            loss_dict["loss"] = fm_loss + aux_weight * total_aux
    else:
        pts3d_data = {
            "x_t": x_t,
            "dx_t": dx_t,
            "v_pred": v_pred,
        }

    loss = loss_dict

    result = dict(view=batch, pred=pred_dict, data=pts3d_data, loss=loss)
    return result
