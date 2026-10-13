"""Torch helpers for turning calibrated depth maps into ShapeVAE surfaces."""

from __future__ import annotations

import torch


def unproject_depth_to_world(depth, intrinsics, extrinsics):
    """Unproject OpenCV-depth images through world-to-camera matrices.

    Args:
        depth: ``[B, V, H, W]`` metric Z depth.
        intrinsics: ``[B, V, 3, 3]`` pixel-space camera matrices.
        extrinsics: ``[B, V, 4, 4]`` world-to-camera transforms.
    """
    if depth.ndim != 4:
        raise ValueError(f"Expected depth [B,V,H,W], got {tuple(depth.shape)}")
    batch, views, height, width = depth.shape
    if intrinsics.shape != (batch, views, 3, 3):
        raise ValueError(f"Unexpected intrinsics shape {tuple(intrinsics.shape)}")
    if extrinsics.shape != (batch, views, 4, 4):
        raise ValueError(f"Unexpected extrinsics shape {tuple(extrinsics.shape)}")

    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    ones = torch.ones_like(x)
    pixels = torch.stack((x, y, ones), dim=-1).reshape(1, 1, -1, 3)
    rays = pixels @ torch.linalg.inv(intrinsics).transpose(-1, -2)
    camera_points = rays * depth.reshape(batch, views, -1, 1)
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    world = (camera_points - translation.unsqueeze(-2)) @ rotation
    return world.reshape(batch, views, height, width, 3)


def _normal_maps(pointmaps, num_views, height, width):
    batch = pointmaps.shape[0]
    points = pointmaps.reshape(batch, num_views, height, width, 3)
    dx = torch.zeros_like(points)
    dy = torch.zeros_like(points)
    dx[:, :, :, 1:-1] = points[:, :, :, 2:] - points[:, :, :, :-2]
    dy[:, :, 1:-1, :] = points[:, :, 2:, :] - points[:, :, :-2, :]
    dx[:, :, :, 0] = points[:, :, :, 1] - points[:, :, :, 0]
    dx[:, :, :, -1] = points[:, :, :, -1] - points[:, :, :, -2]
    dy[:, :, 0, :] = points[:, :, 1, :] - points[:, :, 0, :]
    dy[:, :, -1, :] = points[:, :, -1, :] - points[:, :, -2, :]
    normals = torch.linalg.cross(dx, dy, dim=-1)
    return torch.nn.functional.normalize(normals, dim=-1, eps=1e-8).reshape(batch, -1, 3)


def _normalize(points, mode):
    if mode in ("none", "raw", "skip"):
        return points
    if mode in ("cube", "unit_box"):
        lower, upper = points.amin(0), points.amax(0)
        center = (lower + upper) * 0.5
        padding = 1.0 if mode == "cube" else 1.01
        scale = 2.0 / ((upper - lower).amax().clamp_min(1e-8) * padding)
        return (points - center) * scale
    if mode == "median":
        target = 1.0
    elif str(mode).startswith("median_"):
        target = float(str(mode).rsplit("_", 1)[1])
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")
    scale = target / torch.linalg.vector_norm(points, dim=1).median().clamp(0.01, 100.0)
    return (points * scale).clamp(-1000.0, 1000.0)


def sample_depth_pointmap_for_vae(
    pointmaps,
    confidence,
    num_views,
    height,
    width,
    random_count,
    sharp_count,
    point_feats,
    sharp_threshold=0.985,
    confidence_percentile=0.0,
    normalization="none",
    seed=0,
):
    """Select deterministic random/edge samples and construct ShapeVAE inputs."""
    if pointmaps.ndim != 3 or pointmaps.shape[-1] != 3:
        raise ValueError(f"Expected pointmaps [B,N,3], got {tuple(pointmaps.shape)}")
    expected = num_views * height * width
    if pointmaps.shape[1] != expected:
        raise ValueError(f"Expected {expected} points, got {pointmaps.shape[1]}")
    normals = _normal_maps(pointmaps, num_views, height, width)
    results = []
    for batch_index, (points, point_normals) in enumerate(zip(pointmaps, normals)):
        valid = torch.isfinite(points).all(1) & torch.isfinite(point_normals).all(1)
        valid &= torch.linalg.vector_norm(points, dim=1) > 0
        if confidence is not None:
            scores = confidence[batch_index].reshape(-1)
            valid &= torch.isfinite(scores)
            if confidence_percentile > 0 and valid.any():
                cutoff = torch.quantile(scores[valid].float(), confidence_percentile / 100.0)
                valid &= scores >= cutoff

        # Sharp pixels have a sufficiently different normal from a horizontal
        # or vertical neighbor in the same cubemap face.
        normal_grid = point_normals.reshape(num_views, height, width, 3)
        sharp = torch.zeros((num_views, height, width), dtype=torch.bool, device=points.device)
        sharp[:, :, 1:] |= (normal_grid[:, :, 1:] * normal_grid[:, :, :-1]).sum(-1) < sharp_threshold
        sharp[:, 1:, :] |= (normal_grid[:, 1:, :] * normal_grid[:, :-1, :]).sum(-1) < sharp_threshold
        sharp = sharp.reshape(-1) & valid
        regular_indices = torch.nonzero(valid, as_tuple=False).flatten()
        sharp_indices = torch.nonzero(sharp, as_tuple=False).flatten()
        if len(regular_indices) < random_count:
            raise ValueError(f"Only {len(regular_indices)} valid depth points for {random_count} random samples")
        if sharp_count and not len(sharp_indices):
            sharp_indices = regular_indices

        generator = torch.Generator(device=points.device).manual_seed(seed + batch_index)
        random_pick = regular_indices[
            torch.randperm(len(regular_indices), generator=generator, device=points.device)[:random_count]
        ]
        if sharp_count:
            order = torch.randperm(len(sharp_indices), generator=generator, device=points.device)
            if len(order) < sharp_count:
                order = order.repeat((sharp_count + len(order) - 1) // len(order))
            chosen = torch.cat((random_pick, sharp_indices[order[:sharp_count]]))
        else:
            chosen = random_pick
        xyz = _normalize(points[chosen], normalization)
        if point_feats == 0:
            surface = xyz
        elif point_feats in (3, 4):
            surface = torch.cat((xyz, point_normals[chosen]), dim=1)
            if point_feats == 4:
                labels = torch.cat((
                    torch.zeros(random_count, device=points.device, dtype=points.dtype),
                    torch.ones(sharp_count, device=points.device, dtype=points.dtype),
                )).unsqueeze(1)
                surface = torch.cat((surface, labels), dim=1)
        else:
            raise ValueError(f"Unsupported ShapeVAE point_feats={point_feats}")
        results.append(surface)
    return torch.stack(results)
