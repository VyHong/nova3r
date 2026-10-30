"""Saturated UDF supervision outside the transformed padded sampling box."""

import math

import torch


def append_horizontal_empty_points(batch, count, sdf_label_scale, vertical_axis=1):
    """Append queries in [-2, 2]^3 outside the padded sampling box.

    The function name and vertical_axis argument are retained for config/API
    compatibility; all three axes now span [-2, 2].
    Bounds and queries must already share normalized training coordinates. The
    eight ordered corners preserve the padded sampling box orientation.
    Extra feature columns remain zero (volume class, RGB and RGB-valid mask).
    If the domain is fully covered, reuse existing labeled queries for that
    example to retain a rectangular batch without inventing empty-space labels.
    """
    if count == 0:
        return batch
    if count < 0 or vertical_axis not in (0, 1, 2):
        raise ValueError("Empty-space count must be non-negative and vertical axis 0, 1, or 2")
    if not math.isfinite(sdf_label_scale) or sdf_label_scale <= 0:
        raise ValueError("Empty-space sampling requires a positive finite sdf_label_scale")
    if "geo_points" not in batch or "sampling_bounds_corners" not in batch:
        raise ValueError("Empty-space sampling requires geo_points and full sampling bounds")
    points = batch["geo_points"]
    corners = batch["sampling_bounds_corners"].float()
    if not torch.isfinite(corners).all():
        raise ValueError("Empty-space sampling requires padded bounds in the UDF NPZ")
    lower, upper = corners.amin(dim=1), corners.amax(dim=1)
    if torch.any(upper <= lower):
        raise ValueError("Empty-space sampling requires positive bounding-box extents")
    # Slightly exceed truncation so reduced-precision storage cannot undershoot.
    distance = (1.0 + 1e-3) / sdf_label_scale
    extra = points.new_zeros((points.shape[0], count, points.shape[-1]))
    extra[..., 3] = distance
    for index in range(points.shape[0]):
        # Corner ordering is (000, 001, 010, 011, 100, 101, 110, 111).
        # Recover the transformed box's orthonormal axes, retaining rotation.
        origin = corners[index, 0]
        edges = corners[index, [4, 2, 1]] - origin
        lengths = edges.norm(dim=-1)
        if torch.any(lengths <= 0):
            raise ValueError("Sampling box edges must have positive lengths")
        axes = edges / lengths[:, None]

        def outside(candidates):
            local = (candidates - origin) @ axes.T
            return ((local < 0) | (local > lengths)).any(dim=-1)

        domain_low = corners.new_full((3,), -2)
        domain_high = corners.new_full((3,), 2)
        bits = corners.new_tensor([[x, y, z] for x in (0, 1)
                                  for y in (0, 1) for z in (0, 1)])
        domain_corners = domain_low + bits * (domain_high - domain_low)
        accepted = []
        remaining = count
        # A convex box containing every domain corner covers the whole domain.
        if outside(domain_corners).any():
            for _ in range(64):
                candidates = domain_low + torch.rand(
                    (min(max(remaining * 2, 1024), 65536), 3), device=points.device
                ) * (domain_high - domain_low)
                valid = candidates[outside(candidates)][:remaining]
                if len(valid):
                    accepted.append(valid)
                    remaining -= len(valid)
                if remaining == 0:
                    break
        if accepted:
            valid = torch.cat(accepted)
            extra[index, :len(valid), :3] = valid
            # Bound runtime for vanishingly small complements. Reuse only
            # verified empty queries if rejection sampling hit its budget.
            if remaining:
                extra[index, len(valid):, :3] = valid[
                    torch.randint(len(valid), (remaining,), device=points.device)
                ]
        else:
            if points.shape[1] == 0:
                raise ValueError("No empty space or existing geometry queries available")
            indices = torch.randint(points.shape[1], (count,), device=points.device)
            extra[index] = points[index, indices]
    result = dict(batch)
    result["geo_points"] = torch.cat((points, extra), dim=1)
    return result
