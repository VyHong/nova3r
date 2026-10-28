"""Uniform, saturated UDF supervision outside an enclosing geometry box."""

import math

import torch


def append_horizontal_empty_points(batch, count, sdf_label_scale, vertical_axis=1):
    """Append queries in [-2, 2]^2 outside the box plus a truncation margin.

    Bounds and queries must already share normalized training coordinates. The
    eight transformed box corners conservatively enclose even rotated meshes.
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
    if "geo_points" not in batch or "mesh_bounds_corners" not in batch:
        raise ValueError("Empty-space sampling requires geo_points and full mesh bounds")
    points = batch["geo_points"]
    corners = batch["mesh_bounds_corners"].float()
    if not torch.isfinite(corners).all():
        raise ValueError("Empty-space sampling requires mesh_bounds or bounds in the UDF NPZ")
    lower, upper = corners.amin(dim=1), corners.amax(dim=1)
    if torch.any(upper <= lower):
        raise ValueError("Empty-space sampling requires positive bounding-box extents")
    horizontal = [axis for axis in range(3) if axis != vertical_axis]
    # Slightly exceed truncation so reduced-precision storage cannot undershoot.
    distance = (1.0 + 1e-3) / sdf_label_scale
    extra = points.new_zeros((points.shape[0], count, points.shape[-1]))
    extra[..., 3] = distance
    for index in range(points.shape[0]):
        a, b = horizontal
        lo = (lower[index, horizontal] - distance).clamp(-2, 2)
        hi = (upper[index, horizontal] + distance).clamp(-2, 2)
        # Disjoint rectangles partition the complement of the excluded box.
        minus_two, plus_two = lo.new_tensor(-2), lo.new_tensor(2)
        rect_low = torch.stack([
            torch.stack((minus_two, minus_two)),  # left strip
            torch.stack((hi[0], minus_two)),      # right strip
            torch.stack((lo[0], minus_two)),      # bottom middle
            torch.stack((lo[0], hi[1])),          # top middle
        ])
        rect_high = torch.stack([
            torch.stack((lo[0], plus_two)),
            torch.stack((plus_two, plus_two)),
            torch.stack((hi[0], lo[1])),
            torch.stack((hi[0], plus_two)),
        ])
        areas = (rect_high - rect_low).clamp_min(0).prod(dim=-1)
        if areas.sum() <= 0:
            if points.shape[1] == 0:
                raise ValueError("No empty space or existing geometry queries available")
            indices = torch.randint(points.shape[1], (count,), device=points.device)
            extra[index] = points[index, indices]
            continue
        choices = torch.multinomial(areas, count, replacement=True)
        uv = rect_low[choices] + torch.rand((count, 2), device=points.device) * (rect_high - rect_low)[choices]
        extra[index, :, a] = uv[:, 0]
        extra[index, :, b] = uv[:, 1]
        extra[index, :, vertical_axis] = lower[index, vertical_axis] + torch.rand(
            count, device=points.device
        ) * (upper[index, vertical_axis] - lower[index, vertical_axis])
    result = dict(batch)
    result["geo_points"] = torch.cat((points, extra), dim=1)
    return result
