"""Shared debug export for dataset items and collated Lightning batches."""

import json
from pathlib import Path

import numpy as np
import torch


def _at(value, index, batched):
    if not batched:
        return value
    if isinstance(value, torch.Tensor):
        return value[index]
    if isinstance(value, (list, tuple)):
        # default_collate transposes a list of per-view tensors.
        if value and isinstance(value[0], torch.Tensor):
            return [item[index] for item in value]
        return value[index]
    return value


def _json_value(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _summary(value):
    if isinstance(value, torch.Tensor):
        value = value.detach()
        result = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
        if value.numel() and value.is_floating_point():
            finite = value[torch.isfinite(value)].float()
            result["finite"] = finite.numel() == value.numel()
            if finite.numel():
                result.update(
                    mean=float(finite.mean().cpu()),
                    min=float(finite.min().cpu()),
                    max=float(finite.max().cpu()),
                )
        return result
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "items": [_summary(item) for item in value],
        }
    return {"type": type(value).__name__, "value": _json_value(value)}


def _safe_name(value):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value))


def _write_points(path, value, color):
    array = torch.as_tensor(value).detach().cpu().numpy()
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"Expected point data [N, >=3], got {array.shape}")
    np.save(path.with_suffix(".npy"), array)
    normals = array[:, 3:6] if array.shape[1] >= 6 else None
    labels = array[:, -1] > 0.5 if array.shape[1] >= 7 else None
    colors = np.tile(np.asarray(color, dtype=np.uint8), (len(array), 1))
    if labels is not None:
        colors[~labels] = (51, 140, 255)
        colors[labels] = (255, 64, 26)

    with path.open("w", encoding="utf-8") as file:
        properties = ["property float x", "property float y", "property float z"]
        if normals is not None:
            properties += ["property float nx", "property float ny", "property float nz"]
        properties += ["property uchar red", "property uchar green", "property uchar blue"]
        file.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(array)}\n{'\n'.join(properties)}\nend_header\n"
        )
        for row, rgb in zip(array, colors):
            values = list(row[:6] if normals is not None else row[:3]) + list(rgb)
            file.write(" ".join(map(str, values)) + "\n")


def _write_images(path, images):
    from PIL import Image

    if isinstance(images, torch.Tensor):
        images = list(images)
    tiles = []
    for image in images:
        array = torch.as_tensor(image).detach().cpu().float()
        if array.ndim != 3:
            raise ValueError(f"Expected image [C,H,W], got {tuple(array.shape)}")
        if array.shape[0] in (1, 3, 4):
            array = array.permute(1, 2, 0)
        array = np.clip(array.numpy() * 255.0, 0, 255).astype(np.uint8)
        if array.shape[-1] == 1:
            array = array[..., 0]
        tiles.append(array)
    if tiles:
        Image.fromarray(np.concatenate(tiles, axis=1)).save(path)


def save_batch_debug(batch, output_dir, *, max_samples=None, batched=None, prefix=""):
    """Export dataset/Lightning batch contents in one consistent layout.

    Args:
        batch: Mapping containing any of ``images``, ``surface``, ``geo_points``,
            ``extrinsics`` and ``intrinsics``. Additional values are summarized.
        output_dir: Root directory for the export.
        max_samples: Maximum collated samples to export; ``None`` exports all.
        batched: Whether the mapping has a leading batch dimension. When omitted,
            this is inferred from ``surface`` or ``images``.
        prefix: Optional filename prefix.

    Returns:
        List of directories written, one per exported sample.
    """
    output_dir = Path(output_dir)
    surface = batch.get("surface")
    images = batch.get("images")
    if batched is None:
        batched = bool(
            isinstance(surface, torch.Tensor) and surface.ndim >= 3
            or isinstance(images, torch.Tensor) and images.ndim >= 5
            or isinstance(images, (list, tuple))
            and images
            and isinstance(images[0], torch.Tensor)
            and images[0].ndim >= 4
        )
    batch_size = int(surface.shape[0]) if batched and isinstance(surface, torch.Tensor) else 1
    if batched and not isinstance(surface, torch.Tensor):
        if isinstance(images, torch.Tensor):
            batch_size = int(images.shape[0])
        elif images:
            batch_size = int(images[0].shape[0])
    count = batch_size if max_samples is None or max_samples < 0 else min(batch_size, max_samples)

    written = []
    for index in range(count):
        item = {key: _at(value, index, batched) for key, value in batch.items()}
        seq_name = item.get("seq_name", "sample")
        sample_id = item.get("id", index)
        stem = _safe_name(f"{prefix}{index:03d}_{seq_name}_{sample_id}")
        sample_dir = output_dir / stem if batched else output_dir
        sample_dir.mkdir(parents=True, exist_ok=True)

        if item.get("images") is not None:
            _write_images(sample_dir / "images.png", item["images"])
        point_fields = {
            "surface": (51, 140, 255),
            "geo_points": (255, 140, 51),
            "cam_points": (80, 200, 120),
            "world_points": (180, 180, 180),
        }
        for key, color in point_fields.items():
            if item.get(key) is not None:
                _write_points(sample_dir / f"{key}.ply", item[key], color)
        for key in ("extrinsics", "intrinsics"):
            if item.get(key) is not None:
                array = torch.as_tensor(item[key]).detach().cpu().numpy()
                np.save(sample_dir / f"{key}.npy", array)
                (sample_dir / f"{key}.json").write_text(
                    json.dumps(array.tolist(), indent=2), encoding="utf-8"
                )
        (sample_dir / "summary.json").write_text(
            json.dumps({key: _summary(value) for key, value in item.items()}, indent=2),
            encoding="utf-8",
        )
        torch.save(_to_cpu(item), sample_dir / "batch.pt")
        written.append(sample_dir)
    return written
