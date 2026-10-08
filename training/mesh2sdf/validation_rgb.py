"""RGB rendering and LPIPS evaluation for colored validation meshes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
import torch


def _mesh_vertex_rgb(mesh) -> np.ndarray:
    visual = mesh.visual if getattr(mesh.visual, "kind", None) == "vertex" else mesh.visual.to_color()
    colors = np.asarray(visual.vertex_colors)
    if colors.ndim != 2 or len(colors) != len(mesh.vertices) or colors.shape[1] < 3:
        raise ValueError("LPIPS validation requires RGB vertex colors on the reconstructed mesh")
    return colors[:, :3].astype(np.float32) / 255.0


def _camera_rays(intrinsics: np.ndarray, height: int, width: int) -> np.ndarray:
    """COLMAP/OpenCV camera rays: +X right, +Y down, and +Z forward."""
    y, x = np.meshgrid(
        np.arange(height, dtype=np.float32) + 0.5,
        np.arange(width, dtype=np.float32) + 0.5,
        indexing="ij",
    )
    directions = np.stack(
        (
            (x - intrinsics[0, 2]) / intrinsics[0, 0],
            (y - intrinsics[1, 2]) / intrinsics[1, 1],
            np.ones_like(x),
        ),
        axis=-1,
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return directions.reshape(-1, 3)


def _raycasting_scene(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices), o3d.core.Tensor(faces),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    return scene, faces


def render_colored_mesh_views(
    mesh,
    intrinsics,
    extrinsics,
    height: int,
    width: int,
    *,
    chunk_size: int = 262144,
) -> np.ndarray:
    """Ray-render RGB views from a mesh using world-to-camera matrices."""
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsics [V,3,3], got {intrinsics.shape}")
    if extrinsics.shape != (len(intrinsics), 4, 4):
        raise ValueError(f"Expected extrinsics [V,4,4], got {extrinsics.shape}")
    if chunk_size < 1:
        raise ValueError("Ray chunk size must be positive")

    scene, faces = _raycasting_scene(mesh)
    colors = _mesh_vertex_rgb(mesh)
    outputs = []
    for camera_intrinsics, world_to_camera in zip(intrinsics, extrinsics):
        camera_to_world = np.linalg.inv(world_to_camera)
        directions = _camera_rays(camera_intrinsics, height, width) @ camera_to_world[:3, :3].T
        origin = camera_to_world[:3, 3]
        rgb = np.zeros((height * width, 3), dtype=np.float32)
        for start in range(0, len(directions), chunk_size):
            stop = min(start + chunk_size, len(directions))
            direction_chunk = directions[start:stop]
            rays = np.concatenate(
                (np.broadcast_to(origin, direction_chunk.shape), direction_chunk), axis=1
            )
            hits = scene.cast_rays(
                o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
            )
            distance = hits["t_hit"].numpy()
            valid = np.isfinite(distance) & (distance > 0)
            if not valid.any():
                continue
            face_ids = hits["primitive_ids"].numpy()[valid].astype(np.int64)
            uv = hits["primitive_uvs"].numpy()[valid]
            barycentric = np.column_stack(
                (1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1])
            )
            rgb[start:stop][valid] = np.sum(
                colors[faces[face_ids]] * barycentric[..., None], axis=1
            )
        outputs.append(rgb.reshape(height, width, 3))
    return np.stack(outputs)


def reference_images(batch, sample_idx: int) -> torch.Tensor:
    """Return one sample's validation views as [V,3,H,W] in [0,1]."""
    images = batch.get("images")
    if isinstance(images, torch.Tensor):
        result = images[sample_idx]
    elif isinstance(images, (list, tuple)):
        result = torch.stack([view[sample_idx] for view in images])
    else:
        raise ValueError("RGB validation requires batch['images']")
    if result.ndim != 4 or result.shape[1] != 3:
        raise ValueError(f"Expected validation images [V,3,H,W], got {tuple(result.shape)}")
    return result.detach().float().clamp(0, 1)


class ValidationLPIPSEvaluator:
    """Lazily owned renderer and pretrained LPIPS network."""

    def __init__(self, net_type: str, device: torch.device, ray_chunk_size: int):
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.device = device
        self.ray_chunk_size = ray_chunk_size
        self.metric = LearnedPerceptualImagePatchSimilarity(
            net_type=net_type,
            normalize=True,
            reduction="none",
        ).to(device).eval()

    @torch.inference_mode()
    def evaluate(self, mesh, batch, sample_idx: int, output_dir: Path):
        target = reference_images(batch, sample_idx)
        intrinsics = batch["intrinsics"][sample_idx].detach().float().cpu().numpy()
        extrinsics = batch["extrinsics"][sample_idx].detach().float().cpu().numpy()
        height, width = target.shape[-2:]
        rendered = render_colored_mesh_views(
            mesh,
            intrinsics,
            extrinsics,
            height,
            width,
            chunk_size=self.ray_chunk_size,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for view_idx, image in enumerate(rendered):
            pixels = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
            Image.fromarray(pixels, mode="RGB").save(output_dir / f"rgb_{view_idx:04d}.png")

        prediction = torch.from_numpy(rendered).permute(0, 3, 1, 2).to(self.device)
        target = target.to(self.device)
        scores = self.metric(prediction.clamp_(0, 1), target)
        self.metric.reset()
        return [float(value) for value in scores.reshape(-1).cpu()]
