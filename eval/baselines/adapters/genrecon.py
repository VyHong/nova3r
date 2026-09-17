"""GenRecon adapter for one posed SAGE cubemap panorama."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch

from eval.baselines.common import SageSample, load_cameras, load_scene_points, sample_mesh_surface


# GenRecon builds SAGE rooms in Blender's Z-up frame. The evaluator's exported
# SAGE surfaces and cameras use the glTF/Y-up frame: (x, y, z) -> (x, z, -y).
SAGE_ROOM_TO_EVAL = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)


def resolve_room_id(scene: str, rooms_raw_root: Path) -> str:
    """Resolve evaluator scene names such as ``..._layout_deadbeef`` to SAGE IDs."""
    candidates = [scene]
    if "_layout_" in scene:
        candidates.append(scene.rsplit("_layout_", 1)[1])
    candidates.append(scene.rsplit("_", 1)[-1])
    for candidate in dict.fromkeys(candidates):
        layout = rooms_raw_root / candidate / f"layout_{candidate}.json"
        if layout.is_file():
            return candidate
    checked = [str(rooms_raw_root / item / f"layout_{item}.json") for item in dict.fromkeys(candidates)]
    raise FileNotFoundError("Could not resolve the GenRecon SAGE room layout; checked:\n  " + "\n  ".join(checked))


def infer_sage_floorplan(surface_path: Path) -> tuple[float, float, float, float]:
    """Infer GenRecon's (x, y) room footprint from an evaluator-frame surface.

    SAGE rooms are axis aligned and originate at (0, 0) in their Blender frame.
    Evaluator coordinates map Blender ``(x, y, z)`` to ``(x, z, -y)``, so the
    footprint dimensions are the evaluator X and Z extents. Wall cubes can
    extend as much as their 0.1 m thickness past the nominal floor rectangle;
    generated SAGE dimensions lie on a 0.5 m grid, so remove that small shell
    overshoot when the measured extent is close to the grid.
    """
    points = load_scene_points(surface_path)
    if not len(points):
        raise ValueError(f"Cannot infer a room footprint from an empty surface: {surface_path}")
    extent = np.ptp(points, axis=0)
    if not np.isfinite(extent).all():
        raise ValueError(f"Surface has non-finite bounds: {surface_path}")

    def snap_dimension(value: float) -> float:
        snapped = round(value * 2.0) / 2.0
        return snapped if snapped > 0 and abs(value - snapped) <= 0.11 else value

    width = snap_dimension(float(extent[0]))
    length = snap_dimension(float(extent[2]))
    if width <= 0 or length <= 0:
        raise ValueError(
            f"Invalid inferred SAGE footprint width={width}, length={length}: {surface_path}"
        )
    return 0.0, width, 0.0, length


def normalize_intrinsics(intrinsics: np.ndarray, image_sizes: list[tuple[int, int]]) -> torch.Tensor:
    """Convert pixel-space OpenCV intrinsics to GenRecon's normalized convention."""
    normalized = torch.as_tensor(intrinsics, dtype=torch.float32).clone()
    if len(normalized) != len(image_sizes):
        raise ValueError("One image size is required for every intrinsic matrix")
    for intrinsic, (width, height) in zip(normalized, image_sizes):
        if width < 1 or height < 1:
            raise ValueError(f"Invalid image size {(width, height)}")
        intrinsic[0] /= width
        intrinsic[1] /= height
    return normalized


def _homogeneous_extrinsics(extrinsics: np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(extrinsics, dtype=torch.float32)
    if tensor.ndim != 3 or tensor.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Expected camera extrinsics shaped [N, 3, 4] or [N, 4, 4], got {tuple(tensor.shape)}")
    if tensor.shape[1:] == (3, 4):
        bottom = torch.zeros((len(tensor), 1, 4), dtype=tensor.dtype)
        bottom[:, 0, 3] = 1
        tensor = torch.cat((tensor, bottom), dim=1)
    return tensor


def _transform_points(points: torch.Tensor, transform: torch.Tensor) -> np.ndarray:
    points = points.detach().cpu().float()
    transform = transform.detach().cpu().float()
    return (points @ transform[:3, :3].T + transform[:3, 3]).numpy()


def chunk_transforms_to_eval_frame(
    original_to_chunk: list[torch.Tensor],
    chunk_to_original: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Express GenRecon's Blender-room chunk transforms in evaluator coordinates."""
    eval_to_room = torch.linalg.inv(SAGE_ROOM_TO_EVAL)
    return (
        [transform.float() @ eval_to_room for transform in original_to_chunk],
        [SAGE_ROOM_TO_EVAL @ transform.float() for transform in chunk_to_original],
    )


class GenReconAdapter:
    """Run GenRecon on the six perspective faces belonging to one SAGE panorama."""

    def __init__(
        self,
        repo: Path,
        data_root: Path,
        ss_ckpt: Path,
        shape_ckpt: Path,
        tex_ckpt: Path,
        pipeline_config: Path,
        device: str,
        face_order: list[int],
        work_root: Path,
        surface_points: int,
        seed: int,
        occ_threshold: float = -1.0,
        boundary_width: int = 1,
        proj_batch_voxels: int | None = None,
    ):
        repo = repo.resolve()
        stage_config_files = {
            "sparse_structure_flow_model": repo / "configs/gen/ss_flow_img/genrecon.json",
            "shape_slat_flow_model_512": repo / "configs/gen/slat_flow_img2shape/genrecon_512.json",
            "tex_slat_flow_model_512": repo / "configs/gen/slat_flow_imgshape2tex/genrecon_512.json",
        }
        required = {
            "GenRecon repository": repo / "genrecon" / "pipelines" / "full_scene_images_to_3d.py",
            "sparse-structure checkpoint": ss_ckpt,
            "shape checkpoint": shape_ckpt,
            "texture checkpoint": tex_ckpt,
            "pipeline config": pipeline_config,
            "sparse-structure training config": stage_config_files["sparse_structure_flow_model"],
            "shape training config": stage_config_files["shape_slat_flow_model_512"],
            "texture training config": stage_config_files["tex_slat_flow_model_512"],
        }
        missing = [f"{label}: {path}" for label, path in required.items() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError("Missing GenRecon inputs:\n  " + "\n  ".join(missing))
        # Layout JSONs are preferred when available. Evaluation surfaces provide
        # an equivalent room-footprint fallback, so a prepared GenRecon data root
        # is not required for the SAGE evaluator.
        rooms_raw_root = data_root.resolve() / "rooms_raw"
        if surface_points < 1:
            raise ValueError("surface_points must be positive")
        if boundary_width < 0:
            raise ValueError("boundary_width must be non-negative")

        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        pipeline_module = importlib.import_module("genrecon.pipelines.full_scene_images_to_3d")
        chunks_module = importlib.import_module("inference.get_chunks")
        images_module = importlib.import_module("genrecon.pipelines.types")
        components_module = importlib.import_module("genrecon.datasets.components")

        self.SageGtChunker = chunks_module.SageGtChunker
        self.SelectedImages = images_module.SelectedImages
        self.chunk_visible_in_camera = components_module.chunk_visible_in_camera
        self.data_root = data_root.resolve()
        self.rooms_raw_root = rooms_raw_root
        self.device = torch.device(device)
        self.face_order = face_order
        self.work_root = work_root.resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.surface_points = surface_points
        self.seed = seed
        self.occ_threshold = occ_threshold
        self.boundary_width = boundary_width

        stage_models = {
            "sparse_structure_flow_model": str(ss_ckpt.resolve()),
            "shape_slat_flow_model_512": str(shape_ckpt.resolve()),
            "tex_slat_flow_model_512": str(tex_ckpt.resolve()),
        }
        self.pipeline = pipeline_module.FullSceneImagesTo3DPipeline.from_finetuned(
            stage_models=stage_models,
            pipeline_config_file=str(pipeline_config.resolve()),
            stage_config_files={key: str(path) for key, path in stage_config_files.items()},
        )
        if proj_batch_voxels is not None:
            self.pipeline.proj_batch_voxels = proj_batch_voxels
        self.pipeline.to(self.device)

    def _get_chunks(self, sample: SageSample, sample_work: Path):
        """Build GenRecon chunks from a layout JSON or the evaluation surface."""
        chunker = self.SageGtChunker()
        try:
            room_id = resolve_room_id(sample.scene, self.rooms_raw_root)
        except FileNotFoundError as layout_error:
            x_min, x_max, y_min, y_max = infer_sage_floorplan(sample.surface_path)
            z_floor, _, delta_z = chunker._get_floor_ceiling_height()
            chunk_size = delta_z + 0.15
            centers = chunker._define_centers(
                x_min,
                x_max,
                y_min,
                y_max,
                z_floor,
                chunk_size,
                sample_work,
            )
            m_o2c, m_c2o, relative_translations = chunker._get_transforms(
                centers,
                chunk_size,
                sample_work,
            )
            source = {
                "type": "evaluation_surface_bounds",
                "surface_path": str(sample.surface_path),
                "floorplan": {
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                },
                "layout_lookup_error": str(layout_error),
            }
            return centers, m_o2c, m_c2o, relative_translations, source

        room_render_path = self.data_root / "renders_room" / room_id
        centers, m_o2c, m_c2o, relative_translations = chunker.get_chunks(
            room_render_path,
            sample_work,
        )
        source = {
            "type": "sage_layout_json",
            "layout_path": str(self.rooms_raw_root / room_id / f"layout_{room_id}.json"),
        }
        return centers, m_o2c, m_c2o, relative_translations, source

    @staticmethod
    def _load_face_tensors(paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        tensors_512, tensors_1024, sizes = [], [], []
        for path in paths:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                sizes.append(image.size)
                for size, output in ((512, tensors_512), (1024, tensors_1024)):
                    resized = image.resize((size, size), Image.Resampling.LANCZOS)
                    array = np.array(resized, dtype=np.uint8, copy=True)
                    output.append(torch.from_numpy(array).permute(2, 0, 1).float() / 255.0)
        return torch.stack(tensors_512), torch.stack(tensors_1024), sizes

    def _select_images(
        self,
        paths: list[Path],
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        m_o2c: list[torch.Tensor],
        m_c2o: list[torch.Tensor],
        seed: int,
    ):
        images_512, images_1024, sizes = self._load_face_tensors(paths)
        normalized_intrinsics = normalize_intrinsics(intrinsics, sizes)
        world_to_camera = _homogeneous_extrinsics(extrinsics)
        extrinsics_c0 = world_to_camera @ m_c2o[0].to(dtype=torch.float32)

        rng = random.Random(seed)
        selected_indices = []
        for chunk_index, chunk_to_world in enumerate(m_c2o):
            order = list(range(len(paths)))
            rng.shuffle(order)
            visible = [
                index
                for index in order
                if self.chunk_visible_in_camera(
                    world_to_camera[index] @ chunk_to_world.to(dtype=torch.float32),
                    normalized_intrinsics[index],
                )
            ]
            if visible:
                selected_indices.append(visible[0])
                continue

            # Match GenRecon's fallback: choose the camera closest to the chunk origin.
            selected_indices.append(
                min(
                    range(len(paths)),
                    key=lambda index: float(
                        torch.linalg.norm(
                            (
                                m_o2c[chunk_index].to(dtype=torch.float32)
                                @ torch.linalg.inv(world_to_camera[index])
                            )[:3, 3]
                        )
                    ),
                )
            )

        selected = self.SelectedImages(
            scene_images_1024=images_1024,
            scene_images_512=images_512,
            scene_intrinsics=normalized_intrinsics,
            scene_extrinsics_c0=extrinsics_c0,
            cond2d_images_1024=[images_1024[index] for index in selected_indices],
            cond2d_images_512=[images_512[index] for index in selected_indices],
            cond2d_intrinsics=[normalized_intrinsics[index] for index in selected_indices],
            cond2d_extrinsics_c0=[extrinsics_c0[index] for index in selected_indices],
            chunk_indices=list(range(len(m_o2c))),
        )
        return selected, selected_indices

    def predict(self, sample: SageSample) -> np.ndarray:
        sample_seed = self.seed + sample.frame
        sample_work = self.work_root / f"{sample.scene}_{sample.frame:05d}"
        sample_work.mkdir(parents=True, exist_ok=True)

        _, m_o2c_room, m_c2o_room, relative_translations, chunk_source = self._get_chunks(
            sample,
            sample_work,
        )
        (sample_work / "chunk_source.json").write_text(json.dumps(chunk_source, indent=2))
        m_o2c, m_c2o = chunk_transforms_to_eval_frame(m_o2c_room, m_c2o_room)
        paths, intrinsics, extrinsics = load_cameras(sample, self.face_order)
        selected, selected_indices = self._select_images(
            paths,
            intrinsics,
            extrinsics,
            m_o2c,
            m_c2o,
            sample_seed,
        )
        (sample_work / "cubemap_selection.json").write_text(
            json.dumps(
                {
                    "scene_faces": [str(path) for path in paths],
                    "chunk_face_indices": selected_indices,
                    "chunk_face_paths": [str(paths[index]) for index in selected_indices],
                },
                indent=2,
            )
        )

        slat_params = {"boundary_sensitive": True, "boundary_width": self.boundary_width}
        mesh, _ = self.pipeline.run(
            selected,
            relative_translations,
            seed=sample_seed,
            pipeline_type="512",
            sparse_structure_sampler_params={},
            shape_slat_sampler_params=slat_params,
            tex_slat_sampler_params=slat_params,
            occ_threshold=self.occ_threshold,
            geometry_only=True,
        )
        vertices_world = _transform_points(mesh.vertices, m_c2o[0])
        faces = mesh.faces.detach().cpu().numpy()
        points = sample_mesh_surface(vertices_world, faces, self.surface_points, sample_seed)
        del mesh
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return points
