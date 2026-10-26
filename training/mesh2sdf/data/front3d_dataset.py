"""3D-FRONT cubemap, UDF, feature, and latent dataset."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from nova3r.utils.three_d_front import parse_panorama_sample, world_to_opencv_camera
from training.mesh2sdf.data.sdf_dataset_mixin import SDFDatasetMixin
from training.nova3r.data.dataset_utils import read_image_cv2


class Front3DSDFDataset(SDFDatasetMixin):
    """Load 3D-FRONT cubemaps and room UDF data like SAGE/ScanNet++."""

    AVAILABLE_MODALITIES = frozenset({
        "images", "cameras", "surface", "geo_points", "conditioning_latent",
        "visible_geometry_latent", "image_features", "gt_latent",
    })
    def __init__(
        self, common_conf, data_root, split="train", samples_list_path=None,
        format="mesh", sdf_data_root=None, pose_root=None, anon_mask_root=None,
        dataset_prefix="3dfront", distance_suffix="udf",
        load_conditioning_latent=False, conditioning_latent_root=None,
        image_feature_root=None, gt_latent_root=None, gt_latent_source="latent",
        subseq_ids=None, modalities=None, load_visible_geometry_latent=False,
        visible_geometry_latent_root=None,
        geometry_coordinate_frame=None,
    ):
        super().__init__(common_conf)
        if geometry_coordinate_frame is not None:
            self.configure_geometry_coordinate_frame(geometry_coordinate_frame)
        self.data_root = Path(data_root)
        self.sdf_data_root = Path(sdf_data_root) if sdf_data_root else self.data_root
        self.pose_root = Path(pose_root) if pose_root is not None else self.data_root
        self.anon_mask_root = Path(anon_mask_root) if anon_mask_root else None
        self.dataset_prefix = dataset_prefix
        self.split = split
        self.format = format
        self.distance_suffix = str(distance_suffix).lower()
        if self.distance_suffix not in {"sdf", "udf"}:
            raise ValueError("distance_suffix must be 'sdf' or 'udf'")
        self.configure_subseq_ids(subseq_ids)

        if modalities is None:
            self.modalities = {"images", "cameras", "surface", "geo_points"}
            if load_conditioning_latent:
                self.modalities.add("conditioning_latent")
            if load_visible_geometry_latent:
                self.modalities.add("visible_geometry_latent")
            if image_feature_root is not None:
                self.modalities.add("image_features")
        else:
            self.modalities = {str(modality) for modality in modalities}
            unknown = self.modalities - self.AVAILABLE_MODALITIES
            if unknown:
                raise ValueError(f"Unknown 3D-FRONT modalities: {sorted(unknown)}")
        if gt_latent_root is not None:
            self.modalities.add("gt_latent")
        if "gt_latent" in self.modalities and gt_latent_root is None:
            raise ValueError("gt_latent_root is required for the gt_latent modality")
        self.configure_gt_latents(gt_latent_root, gt_latent_source)
        self.configure_conditioning_latents(
            "conditioning_latent" in self.modalities, conditioning_latent_root
        )
        self.configure_visible_geometry_latents(
            "visible_geometry_latent" in self.modalities, visible_geometry_latent_root
        )
        self.image_feature_root = Path(image_feature_root) if image_feature_root else None
        if "image_features" in self.modalities and (
            self.image_feature_root is None or not self.image_feature_root.is_dir()
        ):
            raise ValueError(f"Image feature root does not exist: {self.image_feature_root}")

        if samples_list_path is None:
            self.samples_list = self._discover_samples()
        else:
            self.samples_list = json.loads(Path(samples_list_path).read_text())
        parsed = [parse_panorama_sample(sample) for sample in self.samples_list]
        self.sequence_list = list(dict.fromkeys(sample.room for sample in parsed))
        self.data_store = {}
        self._load_metadata(parsed)
        self.debug_vis_dir = "debug_points/3dfront_udf_vis"

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        sample = parse_panorama_sample(self.samples_list[index])
        return self.get_data(seq_name=sample.room, id=sample.frame)

    @staticmethod
    def _cube_dir_name(room_name, frame):
        return f"{room_name}_panorama_{int(frame):03d}_cube_95"

    def _discover_samples(self):
        samples = []
        for cube_dir in sorted(self.data_root.glob("*/*/*_panorama_*_cube_95")):
            if not self._has_complete_cube_faces(cube_dir):
                continue
            room = cube_dir.parent.relative_to(self.data_root)
            frame = int(cube_dir.name[:-len("_cube_95")].rsplit("_panorama_", 1)[1])
            samples.append(f"{room.as_posix()}/{room.name}_panorama_{frame:03d}.png")
        return samples

    def _load_metadata(self, parsed_samples):
        for sample in parsed_samples:
            room = Path(sample.room)
            frame_id = f"{sample.frame:05d}"
            cube_name = self._cube_dir_name(room.name, sample.frame)
            cube_dir = self.data_root / room / cube_name
            mask_dir = self.anon_mask_root / room / cube_name if self.anon_mask_root else None
            prefix = self.sdf_data_root / room.parent / f"{room.name}_full"
            self.data_store.setdefault(sample.room, {})[frame_id] = {
                "subsequence_images": [cube_dir / f"{face:04d}.jpg" for face in range(6)],
                "subsequence_anon_masks": (
                    [mask_dir / f"{face:04d}.png" for face in range(6)]
                    if mask_dir is not None else None
                ),
                "camera_data": cube_dir / "camera_data.json",
                "pose_path": self.pose_root / room / "extrinsics.json",
                "panorama_name": f"{room.name}_panorama_{sample.frame:03d}.png",
                "surface": Path(f"{prefix}_surface.npz"),
                "geo_points": Path(f"{prefix}_{self.distance_suffix}.npz"),
            }

    def _load_panorama_w2c(self, annotation):
        poses = json.loads(Path(annotation["pose_path"]).read_text())
        blender_w2c = np.asarray(poses[annotation["panorama_name"]], dtype=np.float32)
        if blender_w2c.shape != (4, 4):
            raise ValueError(f"Expected 4x4 panorama pose, got {blender_w2c.shape}")
        return world_to_opencv_camera(blender_w2c)

    def _load_image_features(self, seq_name, annotation):
        frame = int(Path(annotation["panorama_name"]).stem.rsplit("_", 1)[1])
        path = self.image_feature_root / seq_name / f"{frame:05d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing panorama image features: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return self._load_panorama_image_features(checkpoint, path)

    def _load_views(self, annotation, subseq_ids, load_images, load_cameras, need_transform):
        panorama_w2c = self._load_panorama_w2c(annotation) if need_transform else None
        # Geometry-only ShapeVAE training uses the panorama camera frame and
        # does not require materialized cubemap faces or camera_data.json.
        if need_transform and not load_images and not load_cameras:
            return [], [], [], panorama_w2c
        camera_data = json.loads(Path(annotation["camera_data"]).read_text()) if need_transform else None
        images, extrinsics, intrinsics = [], [], []
        first_w2c = None
        for order, subseq_id in enumerate(subseq_ids):
            face = int(subseq_id)
            image_path = annotation["subsequence_images"][face]
            original_size = None
            if load_images:
                image = read_image_cv2(str(image_path))
                original_size = image.shape[:2]
                image = self.resize_image(image, cv2.INTER_LANCZOS4)
                mask_paths = annotation["subsequence_anon_masks"]
                if mask_paths is not None:
                    mask = cv2.imread(str(mask_paths[face]), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise FileNotFoundError(f"Missing 3D-FRONT mask: {mask_paths[face]}")
                    mask = self.resize_image(mask, cv2.INTER_NEAREST)
                    image[mask == 0] = 0
                images.append(self._image_to_tensor(image))
            if not need_transform:
                continue
            face_data = camera_data[f"{face:04d}"]
            image_w2c = np.asarray(face_data["extrinsics"], dtype=np.float32) @ panorama_w2c
            if order == 0:
                first_w2c = image_w2c
            if load_cameras:
                if original_size is None:
                    original_size = read_image_cv2(str(image_path)).shape[:2]
                intrinsics.append(self._intrinsics_to_resized_pixels(
                    np.asarray(face_data["intrinsics"], dtype=np.float32),
                    original_size, (self.img_size, self.img_size),
                ))
                extrinsics.append(torch.from_numpy(image_w2c).float())
        return images, extrinsics, intrinsics, first_w2c

    def _load_geometry(self, annotation, rng, world_to_camera, load_surface, load_geo_points):
        surface = geo_points = None
        if load_surface:
            with np.load(annotation["surface"], allow_pickle=True) as data:
                surface = self.load_surface_points(
                    rng, random_surface=data["random_surface"],
                    sharpedge_surface=data["sharp_surface"] if self.pc_sharpedge_size else None,
                    **self.surface_rgb_kwargs(data),
                )
        if load_geo_points:
            with np.load(annotation["geo_points"], allow_pickle=True) as data:
                geo_points = self.load_sdf_points(
                    rng, vol_points=data["vol_points"], vol_label=data["vol_label"],
                    near_points=data["random_near_points"], near_label=data["random_near_label"],
                    sharpedge_points=data["sharp_near_points"], sharpedge_label=data["sharp_near_label"],
                    **self.sdf_rgb_kwargs(data),
                )
        geo_points = self.add_exact_surface_points(rng, surface, geo_points)
        transform = torch.from_numpy(world_to_camera).float()
        transform = self.geometry_input_transform(transform)
        if surface is not None:
            surface[:, :3] = surface[:, :3] @ transform[:3, :3].T + transform[:3, 3]
            if self.return_normal:
                surface[:, 3:6] = surface[:, 3:6] @ transform[:3, :3].T
        if geo_points is not None:
            geo_points[:, :3] = geo_points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        return surface, geo_points

    def get_data(self, seq_name=None, id=None, seq_index=None, img_per_seq=1, subseq_ids=None, aspect_ratio=1.0):
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]
        frame_id = f"{int(id):05d}"
        ids = [frame_id]
        annotation = self.data_store[seq_name][frame_id]
        subseq_ids = self.select_subseq_ids(subseq_ids)
        load_images = "images" in self.modalities
        load_cameras = "cameras" in self.modalities
        load_surface = "surface" in self.modalities
        load_geo_points = "geo_points" in self.modalities
        load_geometry = load_surface or load_geo_points
        images, extrinsics, intrinsics, first_w2c = self._load_views(
            annotation, subseq_ids, load_images, load_cameras, load_cameras or load_geometry
        )
        surface = geo_points = None
        if load_geometry:
            rng = self._pointcloud_rng(seq_name, frame_id, int(subseq_ids[0]))
            surface, geo_points = self._load_geometry(
                annotation, rng, first_w2c, load_surface, load_geo_points
            )
        batch = {
            "seq_name": f"{self.dataset_prefix}_{seq_name.replace('/', '_')}",
            "id": frame_id,
            "subseq_ids": torch.as_tensor(subseq_ids, dtype=torch.long),
        }
        if load_images:
            batch["images"] = images
        if load_cameras:
            batch["extrinsics"] = self._normalize_extrinsics(
                torch.stack(extrinsics).unsqueeze(0), translation_scale=1.0
            ).squeeze(0)
            batch["intrinsics"] = torch.from_numpy(np.asarray(intrinsics))
        if surface is not None:
            batch["surface"] = surface
        if geo_points is not None:
            batch["geo_points"] = geo_points
        if self.load_conditioning_latent:
            batch["conditioning_latents"] = self._load_conditioning_latent(seq_name, ids)
        if self.load_visible_geometry_latent:
            batch["visible_geometry_latents"] = self._load_visible_geometry_latent(seq_name, ids)
        if "image_features" in self.modalities:
            batch.update(self._load_image_features(seq_name, annotation))
        if self.load_gt_latent:
            latent = self._load_gt_latent(seq_name, ids)
            batch.update(latent if isinstance(latent, dict) else {"gt_latents": latent})
        return batch


__all__ = ["Front3DSDFDataset"]
