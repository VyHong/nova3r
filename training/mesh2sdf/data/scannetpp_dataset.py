"""ScanNet++ cubemap image and SDF dataset for Mesh2SDF training."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from training.nova3r.data.dataset_utils import read_image_cv2
from training.mesh2sdf.data.sdf_dataset_mixin import SDFDatasetMixin
from nova3r.utils.batch_debug import save_batch_debug
from eval.baselines.common import resolve_scannetpp_room


class ScanNetPPSDFDataset(SDFDatasetMixin):
    """Load ScanNet++ cubemap panoramas with scanner poses and UDF samples."""

    AVAILABLE_MODALITIES = frozenset({
        "images", "cameras", "surface", "geo_points", "conditioning_latent",
        "visible_geometry_latent", "image_features", "gt_latent",
    })

    SCANNER_AXIS_CONVERSION = np.asarray(
        [[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )

    def __init__(
        self,
        common_conf,
        data_root,
        split="train",
        samples_list_path=None,
        format="mesh",
        sdf_data_root=None,
        pose_root="/mnt/projects/theses/vyhong/scannetpp/scan_poses",
        pose_filename="scanner_poses.json",
        surface_filename="mesh_aligned_0.05_surface.npz",
        geo_points_filename="mesh_aligned_0.05_udf.npz",
        dataset_prefix="scannetpp",
        load_conditioning_latent=False,
        conditioning_latent_root=None,
        image_feature_root=None,
        subseq_ids=None,
        load_visible_geometry_latent=False,
        visible_geometry_latent_root=None,
        modalities=None,
        gt_latent_root=None,
        gt_latent_source="latent",
        geometry_coordinate_frame=None,
        room_data_root=None,
        room_gt_latent_root=None,
    ):
        super().__init__(common_conf)
        if geometry_coordinate_frame is not None:
            self.configure_geometry_coordinate_frame(geometry_coordinate_frame)
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.data_root = Path(data_root)
        self.sdf_data_root = Path(sdf_data_root) if sdf_data_root else self.data_root
        self.pose_root = Path(pose_root)
        self.pose_filename = pose_filename
        self.surface_filename = surface_filename
        self.geo_points_filename = geo_points_filename
        self.room_data_root = Path(room_data_root) if room_data_root is not None else None
        self.room_gt_latent_root = Path(room_gt_latent_root) if room_gt_latent_root is not None else None
        if self.room_data_root is not None and not (self.room_data_root / "rooms").is_dir():
            raise FileNotFoundError(f"Missing ScanNet++ room exports: {self.room_data_root / 'rooms'}")
        self.dataset_prefix = dataset_prefix
        self.split = split
        self.format = format
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
                raise ValueError(f"Unknown ScanNet++ modalities: {sorted(unknown)}")
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
        self.image_feature_root = (
            Path(image_feature_root) if image_feature_root is not None else None
        )
        if "image_features" in self.modalities and (
            self.image_feature_root is None or not self.image_feature_root.is_dir()
        ):
            raise ValueError(
                f"Image feature root does not exist: {self.image_feature_root}"
            )

        if samples_list_path is None:
            self.samples_list = self._discover_samples()
        else:
            with open(samples_list_path, "r") as file:
                self.samples_list = json.load(file)
        self.sequence_list = list(dict.fromkeys(item.split(" ")[0] for item in self.samples_list))

        self.data_store = {}
        self._load_metadata()

        self.pc_size = common_conf.pc_size
        self.pc_sharpedge_size = common_conf.pc_sharpedge_size
        self.pointcloud_sample = getattr(common_conf, "pointcloud_sample", True)
        self.return_normal = common_conf.return_normal
        self.sharpedge_label = common_conf.sharpedge_label

        self.sdf_size = common_conf.sdf_size
        self.sdf_near_size = common_conf.sdf_near_size
        self.sdf_sharpedge_size = common_conf.sdf_sharpedge_size
        self.sdf_label = common_conf.sdf_label
        self.debug_vis_dir = "debug_points/scannetpp_udf_vis"

    def __len__(self):
        return len(self.samples_list)

    def _pointcloud_rng(self, seq_name, id, first_subseq_id):
        if self.split != "train":
            return self.deterministic_rng(type(self).__name__, self.split, seq_name, id, int(first_subseq_id))
        if self.pointcloud_sample:
            return np.random.default_rng()

        return self.deterministic_rng(type(self).__name__, self.split, seq_name, id, int(first_subseq_id))

    def resize_image(self, image, interpolation=cv2.INTER_LANCZOS4):
        return cv2.resize(image, (self.img_size, self.img_size), interpolation=interpolation)

    @staticmethod
    def _image_to_tensor(image):
        image = np.ascontiguousarray(image)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        return tensor.div_(255.0)

    @staticmethod
    def _intrinsics_to_resized_pixels(intrinsics, orig_size_hw, new_size_hw):
        intrinsics = intrinsics.copy()
        uses_normalized_intrinsics = (
            abs(intrinsics[0, 0]) <= 2.0 and abs(intrinsics[1, 1]) <= 2.0 and abs(intrinsics[0, 2]) <= 2.0 and abs(intrinsics[1, 2]) <= 2.0
        )
        if uses_normalized_intrinsics:
            intrinsics[0, :] *= float(orig_size_hw[1])
            intrinsics[1, :] *= float(orig_size_hw[0])

        intrinsics[0, :] *= new_size_hw[1] / float(orig_size_hw[1])
        intrinsics[1, :] *= new_size_hw[0] / float(orig_size_hw[0])
        return intrinsics

    def _discover_samples(self):
        if not self.data_root.is_dir():
            raise ValueError(f"Data root directory does not exist: {self.data_root}")

        samples = []
        for image_dir in sorted(self.data_root.glob("*/panocam/images")):
            scene = image_dir.parents[1].name
            for cube_dir in sorted(image_dir.glob("*_cube_95"), key=self._cube_dir_sort_key):
                if not cube_dir.is_dir():
                    continue
                frame_id = self._frame_id_from_cube_dir(cube_dir)
                if self._has_complete_cube_faces(cube_dir):
                    samples.append(f"{scene} {frame_id:05d}")
        return samples

    @staticmethod
    def _cube_dir_sort_key(path):
        return int(path.name[: -len("_cube_95")])

    @staticmethod
    def _frame_id_from_cube_dir(path):
        return int(path.name[: -len("_cube_95")])

    @staticmethod
    def _has_complete_cube_faces(path):
        return all((path / f"{face_idx:04d}.jpg").is_file() for face_idx in range(6))

    def _load_metadata(self):
        for sample in self.samples_list:
            scene, frame_id = sample.split(" ")
            frame_idx = int(frame_id)
            room = (
                resolve_scannetpp_room(self.room_data_root, scene, frame_idx)
                if self.room_data_root is not None else None
            )
            image_dir = self.data_root / scene / "panocam" / "images" / f"{frame_idx}_cube_95"
            anon_mask_dir = self.data_root / scene / "panocam" / "anon_mask" / f"{frame_idx}_cube_95"
            self.data_store.setdefault(scene, {})[frame_id] = {
                "image_dir": image_dir,
                "subsequence_images": [image_dir / f"{face_idx:04d}.jpg" for face_idx in range(6)],
                "subsequence_anon_masks": [anon_mask_dir / f"{face_idx:04d}.png" for face_idx in range(6)],
                "camera_data": image_dir / "camera_data.json",
                "pose_path": self.pose_root / scene / "scans" / self.pose_filename,
                "surface": room / "mesh_surface.npz" if room is not None else self.sdf_data_root / scene / "scans" / self.surface_filename,
                "geo_points": room / "mesh_udf.npz" if room is not None else self.sdf_data_root / scene / "scans" / self.geo_points_filename,
                "room": room,
                "frame_idx": frame_idx,
            }

    def _load_scanner_w2c(self, pose_path, frame_idx):
        with open(pose_path, "r") as file:
            scanner_poses = json.load(file)
        if frame_idx >= len(scanner_poses):
            raise IndexError(f"Pose index {frame_idx} is out of range for {pose_path} with {len(scanner_poses)} poses.")

        scanner_c2w = np.asarray(scanner_poses[frame_idx], dtype=np.float32)
        if scanner_c2w.shape != (4, 4):
            raise ValueError(f"Expected 4x4 scanner pose at index {frame_idx} in {pose_path}, got {scanner_c2w.shape}.")
        scanner_w2c = np.linalg.inv(scanner_c2w).astype(np.float32)
        basis = self.SCANNER_AXIS_CONVERSION
        return basis @ scanner_w2c @ basis.T

    def _load_image_features(self, seq_name, frame_id):
        feature_path = self.image_feature_root / seq_name / f"{int(frame_id):05d}.pt"
        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing panorama image features: {feature_path}")
        checkpoint = torch.load(feature_path, map_location="cpu", weights_only=True)
        return self._load_panorama_image_features(checkpoint, feature_path)

    def _load_geometry(self, anno, rng, total_transform, *, load_surface=True, load_geo_points=True):
        surface_path = Path(anno["surface"])
        geo_points_path = Path(anno["geo_points"])
        if (load_surface and not surface_path.is_file()) or (load_geo_points and not geo_points_path.is_file()):
            raise FileNotFoundError(
                "Missing ScanNet++ UDF files. Expected "
                f"surface={surface_path} and geo_points={geo_points_path}. "
                "Generate them or pass sdf_data_root/surface_filename/geo_points_filename in the dataset config."
            )

        surface = geo_points = mesh_bounds = None
        if load_surface:
            with np.load(surface_path, allow_pickle=True) as surface_data:
                surface = self.load_surface_points(
                    rng,
                    random_surface=surface_data["random_surface"],
                    sharpedge_surface=(
                        surface_data["sharp_surface"] if self.pc_sharpedge_size > 0 else None
                    ),
                    **self.surface_rgb_kwargs(surface_data),
                )

        if load_geo_points:
            with np.load(geo_points_path, allow_pickle=True) as geo_points_data:
                mesh_bounds = self.load_mesh_bounds(geo_points_data)
                geo_points = self.load_sdf_points(
                    rng,
                    vol_points=geo_points_data["vol_points"],
                    vol_label=geo_points_data["vol_label"],
                    near_points=geo_points_data["random_near_points"],
                    near_label=geo_points_data["random_near_label"],
                    sharpedge_points=geo_points_data["sharp_near_points"],
                    sharpedge_label=geo_points_data["sharp_near_label"],
                    **self.sdf_rgb_kwargs(geo_points_data),
                )

        geo_points = self.add_exact_surface_points(rng, surface, geo_points)

        total_transform = torch.from_numpy(total_transform).float()
        pointcloud_axis_conversion = torch.from_numpy(self.SCANNER_AXIS_CONVERSION).float()
        total_transform = total_transform @ pointcloud_axis_conversion
        total_transform = self.geometry_input_transform(total_transform)
        if surface is not None:
            surface[:, :3] = (total_transform[:3, :3] @ surface[:, :3].T).T + total_transform[:3, 3]
            if self.return_normal:
                surface[:, 3:6] = (total_transform[:3, :3] @ surface[:, 3:6].T).T
        if geo_points is not None:
            geo_points[:, :3] = (total_transform[:3, :3] @ geo_points[:, :3].T).T + total_transform[:3, 3]
        if mesh_bounds is not None:
            mesh_bounds = mesh_bounds @ total_transform[:3, :3].T + total_transform[:3, 3]
        return surface, geo_points, mesh_bounds

    def get_data(
        self,
        seq_name=None,
        id=None,
        seq_index=None,
        img_per_seq=1,
        subseq_ids=None,
        aspect_ratio=1.0,
    ):
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]
        subseq_ids = self.select_subseq_ids(subseq_ids)

        metadata = self.data_store[seq_name]

        ids = [f"{int(id):05d}"]

        annos = [metadata[item] for item in ids]
        images, extrinsics, intrinsics = [], [], []
        camera_translation_scale = 1.0
        surface = None
        geo_points = None

        load_images = "images" in self.modalities
        load_cameras = "cameras" in self.modalities
        load_surface = "surface" in self.modalities
        load_geo_points = "geo_points" in self.modalities
        load_geometry = load_surface or load_geo_points
        need_camera_transform = load_cameras or load_geometry

        for anno in annos:
            if need_camera_transform:
                with open(anno["camera_data"], "r") as file:
                    camera_data = json.load(file)
                scanner_w2c = self._load_scanner_w2c(anno["pose_path"], anno["frame_idx"])

            for subseq_order, subseq_id in enumerate(subseq_ids):
                face_key = f"{int(subseq_id):04d}"
                image_path = anno["subsequence_images"][int(subseq_id)]
                orig_size_hw = None
                if load_images:
                    image = read_image_cv2(str(image_path))
                    orig_size_hw = image.shape[:2]
                    image = self.resize_image(image, cv2.INTER_LANCZOS4)
                    image = self._image_to_tensor(image)

                    anon_mask_path = anno["subsequence_anon_masks"][int(subseq_id)]
                    anon_mask = cv2.imread(str(anon_mask_path), cv2.IMREAD_GRAYSCALE)
                    if anon_mask is None:
                        raise FileNotFoundError(f"Missing ScanNet++ anonymization mask: {anon_mask_path}")
                    anon_mask = self.resize_image(anon_mask, cv2.INTER_NEAREST)
                    invalid_mask = torch.from_numpy(np.ascontiguousarray(anon_mask == 0))
                    image[:, invalid_mask] = 0.0
                    images.append(image)

                if not need_camera_transform:
                    continue

                face_w2c = np.asarray(camera_data[face_key]["extrinsics"], dtype=np.float32)
                image_extrinsics = face_w2c @ scanner_w2c
                if subseq_order == 0 and load_geometry:
                    rng = self._pointcloud_rng(seq_name, ids[0], int(subseq_ids[0]))
                    surface, geo_points, mesh_bounds = self._load_geometry(
                        anno, rng, image_extrinsics,
                        load_surface=load_surface, load_geo_points=load_geo_points,
                    )

                if load_cameras:
                    if orig_size_hw is None:
                        orig_size_hw = read_image_cv2(str(image_path)).shape[:2]
                    intrinsics.append(self._intrinsics_to_resized_pixels(
                        np.asarray(camera_data[face_key]["intrinsics"], dtype=np.float32),
                        orig_size_hw=orig_size_hw,
                        new_size_hw=(self.img_size, self.img_size),
                    ))
                    extrinsics.append(torch.from_numpy(image_extrinsics).float())

        batch = {
            "seq_name": f"{self.dataset_prefix}_{seq_name}",
            "id": ids[0] if len(ids) == 1 else ids,
            "subseq_ids": torch.as_tensor(subseq_ids, dtype=torch.long),
        }
        if load_images:
            batch["images"] = images
        if load_cameras:
            batch["extrinsics"] = self._normalize_extrinsics(
                torch.stack(extrinsics).unsqueeze(0),
                translation_scale=camera_translation_scale,
            ).squeeze(0)
            batch["intrinsics"] = torch.from_numpy(np.asarray(intrinsics))
        if surface is not None:
            batch["surface"] = surface
        if geo_points is not None:
            batch["geo_points"] = geo_points
            batch["mesh_bounds_corners"] = mesh_bounds
        if self.load_conditioning_latent:
            batch["conditioning_latents"] = self._load_conditioning_latent(seq_name, ids)
        if self.load_visible_geometry_latent:
            batch["visible_geometry_latents"] = self._load_visible_geometry_latent(seq_name, ids)
        if "image_features" in self.modalities:
            batch.update(self._load_image_features(seq_name, ids[0]))
        if self.load_gt_latent:
            room = annos[0]["room"]
            if room is not None and self.room_gt_latent_root is None:
                raise ValueError("room_gt_latent_root is required for room-specific GT latents")
            gt_latent = self._load_gt_latent(
                seq_name, ids,
                latent_root=self.room_gt_latent_root if room is not None else None,
            )
            if isinstance(gt_latent, dict):
                batch.update(gt_latent)
            else:
                batch["gt_latents"] = gt_latent
        return batch

    def __getitem__(self, index):
        seq_name, frame_id = self.samples_list[index].split(" ")
        return self.get_data(seq_name=seq_name, id=frame_id)


def _parse_subseq_ids(value):
    ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ids:
        raise ValueError("subseq ids cannot be empty.")
    if any(item < 0 or item > 5 for item in ids):
        raise ValueError("subseq ids must be cubemap face ids in [0, 5].")
    return np.asarray(ids, dtype=np.int64)


def _main():
    import argparse
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description="Smoke-test ScanNetPPSDFDataset and export one debug sample.")
    parser.add_argument("--data-root", default="/mnt/projects/theses/vyhong/scannetpp/cubemap_data/cubemaps")
    parser.add_argument("--sdf-data-root", default="/mnt/projects/theses/vyhong/scannetpp/udf")
    parser.add_argument("--pose-root", default="/mnt/projects/theses/vyhong/scannetpp/scan_poses")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--samples-list-path", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--frame-id", default=None)
    parser.add_argument(
        "--subseq-ids",
        default="0,1,2,3,4,5",
        help="Comma-separated cubemap face order. The first face anchors surface/geo_points coordinates.",
    )
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--pc-size", type=int, default=8192)
    parser.add_argument("--pc-sharpedge-size", type=int, default=2048)
    parser.add_argument("--sdf-size", type=int, default=32768)
    parser.add_argument("--sdf-near-size", type=int, default=32768)
    parser.add_argument("--sdf-sharpedge-size", type=int, default=0)
    parser.add_argument("--output-dir", default="debug_points/scannetpp_dataset_main")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    samples_list_path = args.samples_list_path
    if samples_list_path is None:
        default_list = Path("data/scannetpp") / f"{args.split}_list.json"
        samples_list_path = str(default_list) if default_list.is_file() else None

    common_conf = SimpleNamespace(
        img_size=args.img_size,
        patch_size=14,
        aug_scale=None,
        rescale=True,
        rescale_aug=False,
        landscape_check=False,
        allow_duplicate_img=False,
        pc_size=args.pc_size,
        pc_sharpedge_size=args.pc_sharpedge_size,
        pointcloud_sample=False,
        return_normal=True,
        sharpedge_label=True,
        sdf_size=args.sdf_size,
        sdf_near_size=args.sdf_near_size,
        sdf_sharpedge_size=args.sdf_sharpedge_size,
        sdf_label=True,
    )
    dataset = ScanNetPPSDFDataset(
        common_conf,
        data_root=args.data_root,
        split=args.split,
        samples_list_path=samples_list_path,
        sdf_data_root=args.sdf_data_root,
        pose_root=args.pose_root,
    )

    if args.scene is None:
        sample = dataset.samples_list[args.index]
        scene, frame_id = sample.split(" ")
    else:
        scene = args.scene
        frame_id = f"{int(args.frame_id):05d}" if args.frame_id is not None else "00000"

    for i in range(20):
        _ = dataset[i]
    subseq_ids = _parse_subseq_ids(args.subseq_ids)
    print(f"ScanNet++ raw sample: scene={scene} frame={frame_id}")
    print("  no axis conversion, camera transform, normalization, or rotation is applied")

    if not args.no_export:
        output_dir = Path(args.output_dir) / f"{args.split}_{args.index:05d}_scannetpp_{scene}_{frame_id}"
        print(f"  wrote raw extrinsics and point cloud (no rotations) to {output_dir}")


if __name__ == "__main__":
    _main()
