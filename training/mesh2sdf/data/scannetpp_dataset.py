"""ScanNet++ cubemap image and SDF dataset for Mesh2SDF training."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from training.nova3r.data.base_dataset import BaseDataset
from training.nova3r.data.dataset_utils import read_image_cv2


class ScanNetPPSDFDataset(BaseDataset):
    """Load ScanNet++ cubemap panoramas with scanner poses and UDF samples."""

    RAW_TO_COLMAP = np.asarray(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
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
    ):
        super().__init__(common_conf)
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.data_root = Path(data_root)
        self.sdf_data_root = Path(sdf_data_root) if sdf_data_root else self.data_root
        self.pose_root = Path(pose_root)
        self.pose_filename = pose_filename
        self.surface_filename = surface_filename
        self.geo_points_filename = geo_points_filename
        self.dataset_prefix = dataset_prefix
        self.split = split
        self.format = format

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
        if self.pointcloud_sample:
            return np.random.default_rng()

        seed = abs(hash(f"{seq_name}_{id}_{int(first_subseq_id)}")) % (2**32)
        return np.random.default_rng(seed)

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
            abs(intrinsics[0, 0]) <= 2.0
            and abs(intrinsics[1, 1]) <= 2.0
            and abs(intrinsics[0, 2]) <= 2.0
            and abs(intrinsics[1, 2]) <= 2.0
        )
        if uses_normalized_intrinsics:
            intrinsics[0, :] *= float(orig_size_hw[1])
            intrinsics[1, :] *= float(orig_size_hw[0])

        intrinsics[0, :] *= new_size_hw[1] / float(orig_size_hw[1])
        intrinsics[1, :] *= new_size_hw[0] / float(orig_size_hw[0])
        return intrinsics

    def load_surface_points(self, rng, random_surface, sharpedge_surface):
        surface_normal = []
        if self.pc_size > 0:
            ind = rng.choice(random_surface.shape[0], self.pc_size, replace=False)
            random_surface = random_surface[ind]
            if self.sharpedge_label:
                sharpedge_label = np.zeros((self.pc_size, 1))
                random_surface = np.concatenate((random_surface, sharpedge_label), axis=1)
            surface_normal.append(random_surface)

        if self.pc_sharpedge_size > 0:
            ind_sharpedge = rng.choice(sharpedge_surface.shape[0], self.pc_sharpedge_size, replace=False)
            sharpedge_surface = sharpedge_surface[ind_sharpedge]
            if self.sharpedge_label:
                sharpedge_label = np.ones((self.pc_sharpedge_size, 1))
                sharpedge_surface = np.concatenate((sharpedge_surface, sharpedge_label), axis=1)
            surface_normal.append(sharpedge_surface)

        surface_normal = np.concatenate(surface_normal, axis=0)
        surface_normal = torch.FloatTensor(surface_normal)
        surface = surface_normal[:, 0:3]
        normal = F.normalize(surface_normal[:, 3:6], p=2, dim=1)
        assert surface.shape[0] == self.pc_size + self.pc_sharpedge_size

        if self.return_normal:
            surface = torch.cat([surface, normal], dim=-1)
        if self.sharpedge_label:
            surface = torch.cat([surface, surface_normal[:, -1:]], dim=-1)
        return surface

    def load_sdf_points(
        self,
        rng,
        vol_points,
        vol_label,
        near_points,
        near_label,
        sharpedge_points,
        sharpedge_label,
    ):
        sdf_points = []
        if self.sdf_size > 0:
            ind = rng.choice(vol_points.shape[0], self.sdf_size, replace=False)
            vol_sdf_points = np.concatenate((vol_points[ind], vol_label[ind][:, None]), axis=1)
            if self.sdf_label:
                sdf_label = np.zeros((self.sdf_size, 1))
                vol_sdf_points = np.concatenate((vol_sdf_points, sdf_label), axis=1)
            sdf_points.append(vol_sdf_points)

        if self.sdf_near_size > 0:
            ind_near = rng.choice(near_points.shape[0], self.sdf_near_size, replace=False)
            near_sdf_points = np.concatenate((near_points[ind_near], near_label[ind_near][:, None]), axis=1)
            if self.sdf_label:
                sdf_label = np.ones((self.sdf_near_size, 1))
                near_sdf_points = np.concatenate((near_sdf_points, sdf_label), axis=1)
            sdf_points.append(near_sdf_points)

        if self.sdf_sharpedge_size > 0:
            ind_sharpedge = rng.choice(sharpedge_points.shape[0], self.sdf_sharpedge_size, replace=False)
            sharpedge_sdf_points = np.concatenate((sharpedge_points[ind_sharpedge], sharpedge_label[ind_sharpedge][:, None]), axis=1)
            if self.sdf_label:
                sdf_label = np.ones((self.sdf_sharpedge_size, 1)) * 2
                sharpedge_sdf_points = np.concatenate((sharpedge_sdf_points, sdf_label), axis=1)
            sdf_points.append(sharpedge_sdf_points)

        return torch.FloatTensor(np.concatenate(sdf_points, axis=0))

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
            image_dir = self.data_root / scene / "panocam" / "images" / f"{frame_idx}_cube_95"
            self.data_store.setdefault(scene, {})[frame_id] = {
                "image_dir": image_dir,
                "subsequence_images": [image_dir / f"{face_idx:04d}.jpg" for face_idx in range(6)],
                "camera_data": image_dir / "camera_data.json",
                "pose_path": self.pose_root / scene / "scans" / self.pose_filename,
                "surface": self.sdf_data_root / scene / "scans" / self.surface_filename,
                "geo_points": self.sdf_data_root / scene / "scans" / self.geo_points_filename,
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
        return np.linalg.inv(scanner_c2w).astype(np.float32)

    def _load_geometry(self, anno, rng, total_transform):
        surface_path = Path(anno["surface"])
        geo_points_path = Path(anno["geo_points"])
        if not surface_path.is_file() or not geo_points_path.is_file():
            raise FileNotFoundError(
                "Missing ScanNet++ UDF files. Expected "
                f"surface={surface_path} and geo_points={geo_points_path}. "
                "Generate them or pass sdf_data_root/surface_filename/geo_points_filename in the dataset config."
            )

        with np.load(surface_path, allow_pickle=True) as surface_data:
            surface = self.load_surface_points(
                rng,
                random_surface=surface_data["random_surface"],
                sharpedge_surface=surface_data["sharp_surface"],
            )

        with np.load(geo_points_path, allow_pickle=True) as geo_points_data:
            geo_points = self.load_sdf_points(
                rng,
                vol_points=geo_points_data["vol_points"],
                vol_label=geo_points_data["vol_label"],
                near_points=geo_points_data["random_near_points"],
                near_label=geo_points_data["random_near_label"],
                sharpedge_points=geo_points_data["sharp_near_points"],
                sharpedge_label=geo_points_data["sharp_near_label"],
            )

        total_transform = torch.from_numpy(total_transform).float()
        surface[:, :3] = (total_transform[:3, :3] @ surface[:, :3].T).T + total_transform[:3, 3]
        if surface.shape[1] >= 6:
            surface[:, 3:6] = (total_transform[:3, :3] @ surface[:, 3:6].T).T
        geo_points[:, :3] = (total_transform[:3, :3] @ geo_points[:, :3].T).T + total_transform[:3, 3]
        return surface, geo_points

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
        if subseq_ids is None:
            subseq_ids = np.arange(6)
            if self.split == "train":
                subseq_ids = np.random.choice(subseq_ids, len(subseq_ids), replace=self.allow_duplicate_img)

        metadata = self.data_store[seq_name]
        if id is None:
            keys = list(metadata.keys())
            ids = np.random.choice(keys, img_per_seq, replace=self.allow_duplicate_img).tolist()
        elif isinstance(id, str):
            ids = [id]
        else:
            ids = [f"{int(id):05d}"]

        annos = [metadata[item] for item in ids]
        images, extrinsics, intrinsics = [], [], []
        camera_translation_scale = 1.0
        surface = None
        geo_points = None

        for anno in annos:
            with open(anno["camera_data"], "r") as file:
                camera_data = json.load(file)

            scanner_w2c = self._load_scanner_w2c(anno["pose_path"], anno["frame_idx"])
            for face_order, subseq_id in enumerate(subseq_ids):
                face_key = f"{int(subseq_id):04d}"
                image_path = anno["subsequence_images"][int(subseq_id)]
                image = read_image_cv2(str(image_path))
                orig_size_hw = image.shape[:2]
                image = self.resize_image(image, cv2.INTER_LANCZOS4)
                new_size_hw = image.shape[:2]
                image = self._image_to_tensor(image)

                subseq_intrinsics = np.asarray(camera_data[face_key]["intrinsics"], dtype=np.float32)
                subseq_intrinsics = self._intrinsics_to_resized_pixels(
                    subseq_intrinsics,
                    orig_size_hw=orig_size_hw,
                    new_size_hw=new_size_hw,
                )

                face_w2c = np.asarray(camera_data[face_key]["extrinsics"], dtype=np.float32)
                image_extrinsics = self.RAW_TO_COLMAP @ face_w2c @ scanner_w2c

                if face_order == 0:
                    rng = self._pointcloud_rng(seq_name, ids[0], int(subseq_ids[0]))
                    surface, geo_points = self._load_geometry(anno, rng, image_extrinsics)

                images.append(image)
                extrinsics.append(torch.from_numpy(image_extrinsics).float())
                intrinsics.append(subseq_intrinsics)

        ex_t_batched = torch.stack(extrinsics).unsqueeze(0)
        normalized_extrinsics = self._normalize_extrinsics(
            ex_t_batched,
            translation_scale=camera_translation_scale,
        ).squeeze(0)
        intrinsics = torch.from_numpy(np.asarray(intrinsics))

        return {
            "seq_name": f"{self.dataset_prefix}_{seq_name}",
            "id": ids[0] if len(ids) == 1 else ids,
            "subseq_ids": torch.as_tensor(subseq_ids, dtype=torch.long),
            "images": images,
            "extrinsics": normalized_extrinsics,
            "intrinsics": intrinsics,
            "surface": surface,
            "geo_points": geo_points,
        }

    def __getitem__(self, index):
        seq_name, frame_id = self.samples_list[index].split(" ")
        return self.get_data(seq_name=seq_name, id=frame_id)

    def _normalize_extrinsics(self, ex_t, translation_scale=None):
        if ex_t is None:
            return None
        transform = torch.linalg.inv(ex_t[:, :1])
        ex_t_norm = ex_t @ transform
        if translation_scale is None:
            c2ws = torch.linalg.inv(ex_t_norm)
            translations = c2ws[..., :3, 3]
            dists = translations.norm(dim=-1)
            translation_scale = torch.median(dists).clamp(min=1e-1)
        else:
            translation_scale = torch.as_tensor(
                translation_scale,
                dtype=ex_t_norm.dtype,
                device=ex_t_norm.device,
            ).clamp(min=1e-6)

        ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / translation_scale
        return ex_t_norm

    def dynamic_pad_collate_fn(self, batch):
        return default_collate(batch)


def _parse_subseq_ids(value):
    ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ids:
        raise ValueError("subseq ids cannot be empty.")
    if any(item < 0 or item > 5 for item in ids):
        raise ValueError("subseq ids must be cubemap face ids in [0, 5].")
    return np.asarray(ids, dtype=np.int64)


def _finite_summary(item):
    return {
        "images": all(torch.isfinite(image).all().item() for image in item["images"]),
        "extrinsics": torch.isfinite(item["extrinsics"]).all().item(),
        "intrinsics": torch.isfinite(item["intrinsics"]).all().item(),
        "surface": torch.isfinite(item["surface"]).all().item(),
        "geo_points": torch.isfinite(item["geo_points"]).all().item(),
    }


def _write_ply(points, path, normals=None):
    points = np.asarray(points, dtype=np.float32)
    if normals is None:
        colors = np.full((points.shape[0], 3), 180, dtype=np.uint8)
    else:
        normals = np.asarray(normals, dtype=np.float32)
        colors = np.clip((normals * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    with open(path, "w", encoding="utf-8") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points, colors):
            file.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _write_axis_ply(path, length=1.0, steps=64):
    points = [np.zeros(3, dtype=np.float32)]
    colors = [np.asarray([255, 255, 255], dtype=np.uint8)]
    axes = [
        (np.asarray([1.0, 0.0, 0.0], dtype=np.float32), np.asarray([255, 0, 0], dtype=np.uint8)),
        (np.asarray([0.0, 1.0, 0.0], dtype=np.float32), np.asarray([0, 255, 0], dtype=np.uint8)),
        (np.asarray([0.0, 0.0, 1.0], dtype=np.float32), np.asarray([0, 0, 255], dtype=np.uint8)),
    ]
    for axis, color in axes:
        for scale in np.linspace(0.0, length, steps):
            points.append(axis * scale)
            colors.append(color)

    with open(path, "w", encoding="utf-8") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points, colors):
            file.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _write_image_grid(images, path, labels):
    from PIL import Image, ImageDraw

    pil_images = []
    for image in images:
        array = image.detach().cpu().clamp(0, 1).numpy()
        array = np.transpose(array, (1, 2, 0))
        pil_images.append(Image.fromarray((array * 255.0 + 0.5).astype(np.uint8)))

    width, height = pil_images[0].size
    pad = 8
    label_height = 20
    cols = 3
    rows = int(np.ceil(len(pil_images) / cols))
    canvas = Image.new(
        "RGB",
        (cols * width + (cols + 1) * pad, rows * (height + label_height) + (rows + 1) * pad),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for idx, image in enumerate(pil_images):
        row, col = divmod(idx, cols)
        x = pad + col * (width + pad)
        y = pad + row * (height + label_height + pad)
        draw.text((x, y), labels[idx], fill=(0, 0, 0))
        canvas.paste(image, (x, y + label_height))
    canvas.save(path)


def _export_debug_item(item, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    surface = item["surface"].detach().cpu()
    normals = surface[:, 3:6].numpy() if surface.shape[1] >= 6 else None
    _write_ply(surface[:, :3].numpy(), output_dir / "surface.ply", normals=normals)
    _write_axis_ply(output_dir / "colmap_camera_axes_origin.ply")
    _write_image_grid(
        item["images"],
        output_dir / "images.png",
        [f"face {int(face_id)}" for face_id in item["subseq_ids"].detach().cpu().tolist()],
    )

    with open(output_dir / "normalized_extrinsics.json", "w", encoding="utf-8") as file:
        json.dump(item["extrinsics"].detach().cpu().numpy().tolist(), file, indent=2)
    with open(output_dir / "intrinsics.json", "w", encoding="utf-8") as file:
        json.dump(item["intrinsics"].detach().cpu().numpy().tolist(), file, indent=2)

    metadata = {
        "seq_name": item["seq_name"],
        "frame_id": item["id"],
        "subseq_ids": item["subseq_ids"].detach().cpu().tolist(),
        "surface_coordinate_frame": (
            "first selected cubemap face camera frame; "
            "COLMAP/OpenCV axes are +X right, +Y down, +Z forward"
        ),
        "first_face_id": int(item["subseq_ids"][0].item()),
        "image_shapes": [list(image.shape) for image in item["images"]],
        "extrinsics_shape": list(item["extrinsics"].shape),
        "intrinsics_shape": list(item["intrinsics"].shape),
        "surface_shape": list(item["surface"].shape),
        "geo_points_shape": list(item["geo_points"].shape),
        "all_finite": _finite_summary(item),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def _main():
    import argparse
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description="Smoke-test ScanNetPPSDFDataset and export one debug sample.")
    parser.add_argument("--data-root", default="/tmp/datasets/scannetpp/cubemap_data/cubemap_data")
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

    subseq_ids = _parse_subseq_ids(args.subseq_ids)
    item = dataset.get_data(seq_name=scene, id=frame_id, subseq_ids=subseq_ids)
    finite = _finite_summary(item)

    print(f"Loaded ScanNetPPSDFDataset split={args.split}")
    print(f"  samples={len(dataset.samples_list)} scenes={len(dataset.sequence_list)}")
    print(f"  sample={item['seq_name']} frame={item['id']} subseq_ids={item['subseq_ids'].tolist()}")
    print(f"  first face anchors surface/geo_points coordinates: face {int(item['subseq_ids'][0].item())}")
    print(f"  images={len(item['images'])} image_shape={tuple(item['images'][0].shape)}")
    print(f"  extrinsics={tuple(item['extrinsics'].shape)} intrinsics={tuple(item['intrinsics'].shape)}")
    print(f"  surface={tuple(item['surface'].shape)} geo_points={tuple(item['geo_points'].shape)}")
    print(f"  finite={finite}")
    print("  extrinsics convention: normalized COLMAP/OpenCV world-to-camera")

    if not args.no_export:
        output_dir = Path(args.output_dir) / f"{args.split}_{args.index:05d}_{item['seq_name']}_{item['id']}"
        _export_debug_item(item, output_dir)
        print(f"  wrote debug files to {output_dir}")


if __name__ == "__main__":
    _main()
