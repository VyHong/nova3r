"""SAGE-10K cubemap image and UDF dataset for Mesh2SDF training."""

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

from training.mesh2sdf.data.sdf_dataset_mixin import SDFDatasetMixin
from training.nova3r.data.dataset_utils import read_image_cv2
from nova3r.utils.batch_debug import save_batch_debug


class SAGE10KSDFDataset(SDFDatasetMixin):
    """Load rendered SAGE-10K cubemaps and scene-level UDF samples.

    Expected image layout::

        data_root/<scene>/<scene>_panorama_<frame>_cube_95/
            0000.jpg ... 0005.jpg
            camera_data.json

    ``camera_data.json`` provides each face-to-panorama transform. Panorama
    world-to-camera poses are loaded from ``pose_root/<scene>.json`` and
    composed at load time. Geometry defaults to
    ``sdf_data_root/<scene>/scene_{surface,udf}.npz``.
    """

    PANORAMA_TO_OPENCV = np.diag(np.asarray([1.0, -1.0, -1.0, 1.0], dtype=np.float32))
    SURFACE_FILENAME = "scene_surface.npz"
    GEO_POINTS_FILENAME = "scene_udf.npz"

    def __init__(
        self,
        common_conf,
        data_root,
        split="train",
        samples_list_path=None,
        format="mesh",
        sdf_data_root=None,
        pose_root="/mnt/projects/theses/vyhong/sage-10k/poses",
        dataset_prefix="sage10k",
        load_conditioning_latent=False,
        conditioning_latent_root=None,
        subseq_ids=None,
    ):
        super().__init__(common_conf)
        self.data_root = Path(data_root)
        self.sdf_data_root = Path(sdf_data_root) if sdf_data_root else self.data_root
        self.pose_root = Path(pose_root)
        self.surface_filename = self.SURFACE_FILENAME
        self.geo_points_filename = self.GEO_POINTS_FILENAME
        self.dataset_prefix = dataset_prefix
        self.split = split
        self.format = format
        self.configure_subseq_ids(subseq_ids)
        self.configure_conditioning_latents(load_conditioning_latent, conditioning_latent_root)
        if samples_list_path is None:
            self.samples_list = self._discover_samples()
        else:
            with open(samples_list_path, "r", encoding="utf-8") as file:
                self.samples_list = json.load(file)
        self.sequence_list = list(dict.fromkeys(item.split(" ")[0] for item in self.samples_list))
        self.data_store = {}
        self._load_metadata()
        self.debug_vis_dir = "debug_points/sage10k_udf_vis"

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        seq_name, frame_id = self.samples_list[index].split(" ")
        return self.get_data(seq_name=seq_name, id=frame_id)

    @staticmethod
    def _panorama_index(cube_dir):
        marker = "_panorama_"
        prefix = cube_dir.name[: -len("_cube_95")]
        if marker not in prefix:
            raise ValueError(f"Invalid SAGE cubemap directory name: {cube_dir.name}")
        return int(prefix.rsplit(marker, 1)[1])

    def _discover_samples(self):
        if not self.data_root.is_dir():
            raise ValueError(f"Data root directory does not exist: {self.data_root}")

        samples = []
        for scene_dir in sorted(path for path in self.data_root.iterdir() if path.is_dir()):
            cubes = sorted(scene_dir.glob("*_panorama_*_cube_95"), key=self._panorama_index)
            for cube_dir in cubes:
                if self._has_complete_cube_faces(cube_dir):
                    samples.append(f"{scene_dir.name} {self._panorama_index(cube_dir):05d}")
        return samples

    def _load_metadata(self):
        for sample in self.samples_list:
            scene, frame_id = sample.split()
            frame_idx = int(frame_id)
            cube_dir = self.data_root / scene / f"{scene}_panorama_{frame_idx:03d}_cube_95"
            self.data_store.setdefault(scene, {})[frame_id] = {
                "image_dir": cube_dir,
                "subsequence_images": [cube_dir / f"{face_idx:04d}.jpg" for face_idx in range(6)],
                "camera_data": cube_dir / "camera_data.json",
                "pose_path": self.pose_root / f"{scene}.json",
                "panorama_name": f"{scene}_panorama_{frame_idx:03d}.png",
                "surface": self.sdf_data_root / scene / self.surface_filename,
                "geo_points": self.sdf_data_root / scene / self.geo_points_filename,
                "frame_idx": frame_idx,
            }

    def _load_panorama_w2c(self, pose_path, panorama_name):
        pose_path = Path(pose_path)
        poses = json.loads(pose_path.read_text())
        panorama_w2c = np.asarray(poses[panorama_name], dtype=np.float32)
        return self.PANORAMA_TO_OPENCV @ panorama_w2c

    def _load_geometry(self, anno, rng, world_to_first_camera):
        surface_path = Path(anno["surface"])
        geo_points_path = Path(anno["geo_points"])

        with np.load(surface_path, allow_pickle=True) as data:
            surface = self.load_surface_points(
                rng,
                random_surface=data["random_surface"],
                sharpedge_surface=data["sharp_surface"],
            )
        with np.load(geo_points_path, allow_pickle=True) as data:
            geo_points = self.load_sdf_points(
                rng,
                vol_points=data["vol_points"],
                vol_label=data["vol_label"],
                near_points=data["random_near_points"],
                near_label=data["random_near_label"],
                sharpedge_points=data["sharp_near_points"],
                sharpedge_label=data["sharp_near_label"],
            )

        # The UDF points and panorama poses both use the exported glTF scene
        # coordinates, so image_w2c is already the complete geometry transform.
        transform = torch.from_numpy(world_to_first_camera).float()
        surface[:, :3] = (transform[:3, :3] @ surface[:, :3].T).T + transform[:3, 3]
        if surface.shape[1] >= 6:
            surface[:, 3:6] = (transform[:3, :3] @ surface[:, 3:6].T).T
        geo_points[:, :3] = (transform[:3, :3] @ geo_points[:, :3].T).T + transform[:3, 3]
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
        subseq_ids = self.select_subseq_ids(subseq_ids)

        metadata = self.data_store[seq_name]

        ids = [f"{int(id):05d}"]

        images, extrinsics, intrinsics = [], [], []
        surface = geo_points = None
        for anno in (metadata[item] for item in ids):
            camera_data = json.loads(Path(anno["camera_data"]).read_text())
            panorama_w2c = self._load_panorama_w2c(anno["pose_path"], anno["panorama_name"])
            for subseq_order, subseq_id in enumerate(subseq_ids):
                face_key = f"{int(subseq_id):04d}"
                face = camera_data[face_key]

                image = read_image_cv2(str(anno["subsequence_images"][int(subseq_id)]))
                original_size = image.shape[:2]
                image = self.resize_image(image, cv2.INTER_LANCZOS4)
                images.append(self._image_to_tensor(image))
                intrinsics.append(
                    self._intrinsics_to_resized_pixels(
                        np.asarray(face["intrinsics"], dtype=np.float32),
                        original_size,
                        image.shape[:2],
                    )
                )
                face_w2c = np.asarray(face["extrinsics"], dtype=np.float32)
                image_extrinsics = face_w2c @ panorama_w2c
                extrinsics.append(torch.from_numpy(image_extrinsics))

                if subseq_order == 0:
                    rng = self._pointcloud_rng(seq_name, ids[0], int(subseq_ids[0]))
                    surface, geo_points = self._load_geometry(anno, rng, image_extrinsics)

        normalized_extrinsics = self._normalize_extrinsics(torch.stack(extrinsics).unsqueeze(0), translation_scale=1.0).squeeze(0)

        batch = {
            "seq_name": f"{self.dataset_prefix}_{seq_name}",
            "id": ids[0] if len(ids) == 1 else ids,
            "subseq_ids": torch.as_tensor(subseq_ids, dtype=torch.long),
            "images": images,
            "extrinsics": normalized_extrinsics,
            "intrinsics": torch.from_numpy(np.asarray(intrinsics)),
            "surface": surface,
            "geo_points": geo_points,
        }
        if self.load_conditioning_latent:
            batch["conditioning_latents"] = self._load_conditioning_latent(seq_name, ids)
        return batch


def _main():
    import argparse
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description="Load one SAGE-10K sample and inspect the dataset return values.")
    parser.add_argument("--data-root", default="/tmp/datasets/sage-10k/cubemap_data/cubemaps")
    parser.add_argument("--sdf-data-root", default="/tmp/datasets/sage-10k/udf")
    parser.add_argument("--pose-root", default="/mnt/projects/theses/vyhong/sage-10k/poses")
    parser.add_argument("--samples-list-path", default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--frame-id", type=int, default=None)
    parser.add_argument("--subseq-ids", default="4,0,5,1,3,2", help="Comma-separated list of subsequence IDs to load")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--pc-size", type=int, default=8192)
    parser.add_argument("--pc-sharpedge-size", type=int, default=2048)
    parser.add_argument("--sdf-size", type=int, default=32768)
    parser.add_argument("--sdf-near-size", type=int, default=32768)
    parser.add_argument("--sdf-sharpedge-size", type=int, default=0)
    parser.add_argument("--preview-rows", type=int, default=3)
    parser.add_argument("--output-dir", default="debug_points/sage10k_dataset_main")
    args = parser.parse_args()

    subseq_ids = [int(item) for item in args.subseq_ids.split(",") if item.strip()]
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
    dataset = SAGE10KSDFDataset(
        common_conf=common_conf,
        data_root=args.data_root,
        split="test",
        samples_list_path=args.samples_list_path,
        sdf_data_root=args.sdf_data_root,
        pose_root=args.pose_root,
        subseq_ids=subseq_ids,
    )
    if not dataset.samples_list:
        raise RuntimeError(f"No complete cubemaps found under {args.data_root}")

    if args.scene is not None:
        frame_id = f"{(args.frame_id or 0):05d}"
        sample = f"{args.scene} {frame_id}"
        if sample not in dataset.samples_list:
            raise ValueError(f"Sample is not a discovered complete cubemap: {sample}")
    elif args.index is not None:
        sample = dataset.samples_list[args.index]
    else:
        sample = next(
            (
                item
                for item in dataset.samples_list
                if (dataset.data_store[item.split()[0]][item.split()[1]]["surface"]).is_file()
                and (dataset.data_store[item.split()[0]][item.split()[1]]["geo_points"]).is_file()
            ),
            None,
        )
        if sample is None:
            raise RuntimeError("No discovered cubemap has both scene_surface.npz and scene_udf.npz " f"under {args.sdf_data_root}")

    scene, frame_id = sample.split()
    item = dataset.get_data(seq_name=scene, id=frame_id, subseq_ids=subseq_ids)
    print(f"Loaded {sample} from {len(dataset.samples_list)} discovered sample(s)")

    output_dir = Path(args.output_dir) / f"{scene}_{frame_id}"
    save_batch_debug(item, output_dir, batched=False)
    print(f"Saved cubemap, extrinsics, surface points, and geo points to {output_dir}")


if __name__ == "__main__":
    _main()
