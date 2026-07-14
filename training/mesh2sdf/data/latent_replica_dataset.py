import json
from pathlib import Path

import numpy as np
import torch

from training.mesh2sdf.data.replica_dataset import ReplicaPanoSDFDataset
from training.nova3r.data.datasets.replica_utils.igibson_utils import ReplicaPanoScene


class LatentReplicaPanoSDFDataset(ReplicaPanoSDFDataset):
    """Replica dataset variant that also loads depth-pred aggregator points."""

    def _load_metadata(self):
        super()._load_metadata()

        self.aggregator_data_root = Path("datasets/ReplicaPano/depth_pred")
        for scene, sequence_metadata in self.data_store.items():
            for seq_entry_id, seq_entry_metadata in sequence_metadata.items():
                aggregator_path = self._aggregator_points_path(scene, seq_entry_id)
                latent_path = self._aggregator_latent_path(scene, seq_entry_id)
                seq_entry_metadata["aggregator_points"] = str(aggregator_path)
                seq_entry_metadata["aggregator_latents"] = str(latent_path)

    def _aggregator_points_path(self, scene, seq_entry_id=None):
        return self.aggregator_data_root / f"{scene}_{seq_entry_id}_surface.npz"

    def _aggregator_latent_path(self, scene, seq_entry_id=None):
        return self.aggregator_data_root / f"{scene}_{seq_entry_id}_latent.npz"

    def get_data(
        self,
        seq_name: str = None,
        id: int = None,
        seq_index: int = None,
        img_per_seq: int = 1,
        subseq_ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]
        if subseq_ids is None:
            subseq_ids = np.arange(6)
            if self.split == "no train":
                subseq_ids = np.random.choice(subseq_ids, len(subseq_ids), replace=self.allow_duplicate_img)

        metadata = self.data_store[seq_name]

        if id is None:
            ids = np.random.choice(len(metadata), img_per_seq, replace=self.allow_duplicate_img)
        if isinstance(id, str):
            ids = [id]

        annos = [metadata[i] for i in ids]

        images, extrinsics, intrinsics = [], [], []
        camera_translation_scale = None
        aggregator_latents = None

        for anno in annos:
            replica_scene = ReplicaPanoScene.from_pickle(anno["pkl_path"])

            with open(anno["camera_data"], "r") as f:
                camera_data = json.load(f)

            for i, subseq_id in enumerate(subseq_ids):
                subseq_intrinsics = np.array(camera_data[f"{subseq_id:04d}"]["intrinsics"], dtype=np.float32)
                image_size_hw = (self.img_size, self.img_size)
                subseq_intrinsics = self._intrinsics_to_resized_pixels(
                    subseq_intrinsics,
                    orig_size_hw=image_size_hw,
                    new_size_hw=image_size_hw,
                )
                subseq_w2c = np.array(camera_data[f"{subseq_id:04d}"]["extrinsics"])
                T_to_colmap = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

                scene_w2c = replica_scene.transform_3d.camera["world2cam3d"]
                colmap_scene_w2c = T_to_colmap @ scene_w2c @ T_to_colmap.T
                image_extrinsics = subseq_w2c @ colmap_scene_w2c

                if i == 0:
                    rng = self._pointcloud_rng(seq_name, id, subseq_ids[0])
                    surface = np.load(anno["surface"], allow_pickle=True)
                    geo_points = np.load(anno["geo_points"], allow_pickle=True)
                    aggregator_points = np.load(anno["aggregator_points"], allow_pickle=True)
                    latent_data = np.load(anno["aggregator_latents"], allow_pickle=True)
                    latent_view_ids = np.asarray(subseq_ids, dtype=np.int64)
                    patch_tokens = torch.from_numpy(latent_data["patch_tokens"][0, latent_view_ids]).float()
                    camera_token = torch.from_numpy(latent_data["camera_token"][0, latent_view_ids]).float()
                    aggregator_latents = torch.cat([camera_token[:, None], patch_tokens], dim=1)

                    scale = surface["scale"]
                    # Rotation-only camera-frame test: use the selected camera
                    # rotation, but do not apply the camera translation.
                    total_transform = (image_extrinsics @ T_to_colmap).astype(np.float32)
                    total_transform[:3, 3] = 0.0
                    centroid = surface["centroid"]
                    transformed_centroid = total_transform[:3, :3] @ centroid
                    total_transform[:3, 3] += transformed_centroid
                    total_transform[:3, 3] /= scale

                    aggregator_transform = image_extrinsics.astype(np.float32).copy()
                    aggregator_transform[:3, 3] = 0.0
                    aggregator_scale = aggregator_points["scale"]
                    aggregator_centroid = aggregator_points["centroid"]
                    transformed_aggregator_centroid = aggregator_transform[:3, :3] @ aggregator_centroid
                    aggregator_transform[:3, 3] += transformed_aggregator_centroid
                    aggregator_transform[:3, 3] /= aggregator_scale

                    surface = self.load_surface_points(
                        rng,
                        random_surface=surface["random_surface"],
                        sharpedge_surface=surface["sharp_surface"],
                    )
                    aggregator_points = self.load_surface_points(
                        rng,
                        random_surface=aggregator_points["random_surface"],
                        sharpedge_surface=aggregator_points["sharp_surface"],
                    )
                    geo_points = self.load_sdf_points(
                        rng,
                        vol_points=geo_points["vol_points"],
                        vol_label=geo_points["vol_label"],
                        near_points=geo_points["random_near_points"],
                        near_label=geo_points["random_near_label"],
                        sharpedge_points=geo_points["sharp_near_points"],
                        sharpedge_label=geo_points["sharp_near_label"],
                    )

                    total_transform = torch.from_numpy(total_transform).float()
                    surface[:, :3] = (total_transform[:3, :3] @ surface[:, :3].T).T + total_transform[:3, 3]
                    surface[:, 3:6] = (total_transform[:3, :3] @ surface[:, 3:6].T).T
                    geo_points[:, :3] = (total_transform[:3, :3] @ geo_points[:, :3].T).T + total_transform[:3, 3]

                    aggregator_transform = torch.from_numpy(aggregator_transform).float()
                    aggregator_points[:, :3] = (aggregator_transform[:3, :3] @ aggregator_points[:, :3].T).T + aggregator_transform[:3, 3]
                    aggregator_points[:, 3:6] = (aggregator_transform[:3, :3] @ aggregator_points[:, 3:6].T).T

                    camera_translation_scale = float(scale)

                images.append(torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32))
                extrinsics.append(torch.from_numpy(image_extrinsics).float())
                intrinsics.append(subseq_intrinsics)

        ex_t_batched = torch.stack(extrinsics).unsqueeze(0)
        normalized_extrinsics = self._normalize_extrinsics(
            ex_t_batched,
            translation_scale=camera_translation_scale,
        ).squeeze(0)

        del replica_scene
        intrinsics = torch.from_numpy(np.array(intrinsics))
        set_name = "replica_pano"

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "id": id,
            "subseq_ids": torch.as_tensor(subseq_ids, dtype=torch.long),
            "images": images,
            "extrinsics": normalized_extrinsics,
            "intrinsics": intrinsics,
            "surface": surface,
            "geo_points": geo_points,
            "aggregator_points": aggregator_points,
            "aggregator_latents": aggregator_latents,
        }

        return batch

    def _normalize_extrinsics(
        self,
        ex_t: torch.Tensor | None,
        translation_scale: torch.Tensor | float | None = None,
    ) -> torch.Tensor | None:
        return super()._normalize_extrinsics(ex_t, translation_scale=translation_scale)
