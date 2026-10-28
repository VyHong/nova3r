import json
from pathlib import Path
import tempfile
import unittest
import importlib.util

import numpy as np
import torch

from eval.baselines.adapters.genrecon import GenReconAdapter, chunk_transforms_to_eval_frame

from eval.baselines.common import (
    _panorama_pose,
    read_manifest,
    resolve_erp_path,
    resolve_sample,
)
from nova3r.utils.three_d_front import GLTF_TO_BLENDER


class ThreeDFrontBaselineTests(unittest.TestCase):
    def test_genrecon_chunks_cover_translated_y_up_room(self):
        chunker_path = Path(__file__).resolve().parents[3] / "third_party/GenRecon/inference/get_chunks.py"
        spec = importlib.util.spec_from_file_location("genrecon_test_chunks", chunker_path)
        chunks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chunks)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = resolve_sample("dwelling/Bedroom-7", 3, root, root, root, "3dfront")
            sample.surface_path.parent.mkdir(parents=True, exist_ok=True)
            corners = np.asarray([
                [x, y, z] for x in (-6., -2.) for y in (1., 4.) for z in (8., 13.)
            ], dtype=np.float32)
            np.savez(sample.surface_path, random_surface=corners)
            adapter = GenReconAdapter.__new__(GenReconAdapter)
            adapter.ScannetGtChunker = chunks.ScannetGtChunker
            work = root / "work"
            work.mkdir()
            _, o2c, c2o, _, source = adapter._get_chunks(sample, work)
            self.assertAlmostEqual(source["chunk_size"], 3.6)
            self.assertEqual(source["bounds"]["z_floor"], 1.)
            o2c, c2o = chunk_transforms_to_eval_frame(o2c, c2o, "3dfront")
            points = torch.from_numpy(corners)
            covered = torch.zeros(len(points), dtype=torch.bool)
            for forward, inverse in zip(o2c, c2o):
                torch.testing.assert_close(forward @ inverse, torch.eye(4), atol=1e-5, rtol=1e-5)
                local = points @ forward[:3, :3].T + forward[:3, 3]
                covered |= (local.abs() <= 0.50001).all(dim=1)
            self.assertTrue(covered.all())

    def test_path_manifest_and_layout_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "split.json"
            manifest.write_text(json.dumps([
                "dwelling/Bedroom-7/Bedroom-7_panorama_003.png"
            ]))
            entries = read_manifest(manifest, "3dfront")
            self.assertEqual(entries, [("dwelling/Bedroom-7", 3)])

            sample = resolve_sample(
                *entries[0],
                root / "cubes",
                root / "renders",
                root / "surfaces",
                "3dfront",
            )
            self.assertEqual(
                sample.pose_path,
                root / "renders/dwelling/Bedroom-7/extrinsics.json",
            )
            self.assertEqual(
                sample.surface_path,
                root / "surfaces/dwelling/Bedroom-7_full_surface.npz",
            )
            image = root / (
                "renders/dwelling/Bedroom-7/Bedroom-7_panorama_003.png"
            )
            image.parent.mkdir(parents=True)
            image.touch()
            self.assertEqual(resolve_erp_path(sample, root / "renders"), image)

    def test_pose_maps_gltf_camera_center_to_opencv_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = resolve_sample(
                "dwelling/Bedroom-7",
                3,
                root / "cubes",
                root / "renders",
                root / "surfaces",
                "3dfront",
            )
            sample.pose_path.parent.mkdir(parents=True)
            blender_w2c = np.asarray(
                [[0, -1, 0, 2], [1, 0, 0, -3], [0, 0, 1, 4], [0, 0, 0, 1]],
                dtype=np.float32,
            )
            sample.pose_path.write_text(json.dumps({
                "Bedroom-7_panorama_003.png": blender_w2c.tolist()
            }))
            gltf_camera_center = (
                np.linalg.inv(GLTF_TO_BLENDER)
                @ np.linalg.inv(blender_w2c)
                @ np.asarray([0, 0, 0, 1], dtype=np.float32)
            )
            camera_center = _panorama_pose(sample) @ gltf_camera_center
            np.testing.assert_allclose(camera_center, [0, 0, 0, 1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
