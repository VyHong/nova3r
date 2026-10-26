import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from eval.baselines.common import (
    _panorama_pose,
    read_manifest,
    resolve_erp_path,
    resolve_sample,
)
from nova3r.utils.three_d_front import GLTF_TO_BLENDER


class ThreeDFrontBaselineTests(unittest.TestCase):
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
