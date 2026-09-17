"""Exercise conversion, cleanup, and resume without Blender."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render_centered_sage10k as driver


@unittest.skipUnless(importlib.util.find_spec("utils3d"), "Run in the nova3r environment")
class CubemapsOnlyTests(unittest.TestCase):
    def setUp(self):
        import cv2

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scene = "test_scene"
        self.pano = self.root / "panoramas" / self.scene
        self.pano.mkdir(parents=True)
        self.image = self.pano / f"{self.scene}_panorama_000.png"
        cv2.imwrite(str(self.image), np.full((64, 128, 3), 120, dtype=np.uint8))
        driver.atomic_json(self.pano / "extrinsics.json", {self.image.name: np.eye(4).tolist()})
        scene_list = self.root / "input.json"
        driver.atomic_json(scene_list, [f"{self.scene} 00000"])
        self.argv = ["driver", "--scene-list", str(scene_list), "--output-root", str(self.root),
                     "--scene-root", str(self.root / "missing-zips"),
                     "--surface-root", str(self.root / "missing-surfaces"),
                     "--convert-only", "--width", "128", "--height", "64", "--res", "32"]
        with patch.object(sys, "argv", self.argv):
            self.args = driver.parse_args()
        driver.atomic_json(self.pano / ".complete.json", {
            "num_panoramas": 1, "world_coordinate_system": "gltf_y_up",
            "render_settings": driver.render_settings(self.args),
        })
        self.cube = driver.cube_path(self.root, self.scene)

    def invoke(self, runner):
        with patch.object(sys, "argv", self.argv), patch.object(driver, "run", runner):
            return driver.main()

    def actual_projection(self, command):
        command = list(map(str, command))
        if str(driver.CONVERTER) not in command:
            self.assertNotEqual(command[0], "bash", "Conversion-only launched Blender")
            return 0
        result = subprocess.run([sys.executable] + command[command.index(str(driver.CONVERTER)):],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.returncode

    def test_conversion_removes_normalization_and_keeps_panorama(self):
        self.assertEqual(self.invoke(self.actual_projection), 0)
        self.assertTrue(self.image.is_file())
        self.assertTrue((self.pano / "extrinsics.json").is_file())
        self.assertTrue((self.pano / ".complete.json").is_file())
        self.assertFalse((self.cube / "normalization.json").exists())
        self.assertEqual(len(list(self.cube.glob("*.jpg"))), 6)
        self.assertTrue(driver.cubemap_complete(self.root, self.scene, self.args))
        times = {p.name: p.stat().st_mtime_ns for p in self.cube.glob("*.jpg")}
        def no_work(command):
            self.fail(f"Completed cubemap triggered work: {command}")
        # Even normal rendering mode must skip completed cubes without ZIPs.
        self.argv.remove("--convert-only")
        self.assertEqual(self.invoke(no_work), 0)
        self.assertEqual(times, {p.name: p.stat().st_mtime_ns for p in self.cube.glob("*.jpg")})

    def test_failed_projection_preserves_panorama(self):
        def fail_conversion(command):
            return 1 if driver.CONVERTER in command else 0
        self.assertEqual(self.invoke(fail_conversion), 1)
        self.assertTrue(self.image.is_file())
        self.assertTrue((self.pano / "extrinsics.json").is_file())
        self.assertIn(self.scene, json.loads((self.root / "failures.json").read_text()))

    def test_reuses_legacy_normalized_cubemap(self):
        import cv2

        self.cube.mkdir(parents=True)
        metadata = {}
        for i in range(6):
            cv2.imwrite(str(self.cube / f"{i:04d}.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))
            metadata[f"{i:04d}"] = {"world_extrinsics": np.eye(4).tolist(),
                                   "normalized_world_extrinsics": np.eye(4).tolist()}
        driver.atomic_json(self.cube / "camera_data.json", metadata)
        driver.atomic_json(self.pano / "normalization.json", {"scale": 0.25})
        driver.atomic_json(self.cube / "normalization.json", {"scale": 0.25})
        signature = driver.projection_signature(self.image, self.pano, "cube", 95, 32)
        signature.pop("render_settings")
        driver.atomic_json(self.cube / ".complete.json", signature)
        driver.atomic_json(self.root / "poses" / f"{self.scene}.json", {})
        def no_work(command):
            self.fail(f"Legacy RGB was unnecessarily regenerated: {command}")
        self.assertEqual(self.invoke(no_work), 0)
        self.assertTrue(self.image.is_file())
        self.assertFalse((self.cube / "normalization.json").exists())
        self.assertTrue((self.root / "poses").exists())
        self.assertTrue(driver.cubemap_complete(self.root, self.scene, self.args))
        self.assertTrue(all("normalized_world_extrinsics" not in face for face in
                            json.loads((self.cube / "camera_data.json").read_text()).values()))


if __name__ == "__main__":
    unittest.main()
