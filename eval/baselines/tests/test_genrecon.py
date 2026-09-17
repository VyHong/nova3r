from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

from eval.baselines.common import sample_mesh_surface
from eval.baselines.adapters.genrecon import (
    GenReconAdapter,
    chunk_transforms_to_eval_frame,
    infer_sage_floorplan,
    normalize_intrinsics,
    resolve_room_id,
)


class GenReconAdapterTest(unittest.TestCase):
    def test_resolve_room_id_from_generated_scene_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            room = root / "deadbeef"
            room.mkdir()
            (room / "layout_deadbeef.json").write_text("{}")
            self.assertEqual(resolve_room_id("20260101_120000_layout_deadbeef", root), "deadbeef")

    def test_infer_floorplan_from_evaluator_surface_and_strip_wall_overshoot(self):
        with tempfile.TemporaryDirectory() as directory:
            surface = Path(directory) / "scene_surface.npz"
            # Evaluator axes are (raw x, raw z, -raw y). The 5 cm excess on
            # the two horizontal dimensions represents finite wall thickness.
            points = np.asarray(
                [
                    [0.0, -0.1, -6.05],
                    [5.05, 2.8, 0.0],
                    [2.5, 1.4, -3.0],
                ],
                np.float32,
            )
            np.savez(surface, random_surface=points)
            self.assertEqual(infer_sage_floorplan(surface), (0.0, 5.0, 0.0, 6.0))

    def test_normalize_pixel_intrinsics(self):
        intrinsics = np.asarray([[[200, 0, 100], [0, 100, 50], [0, 0, 1]]], np.float32)
        normalized = normalize_intrinsics(intrinsics, [(200, 100)])
        np.testing.assert_allclose(normalized.numpy(), [[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]])

    def test_surface_sampling_is_deterministic_and_on_triangle(self):
        vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        faces = np.asarray([[0, 1, 2]], np.int64)
        first = sample_mesh_surface(vertices, faces, count=100, seed=7)
        second = sample_mesh_surface(vertices, faces, count=100, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[:, :2] >= 0))
        self.assertTrue(np.all(first[:, 0] + first[:, 1] <= 1 + 1e-6))
        np.testing.assert_array_equal(first[:, 2], np.zeros(100, np.float32))

    def test_selected_camera_extrinsics_are_rebased_to_chunk_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "face.png"
            Image.new("RGB", (4, 4), "white").save(image_path)
            adapter = GenReconAdapter.__new__(GenReconAdapter)
            adapter.SelectedImages = lambda **kwargs: SimpleNamespace(**kwargs)
            adapter.chunk_visible_in_camera = lambda extrinsic, intrinsic: True
            intrinsic = np.asarray([[[2, 0, 2], [0, 2, 2], [0, 0, 1]]], np.float32)
            world_to_camera = np.eye(4, dtype=np.float32)[None]
            chunk_to_world = torch.eye(4)
            chunk_to_world[0, 3] = 3
            world_to_chunk = torch.linalg.inv(chunk_to_world)

            selected, indices = adapter._select_images(
                [image_path],
                intrinsic,
                world_to_camera,
                [world_to_chunk],
                [chunk_to_world],
                seed=0,
            )

            self.assertEqual(indices, [0])
            torch.testing.assert_close(selected.scene_extrinsics_c0, chunk_to_world.unsqueeze(0))
            expected_intrinsics = torch.tensor([[[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1.0]]])
            torch.testing.assert_close(selected.scene_intrinsics, expected_intrinsics)

    def test_chunk_transforms_convert_blender_room_to_eval_axes(self):
        room_to_chunk = torch.eye(4)
        chunk_to_room = torch.eye(4)
        eval_to_chunk, chunk_to_eval = chunk_transforms_to_eval_frame(
            [room_to_chunk],
            [chunk_to_room],
        )
        room_point = torch.tensor([[1.0, 2.0, 3.0]])
        eval_point = torch.tensor([[1.0, 3.0, -2.0]])
        room_h = torch.cat((room_point, torch.ones(1, 1)), dim=1)
        eval_h = torch.cat((eval_point, torch.ones(1, 1)), dim=1)
        torch.testing.assert_close((chunk_to_eval[0] @ room_h.T).T, eval_h)
        torch.testing.assert_close((eval_to_chunk[0] @ eval_h.T).T, room_h)


if __name__ == "__main__":
    unittest.main()
