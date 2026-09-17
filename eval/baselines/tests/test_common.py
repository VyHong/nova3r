import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from eval.baselines.common import align_prediction_to_target, compute_ssi3d_fscores, read_manifest, unproject_depth
from eval.mv_recon.metric import accuracy, completion


class CommonTest(unittest.TestCase):
    def test_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(["scene_a 00003"]))
            self.assertEqual(read_manifest(path), [("scene_a", 3)])

    def test_directional_chamfer(self):
        pred = np.asarray([[0, 0, 0], [2, 0, 0]], np.float32)
        target = np.asarray([[0, 0, 0]], np.float32)
        forward, _ = accuracy(target, pred)
        backward, _ = completion(target, pred)
        self.assertEqual(forward, 1.0)
        self.assertEqual(backward, 0.0)

    def test_unproject_identity_camera(self):
        depth = np.asarray([[[2.0]]], np.float32)
        intrinsics = np.asarray([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], np.float32)
        extrinsics = np.eye(4, dtype=np.float32)[None, :3]
        np.testing.assert_allclose(unproject_depth(depth, intrinsics, extrinsics), [[0, 0, 2]])

    def test_alignment_uses_evaluate_pcd_fit(self):
        prediction = np.asarray([[0, 0, 0], [1, 1, 1]], np.float32)
        target = np.asarray([[1, 2, 3]], np.float32)

        def fake_alignment(prediction_tensor, target_tensor, **kwargs):
            self.assertEqual(kwargs, {"max_iterations": 300, "lr": 0.02, "return_transform": True})
            self.assertEqual(tuple(prediction_tensor.shape), (1, 2, 3))
            self.assertEqual(tuple(target_tensor.shape), (1, 1, 3))
            shift = torch.tensor([[[1.0, 2.0, 3.0]]])
            scale = torch.tensor([[[2.0]]])
            return scale * prediction_tensor + shift, shift, scale

        with patch("eval.mv_recon.metric.scale_shift_alignment_chamfer", side_effect=fake_alignment):
            aligned, scale, shift = align_prediction_to_target(prediction, target, "cpu", seed=0)

        np.testing.assert_allclose(aligned, [[1, 2, 3], [3, 4, 5]])
        self.assertEqual(scale, 2.0)
        np.testing.assert_allclose(shift, [1, 2, 3])

    def test_ssi3d_fscores(self):
        prediction = np.asarray([[0, 0, 0], [2, 0, 0]], np.float32)
        target = np.asarray([[0, 0, 0]], np.float32)
        for thresholds in ([1.0, 3.0], (1.0, 3.0)):
            with self.subTest(thresholds=thresholds):
                scores = compute_ssi3d_fscores(prediction, target, thresholds, "cpu")
                self.assertAlmostEqual(scores["f_score_1.0_full"], 2 / 3, places=6)
                self.assertAlmostEqual(scores["f_score_3.0_full"], 1.0, places=6)

    def test_ssi3d_scalar_threshold(self):
        from eval.mv_recon.metric import SSI3DScore

        scorer = SSI3DScore(num_eval_pts=2, fs_thres=1.0, pts_sampling_mode="uniform")
        prediction = torch.tensor([[[0., 0., 0.], [2., 0., 0.]]])
        target = torch.tensor([[[0., 0., 0.]]])
        details = scorer.chamfer_and_fscore(prediction, target, eval_layers=None)
        self.assertAlmostEqual(details["f_score_1.0_full"][0], 2 / 3, places=6)


if __name__ == "__main__":
    unittest.main()
