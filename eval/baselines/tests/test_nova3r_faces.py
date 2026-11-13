import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from eval.baselines.adapters.nova3r import Nova3RAdapter


class IndependentFacesTests(unittest.TestCase):
    def test_six_single_image_calls_with_corresponding_camera(self):
        adapter = Nova3RAdapter.__new__(Nova3RAdapter)
        adapter.input_projection = 'cubemap'
        adapter.face_order = [4, 0, 5, 1, 3, 2]
        adapter.erp_root = None
        adapter.erp_fov = 95
        adapter.independent_faces = True
        adapter.seed = 7
        poses = np.repeat(np.eye(4)[None], 6, axis=0)
        poses[:, 0, 3] = np.arange(6)
        paths = [Path(f'{face}.jpg') for face in adapter.face_order]
        from types import SimpleNamespace
        sample = SimpleNamespace(frame=2)
        with patch('eval.baselines.adapters.nova3r.load_input_cameras', return_value=(paths, None, poses, None)), patch.object(adapter, '_predict_views', return_value=np.zeros((2,3))) as predict:
            points = adapter.predict(sample)
        self.assertEqual(predict.call_count, 6)
        for index, call in enumerate(predict.call_args_list):
            self.assertEqual(call.args[0], [paths[index]])
            self.assertEqual(call.args[1], 1)
            np.testing.assert_allclose(adapter.last_face_pointclouds[str(adapter.face_order[index])][:,0], -index)
        self.assertEqual(points.shape, (12,3))

    def test_face_target_includes_occluded_depths_excludes_rear_outside_and_masked(self):
        from eval.baselines.common import face_target_mask
        points = np.array([[0,0,1], [0,0,-1], [3,0,1], [0,0,2], [-0.6,0,1]], dtype=float)
        intrinsic = np.array([[2,0,2], [0,2,2], [0,0,1]], dtype=float)
        valid = np.ones((4,4), dtype=bool)
        valid[2,0] = False
        mask = face_target_mask(points, np.eye(4), intrinsic, valid)
        np.testing.assert_array_equal(mask, [True,False,False,True,False])
