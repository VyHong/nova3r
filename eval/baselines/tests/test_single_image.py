"""Test isolated inference transport, cubemap placement, and TUDF extraction."""
from pathlib import Path
import tempfile
import ast
import importlib.util
import json
import sys
import types
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from eval.baselines.adapters.single_image import SingleImageAdapter
from eval.baselines.adapters.single_image_worker import volfill_points


UPSTREAM = Path(__file__).resolve().parents[3] / 'third_party/VolFill'


def upstream_save_function():
    source = ast.parse((UPSTREAM / 'volfill/amodal/inference_latent_visible.py').read_text())
    node = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == '_save_result')
    return ast.unparse(node)


def upstream_modules():
    module = types.ModuleType('volfill.amodal.inference_latent_visible')
    namespace = dict(np=np, Path=Path, json=json, Dict=dict, Image=Image)
    exec(upstream_save_function(), namespace)
    module._save_result = namespace['_save_result']
    spec = importlib.util.spec_from_file_location('volfill.visualize', UPSTREAM / 'volfill/visualize.py')
    visualize = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(visualize)
    return {'volfill.amodal.inference_latent_visible': module, 'volfill.visualize': visualize}


class SingleImageTests(unittest.TestCase):
    def test_volfill_grid_axes_voxel_centres_and_denormalization(self):
        grid = torch.ones(1,1,2,3,4)
        grid[0,0,1,2,3] = -1
        result = dict(tudf=grid, bbox_min=torch.tensor([10.,20.,30.]),
                      extent=torch.tensor([4.,6.,8.]))
        with patch.dict(sys.modules, upstream_modules()):
            points = volfill_points(result, 3., .8)
        np.testing.assert_allclose(points, [[13.5,25.,36.]])

    def test_real_worker_protocol_and_six_face_placement(self):
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'volfill' / 'amodal'
            package.mkdir(parents=True)
            (root / 'volfill' / '__init__.py').touch()
            (package / '__init__.py').touch()
            (package / 'inference_latent_visible.py').write_text('''import torch
class LatentTUDFVisibleInference:
    truncation_voxels = 3.
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        print('model diagnostic on stdout')
        return cls()
    def __call__(self, image):
        print('inference diagnostic on stdout')
        return dict(tudf=torch.full((1,1,2,2,2), -1.),
                    bbox_min=torch.tensor([0.,0.,1.]), extent=torch.ones(3))
''')
            with (package / 'inference_latent_visible.py').open('a') as stream:
                stream.write("\nimport numpy as np\nimport json\nfrom pathlib import Path\nfrom PIL import Image\nfrom typing import Dict\n" + upstream_save_function() + "\n")
            import shutil
            shutil.copy2(UPSTREAM / 'volfill/visualize.py', root / 'volfill/visualize.py')
            path = root / 'face.png'
            Image.new('RGB',(8,8)).save(path)
            adapter = SingleImageAdapter('volfill', root, sys.executable, 'cpu',
                                         [4,0,5,1,3,2], dict(checkpoint='fake', steps=1, cfg_strength=1, threshold=.8), 0)
            poses = np.repeat(np.eye(4)[None],6,axis=0)
            poses[:,0,3] = np.arange(6)
            try:
                with patch('eval.baselines.adapters.single_image.load_cameras', return_value=([path]*6, None, poses)):
                    points = adapter.predict(SimpleNamespace(frame=0))
                self.assertEqual(points.shape,(48,3))
                self.assertEqual(len(adapter.last_face_pointclouds),6)
                for index,face in enumerate(adapter.face_order):
                    np.testing.assert_allclose(adapter.last_face_pointclouds[str(face)][:,0].mean(), .5-index)
            finally:
                adapter.close()

    def test_empty_face_is_skipped_and_recorded(self):
        adapter = SingleImageAdapter.__new__(SingleImageAdapter)
        adapter.face_order = [4, 0]
        adapter.seed = 0
        adapter.raw_dir = None
        adapter.process = SimpleNamespace(stdin=SimpleNamespace(write=lambda x: None, flush=lambda: None))
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / 'empty.npy'
            valid = Path(directory) / 'valid.npy'
            np.save(empty, np.empty((0, 3)))
            np.save(valid, np.ones((2, 3)))
            with patch.object(adapter, '_read_reply', return_value={'outputs': [str(empty), str(valid)]}), patch(
                'eval.baselines.adapters.single_image.load_cameras',
                return_value=([Path('unused')] * 2, None, np.repeat(np.eye(4)[None], 2, axis=0)),
            ):
                points = adapter.predict(SimpleNamespace(frame=0))
            self.assertEqual(adapter.last_empty_faces, ['4'])
            self.assertEqual(list(adapter.last_face_pointclouds), ['0'])
            np.testing.assert_allclose(points, np.ones((2, 3)))
        del adapter.process

    def test_merged_layers_drop_empty_entries_before_world_placement(self):
        adapter = SingleImageAdapter.__new__(SingleImageAdapter)
        adapter.face_order = [0, 1]
        adapter.seed = 0
        adapter.raw_dir = None
        adapter.drop_zero_points = True
        adapter.process = SimpleNamespace(stdin=SimpleNamespace(write=lambda value: None, flush=lambda: None))
        try:
            with tempfile.TemporaryDirectory() as directory:
                first, second = Path(directory) / 'first.npy', Path(directory) / 'second.npy'
                np.save(first, np.array([[[[0., 0., 0.], [1., 0., 2.], [np.nan, 1., 2.]]]]))
                np.save(second, np.array([[[[3., 0., 4.], [5., 0., 6.], [np.inf, 0., 1.]]]]))
                poses = np.repeat(np.eye(4)[None], 2, axis=0)
                poses[1, 0, 3] = -10
                with patch.object(adapter, '_read_reply', return_value={'outputs': [str(first), str(second)]}), patch(
                    'eval.baselines.adapters.single_image.load_cameras',
                    return_value=([Path('unused')] * 2, None, poses),
                ):
                    points = adapter.predict(SimpleNamespace(frame=0))
                np.testing.assert_allclose(points, [[1., 0., 2.], [13., 0., 4.], [15., 0., 6.]])
                self.assertEqual(adapter.last_empty_faces, [])
        finally:
            del adapter.process

    def test_alignment_skips_face_without_gt_and_fails_if_all_skipped(self):
        from contextlib import ExitStack
        from eval.baselines import run
        with patch('sys.argv', ['run', '--model', 'lari', '--dataset-layout', 'scannetpp', '--lari-alignment', 'all-chamfer']):
            args = run.parse_args()
        args.visibility_metrics = False
        args.save_predictions = False
        args.eval_points = 3
        cloud = np.ones((3, 3), dtype=np.float32)
        adapter = SimpleNamespace(predict=lambda sample: cloud,
                                  last_face_pointclouds={'0': cloud, '1': cloud})
        sample = SimpleNamespace(key='scene 00000', frame=0)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            image = Path(directory) / 'face.png'
            Image.new('RGB', (4, 4)).save(image)
            stack.enter_context(patch.object(run, 'resolve_sample', return_value=sample))
            stack.enter_context(patch.object(run, 'load_visibility_geometry', return_value=(cloud, [[0, 1, 2]], [[0, 1, 2]], image)))
            stack.enter_context(patch.object(run, 'sample_mesh_surface', return_value=cloud))
            stack.enter_context(patch.object(run, 'load_cameras', return_value=([image, image], [np.eye(3)]*2, [np.eye(4)]*2)))
            stack.enter_context(patch.object(run, 'load_scannetpp_valid_mask', return_value=np.ones((4, 4), bool)))
            args.face_order = [0, 1]
            masks = stack.enter_context(patch.object(run, 'face_target_mask', side_effect=[np.zeros(3, bool), np.ones(3, bool)]))
            align = stack.enter_context(patch.object(run, 'align_prediction_to_target', return_value=(cloud, 1., np.zeros(3))))
            evaluate = stack.enter_context(patch.object(run, 'evaluate_point_clouds', return_value=(cloud, {})))
            record = run.evaluate_sample(args, adapter, 0, ('scene', 0), 1)
            self.assertEqual(record['status'], 'ok')
            self.assertEqual(record['skipped_faces'], {'0': 'no_valid_gt_points'})
            self.assertEqual(list(record['face_alignments']), ['1'])
            self.assertTrue(evaluate.call_args.kwargs['prealigned'])
            align.assert_called_once()
            masks.side_effect = [np.zeros(3, bool), np.zeros(3, bool)]
            record = run.evaluate_sample(args, adapter, 0, ('scene', 0), 1)
            self.assertEqual(record['status'], 'error')
            self.assertIn('No cubemap faces', record['error'])
            self.assertEqual(evaluate.call_count, 1)

    def test_erp_sends_one_image_and_uses_global_alignment(self):
        adapter = SingleImageAdapter.__new__(SingleImageAdapter)
        adapter.input_projection = "erp"
        adapter.erp_root = Path("panoramas")
        adapter.face_order = [4, 0, 5, 1, 3, 2]
        adapter.seed = 0
        adapter.raw_dir = None
        requests = []
        adapter.process = SimpleNamespace(stdin=SimpleNamespace(
            write=lambda value: requests.append(json.loads(value)), flush=lambda: None))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "erp.npy"
            np.save(output, np.ones((2, 3)))
            pose = np.eye(4)
            pose[0, 3] = 2
            with patch.object(adapter, "_read_reply", return_value={"outputs": [str(output)]}), patch(
                "eval.baselines.adapters.single_image.load_erp_input",
                return_value=(Path("panorama.jpg"), pose),
            ), patch("eval.baselines.adapters.single_image.load_cameras") as cameras:
                points = adapter.predict(SimpleNamespace(frame=0))
            cameras.assert_not_called()
            self.assertEqual(len(requests[0]["paths"]), 1)
            self.assertEqual(adapter.last_face_pointclouds, {})
            np.testing.assert_allclose(points, [[-1, 1, 1], [-1, 1, 1]])
        del adapter.process

    def test_volfill_erp_requests_exact_two_to_one_patch_size(self):
        from unittest.mock import Mock
        from eval.baselines.adapters import single_image_worker
        tree = ast.parse(Path(single_image_worker.__file__).read_text())
        load = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_predictor")
        branch = next(node for node in load.body if isinstance(node, ast.If) and node.orelse)
        predict = next(node for node in branch.orelse if isinstance(node, ast.FunctionDef) and node.name == "predict")
        model = Mock(truncation_voxels=3.)
        export = Mock(return_value="cloud")
        namespace = dict(settings={"input_projection": "erp", "threshold": .8}, model=model, volfill_points=export)
        exec(ast.unparse(predict), namespace)
        image = Image.new("RGB", (1024, 512))
        self.assertEqual(namespace["predict"](image), "cloud")
        model.assert_called_once_with(image, max_size=504)
        with self.assertRaisesRegex(ValueError, "exactly 2:1"):
            namespace["predict"](Image.new("RGB", (100, 60)))

    def test_resume_accepts_legacy_volfill_cubemap_settings_only_for_cubemap(self):
        from eval.baselines.run import parse_args, single_image_settings, single_image_settings_match
        with patch('sys.argv', ['run', '--model', 'volfill', '--dataset-layout', 'scannetpp']):
            args = parse_args()
        legacy = single_image_settings(args)
        legacy.pop('input_projection')
        record = dict(inference_settings=legacy)
        self.assertTrue(single_image_settings_match(args, record))
        args.input_projection = 'erp'
        self.assertFalse(single_image_settings_match(args, record))

    def test_lari_first_layer_fit_recovers_scale_and_camera_depth_shift(self):
        from eval.baselines.common import align_lari_pointmap
        height = width = 4
        k = np.array([[2.,0.,2.],[0.,2.,2.],[0.,0.,1.]])
        x,y = np.meshgrid(np.arange(width)+.5,np.arange(height)+.5)
        rays = np.stack([x,y,np.ones_like(x)],-1) @ np.linalg.inv(k).T
        first = rays * 2
        second = rays * 3
        pointmap = np.stack([first,second],axis=2)
        pointmap[:,:,:,2] -= .4
        pointmap /= 3
        vertices = np.array([[-10,-10,2],[10,-10,2],[10,10,2],[-10,10,2]],np.float32)
        aligned,scale,shift,details = align_lari_pointmap(pointmap,vertices,[[0,1,2],[0,2,3]],np.eye(4),k,np.ones((4,4),bool))
        self.assertAlmostEqual(scale,3.,places=5)
        np.testing.assert_allclose(shift,[0,0,.4],atol=1e-5)
        np.testing.assert_allclose(aligned.reshape(4,4,2,3),np.stack([first,second],2),atol=1e-5)
        self.assertEqual(details['correspondence_points'],16)
