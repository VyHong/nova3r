"""Isolated persistent inference workers for LaRI and VolFill."""
from pathlib import Path
import json
import subprocess
import tempfile
import numpy as np
from eval.baselines.common import load_cameras, load_erp_input


class SingleImageAdapter:
    def __init__(self, model, repo, python, device, face_order, settings, seed, input_projection="cubemap", erp_root=None):
        self.input_projection = input_projection
        self.erp_root = erp_root
        self.face_order = face_order
        self.seed = seed
        self.raw_dir = settings.get("raw_dir")
        self.drop_zero_points = model == "lari" and settings.get("alignment") in ("merged_layers_faces_global_gt_scale_translation", "independent_face_frustum_gt_scale_translation")
        self.model_name = model
        self.last_face_pointclouds = {}
        worker = Path(__file__).with_name('single_image_worker.py')
        self.process = subprocess.Popen(
            [str(python), '-u', str(worker), '--model', model, '--repo', str(repo.resolve()),
             '--device', device, '--settings', json.dumps(settings)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        try:
            self._read_reply()
        except Exception:
            self.close()
            raise

    def _read_reply(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Inference worker exited; see its stderr above for dependency/checkpoint errors')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def predict(self, sample):
        erp = getattr(self, "input_projection", "cubemap") == "erp"
        if erp:
            path, extrinsic = load_erp_input(sample, self.erp_root)
            paths, extrinsics, input_ids = [path], [extrinsic], ["erp"]
        else:
            paths, _, extrinsics = load_cameras(sample, self.face_order)
            input_ids = self.face_order
        clouds = []
        self.last_face_pointclouds = {}
        self.last_face_pointmaps = {}
        self.last_empty_faces = []
        with tempfile.TemporaryDirectory(prefix='single_image_benchmark_') as directory:
            request = dict(paths=[str(path.resolve()) for path in paths], output=directory,
                           seed=self.seed + sample.frame * 6,
                           raw_output=(str(Path(self.raw_dir).resolve() / sample.scene.replace('/', '_') / f'{sample.frame:05d}') if self.raw_dir else None))
            self.process.stdin.write(json.dumps(request)+'\n')
            self.process.stdin.flush()
            reply = self._read_reply()
            for face, output, extrinsic in zip(input_ids, reply['outputs'], extrinsics):
                points = np.load(output, allow_pickle=False)
                if points.ndim == 4:
                    self.last_face_pointmaps[str(face)] = points
                points = points.reshape(-1, 3)
                points = points[np.isfinite(points).all(axis=1)]
                if getattr(self, "drop_zero_points", False):
                    points = points[np.any(points != 0, axis=1)]
                if not len(points):
                    self.last_empty_faces.append(str(face))
                    continue
                c2w = np.linalg.inv(extrinsic)
                world = points @ c2w[:3,:3].T + c2w[:3,3]
                clouds.append(world)
                if not erp:
                    self.last_face_pointclouds[str(face)] = world
        if not clouds:
            raise ValueError('No finite predicted points in input images')
        return np.concatenate(clouds)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        if self.process.stdout is not None:
            self.process.stdout.close()

    def __del__(self):
        if hasattr(self, 'process'):
            self.close()
