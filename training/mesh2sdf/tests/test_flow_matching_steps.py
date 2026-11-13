"""The evaluation step count must change integration while preserving noise."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from nova3r.flow_matching.solver import ODESolver


class FlowMatchingStepTests(unittest.TestCase):
    def method(self, wrapper):
        path = Path('nova3r/models/hy3dshape/models/denoisers/hunyuandit.py')
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HunYuanDiTPlain')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_reconstruct_lightning_latents')
        namespace = dict(torch=torch, ODESolver=ODESolver, HunYuanDiTVelocityWrapper=wrapper)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        return namespace[method.name]

    def test_more_steps_reduce_euler_error_with_identical_initial_noise(self):
        runs = []
        class Velocity(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.first = None
                self.calls = 0
                runs.append(self)
            def forward(self, x, t, **extras):
                if self.first is None: self.first = x.clone()
                self.calls += 1
                return x
        method = self.method(Velocity)
        errors = []
        for steps in (100, 200, 400):
            model = SimpleNamespace(flow_matching_steps=steps, latent_scale_factor=1.)
            result = method(model, {}, torch.zeros(1, 2, 3), None)
            errors.append(float((result-runs[-1].first*math.e).abs().mean()))
            self.assertGreaterEqual(runs[-1].calls, steps-1)
            self.assertLessEqual(runs[-1].calls, steps+1)
            torch.testing.assert_close(runs[-1].first, runs[0].first, rtol=0, atol=0)
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])

    def test_invalid_step_counts_fail_before_sampling(self):
        method = self.method(None)
        for steps in (0, -1, 1.5, True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                method(SimpleNamespace(flow_matching_steps=steps), {}, torch.zeros(1), None)


if __name__ == '__main__': unittest.main()
