"""Check conditioning ablations without importing the optional model stack."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from torch import nn


class PluckerAblationTests(unittest.TestCase):
    def test_disabled_embeddings_cannot_affect_conditioning(self):
        path = Path(__file__).resolve().parents[3] / 'nova3r/models/hy3dshape/models/denoisers/hunyuandit.py'
        tree = ast.parse(path.read_text())
        model = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == 'HunYuanDiTPlain')
        forward = next(node for node in model.body
                       if isinstance(node, ast.FunctionDef) and node.name == 'forward')
        rays = Mock(return_value=torch.ones(2, 6))
        namespace = dict(torch=torch, make_equirectangular_plucker_grid=rays)
        exec(compile(ast.Module(body=[forward], type_ignores=[]), str(path), 'exec'), namespace)
        tokens = torch.randn(1, 2, 4)
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                captured = []
                def block(x, c, cond, **kwargs):
                    captured.append(cond)
                    return x
                embedder = nn.Linear(6, 4)
                nn.init.ones_(embedder.weight)
                nn.init.ones_(embedder.bias)
                state = SimpleNamespace(
                    training=False, conditioning_drop_prob=0,
                    use_panorama_plucker=enabled, plucker_embedder=embedder,
                    _uniform_context_value=lambda value, name, default: default if value is None else value,
                    _uniform_grid_size=lambda value, count: (1, 2),
                    t_embedder=lambda t, **kwargs: torch.zeros(1, 1, 4),
                    x_embedder=nn.Identity(), use_pos_emb=False,
                    use_attention_pooling=False, with_decoupled_ca=False,
                    blocks=[block], depth=1, final_layer=nn.Identity(),
                )
                rays.reset_mock()
                namespace['forward'](state, torch.zeros(1, 2, 4), torch.zeros(1),
                                     {'main': tokens})
                torch.testing.assert_close(captured[0], tokens + (7 if enabled else 0))
                self.assertEqual(rays.call_count, int(enabled))


if __name__ == '__main__':
    unittest.main()
