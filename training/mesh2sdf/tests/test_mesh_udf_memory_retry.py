"""CUDA OOM backoff must preserve every query and gradient result."""
import contextlib
import io
import unittest

import torch
from training.mesh2sdf.tools.mesh_udf.mesh_udf import (
    _adaptive_query_batches, get_udf_normals_grid_streamed,
)


class Plane(torch.nn.Module):
    def __init__(self, limit=None):
        super().__init__()
        self.limit = limit
        self.failures = 0
    def forward(self, xyz):
        if self.limit and len(xyz) > self.limit:
            self.failures += 1
            raise torch.cuda.OutOfMemoryError('simulated allocation failure')
        return xyz[:, :1] + 1.1


class MemoryRetryTests(unittest.TestCase):
    def test_backoff_does_not_skip_or_duplicate_queries(self):
        def query(start, stop):
            limit = 4 if start < 4 else 1
            if stop-start > limit: raise torch.cuda.OutOfMemoryError('test OOM')
            return torch.arange(start, stop)
        with contextlib.redirect_stdout(io.StringIO()):
            batches = list(_adaptive_query_batches(11, 8, query, 'cpu', 'test'))
        torch.testing.assert_close(torch.cat([v for _,_,v in batches]), torch.arange(11))
        self.assertEqual([(a,b) for a,b,_ in batches], [(0,4)]+[(i,i+1) for i in range(4,11)])

    def test_minimum_batch_and_non_oom_errors_propagate(self):
        for error in (torch.cuda.OutOfMemoryError('still OOM'), RuntimeError('decoder bug')):
            def query(a,b): raise error
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                list(_adaptive_query_batches(3, 1, query, 'cpu', 'test'))

    def test_streamed_field_and_normals_survive_oom_unchanged(self):
        options = dict(N=9,max_batch=64,device='cpu')
        expected = get_udf_normals_grid_streamed(Plane(),None,**options)
        decoder = Plane(limit=8)
        with contextlib.redirect_stdout(io.StringIO()):
            actual = get_udf_normals_grid_streamed(decoder,None,**options)
        self.assertGreater(decoder.failures, 0)
        for a,b in zip(actual,expected): torch.testing.assert_close(a,b)


if __name__ == '__main__': unittest.main()
