"""Exercise worker scheduling with real spawned processes and lightweight jobs."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from eval.baselines import run


def fake_initialize(args, device, worker_index):
    # Keep CUDA/model dependencies out of orchestration tests.
    args.device = device
    args.worker_index = worker_index
    run._worker_args = args
    run.evaluate_sample = fake_evaluate


def fake_evaluate(args, adapter, index, entry, total):
    if entry[0] == "fail":
        raise RuntimeError("sample failed")
    time.sleep(0.02)
    return dict(sample=entry, index=index, seed=args.seed + index,
                device=args.device, worker=args.worker_index, pid=os.getpid())


class ParallelTests(unittest.TestCase):
    def args(self, **kwargs):
        values = dict(workers=2, devices=["cuda:0", "cuda:1"], device="cuda", seed=17)
        values.update(kwargs)
        return argparse.Namespace(**values)

    def test_spawned_workers_cover_manifest_once_and_reuse_models(self):
        entries = [("scene", frame) for frame in range(8)]
        with patch.object(run, "initialize_worker", fake_initialize):
            records = dict(run.parallel_records(
                self.args(), list(enumerate(entries)), len(entries)
            ))
        self.assertEqual(sorted(records), list(range(8)))
        self.assertEqual({r["device"] for r in records.values()}, {"cuda:0", "cuda:1"})
        self.assertEqual(len({r["pid"] for r in records.values()}), 2)
        for index, record in records.items():
            self.assertEqual(record["sample"], entries[index])
            self.assertEqual(record["seed"], 17 + index)
            self.assertEqual(record["device"], f'cuda:{record["worker"]}')

    def test_parent_writes_manifest_order_after_out_of_order_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.argv", ["run", "--model", "genrecon", "--workers", "2",
                                    "--output-dir", directory]):
                args = run.parse_args()
            entries = [("scene", frame) for frame in range(3)]
            completed = [(index, {"sample": str(index), "status": "ok"})
                         for index in (2, 0, 1)]
            with patch.object(run, "parse_args", return_value=args), \
                 patch.object(run, "read_manifest", return_value=entries), \
                 patch.object(run, "parallel_records", return_value=iter(completed)), \
                 patch.object(run, "build_evaluation_summary", return_value={}), \
                 patch.object(run, "write_evaluation_results") as write, \
                 patch("builtins.print"):
                run.main()
            records = json.loads((Path(directory) / "results.json").read_text())
            self.assertEqual([record["sample"] for record in records], ["0", "1", "2"])
            self.assertEqual(write.call_args.args[0], records)

    def test_worker_exception_propagates(self):
        with patch.object(run, "initialize_worker", fake_initialize):
            with self.assertRaisesRegex(RuntimeError, "sample failed"):
                entries = [("fail", 0), ("scene", 1)]
                list(run.parallel_records(
                    self.args(), list(enumerate(entries)), len(entries)
                ))

    def test_duplicate_output_paths_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            entries = [("scene", 1)] * 2
            list(run.parallel_records(
                self.args(), list(enumerate(entries)), len(entries)
            ))

    def test_cli_device_defaults_and_validation(self):
        with patch("sys.argv", ["run", "--model", "genrecon", "--devices", "cuda:0", "cuda:1"]):
            self.assertEqual(run.parse_args().workers, 2)
        for argv in (["--model", "genrecon", "--workers", "0"],
                     ["--model", "da3", "--workers", "2"],
                     ["--model", "genrecon", "--workers", "1", "--devices", "cuda:0", "cuda:1"]):
            with patch("sys.argv", ["run", *argv]), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    run.parse_args()

    def test_sample_error_record_and_fail_fast(self):
        args = self.args(cubemap_root=Path("/tmp"), pose_root=Path("/tmp"),
                         scene_root=Path("/tmp"), dataset_layout="scannetpp", fail_fast=False)
        adapter = unittest.mock.Mock()
        adapter.predict.side_effect = RuntimeError("inference failed")
        with patch.object(run.traceback, "print_exc"):
            record = run.evaluate_sample(args, adapter, 0, ("scene", 1), 1)
            self.assertEqual(record["status"], "error")
            self.assertEqual(record["error"], "inference failed")
            args.fail_fast = True
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                run.evaluate_sample(args, adapter, 0, ("scene", 1), 1)


if __name__ == "__main__":
    unittest.main()
