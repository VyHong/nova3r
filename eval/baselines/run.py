#!/usr/bin/env python3
"""Run a baseline and report directional Chamfer distances and SSI3D F-scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.baselines.common import (
    DEFAULT_ALIGNMENT_ITERATIONS,
    DEFAULT_ALIGNMENT_LR,
    DEFAULT_EVAL_POINTS,
    DEFAULT_FSCORE_THRESHOLDS,
    build_evaluation_summary,
    evaluate_point_clouds,
    load_scene_points,
    read_manifest,
    resolve_sample,
    sample_points,
    write_evaluation_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("da3", "genrecon"), default="da3")
    parser.add_argument("--split-list", type=Path, default=REPO_ROOT / "data/sage-10k/test_list.json")
    parser.add_argument("--cubemap-root", type=Path, default=Path("/tmp/datasets/sage-10k/cubemap_data/cubemaps"))
    parser.add_argument("--pose-root", type=Path, default=Path("/mnt/projects/theses/vyhong/sage-10k/poses"))
    parser.add_argument("--scene-root", type=Path, default=Path("/tmp/datasets/sage-10k/udf"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scene")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-points",
        type=int,
        default=DEFAULT_EVAL_POINTS,
        help="Max points sampled independently from prediction and GT",
    )
    parser.add_argument(
        "--fscore-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_FSCORE_THRESHOLDS),
    )
    parser.add_argument("--alignment-iterations", type=int, default=DEFAULT_ALIGNMENT_ITERATIONS)
    parser.add_argument("--alignment-lr", type=float, default=DEFAULT_ALIGNMENT_LR)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--da3-model", default="depth-anything/DA3-LARGE")
    parser.add_argument("--da3-repo", type=Path, default=REPO_ROOT / "third_party/Depth-Anything-3")
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--face-order", type=int, nargs=6, default=[4, 0, 5, 1, 3, 2])
    parser.add_argument("--genrecon-repo", type=Path, default=REPO_ROOT / "third_party/GenRecon")
    parser.add_argument(
        "--genrecon-data-root",
        type=Path,
        default=Path("/tmp/datasets/sage-10k/genrecon"),
        help="Optional prepared SAGE root containing rooms_raw; falls back to scene-surface bounds",
    )
    parser.add_argument("--genrecon-ss-ckpt", type=Path, default=REPO_ROOT / "third_party/GenRecon/sparse_structure.pt")
    parser.add_argument("--genrecon-shape-ckpt", type=Path, default=REPO_ROOT / "third_party/GenRecon/shape_slat.pt")
    parser.add_argument("--genrecon-tex-ckpt", type=Path, default=REPO_ROOT / "third_party/GenRecon/texture_slat.pt")
    parser.add_argument(
        "--genrecon-pipeline-config",
        type=Path,
        default=REPO_ROOT / "third_party/GenRecon/configs/pipelines/original.json",
    )
    parser.add_argument("--genrecon-work-dir", type=Path)
    parser.add_argument("--genrecon-surface-points", type=int)
    parser.add_argument("--genrecon-occ-threshold", type=float, default=-1.0)
    parser.add_argument("--genrecon-boundary-width", type=int, default=1)
    parser.add_argument("--genrecon-proj-batch-voxels", type=int)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = REPO_ROOT / f"eval_results/baselines/{args.model}/sage10k_test"
    if sorted(args.face_order) != list(range(6)):
        parser.error("--face-order must be a permutation of 0 1 2 3 4 5")
    if args.eval_points < 1:
        parser.error("--eval-points must be positive")
    if any(threshold <= 0 for threshold in args.fscore_thresholds):
        parser.error("--fscore-thresholds values must be positive")
    if args.alignment_iterations < 1:
        parser.error("--alignment-iterations must be positive")
    if args.alignment_lr <= 0:
        parser.error("--alignment-lr must be positive")
    if args.genrecon_surface_points is not None and args.genrecon_surface_points < 1:
        parser.error("--genrecon-surface-points must be positive")
    if args.genrecon_boundary_width < 0:
        parser.error("--genrecon-boundary-width must be non-negative")
    if args.genrecon_proj_batch_voxels is not None and args.genrecon_proj_batch_voxels < 1:
        parser.error("--genrecon-proj-batch-voxels must be positive")
    return args


def save_ply(points, path):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(path), cloud)


def main():
    args = parse_args()
    entries = read_manifest(args.split_list)
    if args.scene:
        entries = [entry for entry in entries if entry[0] == args.scene]
    if args.limit is not None:
        entries = entries[: args.limit]
    if not entries:
        raise ValueError("No samples selected")

    if args.model == "da3":
        from eval.baselines.adapters.da3 import DA3Adapter

        adapter = DA3Adapter(args.da3_model, args.da3_repo, args.device, args.resolution, args.face_order)
    elif args.model == "genrecon":
        from eval.baselines.adapters.genrecon import GenReconAdapter

        adapter = GenReconAdapter(
            repo=args.genrecon_repo,
            data_root=args.genrecon_data_root,
            ss_ckpt=args.genrecon_ss_ckpt,
            shape_ckpt=args.genrecon_shape_ckpt,
            tex_ckpt=args.genrecon_tex_ckpt,
            pipeline_config=args.genrecon_pipeline_config,
            device=args.device,
            face_order=args.face_order,
            work_root=args.genrecon_work_dir or args.output_dir / "genrecon_work",
            surface_points=args.genrecon_surface_points or args.eval_points,
            seed=args.seed,
            occ_threshold=args.genrecon_occ_threshold,
            boundary_width=args.genrecon_boundary_width,
            proj_batch_voxels=args.genrecon_proj_batch_voxels,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (scene, frame) in enumerate(entries):
        sample = resolve_sample(scene, frame, args.cubemap_root, args.pose_root, args.scene_root)
        print(f"[{index + 1}/{len(entries)}] {sample.key}", flush=True)
        try:
            pred = sample_points(adapter.predict(sample), args.eval_points, args.seed + index)
            target = sample_points(load_scene_points(sample.surface_path), args.eval_points, args.seed + index)
            pred, evaluation = evaluate_point_clouds(
                pred,
                target,
                thresholds=args.fscore_thresholds,
                device=args.device,
                seed=args.seed + index,
                alignment_iterations=args.alignment_iterations,
                alignment_lr=args.alignment_lr,
            )
            record = {
                "sample": sample.key,
                "status": "ok",
                **evaluation,
            }
            if args.save_predictions:
                pred_dir = args.output_dir / "predictions"
                pred_dir.mkdir(exist_ok=True)
                sample_stem = f"{scene}_{frame:05d}"
                save_ply(pred, pred_dir / f"{sample_stem}_pred_aligned.ply")
                save_ply(target, pred_dir / f"{sample_stem}_gt.ply")
        except Exception as exc:
            record = {"sample": sample.key, "status": "error", "error": str(exc)}
            traceback.print_exc()
            if args.fail_fast:
                raise
        records.append(record)
        (args.output_dir / "results.json").write_text(json.dumps(records, indent=2))

    summary = build_evaluation_summary(
        records,
        model=args.model,
        thresholds=args.fscore_thresholds,
        alignment_iterations=args.alignment_iterations,
        alignment_lr=args.alignment_lr,
        metadata={"split": str(args.split_list)},
    )
    write_evaluation_results(records, summary, args.output_dir)
    print(json.dumps(summary, indent=2))
    successful = [record for record in records if record["status"] == "ok"]
    if not successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
