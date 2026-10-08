#!/usr/bin/env python3
"""Run a baseline and report directional Chamfer distances and SSI3D F-scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np

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
    parser.add_argument("--model", choices=("da3", "genrecon", "nova3r", "pager"), default="da3")
    parser.add_argument(
        "--dataset-layout",
        choices=("sage10k", "scannetpp"),
        default="sage10k",
        help="Dataset directory and pose convention (default: sage10k)",
    )
    parser.add_argument("--split-list", type=Path)
    parser.add_argument(
        "--input-projection",
        choices=("cubemap", "erp"),
        default="cubemap",
        help=(
            "Use precomputed cubemap faces or one ERP; DA3 and PaGeR use spherical ERP lifting, "
            "Nova3R needs no intrinsics, and GenRecon retains its pinhole ERP path"
        ),
    )
    parser.add_argument("--cubemap-root", type=Path)
    parser.add_argument(
        "--erp-root",
        type=Path,
        help="Panorama root; supports SAGE and <scene>/panocam/images/<frame> ScanNet++ layouts",
    )
    parser.add_argument(
        "--erp-fov",
        type=float,
        default=95.0,
        help="Assumed horizontal pinhole FOV for GenRecon ERP input only (default: 95)",
    )
    parser.add_argument("--pose-root", type=Path)
    parser.add_argument("--scene-root", type=Path)
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
    parser.add_argument(
        "--da3-model",
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        help="Hugging Face model ID or local DA3 checkpoint directory",
    )
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
    parser.add_argument(
        "--nova3r-checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/scene_n2/checkpoint-last.pth",
    )
    parser.add_argument(
        "--nova3r-config",
        type=Path,
        default=REPO_ROOT / "eval/baselines/configs/nova3r_scene_n2_official.yaml",
        help="Hydra config matching the selected checkpoint",
    )
    parser.add_argument("--nova3r-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(518, 392))
    parser.add_argument("--nova3r-num-views", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--nova3r-num-queries", type=int, default=50_000)
    parser.add_argument("--nova3r-method", choices=("euler", "midpoint", "rk4"))
    parser.add_argument("--pager-repo", type=Path, default=REPO_ROOT / "third_party/PaGeR")
    parser.add_argument("--pager-checkpoint", default="pager")
    parser.add_argument("--pager-scene-mode", choices=("indoor", "outdoor"), default="indoor")
    parser.add_argument("--pager-max-depth", type=float, default=75.0)
    parser.add_argument("--pager-sky-threshold", type=float, default=0.3)
    args = parser.parse_args()
    layout_defaults = {
        "sage10k": {
            "split_list": REPO_ROOT / "data/sage-10k/test_list.json",
            "cubemap_root": Path("/tmp/datasets/sage-10k/cubemap_data/cubemaps"),
            "pose_root": Path("/mnt/projects/theses/vyhong/sage-10k/poses"),
            "scene_root": Path("/tmp/datasets/sage-10k/udf"),
        },
        "scannetpp": {
            "split_list": REPO_ROOT / "data/scannetpp/test_list.json",
            "cubemap_root": Path("/tmp/datasets/scannetpp/cubemap_data"),
            "pose_root": Path("/mnt/projects/theses/vyhong/scannetpp/scan_poses"),
            "scene_root": Path("/tmp/datasets/scannetpp/udf_color"),
        },
    }[args.dataset_layout]
    for name, value in layout_defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.output_dir is None:
        suffix = f"{args.dataset_layout}_test"
        if args.input_projection == "erp":
            suffix += "_erp"
        args.output_dir = REPO_ROOT / f"eval_results/baselines/{args.model}/{suffix}"
    if args.input_projection == "erp" and args.erp_root is None:
        parser.error("--erp-root is required when --input-projection erp")
    if args.model == "pager" and args.input_projection != "erp":
        parser.error("PaGeR follows its official spherical pipeline and requires --input-projection erp")
    if args.model == "genrecon" and args.input_projection == "erp" and not 0 < args.erp_fov < 180:
        parser.error("--erp-fov must be between 0 and 180 degrees")
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
    if any(value < 1 for value in args.nova3r_resolution):
        parser.error("--nova3r-resolution values must be positive")
    if args.nova3r_num_queries < 1:
        parser.error("--nova3r-num-queries must be positive")
    if args.model == "nova3r" and args.input_projection == "erp" and args.nova3r_num_views != 1:
        parser.error("Nova3R ERP input requires --nova3r-num-views 1")
    if args.pager_max_depth <= 0:
        parser.error("--pager-max-depth must be positive")
    if not 0 <= args.pager_sky_threshold <= 1:
        parser.error("--pager-sky-threshold must be between 0 and 1")
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

        adapter = DA3Adapter(
            args.da3_model,
            args.da3_repo,
            args.device,
            args.resolution,
            args.face_order,
            input_projection=args.input_projection,
            erp_root=args.erp_root,
            erp_fov=args.erp_fov,
        )
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
            input_projection=args.input_projection,
            erp_root=args.erp_root,
            erp_fov=args.erp_fov,
        )
    elif args.model == "nova3r":
        from eval.baselines.adapters.nova3r import Nova3RAdapter

        adapter = Nova3RAdapter(
            checkpoint=args.nova3r_checkpoint,
            config=args.nova3r_config,
            device=args.device,
            face_order=args.face_order,
            input_projection=args.input_projection,
            erp_root=args.erp_root,
            erp_fov=args.erp_fov,
            resolution=tuple(args.nova3r_resolution),
            num_views=args.nova3r_num_views,
            num_queries=args.nova3r_num_queries,
            seed=args.seed,
            method=args.nova3r_method,
        )
    elif args.model == "pager":
        from eval.baselines.adapters.pager import PaGeRAdapter

        adapter = PaGeRAdapter(
            repo=args.pager_repo,
            checkpoint=args.pager_checkpoint,
            device=args.device,
            erp_root=args.erp_root,
            scene_mode=args.pager_scene_mode,
            max_depth=args.pager_max_depth,
            sky_threshold=args.pager_sky_threshold,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (scene, frame) in enumerate(entries):
        sample = resolve_sample(
            scene,
            frame,
            args.cubemap_root,
            args.pose_root,
            args.scene_root,
            dataset_layout=args.dataset_layout,
        )
        print(f"[{index + 1}/{len(entries)}] {sample.key}", flush=True)
        try:
            prediction = adapter.predict(sample)
            sample_stem = f"{scene}_{frame:05d}"
            if args.save_predictions and hasattr(adapter, "save_2d_prediction"):
                adapter.save_2d_prediction(
                    args.output_dir / "predictions_2d", sample_stem
                )
            if args.save_predictions and getattr(
                adapter, "last_diagnostic_pointclouds", None
            ):
                diagnostic_dir = args.output_dir / "predictions_diagnostic"
                diagnostic_dir.mkdir(exist_ok=True)
                for name, points in adapter.last_diagnostic_pointclouds.items():
                    save_ply(points, diagnostic_dir / f"{sample_stem}_{name}.ply")
            pred = sample_points(prediction, args.eval_points, args.seed + index)
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
                save_ply(target, pred_dir / f"{sample_stem}_gt.ply")
                if hasattr(adapter, "save_aligned_mesh"):
                    adapter.save_aligned_mesh(
                        pred_dir / f"{sample_stem}_pred_aligned_mesh.ply",
                        evaluation["alignment_scale"],
                        np.asarray(evaluation["alignment_shift"], dtype=np.float32),
                    )
                else:
                    save_ply(pred, pred_dir / f"{sample_stem}_pred_aligned.ply")
        except Exception as exc:
            record = {"sample": sample.key, "status": "error", "error": str(exc)}
            traceback.print_exc()
            if args.fail_fast:
                raise
        records.append(record)
        (args.output_dir / "results.json").write_text(json.dumps(records, indent=2))

    metadata = {
        "split": str(args.split_list),
        "dataset_layout": args.dataset_layout,
        "input_projection": args.input_projection,
        "erp_pinhole_horizontal_fov": (
            args.erp_fov
            if args.input_projection == "erp" and args.model == "genrecon"
            else None
        ),
    }
    if args.model == "da3" and args.input_projection == "erp":
        metadata["da3_erp_depth_interpretation"] = "predicted_intrinsics_pinhole_z_to_range"
    if args.model == "nova3r":
        metadata.update(
            {
                "nova3r_checkpoint": str(args.nova3r_checkpoint),
                "nova3r_num_views": args.nova3r_num_views,
                "nova3r_num_queries": args.nova3r_num_queries,
                "nova3r_face_order": args.face_order if args.input_projection == "cubemap" else None,
            }
        )
    elif args.model == "pager":
        metadata.update(
            {
                "pager_checkpoint": args.pager_checkpoint,
                "pager_scene_mode": args.pager_scene_mode,
                "pager_projection": "spherical_erp",
            }
        )
    summary = build_evaluation_summary(
        records,
        model=args.model,
        thresholds=args.fscore_thresholds,
        alignment_iterations=args.alignment_iterations,
        alignment_lr=args.alignment_lr,
        metadata=metadata,
    )
    write_evaluation_results(records, summary, args.output_dir)
    print(json.dumps(summary, indent=2))
    successful = [record for record in records if record["status"] == "ok"]
    if not successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
