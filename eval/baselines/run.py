#!/usr/bin/env python3
"""Run a baseline and report directional Chamfer distances and SSI3D F-scores."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
import multiprocessing
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.baselines.common import (
    _panorama_pose,
    align_lari_pointmap,
    load_cameras,
    load_scannetpp_valid_mask,
    face_target_mask,
    align_prediction_to_target,
    panorama_visibility_mask,
    load_visibility_geometry,
    sample_mesh_surface,
    compute_visibility_metrics,
    DEFAULT_ALIGNMENT_ITERATIONS,
    DEFAULT_ALIGNMENT_LR,
    DEFAULT_EVAL_POINTS,
    DEFAULT_FSCORE_THRESHOLDS,
    build_evaluation_summary,
    evaluate_point_clouds,
    load_scene_points,
    read_manifest,
    resolve_sample,
    resolve_erp_path,
    sample_points,
    write_evaluation_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("genrecon", "nova3r", "pager", "lari", "volfill"), default="nova3r")
    parser.add_argument(
        "--dataset-layout",
        choices=("sage10k", "scannetpp", "3dfront"),
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
        "--cubemap-fov",
        type=float,
        default=95.0,
        help="FOV encoded in cubemap directory names (default: 95)",
    )
    parser.add_argument(
        "--erp-root",
        type=Path,
        help="Panorama root; defaults to prepared ScanNet++ ERPs or original SAGE/3D-FRONT panorama roots",
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
    parser.add_argument(
        "--workers", type=int, default=None,
        help="GenRecon processes (default: one per --devices entry, otherwise 1)",
    )
    parser.add_argument(
        "--devices", nargs="+",
        help="GenRecon worker devices, e.g. cuda:0 cuda:1; workers cycle over this list",
    )
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
    parser.add_argument("--visibility-mesh-root", type=Path,
                        help="Original GT mesh root, when separate from --scene-root")
    parser.add_argument("--anonymization-root", type=Path,
                        help="ScanNet++ root containing <scene>/scans/mesh_aligned_0.05_mask.txt")
    parser.add_argument("--visibility-tolerance", type=float, default=0.01,
                        help="First-hit distance tolerance in scene units (default: 0.01)")
    parser.add_argument("--visibility-metrics", action="store_true",
                        help="Report visible/occluded GT coverage and write table.md/table.csv")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep successful records from output-dir/results.json and run only unfinished samples",
    )
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
        "--genrecon-scannet-root",
        type=Path,
        default=Path("/tmp/datasets/scannetpp/data"),
        help="ScanNet++ root containing <scene>/dslr/colmap/points3D.txt",
    )
    parser.add_argument(
        "--genrecon-scannet-chunk-source",
        choices=("colmap", "evaluation_surface"),
        default="evaluation_surface",
        help="Source used to estimate ScanNet++ chunks (default: evaluation_surface)",
    )
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
    parser.add_argument("--nova3r-independent-faces", action="store_true",
                        help="Infer each cubemap face separately, GT-align each cloud, then merge")
    parser.add_argument("--nova3r-num-views", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--nova3r-num-queries", type=int, default=50_000)
    parser.add_argument("--nova3r-method", choices=("euler", "midpoint", "rk4"))
    parser.add_argument("--pager-repo", type=Path, default=REPO_ROOT / "third_party/PaGeR")
    parser.add_argument("--pager-checkpoint", default="pager")
    parser.add_argument("--pager-scene-mode", choices=("indoor", "outdoor"), default="indoor")
    parser.add_argument("--pager-max-depth", type=float, default=75.0)
    parser.add_argument("--pager-sky-threshold", type=float, default=0.3)
    parser.add_argument("--lari-repo", type=Path, default=REPO_ROOT / "third_party/LaRI")
    parser.add_argument("--lari-python", default=sys.executable)
    parser.add_argument("--lari-alignment", choices=("native", "all-chamfer", "visible-chamfer", "merged-chamfer"), default="native")
    parser.add_argument("--lari-checkpoint", default="lari_scene_pointmap.pth")
    parser.add_argument("--volfill-repo", type=Path, default=REPO_ROOT / "third_party/VolFill")
    parser.add_argument("--volfill-python", default=sys.executable)
    parser.add_argument("--volfill-checkpoint", default="TuanNgo/VolFill")
    parser.add_argument("--volfill-raw-dir", type=Path, help="Keep upstream TUDF, bbox metadata, and input for each face")
    parser.add_argument("--volfill-steps", type=int, default=50)
    parser.add_argument("--volfill-cfg-strength", type=float, default=3.0)
    parser.add_argument("--volfill-threshold", type=float, default=0.8,
                        help="Surface threshold in voxel units, following upstream visualization")
    args = parser.parse_args()
    args.independent_faces = args.nova3r_independent_faces or (args.model in ("lari", "volfill") and args.input_projection == "cubemap")
    if args.volfill_steps < 1 or args.volfill_threshold < 0:
        parser.error("VolFill requires positive steps and a nonnegative surface threshold")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    if args.model != "genrecon" and (args.workers is not None or args.devices is not None):
        parser.error("--workers and --devices are currently supported only for GenRecon")
    args.workers = args.workers or len(args.devices or [args.device])
    if args.devices and len(args.devices) > args.workers:
        parser.error("--workers must be at least the number of --devices")
    layout_defaults = {
        "sage10k": {
            "split_list": REPO_ROOT / "data/sage-10k/test_list.json",
            "cubemap_root": Path("/tmp/datasets/sage-10k/cubemap_data/cubemaps"),
            "pose_root": Path("/mnt/projects/theses/vyhong/sage-10k/poses"),
            "scene_root": Path("/tmp/datasets/sage-10k/udf"),
        },
        "scannetpp": {
            "split_list": REPO_ROOT / "data/scannetpp/final_test_list.json",
            "cubemap_root": Path("/tmp/datasets/scannetpp/cubemap_data"),
            "pose_root": REPO_ROOT / "datasets/scannetpp/test_scenes",
            "scene_root": REPO_ROOT / "datasets/scannetpp/test_scenes",
        },
        "3dfront": {
            "split_list": REPO_ROOT / "data/3dfront/final_test_list.json",
            "cubemap_root": Path("/tmp/datasets/3dfront/cubemap_data"),
            "pose_root": Path("/tmp/datasets/3dfront/panorama_scenes"),
            "scene_root": Path("/tmp/datasets/3dfront/better_udf"),
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
        args.erp_root = {
            "scannetpp": REPO_ROOT / "datasets/scannetpp/panos/updated_final_test_100_prepared_erp",
            "sage10k": Path("/tmp/datasets/sage-10k/panorama-scenes"),
            "3dfront": Path("/tmp/datasets/3dfront/panorama_scenes"),
        }[args.dataset_layout]
    if args.input_projection == "erp" and not args.erp_root.is_dir():
        parser.error(f"Panorama root does not exist: {args.erp_root}; supply --erp-root")
    if args.model == "pager" and args.input_projection != "erp":
        parser.error("PaGeR follows its official spherical pipeline and requires --input-projection erp")
    if args.model == "genrecon" and args.input_projection == "erp" and not 0 < args.erp_fov < 180:
        parser.error("--erp-fov must be between 0 and 180 degrees")
    if not 0 < args.cubemap_fov < 180:
        parser.error("--cubemap-fov must be between 0 and 180 degrees")
    if sorted(args.face_order) != list(range(6)):
        parser.error("--face-order must be a permutation of 0 1 2 3 4 5")
    if args.visibility_tolerance <= 0:
        parser.error("--visibility-tolerance must be positive")
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
    if args.nova3r_independent_faces and (args.model != "nova3r" or args.input_projection != "cubemap"):
        parser.error("--nova3r-independent-faces requires --model nova3r --input-projection cubemap")
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


def face_alignment_method(args):
    if args.input_projection == "erp":
        return "global_gt_scale_translation"
    if args.model == "lari":
        return {"merged-chamfer": "merged_layers_faces_global_gt_scale_translation",
                "native": "lari_first_layer_scale_camera_z_shift",
                "all-chamfer": "independent_face_frustum_gt_scale_translation",
                "visible-chamfer": "independent_face_visible_gt_chamfer_scale_translation"}[args.lari_alignment]
    return "independent_face_frustum_gt_scale_translation"


def single_image_settings(args):
    if args.model == "lari":
        return {"checkpoint": args.lari_checkpoint, "input_projection": args.input_projection, "alignment": face_alignment_method(args), "iterations": args.alignment_iterations, "lr": args.alignment_lr, "empty_value_policy": "drop_nonfinite_and_exact_zero_xyz" if args.lari_alignment in ("merged-chamfer", "all-chamfer") else "drop_nonfinite"}
    if args.model == "volfill":
        return {"checkpoint": args.volfill_checkpoint, "input_projection": args.input_projection, "steps": args.volfill_steps,
                "cfg_strength": args.volfill_cfg_strength, "threshold": args.volfill_threshold,
                "pipeline": "upstream_export_v2",
                "raw_dir": str(args.volfill_raw_dir) if args.volfill_raw_dir else None}
    return None


def single_image_settings_match(args, record):
    saved = record.get("inference_settings")
    if isinstance(saved, dict):
        saved = dict(saved)
        # Legacy cubemap records predate the explicit projection setting.
        if args.model == "volfill":
            saved.setdefault("input_projection", "cubemap")
    return saved == single_image_settings(args)


def build_adapter(args):
    if args.model in ("lari", "volfill"):
        from eval.baselines.adapters.single_image import SingleImageAdapter

        settings = single_image_settings(args)
        return SingleImageAdapter(
            args.model, getattr(args, f"{args.model}_repo"),
            getattr(args, f"{args.model}_python"), args.device, args.face_order, settings, args.seed,
            input_projection=args.input_projection, erp_root=args.erp_root,
        )
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
            independent_faces=args.nova3r_independent_faces,
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

    return adapter


def evaluate_sample(args, adapter, index, entry, total):
    scene, frame = entry
    sample = resolve_sample(
        scene,
        frame,
        args.cubemap_root,
        args.pose_root,
        args.scene_root,
        dataset_layout=args.dataset_layout,
        cubemap_fov=getattr(args, "cubemap_fov", 95.0),
    )
    print(f"[{index + 1}/{total}] {sample.key}", flush=True)
    sample_started = time.perf_counter()
    try:
        prediction = adapter.predict(sample)
        prediction_seconds = time.perf_counter() - sample_started
        geometry_started = time.perf_counter()
        # 3D-FRONT scene identifiers contain ``scene/room``. Keep prediction
        # artifacts flat and avoid interpreting that slash as a subdirectory.
        sample_stem = f"{scene.replace('/', '_')}_{frame:05d}"
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
        if args.visibility_metrics or getattr(args, "independent_faces", False):
            vertices, faces, evaluation_faces, mesh_path = load_visibility_geometry(
                sample, args.visibility_mesh_root, args.anonymization_root,
            )
            target = sample_mesh_surface(vertices, evaluation_faces, args.eval_points, args.seed + index)
        else:
            target = sample_points(load_scene_points(sample.surface_path), args.eval_points, args.seed + index)
        visible = None
        if args.visibility_metrics or (args.independent_faces and args.model == "lari" and args.lari_alignment == "visible-chamfer"):
            center = np.linalg.inv(_panorama_pose(sample))[:3, 3]
            visible = panorama_visibility_mask(target, center, vertices, faces, args.visibility_tolerance)
        geometry_seconds = time.perf_counter() - geometry_started
        alignment_started = time.perf_counter()
        face_alignments = {}
        skipped_faces = {}
        face_clouds = getattr(adapter, "last_face_pointclouds", {})
        if face_clouds and not (args.model == "lari" and args.lari_alignment == "merged-chamfer"):
            from PIL import Image

            paths, intrinsics, extrinsics = load_cameras(sample, args.face_order)
            cameras = {str(face): (path, intrinsic, extrinsic)
                       for face, path, intrinsic, extrinsic in zip(args.face_order, paths, intrinsics, extrinsics)}
            aligned_faces = []
            for face, cloud in face_clouds.items():
                path, intrinsic, extrinsic = cameras[face]
                with Image.open(path) as image:
                    width, height = image.size
                valid_pixels = load_scannetpp_valid_mask(sample, path, width, height)
                face_mask = face_target_mask(target, extrinsic, intrinsic, valid_pixels)
                if args.model == "lari" and args.lari_alignment == "visible-chamfer":
                    face_mask &= visible
                face_target = target[face_mask]
                if not len(face_target):
                    skipped_faces[face] = "no_valid_gt_points"
                    print(f"Skipping cubemap face {face}: no valid GT points", flush=True)
                    continue
                fit_details = {}
                if args.model == "lari" and args.lari_alignment == "native":
                    aligned_face, scale, shift, fit_details = align_lari_pointmap(
                        adapter.last_face_pointmaps[face], vertices, faces,
                        extrinsic, intrinsic, valid_pixels,
                    )
                    aligned_face = sample_points(aligned_face, args.eval_points, args.seed + index)
                else:
                    aligned_face, scale, shift = align_prediction_to_target(
                        sample_points(cloud, args.eval_points, args.seed + index), face_target,
                        device=args.device, seed=args.seed + index + int(face),
                        max_iterations=args.alignment_iterations, lr=args.alignment_lr,
                    )
                aligned_faces.append(aligned_face)
                face_alignments[face] = {"scale": scale, "shift": shift.tolist(),
                                         "target_points": len(face_target), **fit_details}
            if not aligned_faces:
                raise ValueError("No cubemap faces with valid GT points remain for alignment")
            merged_prediction = np.concatenate(aligned_faces)
            pred = sample_points(merged_prediction, args.eval_points, args.seed + index)
        alignment_seconds = time.perf_counter() - alignment_started
        metrics_started = time.perf_counter()
        pred, evaluation = evaluate_point_clouds(
            pred,
            target,
            thresholds=args.fscore_thresholds,
            device=args.device,
            seed=args.seed + index,
            alignment_iterations=args.alignment_iterations,
            alignment_lr=args.alignment_lr,
            prealigned=bool(face_alignments),
        )
        if face_alignments:
            evaluation["face_alignments"] = face_alignments
            evaluation["alignment_method"] = face_alignment_method(args)
        if args.model == "lari" and args.lari_alignment in ("merged-chamfer", "all-chamfer"):
            evaluation["alignment_method"] = face_alignment_method(args)
            evaluation["empty_value_policy"] = "drop_nonfinite_and_exact_zero_xyz"
        if skipped_faces:
            evaluation["skipped_faces"] = skipped_faces
        empty_faces = getattr(adapter, "last_empty_faces", [])
        if empty_faces:
            evaluation["empty_faces"] = empty_faces
            evaluation["empty_face_metrics"] = {
                face: {"status": "empty", "pred_points": 0,
                       "chamfer_mean": None,
                       **{f"f_score_{threshold}_full": 0.0
                          for threshold in args.fscore_thresholds}}
                for face in empty_faces
            }
        if args.visibility_metrics:
            evaluation["visibility_mesh"] = str(mesh_path)
            evaluation["visibility_protocol"] = "mesh_raycast_v3_directional"
            evaluation["visibility_tolerance"] = args.visibility_tolerance
            evaluation.update(compute_visibility_metrics(
                pred, target, visible, args.fscore_thresholds,
            ))
        record = {
            "sample": sample.key,
            "status": "ok",
            "model": args.model,
            "inference_settings": single_image_settings(args),
            "independent_faces": bool(getattr(args, "independent_faces", False)),
            "nova3r_independent_faces": bool(getattr(args, "nova3r_independent_faces", False)),
            "erp_preprocessing": "aspect_preserving_v1" if args.input_projection == "erp" else None,
            "input_erp": (str(resolve_erp_path(sample, args.erp_root).resolve())
                          if args.input_projection == "erp" else None),
            "timings": {
                "prediction_seconds": prediction_seconds,
                "ground_truth_seconds": geometry_seconds,
                "face_alignment_seconds": alignment_seconds,
                "metrics_seconds": time.perf_counter() - metrics_started,
            },
            **evaluation,
        }
        if args.save_predictions:
            pred_dir = args.output_dir / "predictions"
            pred_dir.mkdir(exist_ok=True)
            if not face_alignments:
                save_ply(target, pred_dir / f"{sample_stem}_gt.ply")
            if hasattr(adapter, "save_aligned_mesh"):
                adapter.save_aligned_mesh(
                    pred_dir / f"{sample_stem}_pred_aligned_mesh.ply",
                    evaluation["alignment_scale"],
                    np.asarray(evaluation["alignment_shift"], dtype=np.float32),
                )
            else:
                save_ply(merged_prediction if face_alignments else pred,
                         pred_dir / f"{sample_stem}_pred_aligned.ply")
        record["timings"]["total_seconds"] = time.perf_counter() - sample_started
    except Exception as exc:
        record = {"sample": sample.key, "status": "error", "error": str(exc)}
        traceback.print_exc()
        if args.fail_fast:
            raise
    return record


_worker_args = None
_worker_adapter = None


def initialize_worker(args, device, worker_index):
    global _worker_args, _worker_adapter
    # Spawn gives each process its own CUDA context and model state. Select the
    # current device too: GenRecon dependencies sometimes use bare .cuda().
    import torch

    torch.set_num_threads(1)
    args.device = device
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
    args.genrecon_work_dir = (
        args.genrecon_work_dir or args.output_dir / "genrecon_work"
    ) / f"worker_{worker_index}"
    _worker_args = args
    _worker_adapter = build_adapter(args)


def evaluate_worker(index, entry, total):
    return evaluate_sample(_worker_args, _worker_adapter, index, entry, total)


def parallel_records(args, indexed_entries, total):
    """Yield completed records with original indices; keep one job per worker."""
    entries = [entry for _, entry in indexed_entries]
    if len(set(entries)) != len(entries):
        raise ValueError("Parallel evaluation requires unique scene/frame entries (prediction paths are shared)")
    devices = args.devices or [args.device]
    jobs = iter(indexed_entries)
    context = multiprocessing.get_context("spawn")
    with ExitStack() as stack:
        pending = {}
        for worker in range(min(args.workers, len(indexed_entries))):
            executor = stack.enter_context(ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
                initializer=initialize_worker,
                initargs=(args, devices[worker % len(devices)], worker),
            ))
            index, entry = next(jobs)
            future = executor.submit(evaluate_worker, index, entry, total)
            pending[future] = (executor, index)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Check every finished job before dispatching more, so fail-fast
            # never queues an additional job after an observed failure.
            completed = [(future, future.result()) for future in done]
            for future, record in completed:
                executor, index = pending.pop(future)
                yield index, record
                job = next(jobs, None)
                if job is not None:
                    next_index, entry = job
                    future = executor.submit(evaluate_worker, next_index, entry, total)
                    pending[future] = (executor, next_index)


def main():
    args = parse_args()
    entries = read_manifest(args.split_list, args.dataset_layout)
    if args.scene:
        entries = [entry for entry in entries if entry[0] == args.scene]
    if args.limit is not None:
        entries = entries[: args.limit]
    if not entries:
        raise ValueError("No samples selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ordered_records = {}
    results_path = args.output_dir / "results.json"
    if args.resume and results_path.is_file():
        prior_records = json.loads(results_path.read_text())
        index_by_key = {
            f"{scene} {frame:05d}": index
            for index, (scene, frame) in enumerate(entries)
        }
        for record in prior_records:
            index = index_by_key.get(record.get("sample"))
            input_matches = (
                args.input_projection != "erp"
                or (record.get("erp_preprocessing") == "aspect_preserving_v1"
                    and bool(record.get("input_erp"))
                    and Path(record["input_erp"]).resolve().is_relative_to(args.erp_root.resolve()))
            )
            if (index is not None and record.get("status") == "ok" and input_matches
                    and (args.model not in ("lari", "volfill")
                         or (record.get("model") == args.model
                             and single_image_settings_match(args, record)))
                    and bool(record.get("nova3r_independent_faces", False)) == args.nova3r_independent_faces
                    and (not args.independent_faces or record.get("alignment_method") == face_alignment_method(args))
                    and (not args.visibility_metrics or (record.get("visibility_protocol") == "mesh_raycast_v3_directional"
                        and record.get("visibility_tolerance") == args.visibility_tolerance))):
                ordered_records[index] = record
        print(
            f"Resuming with {len(ordered_records)}/{len(entries)} successful samples complete",
            flush=True,
        )

    indexed_entries = [
        (index, entry)
        for index, entry in enumerate(entries)
        if index not in ordered_records
    ]
    if indexed_entries and args.model == "genrecon" and (args.workers > 1 or args.devices):
        completed_records = parallel_records(args, indexed_entries, len(entries))
    elif indexed_entries:
        adapter = build_adapter(args)
        completed_records = (
            (index, evaluate_sample(args, adapter, index, entry, len(entries)))
            for index, entry in indexed_entries
        )
    else:
        completed_records = ()

    records = [ordered_records[key] for key in sorted(ordered_records)]
    for index, record in completed_records:
        ordered_records[index] = record
        records = [ordered_records[key] for key in sorted(ordered_records)]
        results_path.write_text(json.dumps(records, indent=2))

    metadata = {
        "visibility_protocol": (
            "mesh_raycast_v3_directional: first hit within visibility_tolerance; masked GT faces excluded; regional CD=GT-to-pred mean; "
            "APD=GT coverage; complete CD=symmetric mean; table=cm and percent"
            if args.visibility_metrics else None
        ),
        "visibility_tolerance": args.visibility_tolerance,
        "visibility_mesh_root": str(args.visibility_mesh_root) if args.visibility_mesh_root else None,
        "split": str(args.split_list),
        "dataset_layout": args.dataset_layout,
        "input_projection": args.input_projection,
        "erp_root": str(args.erp_root.resolve()) if args.erp_root is not None else None,
        "cubemap_fov": args.cubemap_fov if args.input_projection == "cubemap" else None,
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
                "nova3r_num_views": 1 if args.nova3r_independent_faces else args.nova3r_num_views,
                "nova3r_independent_faces": args.nova3r_independent_faces,
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
    if args.model in ("lari", "volfill"):
        metadata.update({"checkpoint": getattr(args, f"{args.model}_checkpoint"),
                         "independent_faces": args.independent_faces,
                         "face_order": args.face_order if args.independent_faces else None})
        if args.model == "volfill":
            metadata.update({"volfill_steps": args.volfill_steps,
                             "volfill_cfg_strength": args.volfill_cfg_strength,
                             "volfill_threshold_voxels": args.volfill_threshold})
    summary = build_evaluation_summary(
        records,
        model=args.model,
        thresholds=args.fscore_thresholds,
        alignment_iterations=args.alignment_iterations,
        alignment_lr=args.alignment_lr,
        metadata=metadata,
    )
    if args.independent_faces:
        summary["alignment"]["method"] = face_alignment_method(args)
        if args.model == "lari" and args.lari_alignment == "native":
            summary["alignment"] = {"method": "lari_first_layer_scale_camera_z_shift",
                                    "solver": "least_squares", "gt_layers": 1}
    write_evaluation_results(records, summary, args.output_dir)
    print(json.dumps(summary, indent=2))
    successful = [record for record in records if record["status"] == "ok"]
    if not successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
