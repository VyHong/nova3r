import os
import sys
from datetime import timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.utils.distributed_env import configure, validate

CONFIGURED_DEVICES = configure()

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import argparse
import torch
from training.nova3r.lightning_module import Nova3RLightningModule
from training.mesh2sdf.lightning_module import Mesh2SDFLightningModule
from lightning_data import Nova3RDataModule
from scripts.inference.demo_nova3r import load_model

def parse_args():
    parser = argparse.ArgumentParser(description="NOVA3R: 3D reconstruction from images")
    parser.add_argument("--config", default="checkpoints/da3_large_hunyuan_mini/.hydra/config_agg.yaml", help="Model and evaluation config")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path; overrides experiment.resume_checkpoint")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--devices", type=int, default=CONFIGURED_DEVICES, help="Number of GPUs; defaults to experiment.gpus")
    parser.add_argument("--ddp-timeout-seconds", type=int, default=1800)
    parser.add_argument("--mesh-method", choices=("mesh_udf", "dualmeshudf", "dcx", "latent2mesh", "mind", "nsdudf", "hrudf"), default="mesh_udf", help="Surface meshing method (default: mesh_udf)")
    parser.add_argument("--mesh-dcx-resolution", type=int, default=None)
    parser.add_argument("--mesh-dcx-iterations", type=int, default=None)
    parser.add_argument("--mesh-dcx-max-batch", type=int, default=None)
    parser.add_argument("--mesh-dcx-threshold", type=float, default=None)
    parser.add_argument("--mesh-dcx-learning-rate", type=float, default=None)
    parser.add_argument("--mesh-dcx-warm-up-end", type=int, default=None)
    parser.add_argument("--mesh-dcx-scan-block-size", type=int, default=None)
    parser.add_argument("--mesh-dcx-scan-lipschitz-safety", type=float, default=None)
    parser.add_argument("--mesh-dcx-projection-batch-size", type=int, default=None)
    parser.add_argument("--mesh-udf-resolution", type=int, default=None)
    parser.add_argument("--mesh-udf-sparse-bricks", action="store_true")
    parser.add_argument("--mesh-udf-brick-cells", type=int, default=None)
    parser.add_argument("--mesh-udf-cull-margin", type=float, default=None)
    parser.add_argument("--mesh-udf-query-batch-size", type=int, default=None)
    parser.add_argument("--mesh-udf-optimized", action="store_true")
    parser.add_argument("--mesh-udf-mc-tile-cells", type=int, default=None)
    parser.add_argument("--test-list", help="Override the configured test split list")
    parser.add_argument("--mesh-dualmesh-max-depth", type=int, default=None)
    parser.add_argument("--mesh-dualmesh-batch-size", type=int, default=None)
    parser.add_argument("--mesh-dualmesh-reliability-threshold", type=float, default=None)
    parser.add_argument("--mesh-dualmesh-subdivision-threshold", type=float, default=None)
    parser.add_argument("--mesh-dualmesh-projection-threshold", type=float, default=None)
    parser.add_argument("--mesh-dualmesh-qef-threshold", type=float, default=None)
    parser.add_argument("--mesh-mind-resolution", type=int, default=None, help="MIND grid resolution (default: 256)")
    parser.add_argument("--mesh-mind-r2", type=float, default=None, help="MIND UDF distance cutoff (default: 0.01)")
    parser.add_argument("--mesh-mind-sample-points", type=int, default=None, help="MIND point samples (default: 250000)")
    parser.add_argument("--mesh-mind-sample-iterations", type=int, default=None, help="MIND point refinement steps (default: 25)")
    parser.add_argument("--mesh-mind-mesh-iterations", type=int, default=None, help="MIND final mesh refinement steps (default: 50)")
    parser.add_argument("--mesh-nsdudf-resolution", type=int, default=None, help="NSDUDF grid samples per axis (default: 129)")
    parser.add_argument("--mesh-nsdudf-query-batch-size", type=int, default=None, help="NSDUDF UDF/gradient query batch (default: 10000)")
    parser.add_argument("--mesh-nsdudf-max-avg-factor", type=float, default=None, help="NSDUDF average-distance cell filter in voxels (default: 1.2)")
    parser.add_argument("--mesh-nsdudf-max-max-factor", type=float, default=None, help="NSDUDF maximum-distance cell filter in voxels (default: 2.0)")
    parser.add_argument("--mesh-nsdudf-target-faces", type=int, default=None, help="Optional post-extraction triangle budget")
    parser.add_argument("--mesh-hrudf-resolution", type=int, default=None, help="HRUDF grid samples per axis (default: 129)")
    parser.add_argument("--mesh-hrudf-query-batch-size", type=int, default=None, help="HRUDF UDF/gradient query batch (default: 10000)")
    parser.add_argument("--mesh-hrudf-recursions", type=int, default=None, help="HRUDF refinement passes (default: 5)")
    parser.add_argument("--test-sample", default=None, help="Mesh only the named test sample")
    args = parser.parse_args()

    return args


def main():

    torch.set_float32_matmul_precision("medium")
    args = parse_args()
    pl.seed_everything(42, workers=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.device in {"cuda", "gpu"}:
        torch.cuda.set_device(local_rank)
        model_device = f"cuda:{local_rank}"
    else:
        model_device = args.device
    model, cfg = load_model(ckpt_path=args.ckpt, device=model_device, config_path=args.config, stage="test")
    if args.mesh_method is not None:
        cfg.mesh_method = args.mesh_method
    if args.mesh_udf_resolution is not None:
        cfg.mesh_mc = args.mesh_udf_resolution
    for key in (
        "mesh_dcx_resolution",
        "mesh_dcx_iterations",
        "mesh_dcx_max_batch",
        "mesh_dcx_threshold",
        "mesh_dcx_learning_rate",
        "mesh_dcx_warm_up_end",
        "mesh_dcx_scan_block_size",
        "mesh_dcx_scan_lipschitz_safety",
        "mesh_dcx_projection_batch_size",
    ):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.mesh_udf_sparse_bricks:
        cfg.mesh_udf_sparse_bricks = True
    if args.mesh_udf_optimized:
        cfg.mesh_udf_streamed_grid = True
        cfg.mesh_udf_fast_cleanup = True
    if args.mesh_udf_mc_tile_cells is not None:
        cfg.mesh_udf_mc_tile_cells = args.mesh_udf_mc_tile_cells
    for key in ("mesh_udf_brick_cells", "mesh_udf_cull_margin", "mesh_udf_query_batch_size"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.mesh_dualmesh_max_depth is not None:
        cfg.mesh_dualmesh_max_depth = args.mesh_dualmesh_max_depth
    if args.mesh_dualmesh_batch_size is not None:
        cfg.mesh_dualmesh_batch_size = args.mesh_dualmesh_batch_size
    if args.mesh_dualmesh_reliability_threshold is not None:
        cfg.mesh_dualmesh_reliability_threshold = args.mesh_dualmesh_reliability_threshold
    for key in (
        "mesh_dualmesh_subdivision_threshold",
        "mesh_dualmesh_projection_threshold",
        "mesh_dualmesh_qef_threshold",
    ):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.mesh_mind_resolution is not None:
        cfg.mesh_mind_resolution = args.mesh_mind_resolution
    if args.mesh_mind_r2 is not None:
        cfg.mesh_mind_r2 = args.mesh_mind_r2
    for key in ("mesh_mind_sample_points", "mesh_mind_sample_iterations", "mesh_mind_mesh_iterations"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    for key in (
        "mesh_nsdudf_resolution",
        "mesh_nsdudf_query_batch_size",
        "mesh_nsdudf_max_avg_factor",
        "mesh_nsdudf_max_max_factor",
        "mesh_nsdudf_target_faces",
    ):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    for key in ("mesh_hrudf_resolution", "mesh_hrudf_query_batch_size", "mesh_hrudf_recursions"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.test_sample is not None:
        cfg.test_sample = args.test_sample
    if args.test_list is not None:
        test_datasets = list(cfg.data.test_datasets.values())
        if len(test_datasets) != 1:
            raise ValueError("--test-list requires exactly one configured test dataset")
        test_datasets[0].params.list_path = args.test_list
        cfg.data.test_batch_size = 1
    data_cfg = cfg.data
    devices = args.devices if args.devices is not None else int(cfg.gpus)
    validate(torch, devices, args.device)

    datamodule = Nova3RDataModule(
        data_cfg,
    )
    module = eval(cfg.lightning_module)(cfg, model)

    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=cfg.output_dir, name="nova3r_training")

    trainer = pl.Trainer(
        accelerator=args.device,
        devices=devices,
        strategy=(DDPStrategy(timeout=timedelta(seconds=args.ddp_timeout_seconds)) if devices > 1 else "auto"),
        logger=logger,
        precision=cfg.amp_dtype,
    )
    trainer.test(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
