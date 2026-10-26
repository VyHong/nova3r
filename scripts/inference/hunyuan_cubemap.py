"""Run the original Hunyuan model per cube face and assemble by extrinsics."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nova3r.models.hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--model-to-camera", type=Path,
                        help="JSON 4x4 model-to-OpenCV-camera transform; default identity")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this batch run")
    cameras = json.loads((args.images / "camera_data.json").read_text())
    model_to_camera = np.eye(4)
    if args.model_to_camera:
        model_to_camera = np.asarray(json.loads(args.model_to_camera.read_text()), dtype=float)
    if model_to_camera.shape != (4, 4) or not np.isfinite(model_to_camera).all():
        raise ValueError("model-to-camera must be a finite 4x4 matrix")
    inputs = {}
    for name, camera in sorted(cameras.items()):
        image_path = args.images / f"{name}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        extrinsic = np.asarray(camera["extrinsics"], dtype=float)
        if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
            raise ValueError(f"Invalid extrinsic for {name}")
        inputs[name] = (image_path, np.linalg.inv(extrinsic) @ model_to_camera)
    args.output.mkdir(parents=True, exist_ok=True)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        str(REPO_ROOT / "checkpoints/hunyuan_dit/model.fp16.ckpt"),
        str(REPO_ROOT / "scripts/inference/hunyuan_shape_v2_1.yaml"),
        device="cuda", dtype=torch.float16,
    )
    scene = trimesh.Scene()
    meshes = []
    records = {}
    colors = [(230, 90, 80, 255), (80, 170, 230, 255), (100, 200, 110, 255),
              (240, 190, 70, 255), (180, 110, 220, 255), (70, 210, 200, 255)]
    for index, (name, (image_path, transform)) in enumerate(inputs.items()):
        raw_path = args.output / f"{name}_raw.glb"
        print(f"Generating {name} from {image_path}", flush=True)
        mesh = pipeline(
            image=Image.open(image_path).convert("RGBA"),
            num_inference_steps=args.steps, guidance_scale=5.0,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            box_v=1.01, octree_resolution=args.resolution,
            mc_level=-1 / 512, num_chunks=8000, output_type="trimesh",
        )[0]
        if mesh is None or not len(mesh.faces):
            raise RuntimeError(f"No mesh extracted for {name}")
        mesh.export(raw_path)
        mesh.apply_transform(transform)
        mesh.export(args.output / f"{name}_world.glb")
        mesh.visual.vertex_colors = colors[index % len(colors)]
        scene.add_geometry(mesh, node_name=name, geom_name=name)
        meshes.append(mesh)
        records[name] = {"model_to_world": transform.tolist(),
                         "vertices": len(mesh.vertices), "faces": len(mesh.faces)}
        print(f"Saved {name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces", flush=True)
    scene.export(args.output / "assembled.glb")
    trimesh.util.concatenate(meshes).export(args.output / "assembled.ply")
    metadata = {
        "images": str(args.images.resolve()), "steps": args.steps,
        "seed": args.seed, "resolution": args.resolution,
        "config": "scripts/inference/hunyuan_shape_v2_1.yaml",
        "checkpoint": "checkpoints/hunyuan_dit/model.fp16.ckpt",
        "model_to_camera": model_to_camera.tolist(),
        "assembly": "inverse(world_to_camera) @ model_to_camera",
        "limitation": "Approximate extrinsic-only concatenation, not fused reconstruction. "
                      "Canonical Hunyuan outputs have no calibrated camera pose, depth, "
                      "or metric scale. Identity model-to-camera is an uncalibrated assumption.",
        "faces": records,
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved assembly to {args.output}", flush=True)


if __name__ == "__main__":
    main()
