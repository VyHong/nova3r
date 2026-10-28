# Preprocessing Runbook

## Setup (repo root)

```bash
conda activate nova3r
pip install "transformers>=4.56.0" torchvision
# If needed for DINOv3 model access:
hf auth login
```

## Mount inputs (skip existing mounts)

```bash
bash mount_datasets.sh
bash bulk_mount_mount_zip.sh /mnt/datasets/scannetpp/scannetpp_panorama.zip /tmp/datasets/scannetpp
bash bulk_mount_mount_zip.sh /mnt/datasets/sage10k/panorama-scenes.zip /tmp/datasets/sage-10k
```

## ScanNet++

DINOv3:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/preprocessing/features/panorama_to_dinov3_features.py \
  /tmp/datasets/scannetpp/scannetpp_panorama \
  datasets/scannetpp/dinov3_latents \
  --dataset-layout scannetpp \
  --backbone dinov3 \
  --batch-size 16 \
  --num-workers 16
```

DINOv2:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/preprocessing/features/panorama_to_dinov3_features.py \
  /tmp/datasets/scannetpp/scannetpp_panorama \
  datasets/scannetpp/dinov2_latents \
  --dataset-layout scannetpp \
  --backbone dinov2 \
  --batch-size 16 \
  --num-workers 16
```

## Sage-10K

DINOv3:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/preprocessing/features/panorama_to_dinov3_features.py \
  /tmp/datasets/sage-10k/panorama-scenes \
  datasets/sage-10k/dinov3_latents \
  --dataset-layout sage10k \
  --backbone dinov3 \
  --batch-size 16 \
  --num-workers 16
```

DINOv2:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/preprocessing/features/panorama_to_dinov3_features.py \
  /tmp/datasets/sage-10k/panorama-scenes \
  datasets/sage-10k/dinov2_latents \
  --dataset-layout sage10k \
  --backbone dinov2 \
  --batch-size 16 \
  --num-workers 16
```

## 3D-FRONT

DINOv3:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/preprocessing/features/panorama_to_dinov3_features.py \
  /tmp/datasets/3dfront/panoramas \
  datasets/3dfront/dinov3_latents \
  --dataset-layout 3dfront \
  --backbone dinov3 \
  --batch-size 16 \
  --num-workers 16
```

DINOv2:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/preprocessing/features/panorama_to_dinov3_features.py \
  /tmp/datasets/3dfront/panoramas \
  datasets/3dfront/dinov2_latents \
  --dataset-layout 3dfront \
  --backbone dinov2 \
  --batch-size 16 \
  --num-workers 16
```

## GT latent posteriors

Saves `*_before_sampling.pt` (FP32 mean and log-variance, computed with BF16
in these examples); existing files are skipped. Commands use separate Hunyuan
output directories to avoid mixing orientations.

GT encoding defaults to `--vae-coordinate-frame hunyuan`: OpenCV camera XYZ
and normals are rotated to `(x, -y, -z)` (+Y up, -Z forward) before normalization
and VAE encoding. The saved `input_transform`, bounds, centers, and coordinate-frame
metadata describe this converted frame. Use `--vae-coordinate-frame opencv` for
the previous behavior. Regenerate existing posteriors with `--overwrite` or use a
new output directory; existing latent tensors cannot be converted by flipping
their channels. Geometry used for training/evaluation and visible-geometry
conditioning must use the same frame as these GT posteriors.

For `Front3DSDFDataset`, `SAGE10KDataset`, and `ScanNetPPSDFDataset`, select
the matching runtime geometry axes in the training/testing config:

```yaml
data:
  geometry_coordinate_frame: hunyuan
```

The default is `hunyuan`. Set `geometry_coordinate_frame: opencv` for legacy
datasets/checkpoints using OpenCV axes. A dataset's `params`
can override `geometry_coordinate_frame` individually. This rotates surface XYZ,
normals, and UDF query XYZ before normalization. Returned extrinsics map the
selected geometry frame into OpenCV camera coordinates, preserving image
projection. RGB, distance labels, images, and precomputed latents are unchanged;
use GT and visible latent files generated in the matching frame.

### ScanNet++ — cubemap face 4 camera

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/preprocessing/latents/surface_to_vae_latent.py \
  --dataset-layout scannetpp \
  --split-list data/scannetpp/full_list_remaining.json \
  --surface-root /tmp/datasets/scannetpp/udf_color \
  --cubemap-root /tmp/datasets/scannetpp/cubemap_data \
  --pose-root /mnt/projects/theses/vyhong/scannetpp/scan_poses \
  --face-id 4 \
  --vae-coordinate-frame hunyuan \
  --checkpoint exp_output/hunyuan_vae/checkpoints/last.ckpt \
  --config checkpoints/hunyuan/.hydra/config.yaml \
  --output-root datasets/scannetpp/hunyuan_gt_latent_posteriors \
  --devices cuda:0 \
  --precision bf16 \
  --batch-size 24 \
  --num-workers 16 \
  --normalization cube \
  --cube-center-camera \
  --only-save-latent-before-sampling \
  --seed 0
```

### ScanNet++ split rooms

Room surface/UDF samples and GT posteriors are already generated for the split rooms.

Add these parameters to a `ScanNetPPSDFDataset` test configuration:

```yaml
room_data_root: datasets/scannetpp/test_scenes
room_gt_latent_root: datasets/scannetpp/room_gt_latents
```

The dataset selects rooms using frame IDs in `test_camera_poses.json`, with
`test_images.txt` as a fallback for older exports. Matching frames use room
geometry and room GT latents; unmatched frames use the configured original
`sdf_data_root` and `gt_latent_root`. Missing room data raises an error.
Image features and visible-geometry conditioning continue to use the original
scene/frame keys and roots. Overlapping room assignments raise an error.
The Hunyuan DiT panorama config enables this for `final_test_10_images.json`.

### Sage-10K — cubemap face 4 camera

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/preprocessing/latents/surface_to_vae_latent.py \
  --dataset-layout sage10k \
  --split-list data/sage-10k/full_list_remaining.json \
  --surface-root /tmp/datasets/sage-10k/udf \
  --cubemap-root /tmp/datasets/sage-10k/cubemap_data/cubemaps \
  --pose-root /mnt/projects/theses/vyhong/sage-10k/poses \
  --face-id 4 \
  --vae-coordinate-frame hunyuan \
  --checkpoint exp_output/hunyuan_vae/checkpoints/last.ckpt \
  --config checkpoints/hunyuan/.hydra/config.yaml \
  --output-root datasets/sage-10k/hunyuan_gt_latent_posteriors \
  --devices cuda:0 \
  --precision bf16 \
  --batch-size 24 \
  --num-workers 16 \
  --normalization cube \
  --cube-center-camera \
  --only-save-latent-before-sampling \
  --seed 0
```

### 3D-FRONT — panorama camera

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/preprocessing/latents/surface_to_vae_latent.py \
  --dataset-layout 3dfront \
  --split-list data/3dfront/panorama_list_20000.json \
  --surface-root /tmp/datasets/3dfront/udf \
  --pose-root /tmp/datasets/3dfront/panoramas \
  --panorama-camera \
  --vae-coordinate-frame hunyuan \
  --checkpoint exp_output/hunyuan_vae/checkpoints/last.ckpt \
  --config checkpoints/hunyuan/.hydra/config.yaml \
  --output-root datasets/3dfront/hunyuan_gt_latent_posteriors \
  --devices cuda:0 \
  --precision bf16 \
  --batch-size 24 \
  --num-workers 16 \
  --normalization cube \
  --cube-center-camera \
  --only-save-latent-before-sampling \
  --seed 0
```

## PaGeR visible latents

PaGeR VAE encoding defaults to `--vae-coordinate-frame hunyuan`: rotate
OpenCV XYZ and normals to `(x, -y, -z)` before cube normalization, retaining
the camera at the origin. Saved bounds, centers, surface previews, and decoded
meshes use this frame. Use `--vae-coordinate-frame opencv` for legacy outputs.
Regenerate old latents using `--overwrite` or a new output directory.
Commands below use new Hunyuan output directories. Point dataset
`gt_latent_root` and `visible_geometry_latent_root` at the corresponding new
outputs (or their mounted archive paths) before training/testing.

### ScanNet++

```bash
CUDA_VISIBLE_DEVICES=0 python -u third_party/PaGeR/tools/infer_sage_vae_latents.py \
  --dataset-layout scannetpp \
  --data-path /tmp/datasets/scannetpp/scannetpp_panorama \
  --split-list data/scannetpp/full_list_remaining.json \
  --output-root datasets/scannetpp/hunyuan_pager_latents \
  --vae-coordinate-frame hunyuan \
  --checkpoint pager \
  --vae-checkpoint exp_output/hunyuan_vae/checkpoints/last.ckpt \
  --vae-config checkpoints/hunyuan/.hydra/config.yaml \
  --devices 0 \
  --batch-size 8 \
  --num-workers 16 \
  --scene-mode indoor \
  --seed 0
```

### Sage-10K

```bash
CUDA_VISIBLE_DEVICES=0 python -u third_party/PaGeR/tools/infer_sage_vae_latents.py \
  --dataset-layout sage10k \
  --data-path /tmp/datasets/sage-10k/panorama-scenes \
  --split-list data/sage-10k/full_list_remaining.json \
  --output-root datasets/sage-10k/hunyuan_pager_latents \
  --vae-coordinate-frame hunyuan \
  --checkpoint pager \
  --vae-checkpoint exp_output/hunyuan_vae/checkpoints/last.ckpt \
  --vae-config checkpoints/hunyuan/.hydra/config.yaml \
  --devices 0 \
  --batch-size 10 \
  --num-workers 16 \
  --scene-mode indoor \
  --seed 0
```

### 3D-FRONT

```bash
CUDA_VISIBLE_DEVICES=0 python -u third_party/PaGeR/tools/infer_sage_vae_latents.py \
  --dataset-layout 3dfront \
  --data-path /tmp/datasets/3dfront/panoramas \
  --output-root datasets/3dfront/hunyuan_pager_latents \
  --vae-coordinate-frame hunyuan \
  --checkpoint pager \
  --vae-checkpoint exp_output/hunyuan_vae/checkpoints/last.ckpt \
  --vae-config checkpoints/hunyuan/.hydra/config.yaml \
  --devices 0 \
  --batch-size 8 \
  --num-workers 16 \
  --scene-mode indoor \
  --seed 0
```

## Quick options

- Smoke test: `--limit 2`.
- Regenerate: `--overwrite`.
- DINO split filter: `--split-list path/to/list.json`.
- DINO two GPUs: run copies with `CUDA_VISIBLE_DEVICES=0` / `1` and `--num-shards 2 --shard-index 0` / `1`.
- PaGeR two GPUs: `CUDA_VISIBLE_DEVICES=0,1` with `--devices 0,1`.
- GT two GPUs: `CUDA_VISIBLE_DEVICES=0,1` with `--devices cuda:0 cuda:1`.
- 3D-FRONT DINO inputs require `.complete.json` in each room directory.

## DA3 missing packages

```bash
pip install moviepy==1.0.3 plyfile pycolmap
```

## PaGer missing packages

```bash
conda run -n nova3r python -m pip install -e .
python -m pip install --upgrade "numpy>=2,<3"
```
