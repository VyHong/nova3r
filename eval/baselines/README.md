# Baseline evaluation

This folder evaluates reconstructive baseline models through small adapters. The
runner owns dataset loading, deterministic point sampling, directional Chamfer
metrics, and result files; adapters only turn one sample into world-space XYZ.

DA3 uses all six SAGE-10K cubemap faces with their known cameras. It follows the
raw preprocessing and model-forward path used by
`scripts/preprocessing/latents/cubemap_to_vae_latent.py`,
then unprojects predicted z-depth through the processed input intrinsics and
input world-to-camera extrinsics. This intentionally bypasses DA3's high-level
postprocessing, which applies Umeyama pose/scale alignment. Ground truth defaults
to `scene_surface.npz`. The accuracy and completion implementations are reused
from `eval/mv_recon/metric.py`. Before measuring them, the prediction is aligned
to the scene with the Chamfer-optimized isotropic scale and XYZ translation also
used by `eval/evaluate_pcd.py`. The baseline defaults to `300` iterations and a
learning rate of `0.1`. Distances are Euclidean, unsquared, and reported in scene
units:

The default checkpoint is the refreshed metric-scale nested model
`depth-anything/DA3NESTED-GIANT-LARGE-1.1`. Override it with `--da3-model` when
comparing another DA3 checkpoint.
With `--save-predictions`, DA3 additionally writes exact 2D arrays to
`predictions_2d/*_da3_2d.npz` and percentile-normalized PNG previews of its raw
depth, spherical range, confidence, and sky maps. By default, ERP depth is
converted from camera-plane Z-depth to range using DA3's predicted pinhole
intrinsics. This deliberately extends one pinhole image plane across the full
ERP and is therefore an approximation outside the forward field of view.
For ERP inputs it writes three unaligned world-space diagnostic clouds under
`predictions_diagnostic/`: `*_erp_pinhole_adjusted.ply` is the evaluated result,
`*_erp_direct_depth_as_range.ply` skips adjustment, and
`*_erp_cubemap_adjusted.ply` uses the bounded nearest-face cubemap correction.
For ScanNet++, zero-valued anonymization-mask pixels are removed from every DA3
ERP and cubemap point cloud after nearest-neighbor resizing to the prediction
resolution. The raw depth remains in the diagnostic NPZ together with
`valid_mask`, so masking can be audited independently.
PaGeR applies the same ScanNet++ ERP validity mask in addition to its native sky
and depth-discontinuity filtering, including removal of the padded bottom band.

- `chamfer_forward`: prediction to nearest scene point (accuracy)
- `chamfer_backward`: scene to nearest prediction (completeness)
- `chamfer_mean`: arithmetic mean of the two directions
- `f_score_0.1_full`, `f_score_0.05_full`, `f_score_0.02_full`: SSI3D F-scores

GenRecon also consumes the same six perspective cubemap faces and their known
cameras; it does not use GenRecon's separately rendered full-room training views.
Its normal SAGE room-layout metadata defines the 2.85 m overlapping chunk grid.
When those layout JSONs are unavailable, the adapter derives the equivalent
axis-aligned room footprint from the evaluation surface under `--scene-root`.
The generated mesh is lifted from chunk-0 coordinates back to the original room
frame, including the Blender Z-up to evaluator `(x, z, -y)` axis conversion, and
sampled uniformly by triangle area before evaluation. A generated scene name
such as `..._layout_deadbeef` is matched to
`<genrecon-data-root>/rooms_raw/deadbeef/layout_deadbeef.json`.

The F-score thresholds default to those in `evaluate_pcd.py` and can be changed
with `--fscore-thresholds`.

DA3, GenRecon, and Nova3R can instead consume the original ERP as one image. Pass
`--input-projection erp --erp-root ROOT`. Nova3R receives no intrinsics. DA3 also
receives no camera parameters: it predicts a 2D depth map, which is lifted through
the ERP's spherical rays and then placed in the world using the panorama pose.
The ERP path applies the predicted-intrinsics pinhole correction described
above. The direct and nearest-face cubemap interpretations remain available as
saved diagnostics.
For GenRecon ERP input, scene-wide `cond_3D` features use a true spherical
longitude/latitude lookup, including the rear hemisphere and horizontal seam.
The direct per-chunk `cond_2D` tokens remain the unchanged square-resized ERP
tokens. Cubemap mode remains the default and continues to use pinhole projection.

Nova3R follows the official demo contract: inputs are resized to 518×392 and
normalized to `[-1, 1]`; a one-view input is duplicated before complete DUSt3R
pairs are built; flow matching uses the checkpoint Hydra configuration; and the
predicted first-camera-frame points are lifted into evaluator world coordinates.
Cubemap inference defaults to all six faces in `--face-order` and presents them
jointly to the model in one call. ERP inference requires
`--nova3r-num-views 1`; that single image is duplicated to preserve the official
single-view demo behavior.

PaGeR is the exception to the pinhole ERP ablation. Its official model is
panoramic, so `--model pager` requires `--input-projection erp` and runs the
official spherical ERP→cubemap→ERP depth path. The resulting spherical cloud is
converted from PaGeR axes into the SAGE/OpenCV world frame before evaluation.

The same PaGeR evaluator supports 3D-FRONT with `--dataset-layout 3dfront`.
Its path-based manifests are parsed as `scene/room` plus panorama index. Saved
Blender-axis panorama poses are composed with the glTF-Y-up world conversion,
so predictions are evaluated directly against `<scene>/<room>_full_surface.npz`.

Use `--dataset-layout scannetpp` for ScanNet++. This selects
`data/scannetpp/test_list.json` and the repository's standard ScanNet++ cubemap,
scanner-pose, and aligned-surface paths. It reads scanner camera-to-world poses
from `<pose-root>/<scene>/scans/scanner_poses.json`, converts camera axes to
OpenCV, and leaves reconstructed points in the raw aligned-scene coordinate
frame used by `<scene-root>/<scene>/scans/mesh_aligned_0.05_surface.npz`.

For example, run DA3 on ScanNet++ ERPs with:

```bash
python -m eval.baselines.run \
  --model da3 \
  --dataset-layout scannetpp \
  --input-projection erp \
  --erp-root /tmp/datasets/scannetpp/scannetpp_panorama \
  --split-list data/scannetpp/test_25_scenes_list.json
```

Run a one-sample smoke test first:

```bash
python -m eval.baselines.run --model da3 --limit 1 --fail-fast
```

Then run the complete test split:

```bash
python -m eval.baselines.run \
  --model da3 \
  --split-list data/sage-10k/test_list.json \
  --cubemap-root /tmp/datasets/sage-10k/cubemap_data/cubemaps \
  --scene-root /tmp/datasets/sage-10k/udf \
  --pose-root /mnt/projects/theses/vyhong/sage-10k/poses \
  --output-dir eval_results/baselines/da3/sage10k_test
```

Run DA3 without camera inputs and spherically lift its predicted ERP depth:

```bash
python -m eval.baselines.run \
  --model da3 \
  --input-projection erp \
  --erp-root /tmp/datasets/sage-10k/rendered \
  --split-list data/sage-10k/test_list.json \
  --scene-root /tmp/datasets/sage-10k/udf \
  --pose-root /mnt/projects/theses/vyhong/sage-10k/poses
```

Run GenRecon from an environment containing its compiled CUDA dependencies:

```bash
python -m eval.baselines.run \
  --model genrecon \
  --split-list data/sage-10k/test_100_images_seed0.json \
  --genrecon-ss-ckpt third_party/GenRecon/sparse_structure.pt \
  --genrecon-shape-ckpt third_party/GenRecon/shape_slat.pt \
  --genrecon-tex-ckpt third_party/GenRecon/texture_slat.pt \
  --output-dir eval_results/baselines/genrecon/sage10k_test \
  --fail-fast
```

Run the ScanNet++ final 100 with prepared 90-degree cubemaps, GT-layout
chunking, two workers on one GPU, and xFormers for dense and sparse attention:

```bash
ATTN_BACKEND=xformers \
SPARSE_ATTN_BACKEND=xformers \
python -m eval.baselines.run \
  --model genrecon \
  --dataset-layout scannetpp \
  --input-projection cubemap \
  --cubemap-root datasets/scannetpp/cubemaps/final_test_100_prepared_cube_90 \
  --cubemap-fov 90 \
  --split-list data/scannetpp/final_test_list.json \
  --genrecon-scannet-chunk-source evaluation_surface \
  --genrecon-ss-ckpt third_party/GenRecon/sparse_structure.pt \
  --genrecon-shape-ckpt third_party/GenRecon/shape_slat.pt \
  --genrecon-tex-ckpt third_party/GenRecon/texture_slat.pt \
  --workers 2 \
  --device cuda:0 \
  --output-dir eval_results/baselines/genrecon/scannetpp_final_test_100_cube90_gt_layout_xformers \
  --save-predictions \
  --fail-fast
```

Parallelize GenRecon across two GPUs by adding `--devices cuda:0 cuda:1`
to the command. This defaults to one persistent model process per listed device.
For multiple processes on one GPU, add `--workers 2 --device cuda:0`; each process
loads its own model and requires its own GPU memory. Use `--workers 4 --devices
cuda:0 cuda:1` for two processes per GPU. Device indices follow
`CUDA_VISIBLE_DEVICES` when set. With neither option, execution stays sequential.

Workers receive new samples as they finish, using the original manifest index
for evaluation seeds. Only the parent writes `results.json` and the summary;
records remain in manifest order even when samples finish out of order. Worker
intermediates live under `genrecon_work/worker_<index>` (or the supplied
`--genrecon-work-dir`). Prediction exports retain their usual paths. Parallel
runs require unique scene/frame entries to avoid colliding prediction filenames.
Sample errors retain the normal error records; `--fail-fast` stops scheduling
new work after an error is observed and waits for already running samples to exit.

The same `--input-projection erp --erp-root ROOT` options apply to GenRecon. In
ERP mode the single image supplies spherical scene-wide `cond_3D` features and
is also the unchanged direct `cond_2D` input for every chunk.

Run Nova3R with the released two-view checkpoint and two cubemap faces:

```bash
python -m eval.baselines.run \
  --model nova3r \
  --nova3r-checkpoint checkpoints/scene_n2/checkpoint-last.pth \
  --nova3r-config eval/baselines/configs/nova3r_scene_n2_official.yaml \
  --nova3r-num-views 2 \
  --split-list data/sage-10k/test_25_images_seed0.json \
  --cubemap-root /tmp/datasets/sage-10k/cubemap_data/cubemaps \
  --scene-root /tmp/datasets/sage-10k/udf \
  --pose-root /mnt/projects/theses/vyhong/sage-10k/poses
```

Run PaGeR on the original panoramas with its indoor scale head:

```bash
python -m eval.baselines.run \
  --model pager \
  --input-projection erp \
  --erp-root /tmp/datasets/sage-10k/rendered \
  --pager-checkpoint pager \
  --pager-scene-mode indoor \
  --split-list data/sage-10k/test_25_images_seed0.json \
  --scene-root /tmp/datasets/sage-10k/udf \
  --pose-root /mnt/projects/theses/vyhong/sage-10k/poses
```

Run PaGeR on the 3D-FRONT final test:

```bash
python -m eval.baselines.run \
  --model pager \
  --dataset-layout 3dfront \
  --input-projection erp \
  --erp-root /tmp/datasets/3dfront/panorama_scenes \
  --split-list data/3dfront/final_test_list.json \
  --pager-checkpoint pager \
  --pager-scene-mode indoor \
  --output-dir eval_results/baselines/pager/3dfront_final_test_100_erp
```

`--genrecon-data-root` is optional. If it contains
`rooms_raw/<room-id>/layout_<room-id>.json`, that metadata is preferred;
otherwise, the chunk footprint is inferred from `scene_surface.npz`. The source
used for each sample is written to `genrecon_work/<sample>/chunk_source.json`.
SAGE surfaces are converted from their evaluator Y-up frame back to GenRecon's
Z-up room frame. ScanNet++ surfaces instead use their native aligned Z-up frame:
X/Y define the floor plan, Z defines floor and ceiling, and no SAGE axis
conversion is applied.
For the single-panorama ScanNet++ evaluation, GenRecon defaults to ground-truth
scene bounds from the evaluation surface because one camera centre cannot
produce a COLMAP sparse reconstruction. This is oracle layout information and
is recorded in each sample's `chunk_source.json`. For multi-view experiments,
`--genrecon-scannet-chunk-source colmap` enables GenRecon's `ScannetChunker` and
reads `<genrecon-scannet-root>/<scene>/dslr/colmap/points3D.txt`.
`--genrecon-surface-points` defaults to `--eval-points`. The adapter retains the
loaded GenRecon pipeline across samples. Per-sample camera/chunk diagnostics are
written below `<output-dir>/genrecon_work`; reduce projection memory with
`--genrecon-proj-batch-voxels` if necessary.

Use `--save-predictions` to retain an aligned prediction and its sampled GT
reference. Point-cloud baselines write `<sample>_pred_aligned.ply`; mesh-capable
baselines write the native `<sample>_pred_aligned_mesh.ply` instead. Both write
`<sample>_gt.ply`. `results.json` is checkpointed after every sample;
`results.csv` and `summary.json` are written at the end. A failed sample is
recorded and evaluation continues unless `--fail-fast` is supplied. Each result
records the fitted `alignment_scale` and `alignment_shift`. The optimizer
settings can be changed with `--alignment-iterations` and `--alignment-lr`.

For GenRecon, the saved prediction retains the model's native triangle
topology. Its vertices receive the same fitted scale and translation as the
sampled evaluation points. Those points are still used internally for metrics
but are not saved as a redundant prediction PLY. The baseline uses GenRecon's
geometry-only path, so the mesh contains geometry but no generated PBR textures.

To add another baseline, create `adapters/<name>.py` with a class exposing
`predict(sample) -> np.ndarray[N, 3]`, then register its name and constructor in
`run.py`. The dataset and metrics remain unchanged.



conda activate nova3r
cd /mnt/home/vyhong/projects/nova3r/third_party/GenRecon

git submodule update --init --recursive

python -m pip install \
  easydict \
  ninja \
  transformers==4.57.3 \
  lpips \
  kornia \
  timm

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')"

. ./setup.sh \
  --flash-attn \
  --nvdiffrast \
  --nvdiffrec \
  --cumesh \
  --o-voxel \
  --flexgemm
