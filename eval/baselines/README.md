# Baseline evaluation

This folder evaluates reconstructive baseline models through small adapters. The
runner owns dataset loading, deterministic point sampling, directional Chamfer
metrics, and result files; adapters only turn one sample into world-space XYZ.

DA3 uses all six SAGE-10K cubemap faces with their known cameras. It follows the
raw preprocessing and model-forward path used by `tools/cubemap_to_vae_latent.py`,
then unprojects predicted z-depth through the processed input intrinsics and
input world-to-camera extrinsics. This intentionally bypasses DA3's high-level
postprocessing, which applies Umeyama pose/scale alignment. Ground truth defaults
to `scene_surface.npz`. The accuracy and completion implementations are reused
from `eval/mv_recon/metric.py`. Before measuring them, the prediction is aligned
to the scene with the Chamfer-optimized isotropic scale and XYZ translation also
used by `eval/evaluate_pcd.py`. The baseline defaults to `300` iterations and a
learning rate of `0.1`. Distances are Euclidean, unsquared, and reported in scene
units:

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

`--genrecon-data-root` is optional. If it contains
`rooms_raw/<room-id>/layout_<room-id>.json`, that metadata is preferred;
otherwise, the chunk footprint is inferred from `scene_surface.npz`. The source
used for each sample is written to `genrecon_work/<sample>/chunk_source.json`.
`--genrecon-surface-points` defaults to `--eval-points`. The adapter retains the
loaded GenRecon pipeline across samples. Per-sample camera/chunk diagnostics are
written below `<output-dir>/genrecon_work`; reduce projection memory with
`--genrecon-proj-batch-voxels` if necessary.

Use `--save-predictions` to retain an aligned prediction and its sampled GT
reference as `<sample>_pred_aligned.ply` and `<sample>_gt.ply`. `results.json` is
checkpointed after every sample; `results.csv` and `summary.json` are written at
the end. A failed sample is recorded and evaluation continues unless
`--fail-fast` is supplied. Each result records the fitted `alignment_scale` and
`alignment_shift`. The optimizer settings can be changed with
`--alignment-iterations` and `--alignment-lr`.

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
