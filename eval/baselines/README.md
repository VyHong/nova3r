# Baseline evaluation

This folder evaluates reconstructive baseline models through small adapters. The
runner owns dataset loading, deterministic point sampling, directional Chamfer
metrics, and result files; adapters only turn one sample into world-space XYZ.

The active baselines are NOVA3R (the default), GenRecon, PaGeR, LaRI,
and VolFill. DA3 is excluded from baseline evaluation.
Predictions are aligned with an isotropic scale and XYZ translation before
measuring Euclidean, unsquared distances in scene units.

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

Run a one-sample smoke test first:

```bash
python -m eval.baselines.run --model nova3r --limit 1 --fail-fast
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

To produce a Visible / Occluded / Complete comparison row for PaGeR, add
`--visibility-metrics` to either PaGeR command above. The runner writes
`table.md` and `table.csv` alongside the usual per-sample results and summary.
This option also works with the other baseline adapters. When resuming with
this option, older records without regional metrics are recomputed.

Visibility uses first-hit raycasting against the original GT triangle mesh.
GT evaluation points are sampled deterministically by triangle area. A point is
visible when its camera distance agrees with the first ray intersection within
`--visibility-tolerance` (default 0.01 scene units). Other points are occluded.
Use `--visibility-mesh-root ROOT` if meshes are separate from the surface NPZs.
The root contains `<scene>/scene.glb` for SAGE, `<scene>/<room>/room.glb`
(or `<scene>/<room>_full.glb`) for 3D-FRONT, and
`<scene>/scans/mesh_aligned_0.05.ply` for ScanNet++.
ScanNet++ requires the original mesh and its `mesh_aligned_0.05_mask.txt`
anonymized vertex list. Set `--anonymization-root ROOT` if the list is stored
separately under `<scene>/scans/`. Triangles touching listed vertices are
excluded from GT evaluation sampling but retained as raycasting occluders.
Frame-matched room meshes under `<scene-root>/rooms/<scene>/<room>/mesh.ply`
take precedence over full scans, including when `--visibility-mesh-root` is set.
Room anonymization matches original masked vertex coordinates within 1e-5 scene
units, rather than applying scan indices to reindexed room vertices. The original
scan and mask must remain available under the scene root or mesh/mask root.
Both evaluation sampling and visibility raycasting use the selected room mesh;
frames without a room use the full scan. Image validity masks still
filter predictions through the adapters. Missing meshes or lists raise errors.
Old hidden-point-removal records are recomputed with `--resume`.
All groups reuse the existing complete-scene scale/translation alignment.
Visible and occluded CD are **one-way GT-to-prediction mean distances**, following
the evaluator's single-direction visible scoring convention. APD at threshold t
is the fraction of that region's GT points whose nearest prediction is less
than t away. Complete CD is the symmetric mean distance, and complete FS retains
the existing SSI3D F-score. These definitions are explicit implementation
choices; equivalence to a paper's protocol requires checking its definitions.

JSON/CSV result metrics retain scene units and fractional scores. The comparison
table multiplies distances and rates by 100, giving centimetres when scene units
are metres and percentages for APD/FS. Each metric is averaged equally across
successful samples with a nonempty corresponding GT region. Empty regions are
null and render as a dash; summary `metric_samples` reports contributing counts once for visible,
occluded, and complete regions. The standard table contains thresholds 0.1, 0.05, and 0.02 (the defaults); retain all three in
`--fscore-thresholds` to populate all columns.

Regional directional distances are also saved as
`visible_chamfer_forward`, `visible_chamfer_backward`, `visible_chamfer_mean`
and the corresponding `occluded_*` keys. Forward compares the entire aligned
prediction to the selected GT region; backward compares that GT region to the
entire prediction. Mean averages those two directions. Consequently, forward
can penalize predictions belonging to the other region. The table's regional
`*_cd` retains its GT-to-prediction definition and equals regional backward.
These extra directional metrics are included in per-sample results and summary.

Mesh2SDF Lightning testing enables these metrics by default for SAGE-10K,
ScanNet++, and 3D-FRONT test datasets. They supply the world-to-geometry transform
so GT meshes are scored in the decoder's first-camera frame, after undoing the
normalization scale. The panorama centre is the shared cubemap optical centre.
`test_visibility_metrics: false` disables the extra mesh metrics;
`test_visibility_tolerance` defaults to 0.01. Override original mesh roots with
`test_visibility_mesh_roots` keyed by `sage10k`, `scannetpp`, and `3dfront`.
The same room selection and anonymization rules apply. Per-mesh `metrics.json`,
test results, summary, and comparison tables include the new scores. Other
training datasets retain their complete-scene evaluation.

ERP evaluation now selects a dataset-specific panorama root when `--erp-root`
is omitted. ScanNet++ defaults to
`datasets/scannetpp/panos/updated_final_test_100_prepared_erp`; SAGE defaults to
`/tmp/datasets/sage-10k/panorama-scenes`, and 3D-FRONT to
`/tmp/datasets/3dfront/panorama_scenes`. A missing root fails explicitly.
Every resolved panorama is checked before inference: images whose width is not
exactly twice their height emit a warning with their path and dimensions.
Explicit `--erp-root` overrides are respected, with the same aspect-ratio check.

Nova3R independent cubemap benchmarking: add `--nova3r-independent-faces`
to a `--model nova3r --input-projection cubemap` run. All six faces in
`--face-order` are inferred separately, each using official single-image
 duplication, regardless of `--nova3r-num-views`. Each cloud is lifted using
its own known cubemap pose and fitted independently to all GT inside that
face's calibrated image, using scale and translation. Occluded surfaces are included in the alignment target; ScanNet++ image masks
exclude invalid face pixels. Mesh visibility is used only for regional scoring. Original
GT meshes are required even without `--visibility-metrics`. Faces with no valid
GT raise an explicit error. Alignment targets are used internally. The six fitted clouds are merged and sampled
for scoring; the merged cloud is not fitted again. This is GT-assisted
per-face alignment, distinct from joint multiview inference. Records include
all six `face_alignments`, and `--save-predictions` exports only the six aligned faces merged into one
`predictions/<sample>_pred_aligned.ply`, before evaluation subsampling. Use a
separate output directory for this protocol. No changes are needed to a
running PaGeR job.

### LaRI and VolFill: independent 90-degree faces

Official sources are cloned into `third_party/LaRI` (`ruili3/lari`, initial
revision `cd98ec85058e2d563bf67b2502dae6a7439a17ae`) and
`third_party/VolFill` (`ngoductuanlhp/VolFill`, initial revision
`a0c8e7779ccee318e7a75bb2d273290d4ffb76ed`). Their inference modules run in
persistent isolated subprocesses, avoiding collisions between their bundled
packages and Nova3R. The model stays loaded across all faces and samples.

LaRI uses the **scene-level** `lari_scene_pointmap.pth` checkpoint from
`ruili3/LaRI`, five point-map layers, and the upstream 512-pixel ImageNet image
normalization. VolFill loads `TuanNgo/VolFill`, including MoGe conditioning,
and converts the predicted TUDF from normalized values to voxel distances.
The default extraction threshold is 0.8 voxel units (upstream visualization
example), with 50 flow steps and CFG 3.0. Use `--volfill-threshold`,
`--volfill-steps`, and `--volfill-cfg-strength` to change these choices.
Checkpoints download on first use. `--lari-checkpoint PATH` accepts local scene
weights; `--volfill-checkpoint` selects a Hugging Face model repository.

Both modes infer all six cubemap faces independently, lift points using the
known face poses, and fit each cloud to all valid GT inside that face frustum,
including occluded geometry. They then merge predictions and evaluate the same
visible, occluded, and complete metrics as Nova3R. `--save-predictions` saves one
merged aligned PLY per sample, without separate face clouds. An empty GT frustum
is recorded as a sample error, as in the Nova3R benchmark; there is no silent
fallback to unrelated scene geometry.

Run from the project's `nova3r` environment:

```bash
bash scripts/eval/benchmark_single_image.sh lari all
bash scripts/eval/benchmark_single_image.sh volfill all
```

Replace `all` with `scannetpp`, `sage10k`, or `3dfront` to select one split.
Each command uses the prepared 90-degree cubemaps, full final-test manifests,
all three thresholds, room-aware GT masks, and resume. Results go to
`eval_results/baselines/<model>/<dataset>_independent_faces_cube90`.
For a direct CLI smoke run, use `python -m eval.baselines.run --model lari`
(or `--model volfill`) with the same dataset flags as Nova3R and add `--limit 1`.
No `--nova3r-independent-faces` flag is needed for these models.

If using the upstream dedicated environments, keep the benchmark process in
`nova3r` and pass `--lari-python /path/to/lari/bin/python` or
`--volfill-python /path/to/volfill/bin/python` to choose the worker interpreter.
Upstream requirements are in each cloned repository; avoid reinstalling their
pinned Torch stacks into an environment running other benchmarks.

On this workspace's PyTorch 2.11/CUDA 13 environment, supplemental VolFill
packages are installed offline from `~/deps.zip` using
`scripts/setup/install_baseline_dependencies.sh`. VolFill-specific packages go
under `$CONDA_PREFIX/share/nova3r-baselines/volfill` and are loaded by its worker.
For reproducibility, the supplemental package list is
`configs/volfill_runtime_requirements.txt`; it intentionally does not install or
replace PyTorch. The worker selects the xFormers sparse-attention backend by
default. Full checkpoint caches are under `checkpoints/hf_baselines`, which the
launcher selects through `HF_HOME` unless already set.

VolFill now uses the upstream export path directly: `_save_result` from
`volfill.amodal.inference_latent_visible`, then `load_sample` and
`field_to_pointcloud` from `volfill.visualize`. There is no separate adapter
implementation of field denormalization or voxel coordinate conversion.
`--volfill-raw-dir DIR` preserves each face's upstream `pred_tudf_256.npz`,
`metadata.json`, and `image.jpg` for inspection. `face_00` through `face_05`
follow `--face-order` (default actual face IDs 4, 0, 5, 1, 3, 2).
Results record `pipeline: upstream_export_v2`; resume does not reuse records
from the previous adapter path. Raw fields are optional to conserve storage.
The old `smoke_scannetpp` output used two flow steps, ten alignment iterations,
and 1,000 evaluation points; it is not a quality benchmark. Use 50 flow steps
and the normal alignment/evaluation settings for comparisons.

LaRI alignment correction: LaRI now retains its H×W×L×3 point maps and fits a
single scale plus camera-Z shift from corresponding first-layer pixels and GT
mesh first-hit raycasts. The least-squares transform family follows the native
LaRI evaluator; this implementation uses only the first layer rather than all
common GT/predicted layers because the datasets provide meshes, not GT LDIs.
All predicted layers share that fit; X/Y translation and rotation are fixed in
camera coordinates. The world-space transform is derived only afterward.
Invalid image-mask pixels and missing mesh hits are excluded from the fit.
This is distinct from Nova3R/VolFill's frustum Chamfer fit. LaRI records identify
`lari_first_layer_scale_camera_z_shift`, the depth shift, correspondence count,
and fitted scale; old LaRI records are invalidated on resume.

LaRI alignment experiments are selectable with `--lari-alignment`:
`native` (default) fits first-layer pixel correspondences, `all-chamfer` fits
flattened clouds to all GT inside each face frustum, and `visible-chamfer`
uses the same flattened-cloud Chamfer optimizer but restricts GT to mesh-visible
points. The latter two share the CLI alignment LR and iteration settings.
The launcher forwards additional options after the dataset argument. Set
`BENCHMARK_SUFFIX` to keep experimental runs in separate output directories.
For example:

```bash
BENCHMARK_SUFFIX=visible_chamfer_lr05 bash scripts/eval/benchmark_single_image.sh \
  lari scannetpp --lari-alignment visible-chamfer --alignment-lr 0.5
```
