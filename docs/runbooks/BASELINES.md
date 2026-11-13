# Baseline Evaluation Runbook

## Install extra baseline dependencies from the ZIP

Start with the original, working `nova3r` environment and cloned baseline
repositories under `third_party/`. The bundle contains extra baseline packages,
their supporting Python dependencies, and compiled GenRecon extensions. It does
**not** contain the `nova3r` environment, PyTorch, torchvision, Triton, CUDA
runtime packages, datasets, or model checkpoints.

```bash

apt update
apt install --reinstall gcc-13 g++-13 libgcc-13-dev
apt install --reinstall libc6-dev -y
conda activate nova3r
bash scripts/setup/install_baseline_dependencies.sh /mnt/home/vyhong/deps.zip
```

The installer validates the bundle's Python, architecture, PyTorch and
torchvision versions, extracts to a temporary directory, and installs the wheels
without accessing the network. PyTorch and its CUDA stack remain in the original
environment. The compiled wheels require a compatible Linux/CUDA/GPU setup;
this is not a universal installer for other PyTorch versions. Some native
extensions may still compile kernels on first use and require a CUDA toolkit.

Common packages install into the activated environment. VolFill's separate
package versions install under `$CONDA_PREFIX/share/nova3r-baselines/volfill`,
which its worker reads automatically. Temporary extraction files are removed
when installation ends. No complete environment copy is created.

```bash
# Validate compatibility and preview package installation without installing:
bash scripts/setup/install_baseline_dependencies.sh /mnt/home/vyhong/deps.zip --dry-run
```

Before installing wheels, the installer compiles and links a small shared library
using the active Python headers and `CC` (default `gcc`). This catches missing
`libgcc` and `libc_nonshared.a` files that otherwise fail only when GenReCon
builds Triton's CUDA helpers at inference time. The ZIP does not contain these
system development libraries. On Debian/Ubuntu, install or repair them with:

```bash
bash scripts/setup/install_baseline_dependencies.sh /mnt/home/vyhong/deps.zip --install-system-deps
```

This explicit option uses apt and requires network access and root or sudo. It
reinstalls `build-essential`, `libc6-dev`, `python3-dev`, and the `libgcc` development
package matching the active GCC major version. It cannot be combined with
`--dry-run`. On other distributions, install equivalent development packages
manually. The default path remains offline and reports actionable errors if
the compiler check fails. A passing compiler check does not validate GPU drivers
or replace an inference smoke test.

Installation finishes with `pip check`. Packages already installed at the same
version are reused. Errors do not roll back packages installed in earlier steps.
Baseline source repositories, their initialized submodules, model weights, and
system GPU drivers remain separate prerequisites.

To rebuild a bundle from a working installation (including the VolFill
supplements in `deps/volfill-runtime`), use:

```bash
python scripts/setup/bundle_baseline_dependencies.py --output /tmp/baseline-extras/deps
```

The resulting `deps/manifest.json` records all packaged distributions and the
required base stack versions. Zip the resulting `deps/` directory to distribute
it. The installer also accepts any custom archive path as its first argument.

## Manual GenRecon installation reference


```bash
apt update
apt install --reinstall gcc-13 g++-13 libgcc-13-dev
apt install --reinstall libc6-dev -y

cd /mnt/home/vyhong/projects/nova3r/third_party/GenRecon

git submodule update --init --recursive

python -m pip install \
  easydict \
  ninja \
  transformers==4.57.3 \
  lpips \
  kornia \
  timm \
  psutil \
  xformers

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')"

. ./setup.sh \
  --nvdiffrast \
  --nvdiffrec \
  --cumesh \
  --o-voxel \
  --flexgemm

```

## Default meshing for our model

Dense `mesh_udf` now defaults to the tuned 768 preset used by the ongoing
ScanNet++ and 3D-FRONT evaluations:

```yaml
mesh_mc: 768
mesh_udf_reference_resolution: 512
mesh_bounds_padding: 0.05
mesh_udf_surface_threshold: 0.01
mesh_udf_normal_band_voxels: 2.0
mesh_udf_query_batch_size: 262144
mesh_udf_vertex_batch_size: 131072
mesh_udf_streamed_grid: true
mesh_udf_fast_cleanup: true
mesh_udf_mc_tile_cells: 0
smooth_borders: false
```

Explicit config and CLI values still override defaults. Set
`mesh_udf_reference_resolution: null` to use native-resolution distance cutoffs.
Sparse-brick extraction defaults to no reference resolution because it does not
support that option. Other mesh methods retain their defaults. Existing saved
configs with explicit older settings retain those settings.
The evaluation launcher uses 16 OMP/MKL threads, 8 OpenBLAS threads and 100 flow
steps; these runtime/model settings are separate from the mesher defaults.


Mesh extraction defaults to `mesh_bounds_padding: 0.05`: each side receives
5% of that axis's full box dimension. This margin applies to both sampling and
snug clipping, preserving boundary surfaces such as floors. Explicit config or
`--mesh-bounds-paddings` values override this default.

Mesh extraction samples on a decoder-coordinate-axis-aligned box enclosing
its transformed snug oriented bounds, then removes every triangle with any
vertex outside those snug bounds. This applies when bounds are inferred from
the surface; explicit `mesh_bounds` remain axis-aligned without an inferred snug
box. Boundary caps are not added. Use `--no-mesh-axis-aligned-snug-clip` with
`training/test.py` to reproduce the previous oriented-grid extraction, or set
`mesh_axis_aligned_snug_clip: false` in the config.

The six-scene qualitative rerun is under `artifacts/qualitative_scenes_1024/`.
It uses MeshUDF resolution 1024, CFG 5, zero padding, surface threshold 0.01,
and `exp_output/hunyuandit_panorama/checkpoints/last-v1.ckpt`.

At high resolution, MeshUDF automatically tiles marching cubes when the native
32-bit face-buffer indexing limit would be exceeded. Tiled fields are rescaled
to preserve the full grid's distance cutoff. The six-scene rerun also uses
streamed-grid sampling and fast cleanup to reduce memory use.

Future qualitative runs prepared by `artifacts/qualitative_scenes_1024/prepare.py`
and its run script use 1% padding. Existing zero-padding result files retain
their original metadata. The two padded 3DFront candidate outputs are the
preferred versions under `3dfront_candidates/ours_padded/`.

## Direct single-ERP baseline ablation

Run LaRI, released NOVA3R, and VolFill across the final ScanNet++, SAGE-10K,
and 3D-FRONT test manifests:

```bash
conda activate nova3r
bash scripts/eval/benchmark_erp.sh all all
```

Arguments can select one model and dataset, e.g. `lari scannetpp`. Additional
runner flags follow those arguments. The launcher requires CUDA and uses
`datasets/scannetpp/panos/updated_final_test_100_prepared_erp` for ScanNet++.
It passes the panorama directly through each model's existing image preprocessing;
no cubemap faces or intrinsics are supplied. NOVA3R uses its official single-view
image duplication. LaRI and VolFill treat ERP as an ordinary image, so this is an
input ablation, with their native predicted XYZ retained rather than reinterpreted
as spherical depth. All models receive one global GT scale/translation fit for
scoring. LaRI's native per-face first-layer alignment is not used in ERP mode.
Camera poses place predictions for evaluation; GT geometry is used for alignment
and metrics, not as model input.

Results, visible/occluded/complete tables, and aligned predictions are saved in
`eval_results/baselines/<model>/<dataset>_single_erp`. Resume is enabled.

ERP preprocessing preserves the panorama aspect ratio: NOVA3R fits the full
panorama into 518×392 with centered black padding (2:1 content is 518×259),
and LaRI fits into 512×512 (2:1 content is 512×256). VolFill uses `max_size=504`, giving exactly 504×252 for 2:1 ERP input,
with both dimensions divisible by 14. Non-2:1 ERP inputs fail explicitly.
Its MoGe branch receives the original image through upstream preprocessing. ERP resume requires the
`aspect_preserving_v1` record marker, recomputing older preprocessing results.

`benchmark_erp.sh all all` runs all nine model/dataset jobs in parallel.
A single-model `all` invocation also parallelizes its three datasets. Logs are
written under `eval_results/baselines/erp_suite/` as `<model>_<dataset>.log`.
Each job holds a model/dataset-specific lock to prevent duplicate launches.
Ensure the GPU has memory for the concurrent models. Resume remains enabled.

## Panorama colors on qualitative meshes

The accepted projection and cleanup recipe for the two padded 3D-FRONT meshes
is saved in [the colorization README](../../artifacts/panorama_mesh_colorization/README.md).
Run `OMP_NUM_THREADS=4 python artifacts/panorama_mesh_colorization/colorize.py`
from the repository root in the `nova3r` environment. The script projects ERP RGB,
colors nearby small hidden islands to remove blue speckles, and leaves other
hidden surfaces muted slate blue. The README records all thresholds and the
coordinate convention. This visualization cleanup does not alter mesh geometry
or evaluation visibility metrics.

## VolFill empty cubemap face recovery

The single-image adapter omits empty/nonfinite face clouds from alignment and
merging. Nonempty faces are fitted normally and the merged cloud is scored
against the unchanged full GT target. Results record omitted face IDs in
`empty_faces`; the total sample fails only if all input faces are empty (or
another inference/alignment error occurs). The original panorama-only empty
prediction cannot be recovered by this policy because it has no other views.

Resume accepts legacy VolFill cubemap settings without an explicit
`input_projection` key, treating them as cubemap only. This preserves successful
old records while retrying failed ones. Before the first recovery, old result
files were backed up under `artifacts/volfill_empty_face_recovery/<dataset>/`.
Recovery logs are in that directory as `<dataset>.log`.

```bash
BENCHMARK_PYTHON=/opt/conda/envs/nova3r/bin/python \
  bash scripts/eval/benchmark_single_image.sh volfill all
```

## LaRI cubemap: merge before alignment

```bash
BENCHMARK_PYTHON=/opt/conda/envs/nova3r/bin/python \
BENCHMARK_SUFFIX=merged_layers_global \
  bash scripts/eval/benchmark_single_image.sh lari all --lari-alignment merged-chamfer
```

This mode flattens all five XYZ layers per face, drops nonfinite and exact-zero
XYZ entries, transforms all six faces into world coordinates, concatenates them,
and samples the merged cloud for a single global GT Chamfer scale/translation fit.
It does not average layers or fit individual faces. Empty faces are omitted; an
all-empty sample fails. The scene point checkpoint has no occupancy prediction
(its mask is a zero placeholder), so finite absent-surface predictions cannot be
identified without a separate segmentation checkpoint. No holes are filled.
The separate output suffix preserves earlier benchmark results.

### Merge LaRI layers within each face, then align faces independently

```bash
BENCHMARK_PYTHON=/opt/conda/envs/nova3r/bin/python \
BENCHMARK_SUFFIX=merged_layers_per_face \
  bash scripts/eval/benchmark_single_image.sh lari all --lari-alignment all-chamfer
```

For each cubemap face, flatten all five layers into one XYZ cloud, discard
NaN/Inf and exact-zero XYZ, and fit that cloud independently to GT points in
its camera frustum using Chamfer scale/translation alignment. Merge the six
aligned face clouds afterward; do not perform another global alignment.

Faces with no valid GT points in their alignment target are skipped and recorded
in `skipped_faces` with reason `no_valid_gt_points`. Their predictions are omitted
from the merged cloud; the full GT scoring target stays unchanged. A sample
fails if no alignable faces remain. Resume retains successful samples and retries
previous failures.

### LaRI evaluation throughput

Visibility geometry loading skips GLB materials and textures. Metrics only
consume vertex/face arrays; the old scene concatenation also built an unused
texture atlas. A real SAGE scene took 40.9–44.8 seconds through the old loader
and 1.28–1.31 seconds with materials skipped, with exactly identical float32
vertices and int64 faces. A synthetic multi-material GLB regression test also
checks geometry and face ordering, alongside the anonymization tests.

A complete one-panorama validation took 82.4 seconds before and 37.1 seconds
after the change, including model startup under concurrent GPU load. Its nine
metrics differed by less than 0.003 percentage points (or 0.002 cm for CD);
the alignment optimizer already has small GPU numerical variability. This
single-scene timing does not guarantee the same gain for every panorama.

Six-face batching was also measured: model inference fell from 0.79 seconds
to 0.47 seconds, with small point-map numerical changes. It was not enabled:
the larger gain comes from geometry loading. Alignment remains 300 iterations
per face at LR 0.1, with the same six faces, 100k-point scoring, seeds and
visibility protocol. New records include `timings` for prediction, GT preparation,
face alignment, metrics and total time.

Raw measurements and metric comparisons are under
`eval_results/baselines/lari/merged_layers_per_face_suite/` in
`geometry_speed_probe.json`, `geometry_full_metric_comparison.json`,
`profile_baseline.log`, `profile_fast_geometry.log` and `batch_speed_probe.json`.
The resumable supervisor is `scripts/eval/resume_lari_corrected.py`; its current
state is `resume_status.json` in that same suite directory.
