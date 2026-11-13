# MeshUDF1024: extraction tolerances, memory safety, and validation

Updated 2026-11-13. This documents the changes made after MeshUDF1024 regressed
against `exp_output/hunyuandit_panorama/meshudf_apd_resolution4/resolution512_full`.

## Current manually selected runs and throughput preset

The user stopped the automatic 200-step run and selected 100 steps. ScanNet++
resumes in `resolution1024_reference512_full`; 3D-FRONT runs in
`3dfront_resolution1024_reference512_steps100_full`, both under
`exp_output/hunyuandit_panorama/meshudf_apd_resolution4/`. The automatic pilot
supervisor remains stopped. Each directory's `launch.json` and
`evaluation_fast.log` identify the current worker and progress.

Both runs now use 262144 field queries, 65536 gradient queries, 131072 vertex
queries, eight OMP/MKL/OpenBLAS threads, and `mesh_udf_fast_cleanup: true`.
Actual resolution 1024, reference cutoffs 512, padding .05, epoch09, CFG5 and
100 Euler steps remain fixed; CUDA OOM batch backoff remains enabled. Completed
meshes/metrics are reused, including compatible 100-step pilot records.

Cached-decoder microbenchmarks measured 262144 gradient points in 0.270 s with
16384-point batches versus 0.139 s with 65536-point batches. Field-query batch
changes alone gave little improvement. Increasing CPU threads and using
524288-point field batches reduced a small complete grid-query benchmark from
1.505 s to 1.051 s; the production preset uses the more moderate 262144 batch.
Field values matched exactly in these probes; gradients had numerical
differences. These timings do not establish an end-to-end speedup or full-set
quality parity.

Fast cleanup was compared with legacy cleanup from the same raw cached mesh:
1.241 s versus 1.824 s, with exactly equal final vertex and face arrays
(347438 vertices, 687343 faces). This is one scene at a smaller diagnostic grid,
not proof for every 1024 scene. Per-stage logs now also time vertex filtering
and cleanup. Raw benchmark results are `batch_speed_benchmark.json`,
`gradient_speed_benchmark.json`, `grid_speed_benchmark.json`, and
`cleanup_speed_benchmark.json` under `meshudf_apd_resolution4/`.

## Status and scope

The extraction change recovered occluded APD and Chamfer performance in a
same-latent scene comparison. It has **not established parity across the full
dataset or every metric**. The initial full 1024 evaluation was paused at two
successful samples at the user's request. The current pilot compares **50, 100,
and 200 Euler steps** on three ScanNet++ scenes. Historical 400-step results are
retained but excluded from selection.

After three complete comparisons, the automatic launcher selects the highest
mean occluded APD5; lower occluded CD, then fewer steps, break ties. It then starts
a fresh full ScanNet++ evaluation with that step count and the memory safeguards
below. The three-scene pilot selects a candidate; the 100-scene run validates it.

Live state is recorded in:

- `exp_output/hunyuandit_panorama/flow_steps_1024_generalization/auto_full_status.json`
- `exp_output/hunyuandit_panorama/flow_steps_1024_generalization/three_scene_selection.json` once selected.
- `exp_output/hunyuandit_panorama/meshudf_apd_resolution4/resolution1024_reference512_steps<N>_full/` once launched.

## Why increasing resolution changed the result

There were two distinct effects.

1. **Padding was different.** The earlier 1024 scene used padding 0.10; the 512
   reference used 0.05. Increasing padding changes the extraction region and its
   physical voxel size. Controlled comparisons now keep padding at 0.05.
2. **Native cell cutoffs shrink with voxel size.** The native MeshUDF kernel uses
   a mean-corner UDF cutoff of `1.05 * voxel_size` and a maximum-corner cutoff of
   `1.74 * voxel_size`. Doubling resolution approximately halves these absolute
   cutoffs. A learned UDF can retain a small positive residual near a surface, so
   finer extraction can reject cells that coarser extraction accepted. The
   normal-query band also shrinks when expressed in voxel units.

A synthetic residual-plane regression test reproduces a surface appearing on
the coarse grid, disappearing on the finer grid, and returning when the finer
grid retains the coarse-grid tolerances.

The BF16 precision probe also found more identical neighboring UDF values at
1024 than at 512. Its effect on the final metrics was not established. Full FP32
extraction is **not** part of this fix and must not be credited for the reported
improvement.

## Extraction fix: reference-resolution cutoffs

New config: `mesh_udf_reference_resolution: 512`.
New CLI: `--mesh-udf-reference-resolution 512`.

For actual grid dimensions `resolution` and reference resolution `R`, compute:

```text
s = min(1, (R - 1) / (max(resolution) - 1))
```

The extractor:

- Keeps the actual 1024 extraction lattice and its query coordinates.
- Sends `UDF * s` to native marching cubes, widening its cell eligibility
  cutoffs to approximately the reference grid's absolute tolerances.
- Expands the configured normal band by `1 / s`, supplying gradients for the
  wider eligible region. At grid maximum 1024 and reference 512, a configured
  two-voxel band becomes approximately 4.0039 fine-grid voxels.
- Combines this scale with the existing per-tile distance correction when
  marching cubes is tiled.
- Uses the **original, unscaled decoder UDF** for final vertex filtering.

Uniform distance scaling preserves interpolation ratios. It can change mesh
topology by admitting additional cells. It does not subdivide a 512 mesh,
remesh in Blender, fill missing regions with ground truth, or change the metric
definitions. Wider eligibility is a coverage/accuracy tradeoff, not a guarantee
that every scene or metric improves.

The reference option is off by default. A reference finer than the actual grid
does not tighten its cutoffs. Sparse-brick extraction with reference cutoffs is
explicitly rejected rather than silently ignoring the option.

Implementation:

- `training/mesh2sdf/tools/mesh_udf/mesh_udf.py`: `_reference_distance_scale`,
  normal-band adjustment, full-grid/tiled native input scaling.
- `training/mesh2sdf/module_options.py`: option parsing and validation.
- `training/mesh2sdf/lightning_module.py`: option forwarding.
- `training/test.py`: CLI override.

## Controlled evidence

Scene: `scannetpp_09c1414f1b_00003_4`. These rows use identical cached decoded
latents, epoch09, CFG5, padding 0.05, and the unchanged evaluation protocol.
CD is cm; APD/F-score are percentages. Visible/occluded CD is backward Chamfer;
complete CD is symmetric Chamfer.

| Metric | 512 replay | Default 1024 | 1024 with reference 512 |
|---|---:|---:|---:|
| Visible CD ↓ | 8.4837 | 8.8838 | 8.5170 |
| Visible APD10 ↑ | 70.6672 | 68.2068 | 70.3447 |
| Visible APD5 ↑ | 29.2855 | 26.3438 | 28.3145 |
| Occluded CD ↓ | 12.9321 | 13.3493 | 12.8353 |
| Occluded APD10 ↑ | 55.5795 | 53.5198 | 55.8689 |
| Occluded APD5 ↑ | 24.5396 | 21.4452 | 25.8196 |
| Complete CD ↓ | 12.1895 | 12.4532 | 12.1520 |
| Complete F10 ↑ | 54.1278 | 52.2396 | 53.4740 |
| Complete F5 ↑ | 22.9416 | 20.7318 | 22.7992 |

This demonstrates recovery of the tested scene's occluded metrics, while
visible APD5 and complete F-scores still trail the 512 replay. Do not present it
as dataset-wide superiority.

Raw evidence:
`exp_output/hunyuandit_panorama/meshudf_resolution_diagnosis/scannetpp_09c1414f1b_00003_4/validated_comparison.json`.

Seed clarification: diffusion reconstruction uses fixed initial noise seed
**42**. The original full-list index for this scene is 2; `42 + index = 44`
controls the deterministic posterior/loss sampling path. Earlier references to
an “inference seed 44” were ambiguous; it is not the diffusion noise seed.

## Memory safeguards for large scenes

1. Stream query coordinates in batches. Dense UDF and normal arrays are
   explicitly allocated on **CPU**, rather than retaining a full XYZ/UDF/normal
   grid on the GPU.
2. The selected full-run preset uses 65536 field queries, 16384 gradient queries
   and 32768 vertex-filter queries per batch.
3. Field, normal, and vertex queries catch `torch.cuda.OutOfMemoryError`, release
   the failed batch's references, clear the CUDA cache, halve the batch, and
   retry from the same position. Query order and resolution are preserved. An
   OOM at batch size one propagates instead of looping forever. Other errors
   are not treated as OOM.
4. The existing native guard automatically uses 128-cell MC tiles when
   `4 * grid_points` exceeds the signed 32-bit allocation/index limit. The
   reference-distance correction is applied consistently within those tiles.
5. The automatic full-run supervisor permits up to three process attempts.
   Later attempts use smaller query/vertex batches and 64-cell MC tiles.
   Successful per-sample meshes/metrics are reused through
   `MESH_UDF_RESUME_COMPLETED`; incomplete/error samples are retried. It marks
   the run complete only after 100 successful records.

The full selected-step run starts in a new process to load these changes. The
old paused worker is terminated when its replacement starts. This avoids
resuming an old in-memory implementation with the wrong step count.

Memory checks passed:

| Check | Result | Scope |
|---|---|---|
| Streamed 1024³ grid + automatic MC tiling | 1,073,741,824 grid points; 17.77 GiB peak CPU RSS; 2,093,058 faces | Analytic field; verifies maximum-size grid machinery, not learned-scene quality |
| Learned decoder, field/gradient queries | 1.60 GiB peak GPU allocation in the probe | Decoder-only; whole-model memory is additional |
| OOM injection tests | Correct ordered field/normal results after backoff | Simulated CUDA allocation failures |

The job's RAM limit was approximately 449 GiB. These checks provide headroom
evidence and exercise retry behavior; they cannot guarantee every future
scene or concurrent GPU workload will fit. Surface complexity also affects
mesh cleanup/output memory.

Memory evidence is under
`exp_output/hunyuandit_panorama/flow_steps_1024_generalization/memory_safety/`.

## Full-run configuration

```yaml
cfg:
  mesh_method: mesh_udf
  mesh_mc: 1024
  mesh_bounds_padding: 0.05
  mesh_udf_reference_resolution: 512
  mesh_udf_normal_band_voxels: 2.0
  mesh_udf_surface_threshold: 0.010
  mesh_udf_streamed_grid: true
  mesh_udf_query_batch_size: 65536
  mesh_udf_vertex_batch_size: 32768
  mesh_udf_mc_tile_cells: 0  # automatic large-grid protection
  mesh_udf_fast_cleanup: false
  smooth_borders: false
  test_flow_matching_steps: 100  # baseline; launcher replaces with pilot winner
```

Do not manually double the normal band when setting reference resolution 512;
the extractor already adjusts it. Border smoothing, Taubin smoothing, Blender
remeshing, and FP32 meshing are not enabled in this evaluation preset.

The flow sampler's previously hard-coded 100 Euler steps are now configurable
through `test_flow_matching_steps` or `--flow-matching-steps`. Default behavior
remains 100 steps. Different step counts reuse identical initial noise and
conditioning but integrate separately. Completed 100/200 results are reused in
the revised pilot; 400-step results remain historical and ineligible.

## Reproduce checks and inspect results

```bash
python -m unittest \
  training.mesh2sdf.tests.test_mesh_udf_reference_resolution \
  training.mesh2sdf.tests.test_mesh_udf_normal_band \
  training.mesh2sdf.tests.test_mesh_udf_tile_scale \
  training.mesh2sdf.tests.test_mesh_udf_memory_retry \
  training.mesh2sdf.tests.test_flow_matching_steps \
  training.mesh2sdf.tests.test_step_pilot_selection

python scripts/eval/stress_meshudf_memory.py \
  --output /tmp/meshudf_1024_memory_check.json
```

`scripts/eval/report_meshudf_resolution.py` compares all nine metrics on matched
successful sample IDs, including per-scene improvements/regressions. Partial
means are not compared against the full 512 mean. Failed and unfinished samples
are counted explicitly. The full-run report is `comparison_to_512.md`/`.json`
inside the selected full-run output directory.

The automatic launcher's PID is recorded in
`flow_steps_1024_generalization/auto_full_launcher.json`. To cancel automatic
execution, stop the supervisor as well as its worker so that retry logic does
not restart a user-cancelled run.
