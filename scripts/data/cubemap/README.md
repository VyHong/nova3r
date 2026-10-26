# Cubemap generation

This folder contains the shared panorama projection code and dataset-specific
batch entry points.

- `generate.py`: convert one equirectangular RGB, depth, and/or categorical mask
  panorama into perspective faces.
- `replica_pano.py`: batch conversion for the ReplicaPano directory layout.
- `scannetpp.py`: batch conversion for ScanNet++ RGB, depth, and anonymization masks.
- `sage10k.py`: batch conversion for rendered SAGE-10k panoramas, including
  composed world-to-face extrinsics.
- `legacy_views.py`: older configurable horizontal-view generator retained for
  compatibility with `experimentation/test_image_gen.py`.

Run these commands from the repository root in the `nova3r` Conda environment.

## ReplicaPano depth cubemaps

The mounted source dataset is read-only, so use a separate writable output root:

```bash
conda run --no-capture-output -n nova3r \
  python scripts/data/cubemap/replica_pano.py \
  --data-root /tmp/datasets/replica_pano \
  --output-root /tmp/datasets/replica_pano_depth \
  --depth-only \
  --workers 16 \
  --shape cube \
  --fov 95 \
  --res 518
```

## ScanNet++ RGB and anonymization-mask cubemaps

```bash
conda run --no-capture-output -n nova3r \
  python scripts/data/cubemap/scannetpp.py \
  --data_root /tmp/scannetpp_data/data \
  --output_root /tmp/scannetpp_data/cubemap_data \
  --workers 16
```

To require and generate matching depth cubemaps:

```bash
conda run --no-capture-output -n nova3r \
  python scripts/data/cubemap/scannetpp.py \
  --data_root /tmp/scannetpp_data/data \
  --output_root /tmp/scannetpp_data/cubemap_data \
  --require-depth \
  --workers 16
```

Use `--depth-only` to leave existing RGB and mask cubemaps untouched. Depth
directories named `depth`, `depths`, `resized_depth`, or `resized_depths` are
auto-detected; use `--depth-dir-name NAME` to specify a different sibling
directory.

## SAGE-10k cubemaps

```bash
conda run --no-capture-output -n nova3r \
  python scripts/data/cubemap/sage10k.py \
  --input-root /tmp/datasets/sage-10k/panoramas \
  --output-root /tmp/datasets/sage-10k/cubemaps \
  --workers 4 \
  --shape cube \
  --fov 95 \
  --res 518
```

Generate only the final-test SAGE and 3D-FRONT 90-degree cubemaps:

```bash
conda run --no-capture-output -n nova3r python scripts/data/cubemap/sage10k.py \
  --input-root /tmp/datasets/sage-10k/panorama-scenes \
  --output-root datasets/sage-10k/cubemaps/final_test_100_prepared_cube_90 \
  --sample-list data/sage-10k/final_test_list.json \
  --dataset-layout sage10k --workers 8 --shape cube --fov 90 --res 518

conda run --no-capture-output -n nova3r python scripts/data/cubemap/sage10k.py \
  --input-root /tmp/datasets/3dfront/panorama_scenes \
  --output-root datasets/3dfront/cubemaps/final_test_100_prepared_cube_90 \
  --sample-list data/3dfront/final_test_list.json \
  --dataset-layout 3dfront --workers 8 --shape cube --fov 90 --res 518
```

## One centered RGB cubemap per SAGE-10k scene

`render_centered_sage10k.py` deduplicates the `SCENE FRAME` list by scene and
renders one view at the exact room midpoint (including half-height), with a
fixed SAGE +Y heading. The supplied list contains 9,796 unique scenes.
`--camera-height 1.7` overrides the height above the floor. The camera is not
moved away from furniture at the room center.

Run from the repository root; create `blender` once if needed with
`conda env create -f environment-panorama.yml`.

```bash
conda run --no-capture-output -n nova3r python scripts/data/cubemap/render_centered_sage10k.py \
  --scene-list data/sage-10k/surface_vae_latent_list.json \
  --scene-root /tmp/datasets/sage-10k/scenes \
  --output-root /tmp/datasets/sage-10k/centered_median \
  --conda-env blender --projection-env nova3r \
  --gpus 0,1 --jobs 32 --batch-size 50 --workers 4
```

Add `--limit 1` to test one scene, or `--dry-run` to check input coverage.
Surface files and median normalization are no longer used; `--surface-root`
is accepted only for compatibility with older commands. The directory name
`centered_median` is retained here so existing rendered panoramas can be reused.

Final data for each scene:

```text
panoramas/<scene>/
  <scene>_panorama_000.png       # centered 2048x1024 panorama
  extrinsics.json                # panorama pose
  .complete.json                 # render settings and completion record
cubemaps/<scene>/<scene>_panorama_000_cube_95/
  0000.jpg ... 0005.jpg          # six 518x518 faces, 95-degree FOV
  camera_data.json               # face intrinsics and local/world extrinsics
  .complete.json                 # settings and completion record for resume
```

The renderer first creates the missing panoramas, then converts them to
cubemaps. The panorama PNG and its extrinsics are retained. The driver removes
only obsolete `normalization.json` files and normalized pose fields left by the
older workflow. Logs and `scene_list.json`, `completed_list.json`, and
`failures.json` remain as progress records. Cubemap camera metadata contains
metric glTF `world_extrinsics`.

Reruns skip completed cubemaps even after the source panorama has been deleted.
To convert existing panoramas or remove normalization metadata from the older
workflow without launching Blender:

```bash
python scripts/data/cubemap/render_centered_sage10k.py \
  --output-root /tmp/datasets/sage-10k/centered_median \
  --projection-env nova3r --convert-only
```

Do not run cleanup concurrently with another driver using the same output root.
An already-running Python driver keeps the old behavior: either restart it to
use the updated pipeline (completed panoramas are reused), or let it finish
and run `--convert-only` afterward. Existing matching cubemaps from the older
workflow are reused without reprojecting their images.

GPU IDs default to `0,1`; `--jobs` is total concurrency across both GPUs.
`--gpus 0` restricts rendering to one GPU. Batch logs are in
`panoramas/logs-batched/`, and failed scenes have `panoramas/<scene>/failure.log`.

## Single panorama

```bash
conda run --no-capture-output -n nova3r \
  python scripts/data/cubemap/generate.py \
  --image /path/to/rgb.png \
  --depth /path/to/depth.png \
  --image_out /path/to/output/rgb \
  --depth_out /path/to/output/depth \
  --shape cube --fov 95 --res 518
```
