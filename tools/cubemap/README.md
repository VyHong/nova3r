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
  python tools/cubemap/replica_pano.py \
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
  python tools/cubemap/scannetpp.py \
  --data_root /tmp/scannetpp_data/data \
  --output_root /tmp/scannetpp_data/cubemap_data \
  --workers 16
```

To require and generate matching depth cubemaps:

```bash
conda run --no-capture-output -n nova3r \
  python tools/cubemap/scannetpp.py \
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
  python tools/cubemap/sage10k.py \
  --input-root /tmp/datasets/sage-10k/panoramas \
  --output-root /tmp/datasets/sage-10k/cubemaps \
  --workers 4 \
  --shape cube \
  --fov 95 \
  --res 518
```

## Single panorama

```bash
conda run --no-capture-output -n nova3r \
  python tools/cubemap/generate.py \
  --image /path/to/rgb.png \
  --depth /path/to/depth.png \
  --image_out /path/to/output/rgb \
  --depth_out /path/to/output/depth \
  --shape cube --fov 95 --res 518
```
