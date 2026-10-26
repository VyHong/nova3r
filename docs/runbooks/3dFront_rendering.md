PYTHONPATH="$PWD/third_party/BlenderProc" \
python third_party/BlenderProc/cli.py download haven \
  /tmp/datasets/polyhaven \
  --types hdris \
  --resolution 2k \
  --categories outdoor \
  --threads 16

## Raw 3D-FRONT scene with HDRI and fitted windows

This renderer loads the complete dwelling for the RGB panorama, so doorways
and open-plan room connections show adjacent geometry. Its GLB contains only
the requested room plus its generated window frames, glass, and door leaves.
Window opacity, transmission, roughness, IOR, and tint vary deterministically
per opening and are saved in `metadata.json`.

```bash
FRONT3D_SCENE=/tmp/datasets/3dfront/3D-FRONT/<scene-id>.json \
FRONT3D_ROOM=LivingDiningRoom-142 \
FRONT3D_OUTPUT="$PWD/artifacts/<output-name>" \
FRONT3D_HDRI=/tmp/datasets/polyhaven/hdris/<asset>/<asset>_2k.hdr \
FRONT3D_HDRI_ROTATION=128 \
FRONT3D_HDRI_STRENGTH=0.85 \
FRONT3D_CEILING_EMISSION_STRENGTH=0.8 \
FRONT3D_LAMP_EMISSION_STRENGTH=7.0 \
FRONT3D_OPEN_DOOR_PROBABILITY=0.45 \
FRONT3D_ADJACENT_DOOR_MARGIN=0.05 \
PYTHONPATH="$PWD/third_party/BlenderProc" \
python \
  scripts/data/3dfront/render_front3d_scene.py
```

Doors are deterministically opened by 12–32 degrees according to
`FRONT3D_OPEN_DOOR_PROBABILITY` (default `0.45`). Set it to `0` for all closed
or `1` for all open. Room-only GLBs omit dwelling-wide `SlabSide` records,
which some 3D-FRONT rooms reference even when those slab edges are far outside
the room footprint. The full-scene panorama still retains those meshes.
Door openings must follow the actual target floor boundary at three points
across their central span. This applies to both target and neighboring room
assignments. The tolerance beyond half the opening depth is
`FRONT3D_ADJACENT_DOOR_MARGIN` meters (default `0.05`).

Run `python scripts/data/3dfront/render_samples.py` to
create ten samples in `artifacts/3dfront_gpu_samples`. Panoramas use OptiX/CUDA,
OptiX denoising, 128 samples, and 1036×518 resolution by default. Override
`FRONT3D_WIDTH` and `FRONT3D_SAMPLES` to change quality. Windows vary between
`picture`, `double_casement`, `triple_panel`, `grid`, `transom`, `awning`, and
`arched_transom`. Set `FRONT3D_WINDOW_STYLE` to force one style. The sample
batch cycles through the styles. Frames vary in finish, thickness, and depth;
awning windows have a tilted upper glazed sash, and arched transoms have an
elliptical mullion inside the rectangular source opening. Metadata records
each target window's style and exported object names. GLBs contain geometry
and materials, while the HDRI and
Cycles lighting are recorded in metadata rather than baked into GLB textures.
The batch alternates five landscapes with five city environments, with three
night scenes at the end. Interior lighting follows BlenderProc's native
3D-FRONT loader: ceiling materials emit at strength `0.8` and lamp materials
emit at strength `7.0` by default. Override these with
`FRONT3D_CEILING_EMISSION_STRENGTH` and `FRONT3D_LAMP_EMISSION_STRENGTH`.
There is no camera-centered fill light. Select a night HDRI using
`FRONT3D_HDRI`; the time-of-day flag does not itself replace the environment.
The ten-sample batch varies ceiling emission from `0.35` to `1.4` and lamp
emission from `4` to `20`; its three night scenes use lamp strengths of
`12`, `16`, and `20`.

Doors vary between `flush`, `two_panel`, `six_panel`, `glazed`, and `louvered`.
Use `FRONT3D_DOOR_STYLE` to force a style. Glazed doors have a real inset glass
panel and louvered doors use individual angled slats. Raised panels appear on
both sides of paneled doors. Wood/paint finishes and metal lever hardware vary
deterministically per opening and are recorded alongside exported part names.

## Batched multi-view rooms

Use the `blender` environment (defined in `environment-panorama.yml`). To
install the additional 3D-FRONT runtime dependencies into an existing environment:

```bash
conda activate blender
python -m pip install -r scripts/data/3dfront/requirements-blender.txt
```

Run `python scripts/data/3dfront/render_rooms_batched.py` in this environment.
Workers use the same Python interpreter as the launcher; set
`BLENDERPROC_PYTHON` to override it. GPU rendering requires a working NVIDIA driver and visible OptiX-capable GPU.
The raw assets must include `/tmp/datasets/3dfront/future_model` in addition
to the scene, architectural texture, CC texture, and HDRI directories.

`render_rooms_batched.py` groups work by dwelling and, by default, renders every
room from the manifest in one persistent Blender process. The complete dwelling,
furniture, images, and materials are loaded once; per-room cameras and generated
window/door geometry are cleaned between rooms. Use `--max-rooms-per-scene N`
only when intentionally limiting the number of rooms from each dwelling. The renderer samples ten positions
with the legacy fast sampler (BlenderProc's `Front3DPointInRoomSampler`, exact
BVH surface clearance, and minimum pose separation), renders all poses
together, then exports one room GLB. The legacy sampler avoids the expensive
four-heading obstacle and coverage ray grids. Each `metadata.json` records
`camera_sampler`, `camera_sampling_attempts`, and `camera_sampling_seconds`.
It writes
`extrinsics.json` in the format consumed by `Front3DSDFDataset`: panorama
filename to row-major Blender-axis world-to-camera matrix.

```bash
python scripts/data/3dfront/render_rooms_batched.py \
  --manifest artifacts/3dfront_gpu_samples/manifest.json \
  --output artifacts/3dfront_room_batches \
  --views 10 --samples 64 --width 1036 --workers 1 --gpu-ids 0 \
  --hdri-root /tmp/datasets/3dfront/hdris
```

For multiple GPUs, use one worker per GPU, such as `--workers 4 --gpu-ids
0,1,2,3`. Dataset shards can run independently with `--shard-count N
--shard-index I`. Resume is automatic unless `--overwrite` is supplied.

The default shared-GLB profile resizes only its embedded textures to 1024 px
and exports them as WebP at quality 80. Panorama source textures remain at
their original resolution because resizing happens after rendering. Override
this with `--glb-texture-size`, `--glb-image-format`, and
`--glb-image-quality`.

### Render 2,100 eligible rooms

First build a deterministic manifest. Eligibility requires a supported room
type, floor geometry, window geometry, and at least four furniture instances.
The builder fails instead of silently producing fewer than the requested
number of rooms.

```bash
python scripts/data/3dfront/build_room_manifest.py \
  --scenes /tmp/datasets/3dfront/3D-FRONT \
  --target 2100 \
  --output artifacts/3dfront_2100_manifest.json
```

Then render ten 1036×518 panoramas and one optimized GLB per room on GPU 0:

Optionally warm the Linux page cache immediately before rendering. This reads
only scene, furniture, architectural-texture, and HDRI files referenced by the
manifest; add `--include-cc` only when enough RAM is available for the much
larger CC-material collection:

```bash
python scripts/data/3dfront/warm_render_cache.py \
  --manifest artifacts/3dfront_2100_manifest.json \
  --workers 16
```

```bash
python scripts/data/3dfront/render_rooms_batched.py \
  --manifest artifacts/3dfront_2100_manifest.json \
  --output artifacts/3dfront_2100 \
  --views 10 \
  --samples 64 \
  --width 1036 \
  --workers 1 \
  --gpu-ids 0 \
  --hdri-root /tmp/datasets/3dfront/hdris
```

The command is resumable: rerun the same command after an interruption and
complete room folders are skipped. Do not use `--overwrite` when resuming.
The terminal progress bar counts completed and already-complete rooms and shows
the current room throughput and estimated remaining time.
Use one worker per GPU, for example `--workers 4 --gpu-ids 0,1,2,3` on a
four-GPU machine. A complete room contains ten panoramas, `room.glb`,
`extrinsics.json`, and `metadata.json`. Successful batch completion also
writes `batch_results.json` and the 21,000-entry `panorama_list.json`.

Set `--hdri-root` (or `FRONT3D_HDRI_ROOT`) to the current HDRI collection.
Lookup is recursive and accepts `.hdr` and `.exr`, so the old
`<asset>/<asset>_2k.hdr` layout is no longer required. Paths saved by an older
manifest are relocated by asset name when the original file no longer exists.
`/tmp` may be cleared by the host, so persistent storage is recommended.
