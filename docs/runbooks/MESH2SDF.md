# Mesh2SDF Runbook

```bash
python scripts/preprocessing/mesh2sdf/process_replica.py --data_root /tmp/datasets/replica_pano/ --output_root datasets/ReplicaPano/median_pad_udf --udf-only

## Blender env
conda env create -f environment-panorama.yml

## Download ambientCG 2K PBR materials (safe to resume)
PYTHONPATH="$PWD/third_party/BlenderProc" \
python third_party/BlenderProc/cli.py download cc_textures \
  /tmp/datasets/ambientcg/cctextures \
  --workers 8

## Use them when rendering 3D-FRONT panoramas
# --architecture-material ambientcg \
# --pbr-material-root /tmp/datasets/ambientcg/cctextures \
# --exterior-ground pbr-plane \
# --time-of-day random

## sage gen

nohup bash training/mesh2sdf/tools/sage-10k/render_panoramas_batched.sh       /tmp/datasets/sage-10k/scenes/       /tmp/datasets/sage10k/       --completion-dir /tmp/datasets/sage-10k/panorama-scenes/       --batch-size 10       --jobs 24       --conda-env blender       --gpus 0       --       --save-scene-glb       --device OPTIX       --num-panoramas 10 &

conda run --no-capture-output -n nova3r python scripts/preprocessing/mesh2sdf/process_sage10k.py --data-root /tmp/datasets/sage-10k/sage10k/ --output-root /tmp/datasets/sage-10k/udf --workers 2

conda run --no-capture-output -n nova3r \
  python scripts/preprocessing/mesh2sdf/process_3dfront.py \
  --data_root /tmp/datasets/3dfront/3D-FRONT-SCENE \
  --output_root /tmp/datasets/3dfront/3D-FRONT-SCENE-UDF \
  --room_id_json 3D-Front/midi_room_ids.json 3D-Front/valid_room_ids.json \
  --workers 2 \
  --udf-only
```
