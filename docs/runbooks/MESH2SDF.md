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
  --data_root artifacts/3dfront_2100 \
  --output_root /tmp/datasets/3dfront/udf \
  --room_id_json artifacts/3dfront_2100_manifest.json \
  --workers 2 \
  --udf-only
```


## HRUDF 512 near-surface preset

For MeshUDF1024 extraction tolerances, memory safeguards, and validation evidence,
see [MeshUDF1024 improvements](MESH_UDF_1024.md).

Saved config: `eval/baselines/configs/hunyuandit_epoch09_hrudf512_near_surface.yaml`.
This preset is a retry for the malformed unfiltered 3D-FRONT reconstruction; room validation is pending.

- Epoch09 checkpoint, CFG 5, bounds padding 0.10.
- HRUDF grid resolution 512, five refinement recursions.
- Query/pseudo-SDF batch size 65536.
- Mean cell UDF cutoff: 1.2 voxel spacings (`mesh_hrudf_max_avg_factor`).
- Maximum corner UDF cutoff: 2.0 voxel spacings (`mesh_hrudf_max_max_factor`).
- CLI seed 43 with the original selected-four list; the target is sample index 1, giving inference seed 43.

The default absolute HRUDF limits (mean 0.1, max 0.2) admitted essentially the full cube and produced spurious sheets on this learned field. The new voxel-relative limits are under validation. An off-center analytic sphere check passed the coordinate/distance/gradient conventions.

```bash
python training/test.py \
  --config eval/baselines/configs/hunyuandit_epoch09_hrudf512_near_surface.yaml \
  --mesh-method hrudf --test-dataset 3dfront \
  --test-list exp_output/hunyuandit_panorama/eval_epoch09_mesh_udf_1024_3dfront_selected4_padding10_seed43/test_list.json \
  --test-sample 3dfront_1ab3b8d4-af97-43f7-bb23-a6233c87ebfd_LivingDiningRoom-129846_00004_4 \
  --seed 43 --devices 1 --low-cpu-mem-usage \
  --output-dir exp_output/hunyuandit_panorama/hrudf512_near_surface
```
