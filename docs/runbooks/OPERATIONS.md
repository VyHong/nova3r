# Operations Runbook

## load image
podman load -i ~/projects/demo/nova3r_image.tar

## run demo with 6 views
python scripts/inference/demo_nova3r.py --images /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0000.jpg /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0001.jpg /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0002.jpg /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0003.jpg /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0004.jpg /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0005.jpg --ckpt checkpoints/scene_n2/checkpoint-last.pth --num_queries 200000

## open image
podman run \
  -v /mnt:/mnt:rw \
  -v /tmp:/tmp:rw \
  -w "$PWD" \
  --device /dev/fuse \
  --cap-add SYS_ADMIN \
  --security-opt label=disable \
  --security-opt seccomp=unconfined \
  --device nvidia.com/gpu=all \
  -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  --name demo \
  --hostname demo \
  --rm -it demo:v1

## commit image
podman commit demo demo:v1

## export image
podman export -o ~/projects/demo/nova3r.tar nova3r:latest

## import image
podman import ~/projects/demo/nova3r.tar nova3r

## save image 
podman save -o ~/projects/demo/nova3r.tar demo:v1

## mount replicaPano
bash bulk_mount_mount_zip.sh /mnt/home/vyhong/projects/nova3r/datasets/ReplicaPano /tmp/datasets/replica_pano -o notrim

## mount 3dfront for rendering
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/3dfront/3D-FRONT.zip /tmp/datasets/3dfront
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/3dfront/3D-FRONT-texture.zip /tmp/datasets/3dfront
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/3dfront/cctextures.zip /tmp/datasets/3dfront
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/3dfront/future_model.zip /tmp/datasets/3dfront
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/3dfront/hdris.zip /tmp/datasets/3dfront/ 

## mount 3dfront
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/3dfront/panoramas.zip /tmp/datasets/3dfront
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/3dfront/udf.zip /tmp/datasets/3dfront
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/3dfront/gt_latents.zip /tmp/datasets/3dfront
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/3dfront/panorama_latents.zip /tmp/datasets/3dfront
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/3dfront/visible_latents.zip /tmp/datasets/3dfront

## mount scannetpp
bash bulk_mount_mount_zip.sh /mnt/datasets/scannetpp/scannetpp_panorama.zip /tmp/datasets/scannetpp

bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/scannetpp/cubemap_data.zip /tmp/datasets/scannetpp -o precache
bash bulk_mount_mount_zip.sh /mnt/home/vyhong/projects/nova3r/datasets/scannetpp/udf_color.zip /tmp/datasets/scannetpp -o precache
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/scannetpp/panorama_latents.zip /tmp/datasets/scannetpp -o precache
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/scannetpp/vis_latent.zip /tmp/datasets/scannetpp -o precache
bash bulk_mount_mount_zip.sh /mnt/home/vyhong/projects/nova3r/datasets/scannetpp/cube_trans_latent.zip /tmp/datasets/scannetpp -o precache

## mount sage-10k
bash bulk_mount_mount_zip.sh /mnt/datasets/sage10k/panorama-scenes.zip /tmp/datasets/sage-10k

bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/sage-10k/udf.zip /tmp/datasets/sage-10k
bash bulk_mount_mount_zip.sh /mnt/home/vyhong/projects/nova3r/datasets/sage-10k/cubemap_data.zip /tmp/datasets/sage-10k -o notrim
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/sage-10k/pano_features.zip /tmp/datasets/sage-10k
bash bulk_mount_mount_zip.sh /mnt/home/vyhong/projects/nova3r/datasets/sage-10k/visible_latents.zip /tmp/datasets/sage-10k
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/sage-10k/latents.zip /tmp/datasets/sage-10k

## Python Path
export PYTHONPATH=$PYTHONPATH:/absolute/path/to/nova3r

## 3D-FRONT UDF surface -> panorama-camera ShapeVAE target latent
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nova3r python -u \
scripts/preprocessing/latents/surface_to_vae_latent.py \
  --dataset-layout 3dfront \
  --split-list data/3dfront/panorama_list_2000.json \
  --surface-root /tmp/3dfront/udf \
  --pose-root /tmp/3dfront/panorama/panoramas_20000_gpu \
  --panorama-camera \
  --checkpoint exp_output/hunyuan_vae/checkpoints/last.ckpt \
  --config checkpoints/hunyuan/.hydra/config.yaml \
  --output-root /tmp/3dfront/gt_latents_camera_centered \
  --device cuda:0 \
  --precision fp32 \
  --batch-size 24 \
  --num-workers 8 \
  --normalization cube \
  --cube-center-camera \
  --save-latent-before-sampling \
  --seed 0

## run training in background
nohup bash projects/nova3r/train.sh &

## tensorboard 
tensorboard --logdir /mnt/home/vyhong/projects/nova3r/exp_output

## eval 
python eval/evaluate_pcd.py --gt_ply /tmp/datasets/replica_pano/office_0_000/office_0_000/office_0_aligned.ply --pred_ply /mnt/home/vyhong/projects/nova3r/exp_output/nova3r_img_cond_finetune_complete/nova3r_training/version_22/val_points/epoch_10/pointcloud.ply --output_file

# Point cloud autoencoding from a SCRREAM scene
python demo_nova3r_ae.py   --input_ply  nova3r/debug_points/room_2_aligned.ply --ckpt checkpoints/scene_ae/checkpoint-last.pth  --num_queries 50000

## install codex
curl -fsSL https://chatgpt.com/codex/install.sh | sh

## download the ambinetcg textures
PYTHONPATH="$PWD/third_party/BlenderProc" \
python third_party/BlenderProc/cli.py download cc_textures \
  /tmp/datasets/ambientcg/cctextures \
  --workers 24

## Horizontal empty-space supervision for ShapeVAE

In the experiment config, `horizontal_empty_points: 4096` appends this many
volume queries per training example after surface/geometry normalization. Set it
to `0` to disable. `empty_space_vertical_axis: 1` keeps training-frame Y within
the enclosing box height and samples X/Z uniformly in `[-2, 2]` outside the box,
leaving a `1 / sdf_label_scale` safety margin. These queries receive saturated
UDF targets; any RGB supervision is masked out. Validation samples are unchanged.

New UDF preprocessing writes exact `mesh_bounds`. Existing NPZ files use their
padded `bounds` as a conservative enclosure; bounds are never inferred from a
sampled point cloud. All eight corners follow the geometry's camera transform
and normalization before their enclosing axis-aligned box is computed. Thus the
axes are training/camera axes, not necessarily gravity-aligned world axes.
Missing bounds raise an error when this feature is enabled. If a box covers
the whole horizontal target domain, the extra slots for that example resample
existing geometry queries with their original distances and features. This
keeps batch shapes consistent without assigning false empty-space labels.
