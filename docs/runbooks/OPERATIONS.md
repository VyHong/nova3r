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

## mount 3dfront
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/3dfront/panoramas.zip /tmp/datasets/3dfront
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/3dfront/udf.zip /tmp/datasets/3dfront
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/3dfront/gt_latents.zip /tmp/datasets/3dfront
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/3dfront/panorama_latents.zip /tmp/datasets/3dfront
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/3dfront/visible_latents.zip /tmp/datasets/3dfront

## mount scannetpp
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/scannetpp/cubemap_data.zip /tmp/datasets/scannetpp -o precache
bash bulk_mount_mount_zip.sh /mnt/home/vyhong/projects/nova3r/datasets/scannetpp/udf_color.zip /tmp/datasets/scannetpp -o precache
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/scannetpp/panorama_latents.zip /tmp/datasets/scannetpp -o precache
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/scannetpp/vis_latent.zip /tmp/datasets/scannetpp -o precache
bash bulk_mount_mount_zip.sh /mnt/home/vyhong/projects/nova3r/datasets/scannetpp/cube_trans_latent.zip /tmp/datasets/scannetpp -o precache

## mount sage-10k
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

## Render 20,000 panoramas (2,000 rooms x 10 views) with 12 sharded workers.
## Each worker handles different rooms; rerunning resumes completed output.
mkdir -p /tmp/datasets/3dfront/panoramas_20000_gpu/logs

seq 0 11 | xargs -P 12 -I '{}' bash -c '
  worker="$1"
  block=$((worker / 3))
  slot=$((worker % 3))
  # CPU ranges permitted by this container: 0-29,72-101,144-173,216-245
  bases=(0 72 144 216)
  first=$((${bases[$block]} + slot * 10))
  last=$((first + 9))

  exec taskset -c "${first}-${last}" \
    env CUDA_VISIBLE_DEVICES=0 \
    conda run --no-capture-output -n blender \
    python -u scripts/data/3dfront/render_panoramas.py \
      /tmp/datasets/3dfront/3D-FRONT-SCENE \
      /tmp/datasets/3dfront/panoramas_20000_gpu \
      --target-complete 2000 \
      --shard-count 12 \
      --shard-index "$worker" \
      --num-panoramas 10 \
      --width 1024 \
      --height 512 \
      --samples 32 \
      --threads 10 \
      --device OPTIX \
      --clearance 0.2 \
      --min-separation 0.15 \
      --max-attempts 100000 \
      --architecture-material ambientcg \
      --pbr-material-root /tmp/datasets/ambientcg/cctextures \
      --exterior-ground pbr-plane \
      --background sky \
      --time-of-day day \
      --light-energy 140 \
      --lamp-energy 35 \
      --minimum-fill-energy 280 \
      >"/tmp/datasets/3dfront/panoramas_20000_gpu/logs/worker_${worker}.log" 2>&1
' _ '{}'

PYTHONPATH="$PWD/third_party/BlenderProc" \
python third_party/BlenderProc/cli.py download cc_textures \
  /tmp/datasets/ambientcg/cctextures \
  --workers 24
