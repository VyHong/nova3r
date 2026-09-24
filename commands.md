## load image
podman load -i ~/projects/demo/nova3r_image.tar

## run demo with 6 views
python demo_nova3r.py --images /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0000.jpg /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0001.jpg /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0002.jpg /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0003.jpg /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0004.jpg /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0005.jpg --ckpt checkpoints/scene_n2/checkpoint-last.pth --num_queries 200000

## run scannet demo 
python demo_nova3r.py --images datasets/scannet/116456116b/0_cube_95/0000.jpg datasets/scannet/116456116b/0_cube_95/0001.jpg datasets/scannet/116456116b/0_cube_95/0002.jpg datasets/scannet/116456116b/0_cube_95/0003.jpg datasets/scannet/116456116b/0_cube_95/0004.jpg datasets/scannet/116456116b/0_cube_95/0005.jpg --ckpt checkpoints/first_train/last.ckpt --num_queries 1000000

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
bash bulk_mount_mount_zip.sh /mnt/datasets/3dfront/3D-FRONT-TEST-SCENE.zip  /tmp/datasets/3dfront
bash bulk_mount_mount_zip.sh /mnt/datasets/3dfront/3D-FRONT-SCENE.zip  /tmp/datasets/3dfront

bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/3dfront/3D-FRONT-TEST-SCENE-UDF.zip /tmp/datasets/3dfront
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/3dfront/3D-FRONT-SCENE-UDF.zip /tmp/datasets/3dfront

## mount scannetpp
bash bulk_mount_mount_zip.sh /mnt/datasets/scannetpp/scannetpp_scans.zip  /tmp/datasets/scannetpp
bash bulk_mount_mount_zip.sh /mnt/datasets/scannetpp/scannetpp_panorama.zip /tmp/datasets/scannetpp

bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/scannetpp/cubemap_data.zip  /tmp/datasets/scannetpp
bash bulk_mount_mount_zip.sh /mnt/home/vyhong/projects/nova3r/udf_color.zip  /tmp/datasets/scannetpp

## ScanNet++ DINOv3 panorama features (train + val, two GPUs)
mkdir -p /tmp/datasets/scannetpp/panorama_latents
for split in train val; do
  pids=""
  for gpu in 0 1; do
    CUDA_VISIBLE_DEVICES="$gpu" nohup conda run --no-capture-output -n nova3r \
      python -u tools/cubemap_to_dinov3_features.py \
      --panorama-root /tmp/datasets/scannetpp/scannetpp_panorama \
      --dataset-layout scannetpp \
      --split-list "data/scannetpp/${split}_list.json" \
      --output-root /tmp/datasets/scannetpp/panorama_latents \
      --resolution 512 \
      --batch-size 8 \
      --inference-dtype bfloat16 \
      --save-dtype float16 \
      --num-shards 2 \
      --shard-index "$gpu" \
      > "/tmp/datasets/scannetpp/panorama_latents/${split}_gpu${gpu}.log" 2>&1 &
    pids="$pids $!"
  done
  wait $pids
done

## ScanNet++ PaGeR visible-geometry latents (train, then val)
cd /mnt/home/vyhong/projects/PaGeR
conda run -n nova3r python -m pip install -e .

CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n nova3r \
  python -u tools/infer_sage_vae_latents.py \
  --dataset-layout scannetpp \
  --split-list /mnt/home/vyhong/projects/nova3r/data/scannetpp/train_list.json \
  --data-path /tmp/datasets/scannetpp/scannetpp_panorama \
  --output-root /tmp/datasets/scannetpp/vis_latent \
  --vae-checkpoint /mnt/home/vyhong/projects/nova3r/exp_output/hunyuan_vae/checkpoints/last.ckpt \
  --vae-config /mnt/home/vyhong/projects/nova3r/checkpoints/hunyuan/.hydra/config.yaml \
  --devices 0,1 \
  --batch-size 8 \
  --num-workers 8 \
  --scene-mode indoor

CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n nova3r \
  python -u tools/infer_sage_vae_latents.py \
  --dataset-layout scannetpp \
  --split-list /mnt/home/vyhong/projects/nova3r/data/scannetpp/val_list.json \
  --data-path /tmp/datasets/scannetpp/scannetpp_panorama \
  --output-root /tmp/datasets/scannetpp/vis_latent \
  --vae-checkpoint /mnt/home/vyhong/projects/nova3r/exp_output/hunyuan_vae/checkpoints/last.ckpt \
  --vae-config /mnt/home/vyhong/projects/nova3r/checkpoints/hunyuan/.hydra/config.yaml \
  --devices 0,1 \
  --batch-size 8 \
  --num-workers 8 \
  --scene-mode indoor

cd /mnt/home/vyhong/projects/nova3r

## mount sage-10k
bash bulk_mount_mount_zip.sh /mnt/datasets/sage10k/scenes.zip  /tmp/datasets/sage-10k
bash bulk_mount_mount_zip.sh /mnt/datasets/sage10k/panorama-scenes.zip  /tmp/datasets/sage-10k

bash bulk_mount_mount_zip.sh /mnt/datasets/sage10k/panorama-scenes.zip  /tmp/datasets/sage-10k

## Create this archive after removing the old udf_colored.zip:
# cd /tmp/sage10k_surfaces_only && zip -r /mnt/projects/theses/vyhong/sage-10k/udf_surfaces.zip .
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/sage-10k/surface.zip /tmp/datasets/sage-10k
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/sage-10k/cubemap_data.zip /tmp/datasets/sage-10k -o notrim
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/sage-10k/panorama_latents.zip /tmp/datasets/sage-10k 
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/sage-10k/vis_latent.zip /tmp/datasets/sage-10k 
bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/sage-10k/cube_trans_latent.zip /tmp/datasets/sage-10k -o notrim

## Python Path
export PYTHONPATH=$PYTHONPATH:/absolute/path/to/nova3r

## run training in background
nohup bash projects/nova3r/train.sh &

## tensorboard 
tensorboard --logdir /mnt/home/vyhong/projects/nova3r/exp_output

## eval 
python eval/evaluate_pcd.py --gt_ply /tmp/datasets/replica_pano/office_0_000/office_0_000/office_0_aligned.ply --pred_ply /mnt/home/vyhong/projects/nova3r/exp_output/nova3r_img_cond_finetune_complete/nova3r_training/version_22/val_points/epoch_10/pointcloud.ply --output_file

## copy scene ply
cp /tmp/datasets/replica_pano/large_apartment_0_001/large_apartment_0_001/large_apartment_0_cropped.ply ./debug_points

# Point cloud autoencoding from a SCRREAM scene
python demo_nova3r_ae.py   --input_ply  nova3r/debug_points/room_2_aligned.ply --ckpt checkpoints/scene_ae/checkpoint-last.pth  --num_queries 50000

## install codex
curl -fsSL https://chatgpt.com/codex/install.sh | sh
