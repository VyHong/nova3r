## load image
podman load -i ~/projects/demo/nova3r_image.tar

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
bash bulk_mount.sh /mnt/home/vyhong/projects/nova3r/datasets/ReplicaPano /tmp/datasets/replica_pano

## run demo with 6 views
python demo_nova3r.py --images /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0000.jpg /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0001.jpg /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0002.jpg /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0003.jpg /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0004.jpg /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0005.jpg --ckpt checkpoints/scene_n2/checkpoint-last.pth --num_queries 200000

## run scannet demo 
python demo_nova3r.py --images datasets/scannet/116456116b/0_cube_95/0000.jpg datasets/scannet/116456116b/0_cube_95/0001.jpg datasets/scannet/116456116b/0_cube_95/0002.jpg datasets/scannet/116456116b/0_cube_95/0003.jpg datasets/scannet/116456116b/0_cube_95/0004.jpg datasets/scannet/116456116b/0_cube_95/0005.jpg --ckpt checkpoints/first_train/last.ckpt --num_queries 1000000

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
python demo_nova3r_ae.py   --input_ply  debug_points/hotel_0_0_small/hotel_0_cropped.ply --ckpt checkpoints/scene_ae/checkpoint-last.pth  --num_queries 50000