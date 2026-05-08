## load image
podman load -i ~/projects/demo/nova3r.tar

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
podman export -o ~/projects/demo/nova3r.tar demo:v1

## import image
podman import ~/projects/demo/nova3r.tar 
podman tag <> demo:v1

## save image 
podman save -o ~/projects/demo/nova3r.tar demo:v1

## mount replicaPano
bash bulk_mount.sh /mnt/home/vyhong/projects/nova3r/datasets/ReplicaPano /tmp/datasets/replica_pano

## run demo with 6 views
python demo_nova3r.py \
  --images /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0000.jpg \
  /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0001.jpg \
  /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0002.jpg \
  /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0003.jpg \
  /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0004.jpg \
  /tmp/datasets/replica_pano/office_0_000/office_0_000/Scene_Info/00000/rgb_cube_95/0005.jpg \
  --ckpt checkpoints/scene_n2/checkpoint-last.pth

## Python Path
export PYTHONPATH=$PYTHONPATH:/absolute/path/to/nova3r