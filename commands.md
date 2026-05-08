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



[ Top 5 Memory Consumers ]
<frozen importlib._bootstrap_external>:757: size=13.4 MiB, count=94763, average=148 B
<frozen importlib._bootstrap>:488: size=1020 KiB, count=9896, average=106 B
/opt/conda/envs/nova3r/lib/python3.12/site-packages/networkx/utils/decorators.py:793: size=978 KiB, count=2520, average=397 B
/opt/conda/envs/nova3r/lib/python3.12/site-packages/networkx/utils/decorators.py:783: size=377 KiB, count=3385, average=114 B
/opt/conda/envs/nova3r/lib/python3.12/site-packages/google/protobuf/text_format.py:344: size=291 KiB, count=12, average=24.2 KiB
[ Top 5 Memory Consumers ]
<frozen importlib._bootstrap_external>:757: size=13.9 MiB, count=99675, average=147 B
/opt/conda/envs/nova3r/lib/python3.12/abc.py:123: size=1186 KiB, count=13109, average=93 B
<frozen importlib._bootstrap>:488: size=1020 KiB, count=9887, average=106 B
/opt/conda/envs/nova3r/lib/python3.12/site-packages/networkx/utils/decorators.py:793: size=978 KiB, count=2520, average=397 B
/opt/conda/envs/nova3r/lib/python3.12/site-packages/networkx/utils/decorators.py:783: size=377 KiB, count=3385, average=114 B
