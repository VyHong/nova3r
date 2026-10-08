apt update
apt install --reinstall gcc-13 g++-13 libgcc-13-dev
apt install --reinstall libc6-dev -y

cd /mnt/home/vyhong/projects/nova3r/third_party/GenRecon

git submodule update --init --recursive

python -m pip install \
  easydict \
  ninja \
  transformers==4.57.3 \
  lpips \
  kornia \
  timm \
  psutil \
  xformers

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')"

. ./setup.sh \
  --nvdiffrast \
  --nvdiffrec \
  --cumesh \
  --o-voxel \
  --flexgemm

FLASH_ATTN_CUDA_ARCHS="86;90" \
MAX_JOBS=2 \
NVCC_THREADS=1 \
python -m pip install flash-attn==2.7.3 --no-build-isolation

cd /mnt/home/vyhong/projects/nova3r

ATTN_BACKEND=xformers MALLOC_ARENA_MAX=2 \
python -m eval.baselines.run \
  --model genrecon \
  --dataset-layout sage10k \
  --input-projection erp \
  --erp-root /tmp/datasets/sage-10k/rendered \
  --split-list data/sage-10k/test_100_images_seed0.json \
  --pose-root /mnt/projects/theses/vyhong/sage-10k/poses \
  --scene-root /tmp/datasets/sage-10k/udf \
  --genrecon-ss-ckpt third_party/GenRecon/sparse_structure.pt \
  --genrecon-shape-ckpt third_party/GenRecon/shape_slat.pt \
  --genrecon-tex-ckpt third_party/GenRecon/texture_slat.pt \
  --output-dir eval_results/baselines/genrecon/sage10k_test \
  --eval-points 50000 \
  --genrecon-surface-points 50000 \
  --genrecon-proj-batch-voxels 2048 \
  --save-predictions \
  --fail-fast
  
# run all baselines
cd /mnt/home/vyhong/projects/nova3r && \
set -euo pipefail

COMMON_ARGS=(
  --dataset-layout sage10k
  --split-list data/sage-10k/test_25_images_seed0.json
  --cubemap-root /tmp/datasets/sage-10k/cubemap_data/cubemaps
  --erp-root /tmp/datasets/sage-10k/panorama-scenes
  --pose-root /mnt/projects/theses/vyhong/sage-10k/poses
  --scene-root /tmp/datasets/sage-10k/udf_colored
  --eval-points 50000
  --save-predictions
  --fail-fast
)

CUDA_VISIBLE_DEVICES=0 python -m eval.baselines.run \
  "${COMMON_ARGS[@]}" \
  --model da3 \
  --input-projection erp \
  --output-dir eval_results/baselines/da3/sage10k_erp

CUDA_VISIBLE_DEVICES=0 python -m eval.baselines.run \
  "${COMMON_ARGS[@]}" \
  --model da3 \
  --input-projection cubemap \
  --output-dir eval_results/baselines/da3/sage10k_cubemap

ATTN_BACKEND=xformers MALLOC_ARENA_MAX=2 CUDA_VISIBLE_DEVICES=0 \
python -m eval.baselines.run \
  "${COMMON_ARGS[@]}" \
  --model genrecon \
  --input-projection erp \
  --genrecon-ss-ckpt third_party/GenRecon/sparse_structure.pt \
  --genrecon-shape-ckpt third_party/GenRecon/shape_slat.pt \
  --genrecon-tex-ckpt third_party/GenRecon/texture_slat.pt \
  --genrecon-surface-points 50000 \
  --genrecon-proj-batch-voxels 2048 \
  --output-dir eval_results/baselines/genrecon/sage10k_erp

ATTN_BACKEND=xformers MALLOC_ARENA_MAX=2 CUDA_VISIBLE_DEVICES=0 \
python -m eval.baselines.run \
  "${COMMON_ARGS[@]}" \
  --model genrecon \
  --input-projection cubemap \
  --genrecon-ss-ckpt third_party/GenRecon/sparse_structure.pt \
  --genrecon-shape-ckpt third_party/GenRecon/shape_slat.pt \
  --genrecon-tex-ckpt third_party/GenRecon/texture_slat.pt \
  --genrecon-surface-points 50000 \
  --genrecon-proj-batch-voxels 2048 \
  --output-dir eval_results/baselines/genrecon/sage10k_cubemap

CUDA_VISIBLE_DEVICES=0 python -m eval.baselines.run \
  "${COMMON_ARGS[@]}" \
  --model nova3r \
  --input-projection erp \
  --nova3r-checkpoint checkpoints/scene_n2/checkpoint-last.pth \
  --nova3r-config eval/baselines/configs/nova3r_scene_n2_official.yaml \
  --nova3r-num-views 1 \
  --output-dir eval_results/baselines/nova3r/sage10k_erp

CUDA_VISIBLE_DEVICES=0 python -m eval.baselines.run \
  "${COMMON_ARGS[@]}" \
  --model nova3r \
  --input-projection cubemap \
  --nova3r-checkpoint checkpoints/scene_n2/checkpoint-last.pth \
  --nova3r-config eval/baselines/configs/nova3r_scene_n2_official.yaml \
  --nova3r-num-views 2 \
  --output-dir eval_results/baselines/nova3r/sage10k_cubemap

CUDA_VISIBLE_DEVICES=0 python -m eval.baselines.run \
  "${COMMON_ARGS[@]}" \
  --model pager \
  --input-projection erp \
  --pager-checkpoint pager \
  --pager-scene-mode indoor \
  --output-dir eval_results/baselines/pager/sage10k_erp