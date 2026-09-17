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
  psutil
  xformers

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')"

. ./setup.sh \
  --nvdiffrast \
  --nvdiffrec \
  --cumesh \
  --o-voxel \
  --flexgemm
  --flash-attn \

FLASH_ATTN_CUDA_ARCHS="86;90" \
MAX_JOBS=2 \
NVCC_THREADS=1 \
pip install flash-attn --no-build-isolation 

ATTN_BACKEND=xformers MALLOC_ARENA_MAX=2 \
python -m eval.baselines.run \
  --model genrecon \
  --split-list data/sage-10k/test_100_images_seed0.json \
  --genrecon-ss-ckpt third_party/GenRecon/sparse_structure.pt \
  --genrecon-shape-ckpt third_party/GenRecon/shape_slat.pt \
  --genrecon-tex-ckpt third_party/GenRecon/texture_slat.pt \
  --output-dir eval_results/baselines/genrecon/sage10k_test \
  --eval-points 50000 \
  --genrecon-surface-points 50000 \
  --genrecon-proj-batch-voxels 2048 \
  --fail-fast