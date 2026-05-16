#!/bin/sh
 
#SBATCH --job-name=train_nova3r
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

# Import image
if podman image exists nova3r; then
    echo "Image nova3r already exists, skipping import."
else
    echo "Importing nova3r container from export..."
    podman import /mnt/home/${USER}/projects/demo/nova3r.tar nova3r
fi

# Start container
echo "Starting container..."
podman run -v /mnt:/mnt:rw \
    -v /usr/bin/start-ssh-server:/usr/bin/start-ssh-server:ro \
    -v /etc/ssh/node_rsa_key:/etc/ssh/node_rsa_key \
    -v /mnt/home/${USER}:/root:rw \
    -v /tmp:/tmp:rw \
    -v /mnt/home/${USER}:/workspaces:rw \
    -w /mnt/home/${USER}/projects/nova3r \
    --device=nvidia.com/gpu=all \
    --network=host -e USER=root --replace \
    --shm-size=16gb \
    -e CUDA_VISIBLE_DEVICES=0 \
    --device=/dev/fuse \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --cap-add=IPC_LOCK \
    --cap-add=DAC_READ_SEARCH \
    --cap-drop=MKNOD \
    --security-opt=apparmor:unconfined \
    --security-opt=seccomp=unconfined \
    --name=nova3r-run -d nova3r sleep infinity


# Run commands inside container
RUN() { podman exec nova3r-run "$@"; }
RUN nvidia-smi

echo "Setting up environment and running commands inside container..."
RUN bash -c "

  source /opt/conda/etc/profile.d/conda.sh
  conda activate nova3r || echo 'Failed to activate conda environment'
  python -c 'import torch; print(f\"Num GPUs: {torch.cuda.device_count()}\")'
  
  echo 'Mounting dataset using FUSE...'
  bash bulk_mount.sh /mnt/home/vyhong/projects/nova3r/datasets/ReplicaPano /tmp/datasets/replica_pano
  
  export PYTHONUNBUFFERED=1
  export PYTHONPATH=\${PYTHONPATH}:/mnt/home/vyhong/projects/nova3r
  
  echo 'Running training...'
  python -u training/train.py --ckpt checkpoints/scene_n2/checkpoint-last.pth
"

# Stop and remove the container
echo "Cleaning up..."
podman rm -f nova3r-run
