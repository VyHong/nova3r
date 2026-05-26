#!/bin/sh
 
#SBATCH --job-name=podman
#SBATCH --output=podman.out
#SBATCH --error=podman.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

# Build image
# - Mount temporary directories for cache and package management
# - Tag the built image as 'example'
# - Specify the Dockerfile location
# - Set the build context directory
# podman build \
#     -v /tmp:/var/cache:rw -v /tmp:/var/lib/apt:rw -v /tmp:/opt/conda/pkgs:rw -v /tmp:/root/.cache:rw \
#     -t example \
#     -f /mnt/general/examples/container/Dockerfile \
#     /mnt/general/examples/container
echo "Loading image..."
podman import ~/projects/demo/nova3r.tar nova3r
# Save image
# rm -f ~/example.tar
# podman save --quiet -o ~/example.tar example

# # Export to enroot
# rm -f example.sqsh
# enroot import -o example.sqsh podman://example

# Load image
# podman image exists example || podman load -i ~/example.tar

# Start container
echo "Starting container..."
podman run -v /mnt:/mnt:rw \
    -v /usr/bin/start-ssh-server:/usr/bin/start-ssh-server:ro \
    -v /etc/ssh/node_rsa_key:/etc/ssh/node_rsa_key \
    -v ~/.ssh/authorized_keys:/root/.ssh/authorized_keys \
    -v /mnt/home/vyhong/.vscode-server-extensions:/root/.vscode-server/extensions \
    -w /mnt/home/${USER}/projects/nova3r \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --cap-add=IPC_LOCK \
    --cap-add=DAC_READ_SEARCH \
    --cap-drop=MKNOD \
    --security-opt=apparmor:unconfined \
    --security-opt=seccomp=unconfined \
    --device=nvidia.com/gpu=all \
    --network=host -e USER=root --replace \
    -e PORT=$(python -c "import random; print(random.randint(20000,30000))") \
    --name=nova3r \
    --shm-size=64gb \
    -e CUDA_VISIBLE_DEVICES=0 \
    --device=/dev/fuse \
    -d \
    nova3r \
    bash -c 'apt update && apt install -y openssh-server && mkdir -p /var/run/sshd && start-ssh-server $PORT && sleep infinity'

sleep 1m

podman logs nova3r

# Run commands inside container
alias RUN='podman exec nova3r'
RUN nvidia-smi
RUN python3 -c 'import torch; print(f"Num GPUs: {torch.cuda.device_count()}")'

sleep infinity

# Commit changes to image
# podman commit example example

# Stop and remove the container
podman rm -f nova3r
