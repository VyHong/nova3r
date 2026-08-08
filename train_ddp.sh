#!/bin/bash

#SBATCH --job-name=train_nova3r_ddp
#SBATCH --output=train_ddp.out
#SBATCH --error=train_ddp.err
#SBATCH --gres=gpu:2
#SBATCH --nodes=1

set -euo pipefail

MODE="${1:-train}"
case "$MODE" in
    train)
        ENTRYPOINT="training/train.py"
        DEFAULT_CONFIG="checkpoints/da3_large_hunyuan/.hydra/config.yaml"
        ;;
    resume)
        ENTRYPOINT="training/resume_train.py"
        DEFAULT_CONFIG="checkpoints/hunyuan/.hydra/config.yaml"
        ;;
    *)
        echo "Usage: sbatch train_ddp.sh [train|resume] [config_path]"
        exit 2
        ;;
esac

CONFIG_PATH="${2:-$DEFAULT_CONFIG}"
PROJECT_DIR="/mnt/home/${USER}/projects/nova3r"
CONTAINER_NAME="nova3r-ddp-${SLURM_JOB_ID:-$$}"

cleanup() {
    podman rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if ! podman image exists nova3r; then
    podman import "/mnt/home/${USER}/projects/demo/nova3r.tar" nova3r
fi

podman run \
    -v /mnt:/mnt:rw \
    -v /usr/bin/start-ssh-server:/usr/bin/start-ssh-server:ro \
    -v /etc/ssh/node_rsa_key:/etc/ssh/node_rsa_key \
    -v "/mnt/home/${USER}:/root:rw" \
    -v /tmp:/tmp:rw \
    -v "/mnt/home/${USER}:/workspaces:rw" \
    -w "$PROJECT_DIR" \
    --device=nvidia.com/gpu=all \
    --network=host \
    -e USER=root \
    -e CUDA_VISIBLE_DEVICES \
    --shm-size=1000gb \
    --device=/dev/fuse \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --cap-add=IPC_LOCK \
    --cap-add=DAC_READ_SEARCH \
    --cap-drop=MKNOD \
    --security-opt=apparmor:unconfined \
    --security-opt=seccomp=unconfined \
    --name="$CONTAINER_NAME" \
    -d nova3r sleep infinity

podman exec "$CONTAINER_NAME" nvidia-smi
podman exec "$CONTAINER_NAME" bash -lc "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate nova3r
    python -c 'import torch; count = torch.cuda.device_count(); assert count == 2, count; print(count)'

    bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/sage-10k/udf.zip  /tmp/datasets/sage-10k
    bash bulk_mount_mount_zip.sh /mnt/projects/theses/vyhong/sage-10k/cubemap_data.zip /tmp/datasets/sage-10k -o notrim

    export PYTHONUNBUFFERED=1
    export PYTHONPATH=\"\${PYTHONPATH:-}:$PROJECT_DIR\"
    python -u \"$ENTRYPOINT\" --config \"$CONFIG_PATH\" --devices 2
"
