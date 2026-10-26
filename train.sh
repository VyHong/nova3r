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
    --shm-size=1000gb \
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

  mount_pids=()
  mount_dataset() {
    bash bulk_mount_mount_zip.sh \"\$@\" &
    mount_pids+=(\$!)
  }

  mount_dataset /mnt/projects/theses/vyhong/scannetpp/cubemap_data.zip /tmp/datasets/scannetpp -o precache
  mount_dataset /mnt/projects/theses/vyhong/scannetpp/udf_color.zip /tmp/datasets/scannetpp -o precache
  mount_dataset /mnt/projects/theses/vyhong/scannetpp/dinov2_latents.zip /tmp/datasets/scannetpp -o precache
  mount_dataset /mnt/projects/theses/vyhong/scannetpp/fix_vis_latents.zip /tmp/datasets/scannetpp -o precache
  mount_dataset /mnt/projects/theses/vyhong/scannetpp/fix_gt_latents.zip /tmp/datasets/scannetpp -o precache

  mount_dataset /mnt/projects/theses/vyhong/sage-10k/udf.zip /tmp/datasets/sage-10k
  mount_dataset /mnt/projects/theses/vyhong/sage-10k/cubemap_data.zip /tmp/datasets/sage-10k -o notrim
  mount_dataset /mnt/projects/theses/vyhong/sage-10k/dinov2_latents.zip /tmp/datasets/sage-10k
  mount_dataset /mnt/projects/theses/vyhong/sage-10k/fix_vis_latents.zip /tmp/datasets/sage-10k
  mount_dataset /mnt/projects/theses/vyhong/sage-10k/fix_gt_latents.zip /tmp/datasets/sage-10k

  mount_dataset /mnt/projects/theses/vyhong/3dfront/panorama_scenes.zip /tmp/datasets/3dfront
  mount_dataset /mnt/projects/theses/vyhong/3dfront/better_udf.zip /tmp/datasets/3dfront
  mount_dataset /mnt/projects/theses/vyhong/3dfront/better_gt_latents.zip /tmp/datasets/3dfront
  mount_dataset /mnt/projects/theses/vyhong/3dfront/better_dinov2_latents.zip /tmp/datasets/3dfront
  mount_dataset /mnt/projects/theses/vyhong/3dfront/better_vis_latents.zip /tmp/datasets/3dfront

  # Keep the allocated GPU active while mount-zip populates its caches. This
  # process is stopped before training claims the GPU.
  python -u scripts/training/gpu_keepalive.py --memory-fraction 0.5 &
  keepalive_pid=\$!
  stop_gpu_keepalive() {
    if kill -0 "\$keepalive_pid" 2>/dev/null; then
      kill "\$keepalive_pid"
      wait "\$keepalive_pid" || true
    fi
  }
  trap stop_gpu_keepalive EXIT INT TERM

  mount_failed=0
  for pid in \"\${mount_pids[@]}\"; do
    if ! wait \"\$pid\"; then
      mount_failed=1
    fi
  done
  stop_gpu_keepalive
  trap - EXIT INT TERM

  if [ \"\$mount_failed\" -ne 0 ]; then
    echo 'Dataset mounting failed; aborting training.' >&2
    exit 1
  fi

  
  export PYTHONUNBUFFERED=1
  export PYTHONPATH=\${PYTHONPATH}:/mnt/home/vyhong/projects/nova3r
  
  echo 'Running training...'
  # Resume the full training state from experiment.resume_checkpoint in the config.
  python -u training/resume_train.py \
    --config checkpoints/hunyuan_dit/config_hunyuandit_panorama_visible_latent.yaml
"
training_status=$?

# Stop and remove the container
echo "Cleaning up..."
podman rm -f nova3r-run
exit "$training_status"
