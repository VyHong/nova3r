# Preprocessing Runbook

## DA3 missing packages

pip install moviepy==1.0.3
pip install plyfile
pip install pycolmap

nohup python scripts/preprocessing/latents/cubemap_to_vae_latent.py \
  --dataset-root /tmp/datasets/replica_pano \
  --split-list data/replica_pano/val_list.json \
  --vae-checkpoint checkpoints/hunyuan_mini/best_all.ckpt \
  --normalization median \
  --batch-size 4 \
  --output-root datasets/ReplicaPano/replica_pano_vae_latents


python scripts/preprocessing/latents/vae_latent_to_scene.py   /tmp/sage10k_surface_vae_latents/20251220_141038_layout_0aca37b0/00000.pt   --checkpoint exp_output/hunyuan_vae_v2_1_udf/checkpoints/last.ckpt   --config checkpoints/hunyuan/.hydra/config.yaml   --output debug_points/reconstruction.ply   --device cuda   --mc-level 0.5


CUDA_VISIBLE_DEVICES=0,1 python scripts/preprocessing/latents/surface_to_vae_latent.py --output-root /tmp/sage10k_surface_vae_latents_unit_box --batch-size 96 --sample-posterior --normalization unit_box --devices cuda:0 cuda:1 --only-save-latent-before-sampling --split-list data/sage-10k/full_list.json

# dino commands
CUDA_VISIBLE_DEVICES=0 python -u scripts/preprocessing/features/panorama_to_dinov3_features.py \
  /tmp/datasets/sage-10k/panorama-scenes \
  /tmp/datasets/sage-10k/dinov3_panorama_features \
  --batch-size 64 \
  --num-workers 8

pip install "transformers>=4.56.0" torchvision

conda activate nova3r
latent_output=/tmp/sage10k_da3_vae_latents_p5
mkdir -p "$latent_output"
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u scripts/preprocessing/latents/cubemap_to_vae_latent.py \
    --dataset-root /tmp/datasets/sage-10k/cubemap_data \
    --dataset-layout sage10k \
    --da3-model debug_points/cubemap_confidence_sweep/da3_local \
    --vae-checkpoint exp_output/hunyuan_vae_mini/checkpoints/last.ckpt \
    --vae-config checkpoints/hunyuan_mini/.hydra/config.yaml \
    --normalization median \
    --confidence-percentile 5 \
    --batch-size 24 \
    --cpu-threads 8 \
    --num-shards 2 --shard-index "$gpu" \
    --device cuda:0 \
    --output-root "$latent_output" \
    > "$latent_output/gpu${gpu}.log" 2>&1 &
done

pip install "transformers>=4.56.0" torchvision
