# da3 missing packages

pip install moviepy==1.0.3
pip install plyfile
pip install pycolmap

nohup python tools/cubemap_to_vae_latent.py \
  --dataset-root /tmp/datasets/replica_pano \
  --split-list data/replica_pano/val_list.json \
  --vae-checkpoint checkpoints/hunyuan_mini/best_all.ckpt \
  --normalization median \
  --batch-size 4 \
  --output-root datasets/ReplicaPano/replica_pano_vae_latents


python tools/vae_latent_to_scene.py   /tmp/sage10k_surface_vae_latents/20251220_141038_layout_0aca37b0/00000.pt   --checkpoint exp_output/hunyuan_vae_v2_1_udf/checkpoints/last.ckpt   --config checkpoints/hunyuan/.hydra/config.yaml   --output debug_points/reconstruction.ply   --device cuda   --mc-level 0.5


python tools/unconditional_dit_noise_sweep.py /tmp/datasets/sage-10k/median_latents/20251220_141038_layout_0aca37b0/00000_before_sampling.pt --posterior-mode sample --posterior-seed 0 --dit-checkpoint exp_output/unconditional_dit_gt_latents/checkpoints/last.ckpt

CUDA_VISIBLE_DEVICES=0,1 python tools/surface_to_vae_latent.py --output-root /tmp/sage10k_surface_vae_latents_unit_box --batch-size 96 --sample-posterior --normalization unit_box --devices cuda:0 cuda:1 --only-save-latent-before-sampling --split-list data/sage-10k/full_list.json

# dino commands
CUDA_VISIBLE_DEVICES=0 python -u tools/cubemap_to_dinov3_features.py \
  --panorama-root /tmp/datasets/sage-10k/panorama-scenes \
  --output-root /tmp/datasets/sage-10k/dinov3_panorama_features \
  --resolution 512 \
  --batch-size 64 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --inference-dtype bfloat16 \
  --save-dtype float16

pip install "transformers>=4.56.0" torchvision

CUDA_VISIBLE_DEVICES=0 conda run -n nova3r \
python tools/cubemap_conditioned_dit_noise_sweep.py \
  /tmp/datasets/sage-10k/median_latents_512/20251220_141038_layout_0aca37b0/00000_before_sampling.pt \
  /tmp/datasets/sage-10k/cubemap_data/cubemaps/20251220_141038_layout_0aca37b0/20251220_141038_layout_0aca37b0_panorama_000_cube_95 \
  --model-checkpoint exp_output/da3_hunyuan_mini_scratch/checkpoints/last_1-e4.ckpt \
  --model-config exp_output/da3_hunyuan_mini_scratch/version_1/hparams.yaml \
  --timesteps 0 0.2 0.4 0.6 0.8 0.9 \
  --reconstruction ode \
  --posterior-mode sample \
  --posterior-seed 0 \
  --noise-seed 0 \
  --output-dir /tmp/conditioned_noise_sweep


conda activate nova3r
latent_output=/tmp/sage10k_da3_vae_latents_p5
mkdir -p "$latent_output"
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u tools/cubemap_to_vae_latent.py \
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
