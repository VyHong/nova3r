pip install moviepy==1.0.3
pip install plyfile
pip install pycolmap
pip install 

  nohup python tools/cubemap_to_vae_latent.py \
    --dataset-root /tmp/datasets/replica_pano \
    --split-list data/replica_pano/val_list.json \
    --vae-checkpoint checkpoints/hunyuan_mini/best_all.ckpt \
    --normalization median \
    --batch-size 4 \
    --output-root datasets/ReplicaPano/replica_pano_vae_latents