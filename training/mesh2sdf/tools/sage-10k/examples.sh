python export_glb.py \
    data/${scene_id}/${layout_id}.json \
    vis/glb/

python camera_traj_gen.py \
    data/${scene_id}/${layout_id}.json \
    --output vis/renders/${scene_id}/${layout_id}.json \
    --render \
    --save_point_cloud

python export_usd.py \
    data/${scene_id}/${layout_id}.json \
    vis/usd/${layout_id}

python load_isaacsim_demo.py \
    data/${scene_id}/${layout_id}.json

# Render 10 resumable, collision-safe equirectangular panoramas for every
# packaged scene. Run this from the kits directory in the nova3r environment.
conda run -n nova3r python render_panoramas.py \
    ../scenes \
    --output-dir ../panoramas \
    --num-panoramas 10
