script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
rendering_dir="$repo_root/scripts/data/sage10k/rendering"

python "$rendering_dir/export_glb.py" \
    data/${scene_id}/${layout_id}.json \
    vis/glb/

python camera_traj_gen.py \
    data/${scene_id}/${layout_id}.json \
    --output vis/renders/${scene_id}/${layout_id}.json \
    --render \
    --save_point_cloud

python "$rendering_dir/export_usd.py" \
    data/${scene_id}/${layout_id}.json \
    vis/usd/${layout_id}

python "$rendering_dir/load_isaacsim_demo.py" \
    data/${scene_id}/${layout_id}.json

# Render 10 resumable, collision-safe equirectangular panoramas for every
# packaged scene. Run this from the kits directory in the nova3r environment.
conda run -n nova3r python "$rendering_dir/render_panoramas.py" \
    ../scenes \
    --output-dir ../panoramas \
    --num-panoramas 10
