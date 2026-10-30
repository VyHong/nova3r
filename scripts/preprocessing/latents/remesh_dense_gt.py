"""Re-encode a GT surface more densely, preserving its saved camera frame."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from scripts.preprocessing.latents.surface_to_vae_latent import make_surface, encode_surfaces
from scripts.preprocessing.latents.vae_latent_to_scene import load_model, load_latent
from training.mesh2sdf.lightning_module import UDFGeoDecoderAdapter
from training.mesh2sdf.tools.mesh_udf.mesh_udf import get_mesh_udf_fast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path, help='Reference mesh JSON metadata')
    parser.add_argument('--num-points', type=int, default=327680)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    saved = torch.load(reference['latent'], map_location='cpu', weights_only=True)
    model = load_model(Path(reference['checkpoint']), Path(reference['config']), torch.device('cuda'))
    encoder = model.encoder
    if encoder.pc_sharpedge_size != 0 or args.num_points % encoder.num_latents:
        raise ValueError('Requires random-only sampling and a point count divisible by num_latents')
    # Both settings must change: otherwise the encoder subsamples back to its original count.
    encoder.pc_size = args.num_points
    encoder.downsample_ratio = args.num_points // encoder.num_latents
    surface = make_surface(Path(saved['source']), saved['input_transform'].numpy(),
                           random_count=args.num_points, sharp_count=0,
                           point_feats=encoder.point_feats, seed=saved['seed'])
    surface[:, :3] -= saved['room_center']
    surface[:, :3] *= saved['normalization_scale']
    if not saved['cube_center_camera'] or saved['normalization'] != 'cube':
        raise ValueError('This comparison requires camera-centered cube normalization')
    started = time.monotonic()
    torch.manual_seed(42)
    with torch.no_grad():
        _, mean, logvar = encode_surfaces(model, surface.unsqueeze(0).cuda(), sample_posterior=False, only_before_sampling=True)
    prefix = args.reference.parent / f'gt_{args.num_points}pts_sample_seed{reference["seed"]}'
    posterior_path = Path(str(prefix) + '_posterior.pt')
    saved.update(posterior_mean=mean.cpu(), posterior_logvar=logvar.cpu(),
                 num_surface_points=args.num_points, encoder_downsample_ratio=encoder.downsample_ratio)
    torch.save(saved, posterior_path)
    latent = load_latent(posterior_path, model, torch.device('cuda'), saved=saved,
                         sample_posterior=True, seed=reference['seed'])
    latent_path = Path(str(prefix) + '_latent.pt')
    torch.save(dict(saved, latent=latent.cpu()), latent_path)
    print(f'Encoded {args.num_points} points; saved {latent_path}', flush=True)
    with torch.no_grad():
        decoded = model.decode(latent)
    adapter = UDFGeoDecoderAdapter(model.geo_decoder, decoded.detach(), reference['bounds'])
    grid_bounds, resolution = adapter.mesh_udf_grid(reference['max_resolution'])
    _, _, mesh = get_mesh_udf_fast(adapter, None, N_MC=resolution,
        gradient=False, eps=0.005, border_gradients=True, smooth_borders=True,
        surface_threshold=reference['surface_threshold'], grid_bounds=grid_bounds,
        device=latent.device, streamed_grid=True, fast_cleanup=True)
    mesh = adapter.mesh_to_world(mesh)
    output = Path(str(prefix) + f'_meshudf_{reference["max_resolution"]}.ply')
    mesh.export(output)
    metadata = dict(reference, latent=str(posterior_path), sampled_latent=str(latent_path),
                    num_surface_points=args.num_points, encoder_downsample_ratio=encoder.downsample_ratio,
                    smooth_borders=True, streamed_grid=True, fast_cleanup=True,
                    normalization_scale=saved['normalization_scale'], grid_resolution=resolution,
                    vertices=len(mesh.vertices), faces=len(mesh.faces),
                    elapsed_seconds=time.monotonic() - started, output=str(output))
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
