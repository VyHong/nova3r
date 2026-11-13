"""Standalone worker; stdout is reserved for JSON, model logging uses stderr."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
import traceback


def volfill_points(result, truncation, threshold, image=None, output_dir=None):
    """Use the repository's CLI serialization and visualization conversion verbatim."""
    import tempfile
    from PIL import Image
    from volfill.amodal.inference_latent_visible import _save_result
    from volfill.visualize import load_sample, field_to_pointcloud

    def export(directory):
        _save_result(result, image if image is not None else Image.new('RGB', (1, 1)),
                     Path(directory), truncation)
        field, metadata = load_sample(Path(directory), 'pred')
        return field_to_pointcloud(field, metadata['bbox_min'], metadata['extent_xyz'],
                                   method='threshold', threshold=threshold)
    if output_dir is not None:
        return export(output_dir)
    with tempfile.TemporaryDirectory(prefix='volfill_upstream_export_') as directory:
        return export(directory)


def load_predictor(args):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    settings = json.loads(args.settings)
    sys.path.insert(0, str(Path(args.repo).resolve()))
    if args.model == 'lari':
        from huggingface_hub import hf_hub_download
        from src.lari.model import LaRIModel
        import torchvision.transforms as transforms
        checkpoint = settings['checkpoint']
        if not Path(checkpoint).is_file():
            if checkpoint != 'lari_scene_pointmap.pth':
                raise FileNotFoundError(checkpoint)
            checkpoint = hf_hub_download('ruili3/LaRI', checkpoint)
        # Scene evaluation recipe; raw point maps already use camera XYZ.
        model = LaRIModel(use_pretrained=None, pretrained_path='', num_output_layer=5, head_type='point')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state.get('model', state), strict=True)
        model.to(args.device).eval()
        normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            [0.485,0.456,0.406], [0.229,0.224,0.225])])
        def predict(image, output_dir=None):
            # ERP ablation resizes the full panorama directly to the model input.
            if settings.get("input_projection", "cubemap") == "cubemap" and image.width != image.height:
                raise ValueError('LaRI cubemap adapter requires square faces')
            if settings.get("input_projection") == "erp":
                image = ImageOps.pad(image, (512, 512), method=Image.Resampling.BILINEAR, color=(0, 0, 0))
            else:
                image = image.resize((512,512), Image.Resampling.BILINEAR)
            points = model(normalize(image)[None].to(args.device))['pts3d'][0]
            return points.detach().float().cpu().numpy()
    else:
        runtime = Path(sys.prefix) / 'share' / 'nova3r-baselines' / 'volfill'
        if not runtime.is_dir():
            runtime = Path(__file__).resolve().parents[3] / 'deps' / 'volfill-runtime'
        if runtime.is_dir():
            sys.path.insert(0, str(runtime))
        os.environ.setdefault('SPARSE_ATTN_BACKEND', 'xformers')
        from volfill.amodal.inference_latent_visible import LatentTUDFVisibleInference
        model = LatentTUDFVisibleInference.from_pretrained(
            settings['checkpoint'], device=args.device,
            steps=settings['steps'], cfg_strength=settings['cfg_strength'])
        def predict(image, output_dir=None):
            if settings.get("input_projection") == "erp":
                if image.width != 2 * image.height:
                    raise ValueError("VolFill ERP input must be exactly 2:1 to preserve aspect ratio")
                # Both dimensions are multiples of 14 without changing the 2:1 ratio.
                result = model(image, max_size=504)
            else:
                result = model(image)
            return volfill_points(result, model.truncation_voxels, settings['threshold'], image, output_dir)
    return predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['lari','volfill'], required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--settings', required=True)
    args = parser.parse_args()
    protocol = sys.stdout
    try:
        with redirect_stdout(sys.stderr):
            predict = load_predictor(args)
        print(json.dumps({'ready': True}), file=protocol, flush=True)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({'error': str(error)}), file=protocol, flush=True)
        return
    import numpy as np
    import torch
    from PIL import Image
    for line in sys.stdin:
        try:
            request = json.loads(line)
            outputs = []
            with redirect_stdout(sys.stderr), torch.inference_mode():
                for index, path in enumerate(request['paths']):
                    print(f"[{args.model}] face {index + 1}/{len(request['paths'])}: {path}", file=sys.stderr, flush=True)
                    torch.manual_seed(request['seed'] + index)
                    np.random.seed((request['seed'] + index) % 2**32)
                    with Image.open(path) as opened:
                        raw_dir = (Path(request['raw_output']) / f'face_{index:02d}'
                                   if request.get('raw_output') and args.model == 'volfill' else None)
                        points = predict(opened.convert('RGB'), raw_dir)
                    output = Path(request['output']) / f'{index}.npy'
                    np.save(output, points)
                    outputs.append(str(output))
            print(json.dumps({'outputs': outputs}), file=protocol, flush=True)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({'error': str(error)}), file=protocol, flush=True)


if __name__ == '__main__':
    main()
