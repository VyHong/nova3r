"""Render completed 512-mesher reconstructions raw and Taubin-smoothed."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Original 1024 sample directory containing view.json')
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', default=['mesh_udf', 'hrudf'])
    parser.add_argument('--comparison-dir', type=Path)
    parser.add_argument('--resolution', type=int, default=512)
    args = parser.parse_args()
    os.chdir(PROJECT)
    source = args.source.resolve()
    root = args.comparison_dir.resolve() if args.comparison_dir else source / 'comparison_512'
    pending = set(args.methods)
    deadline = time.monotonic() + 12 * 3600
    renderer = PROJECT / 'scripts/visualization/render_saved_mesh_view.py'
    while pending:
        for method in sorted(pending.copy()):
            paths = sorted((root / method).glob('nova3r_training/version_*/test/meshes/*/metrics.json'))
            if not paths and (root / method / 'render_input.json').is_file():
                paths = [root / method / 'render_input.json']
            if not paths:
                continue
            record = json.loads(paths[-1].read_text())
            if record.get('status') != 'ok':
                raise RuntimeError(f'{method} reconstruction failed: {record.get("error")}')
            mesh = Path(record['mesh']).resolve()
            smooth = mesh.with_name('test_mesh_render_smoothed.ply')
            print(f'Preparing {method} smoothing', flush=True)
            with (root / f'{method}_smoothing.log').open('w') as log:
                subprocess.run([sys.executable, str(PROJECT / 'scripts/visualization/preprocess_render_mesh.py'),
                                str(mesh), '--output', str(smooth), '--iterations', '30'],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            for label, geometry in [('raw', mesh), ('smoothed', smooth)]:
                output = root / f'{method}_{args.resolution}_{label}.png'
                argv = [str(renderer), str(source / 'view.json'), '--mesh', str(geometry), '--output', str(output)]
                expression = f'import sys,runpy; sys.argv={argv!r}; runpy.run_path({str(renderer)!r},run_name="__main__")'
                print(f'Rendering {output.name}', flush=True)
                with (root / f'{method}_{label}_render.log').open('w') as log:
                    subprocess.run([str(args.blender), '-b', '--threads', '8', '--python-expr', expression],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                if not output.is_file() or output.stat().st_size == 0:
                    raise RuntimeError(f'Render was not saved: {output}')
            pending.remove(method)
            print(f'Finished both {method} images', flush=True)
        if pending:
            if time.monotonic() > deadline:
                raise TimeoutError(f'Waiting for reconstruction: {pending}')
            time.sleep(10)
    print('All comparison images complete', flush=True)

if __name__ == '__main__':
    main()
