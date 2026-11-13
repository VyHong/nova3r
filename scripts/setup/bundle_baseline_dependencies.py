"""Repackage installed baseline additions as offline wheels; never bundle Torch/CUDA.
Run with the working nova3r Python. VolFill supplements must already be installed
under deps/volfill-runtime. Output is staged outside home by default.
"""
import argparse
import json
import platform
import re
from collections import deque
from importlib import metadata
from pathlib import Path, PurePosixPath
from packaging.requirements import Requirement
from packaging.tags import parse_tag, sys_tags
from wheel.wheelfile import WheelFile

ROOT = Path(__file__).resolve().parents[2]
ROOTS = '''ninja easydict transformers lpips kornia timm psutil imageio imageio-ffmpeg
opencv-python-headless trimesh open3d rtree scikit-image matplotlib tqdm pandas
zstandard tensorboard einops addict safetensors huggingface_hub datasets omegaconf
open_clip_torch pytorch360convert plyfile piqp moderngl onnxruntime xformers utils3d
nvdiffrast nvdiffrec_render cumesh flex_gemm o_voxel'''.split()
EXCLUDE = {'torch', 'torchvision', 'torchaudio', 'triton', 'cuda-bindings', 'cuda-pathfinder', 'cuda-toolkit'}
def canonical(s): return re.sub(r'[-_.]+', '-', s).lower()
def excluded(s): return s in EXCLUDE or (s.startswith(('nvidia-', 'cuda-')) and s != 'nvidia-arch')

def make_wheel(dist, output):
    wheel_meta = dist.read_text('WHEEL')
    if not wheel_meta: raise RuntimeError(f'{dist.name} has no wheel metadata')
    tags = [line[5:] for line in wheel_meta.splitlines() if line.startswith('Tag: ')]
    # The first advertised tag is sufficient for this same-platform backup.
    if not tags: raise RuntimeError(f'No compatibility tag for {dist.name}')
    supported=set(sys_tags())
    tags=[tag for tag in tags if parse_tag(tag) & supported]
    if not tags: raise RuntimeError(f'No compatible wheel tag for {dist.name}')
    name = re.sub(r'[-_.]+', '_', dist.name).lower()
    version = dist.version.replace('-', '_')
    destination = output / f'{name}-{version}-{tags[0]}.whl'
    if destination.exists(): return destination.name
    with WheelFile(destination, 'w') as wheel:
        for entry in dist.files or ():
            rel = PurePosixPath(str(entry))
            if rel.name in ('RECORD', 'INSTALLER', 'REQUESTED', 'direct_url.json') or '__pycache__' in rel.parts or rel.suffix == '.pyc': continue
            if '..' in rel.parts:
                # pip regenerates entry-point scripts; unrelated prefix files
                # must not escape the wheel's site-packages install location.
                if rel.parts[:3] == ('..', '..', '..') and len(rel.parts)>3 and rel.parts[3]=='bin': continue
                if rel.parts[:3] == ('..', '..', '..'):
                    source = Path(dist.locate_file(entry))
                    if source.is_file(): wheel.write(source, f'{name}-{version}.data/data/'+str(PurePosixPath(*rel.parts[3:])))
                    continue
                raise RuntimeError(f'Unsupported prefix file in {dist.name}: {entry}')
            if '__editable__' in str(rel) :
                raise RuntimeError(f'{dist.name} references external sources: {entry}')
            source = Path(dist.locate_file(entry))
            archive_rel = str(rel)
            if rel.parts[0].endswith('.dist-info'):
                archive_rel = wheel.dist_info_path + '/' + str(PurePosixPath(*rel.parts[1:]))
            if source.is_file(): wheel.write(source, archive_rel)
    return destination.name

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('/tmp/nova3r-baseline-extras/deps'))
    parser.add_argument('--resume',action='store_true',help='Reuse previously completed wheels')
    args=parser.parse_args();out=args.output
    if out.exists() and not args.resume: raise SystemExit(f'Output already exists: {out}')
    out.mkdir(parents=True,exist_ok=True)
    common={canonical(d.name):d for d in metadata.distributions()}
    supplemental={canonical(d.name):d for d in metadata.distributions(path=[str(ROOT/'deps/volfill-runtime')])}
    if not supplemental: raise SystemExit('Missing installed VolFill supplemental packages')
    queue=deque(ROOTS);selected={};missing=[]
    while queue:
        name=canonical(queue.popleft())
        if name in selected or excluded(name): continue
        dist=common.get(name)
        if dist is None: missing.append(name); continue
        selected[name]=dist
        for raw in dist.requires or []:
            req=Requirement(raw)
            if req.marker is None or req.marker.evaluate({'extra':''}): queue.append(req.name)
    if missing: raise SystemExit(f'Missing installed dependencies: {sorted(set(missing))}')
    import torch, torchvision
    manifest={'cuda':torch.version.cuda, 'python':platform.python_version(), 'platform':platform.machine(),
              'torch':torch.__version__, 'torchvision':torchvision.__version__,
              'excluded_base_stack':sorted(EXCLUDE), 'common':{}, 'volfill':{}}
    for group, distributions in [('common',selected),('volfill',supplemental)]:
        folder=out/'wheels'/group;folder.mkdir(parents=True,exist_ok=True)
        for name,dist in sorted(distributions.items()):
            if excluded(name): continue
            wheel=make_wheel(dist,folder)
            manifest[group][name]={'version':dist.version,'wheel':wheel}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Packaged {len(manifest["common"])} common and {len(manifest["volfill"])} VolFill wheels into {out}',flush=True)
if __name__=='__main__': main()
