"""Remesh a cached same-latent scene outside Lightning autocast for diagnosis."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
import numpy as np
from omegaconf import OmegaConf
PROJECT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PROJECT/'training'),str(PROJECT)]
from training.mesh2sdf.lightning_module import Mesh2SDFLightningModule
from training.mesh2sdf.module_options import MeshOptions
from training.mesh2sdf.oriented_meshing import clip_mesh_to_bounds

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('comparison',type=Path)
a=p.parse_args()
root=a.comparison.resolve()
cache=root/'precision_diagnostic/udf_decoder.pt'
# Trusted local diagnostic artifact created by diagnose_hrudf_precision.py.
bundle=torch.load(cache,map_location='cpu',weights_only=False)
cfg=OmegaConf.load(root/'config.yaml').cfg
module=SimpleNamespace(cfg=cfg,mesh_options=MeshOptions.from_config(cfg))
decoder=SimpleNamespace(geo_decoder=bundle['udf_decoder'].geo_decoder.to('cuda').float())
latents=bundle['latent_batch'].to('cuda').float()
output=root/'hrudf_fp32';output.mkdir(parents=True,exist_ok=True)
torch.set_float32_matmul_precision('highest')
with torch.autocast('cuda',enabled=False),torch.inference_mode(False),torch.enable_grad():
    mesh=Mesh2SDFLightningModule.mesh_from_latents(module,latents,bundle['sample_idx'],bundle['bounds'],decoder,
                    method='hrudf',artifact_dir=output,world_scale=bundle['world_scale'])
if mesh is None or not len(mesh.faces):raise RuntimeError('FP32 extraction produced no triangles')
if cfg.get('test_mesh_clip_bounds') is not None:
    mesh=clip_mesh_to_bounds(mesh,MeshOptions._validate_bounds(cfg.test_mesh_clip_bounds))
mesh.vertices=np.asarray(mesh.vertices)*float(bundle['world_scale'])
original=next((root/'hrudf').glob('nova3r_training/version_*/test/meshes/*/metrics.json'))
alignment=json.loads(original.read_text())
mesh.vertices=np.asarray(mesh.vertices)*alignment['alignment_scale']+np.asarray(alignment['alignment_shift'])
path=output/'test_mesh.ply';mesh.export(path)
record=dict(status='ok',mesh=str(path),vertices=len(mesh.vertices),faces=len(mesh.faces),
            same_cached_scene=True,precision='float32; autocast disabled; highest matmul precision',
            alignment_reference=str(original),evaluation_recomputed=False,
            note='Original HRUDF alignment reused to isolate geometry precision; no new APD scores computed.')
(output/'render_input.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
