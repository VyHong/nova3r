"""Check learned-decoder query memory using the full run's conservative chunks."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'training'),str(ROOT)]
from training.mesh2sdf.lightning_module import UDFGeoDecoderAdapter
from training.mesh2sdf.oriented_meshing import OrientedMeshBounds, enclosing_axis_aligned_bounds
from training.mesh2sdf.tools.mesh_udf.mesh_udf import _adaptive_query_batches


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    bundle=torch.load(args.cache,map_location='cpu',weights_only=False)
    bounds=bundle['bounds']
    if isinstance(bounds,OrientedMeshBounds): bounds=enclosing_axis_aligned_bounds(bounds)
    decoder=UDFGeoDecoderAdapter(bundle['geo_decoder'].cuda().float(),bundle['latents'].cuda().float(),bounds)
    torch.set_float32_matmul_precision('medium')
    g=torch.Generator(device='cuda').manual_seed(123)
    points=(torch.rand((65536,3),device='cuda',generator=g)*2-1)*decoder.grid_half_extents
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        def field(a,b):
            with torch.no_grad(): return decoder(points[a:b]).float().cpu()
        for _,_,values in _adaptive_query_batches(65536,65536,field,'cuda','memory probe field'):
            assert torch.isfinite(values).all()
        def gradient(a,b):
            with torch.enable_grad():
                q=points[a:b].detach().requires_grad_(True)
                value=decoder(q)
                return torch.autograd.grad(value.sum(),q)[0].detach().cpu()
        for _,_,values in _adaptive_query_batches(16384,16384,gradient,'cuda','memory probe normals'):
            assert torch.isfinite(values).all()
    torch.cuda.synchronize()
    record=dict(status='ok',device=torch.cuda.get_device_name(),field_query_batch=65536,
         normal_query_batch=16384,vertex_query_batch=32768,
         learned_decoder_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
         gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/1024**3,
         scope='Real cached decoder forward and autograd, using full-run chunk sizes; whole-model memory is additional.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
