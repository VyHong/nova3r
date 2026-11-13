"""Exercise a full 1024-cubed streamed grid and automatic native MC tiling."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'training'),str(ROOT)]
from training.mesh2sdf.tools.mesh_udf.mesh_udf import get_mesh_udf_fast


class Plane(torch.nn.Module):
    def forward(self, xyz):
        return (xyz[:, :1]-.137).abs()+.00002


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    start=time.perf_counter()
    _,_,mesh=get_mesh_udf_fast(Plane(),None,N_MC=1024,gradient=False,device='cpu',
          streamed_grid=True,query_batch=65536,vertex_max_batch=32768,
          surface_threshold=.01,reference_resolution=512)
    assert mesh is not None and len(mesh.faces)>0
    assert np.isfinite(mesh.vertices).all()
    assert mesh.faces.min()>=0 and mesh.faces.max()<len(mesh.vertices)
    record=dict(status='ok',resolution=[1024,1024,1024],grid_points=1024**3,
          automatic_native_tile_cells=128,vertices=len(mesh.vertices),faces=len(mesh.faces),
          peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
          seconds=time.perf_counter()-start,
          scope='Worst-size dense grid and native tiling; analytic field, not a learned-scene quality test.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
