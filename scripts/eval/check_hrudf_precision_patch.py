"""Compare a local patch at the original 768 voxel spacing in BF16 and FP32."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
PROJECT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PROJECT),str(PROJECT/'training'),str(PROJECT/'third_party/hrudf/custom_mc'),str(PROJECT/'third_party/hrudf')]
from core.meshing import compute_pseudo_sdf,mesh_marching_cubes
from core.utils import load_model
p=argparse.ArgumentParser(description=__doc__);p.add_argument('comparison',type=Path);a=p.parse_args()
root=a.comparison.resolve();out=root/'precision_diagnostic'
b=torch.load(out/'udf_decoder.pt',map_location='cuda',weights_only=False)
decoder=b['udf_decoder'];decoder.geo_decoder.float()
torch.set_float32_matmul_precision('highest')
g=torch.Generator(device='cuda').manual_seed(123)
xyz=torch.rand((8192,3),device='cuda',generator=g)*2-1
with torch.no_grad(),torch.autocast('cuda',enabled=False):values=decoder(xyz).reshape(-1)
n=65;h=2/767;extent=(n-1)*h/2
center=xyz[values.argmin()].clamp(-1+extent,1-extent).detach()
report=dict(center=center.cpu().tolist(),original_voxel_spacing=h,patch_resolution=n,patch_half_extent=extent)
for mode in ['bf16','fp32']:
    enabled=mode=='bf16'
    model,cfg=load_model(str(PROJECT/'third_party/hrudf/weights'),'cuda')
    def query(points):
        with torch.enable_grad():
            points=(points.detach()*extent+center).requires_grad_(True)
            v=decoder(points).reshape(-1)
            grad=torch.autograd.grad(v.sum(),points)[0]
            return v.detach().float().cpu()/extent,F.normalize(grad.detach().float(),dim=-1).cpu()
    with torch.autocast('cuda',enabled=enabled,dtype=torch.bfloat16):
        pseudo=compute_pseudo_sdf(model,query,n_grid_samples=n,batch_size_udf=10000,batch_size_pseudosdf=10000,
                num_recursions=5,recursion_type=cfg['recursion_type'],max_avg_distance_factor=1.2,
                max_max_distance_factor=2.0,verbose=False)
    mesh=mesh_marching_cubes(pseudo[-1])
    if mesh is None:
        report[mode]=dict(no_surface=True);continue
    # Edge incidence measures actual mesh openings, not shading artifacts.
    count=np.bincount(mesh.edges_unique_inverse)
    components=mesh.split(only_watertight=False)
    report[mode]=dict(vertices=len(mesh.vertices),faces=len(mesh.faces),boundary_edges=int((count==1).sum()),
                boundary_edge_fraction=float((count==1).mean()),components=len(components))
    mesh.vertices=np.asarray(mesh.vertices)*extent+center.cpu().numpy()
    mesh.export(out/f'patch_{mode}.ply')
    print(mode,report[mode],flush=True)
(out/'patch_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
