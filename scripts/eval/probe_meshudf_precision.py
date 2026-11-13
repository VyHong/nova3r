"""Measure finite-grid precision on a trusted local geometry cache."""
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'training'), str(ROOT)]
from training.mesh2sdf.lightning_module import UDFGeoDecoderAdapter
from training.mesh2sdf.oriented_meshing import OrientedMeshBounds, enclosing_axis_aligned_bounds

folder = Path(sys.argv[1])
b = torch.load(folder / 'geometry_cache.pt', map_location='cpu', weights_only=False)
bounds = b['bounds']
if isinstance(bounds, OrientedMeshBounds):
    bounds = enclosing_axis_aligned_bounds(bounds)
decoder = UDFGeoDecoderAdapter(b['geo_decoder'].cuda().float(), b['latents'].cuda().float(), bounds)
torch.set_float32_matmul_precision('highest')
g = torch.Generator(device='cuda').manual_seed(123)
xyz = torch.rand((65536, 3), device='cuda', generator=g) * 2 - 1
xyz *= decoder.grid_half_extents
def query(p, mixed):
    with torch.autocast('cuda', enabled=mixed, dtype=torch.bfloat16), torch.enable_grad():
        p = p.detach().requires_grad_(True)
        v = decoder(p).flatten()
        grad = torch.autograd.grad(v.sum(), p)[0]
    return v.detach(), F.normalize(grad.detach(), dim=-1)
v, _ = query(xyz, False)
points = xyz[v.argsort()[:8192]]
report = {}
for resolution in [512, 1024]:
    h = 2 / (resolution - 1)
    shift = points + torch.tensor([h, 0, 0], device='cuda')
    bf, bg = query(points, True)
    fp, fg = query(points, False)
    bf1, _ = query(shift, True)
    fp1, _ = query(shift, False)
    report[resolution] = dict(
        voxel_spacing=h, bf16_equal_adjacent_fraction=float((bf == bf1).float().mean()),
        fp32_equal_adjacent_fraction=float((fp == fp1).float().mean()),
        bf16_distinct_values=bf.unique().numel(), fp32_distinct_values=fp.unique().numel(),
        mean_normal_cosine=float((bg * fg).sum(-1).mean()),
        field_absolute_error_p95=float(torch.quantile((bf - fp).abs(), .95)),
        relative_to_voxel_error_p95=float(torch.quantile((bf - fp).abs(), .95) / h),
    )
(folder / 'precision_probe.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
