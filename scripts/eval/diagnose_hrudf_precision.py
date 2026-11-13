"""Probe meshing precision using the identical reconstruction latents and bounds."""
import json
import os
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
PROJECT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PROJECT/'training'),str(PROJECT)]
import training.mesh2sdf.tools.hrudf as adapter
OUT=Path(os.environ['HRUDF_DIAGNOSTIC_DIR'])
OUT.mkdir(parents=True,exist_ok=True)

def probe(decoder, **kwargs):
    result=dict(autocast_enabled=torch.is_autocast_enabled('cuda'),
                autocast_dtype=str(torch.get_autocast_dtype('cuda')),
                parameter_dtype=str(next(decoder.geo_decoder.parameters()).dtype),
                resolution=kwargs['resolution'])
    device=decoder.latent.device
    h=2/(kwargs['resolution']-1)
    generator=torch.Generator(device=device).manual_seed(123)
    xyz=torch.rand((32768,3),generator=generator,device=device)*2-1
    def query(points, enabled):
        with torch.inference_mode(False),torch.enable_grad(),torch.autocast('cuda',enabled=enabled,dtype=torch.bfloat16):
            points=points.detach().clone().requires_grad_(True)
            value=decoder(points).reshape(-1)
            grad=torch.autograd.grad(value.sum(),points)[0]
        return value.detach().float(), F.normalize(grad.detach().float(),dim=-1)
    bf,bg=query(xyz,True)
    fp,fg=query(xyz,False)
    selected=torch.argsort(fp)[:2048]
    base=xyz[selected]
    shifted=base+torch.tensor([h,0,0],device=device)
    bf0,bg0=query(base,True); bf1,_=query(shifted,True)
    fp0,fg0=query(base,False); fp1,_=query(shifted,False)
    result.update(value_mean_abs_error=float((bf-fp).abs().mean()),
                  value_p95_abs_error=float(torch.quantile((bf-fp).abs(),.95)),
                  mean_gradient_cosine=float((bg0*fg0).sum(-1).mean()),
                  bf16_adjacent_equal_fraction=float((bf0==bf1).float().mean()),
                  fp32_adjacent_equal_fraction=float((fp0==fp1).float().mean()),
                  bf16_distinct_values=int(bf0.unique().numel()),fp32_distinct_values=int(fp0.unique().numel()),
                  voxel_spacing=h)
    # Match HRUDF's corner order in core.utils.df_and_grad_to_input_cells.
    corners=torch.tensor([[0,0,0],[0,0,1],[0,1,1],[0,1,0],[1,0,0],[1,0,1],[1,1,1],[1,1,0]],device=device)
    pts=(base[:512,None,:]+corners[None,:,:]*h).reshape(-1,3)
    bfv,bfg=query(pts,True);fpv,fpg=query(pts,False)
    hrroot=PROJECT/'third_party/hrudf'
    sys.path[:0]=[str(hrroot/'custom_mc'),str(hrroot)]
    from core.utils import load_model
    model,_=load_model(str(PROJECT/'third_party/hrudf/weights'),str(device))
    def make_inputs(v,g):
        return torch.cat((v.reshape(-1,8)/h,g.reshape(-1,24),torch.zeros((512,896),device=device)),dim=1)
    with torch.no_grad():
        with torch.autocast('cuda',enabled=False):
            reference=model(make_inputs(fpv,fpg)).argmax(-1)
            bf_input_only=model(make_inputs(bfv,bfg)).argmax(-1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            actual=model(make_inputs(bfv,bfg)).argmax(-1)
    result['sign_code_disagreement_bf16_field_only']=float((bf_input_only!=reference).float().mean())
    result['sign_code_disagreement_bf16_field_and_classifier']=float((actual!=reference).float().mean())
    (OUT/'precision_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    decoder.cpu()
    for key in ['latent','center','half_extent','grid_half_extents']:
        setattr(decoder,key,getattr(decoder,key).cpu())
    context['latent_batch']=context['latent_batch'].detach().cpu()
    torch.save(dict(udf_decoder=decoder,**context),OUT/'udf_decoder.pt')
    print('Saved same-latent UDF decoder for targeted remeshing',flush=True)
    raise SystemExit(0)

adapter.extract_hrudf=probe
context={}
from training.mesh2sdf.lightning_module import Mesh2SDFLightningModule
original_mesh=Mesh2SDFLightningModule.mesh_from_latents
def capture(self,latent_batch,sample_idx,bounds,decoder,**kwargs):
    context.update(latent_batch=latent_batch,sample_idx=sample_idx,bounds=bounds,world_scale=kwargs.get('world_scale',1.0))
    return original_mesh(self,latent_batch,sample_idx,bounds,decoder,**kwargs)
Mesh2SDFLightningModule.mesh_from_latents=capture
from training.test import main
if __name__=='__main__': main()
