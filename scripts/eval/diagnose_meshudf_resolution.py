"""Compare MeshUDF extraction settings using identical generated scene latents."""
import copy
import json
import os
from pathlib import Path
import sys
import time

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / 'training'), str(PROJECT)]
from training.mesh2sdf.lightning_module import Mesh2SDFLightningModule
from training.mesh2sdf.module_options import MeshOptions

OUT = Path(os.environ['MESH_UDF_DIAGNOSTIC_DIR']).resolve()
OUT.mkdir(parents=True, exist_ok=True)
DEFAULT_VARIANTS = {
    'r1024_padding05_bf16': dict(mesh_mc=1024, mesh_bounds_padding=.05),
    'r1024_padding05_fp32': dict(mesh_mc=1024, mesh_bounds_padding=.05, diagnostic_fp32=True),
    'r1024_padding05_band4': dict(mesh_mc=1024, mesh_bounds_padding=.05, mesh_udf_normal_band_voxels=4.),
}
VARIANTS = json.loads(os.environ['MESH_UDF_DIAGNOSTIC_VARIANTS']) if 'MESH_UDF_DIAGNOSTIC_VARIANTS' in os.environ else DEFAULT_VARIANTS
INDICES = json.loads(Path(os.environ['MESH_UDF_DIAGNOSTIC_INDICES']).read_text()) if os.environ.get('MESH_UDF_DIAGNOSTIC_INDICES') else None
original_step = Mesh2SDFLightningModule.test_step
original_mesh = Mesh2SDFLightningModule.mesh_from_latents
original_dir = Mesh2SDFLightningModule.current_log_dir.fget


def diagnostic_mesh(self, latent_batch, sample_idx, bounds, decoder, **kwargs):
    cache_dir = OUT / 'caches' / self.diagnostic_sample
    if os.environ.get('MESH_UDF_SAVE_GEOMETRY_CACHE', '1') == '1' and not (cache_dir / 'geometry_cache.pt').exists():
        # Local trusted diagnostic cache; geometry-only reruns need no diffusion.
        from omegaconf import OmegaConf
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dict(latents=latent_batch.detach().cpu(), sample_idx=sample_idx,
                        bounds=bounds, world_scale=kwargs.get('world_scale', 1.),
                        geo_decoder=copy.deepcopy(decoder.geo_decoder).cpu(),
                        cfg=OmegaConf.to_container(self.cfg, resolve=True)), cache_dir / 'geometry_cache.pt')
    enabled = bool(self.cfg.get('diagnostic_fp32', False))
    if enabled:
        decoder.geo_decoder.float()
        old_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision('highest')
        try:
            with torch.autocast(device_type=self.device.type, enabled=False):
                return original_mesh(self, latent_batch.float(), sample_idx, bounds, decoder, **kwargs)
        finally:
            torch.set_float32_matmul_precision(old_precision)
    return original_mesh(self, latent_batch, sample_idx, bounds, decoder, **kwargs)


def controlled(self, batch, batch_idx):
    self.diagnostic_sample = self._test_sample_name(batch, 0)
    inference_index = INDICES[self.diagnostic_sample] if INDICES is not None else batch_idx
    original_forward = self.model.lightning_forward
    cached = {}
    generation_seconds = {}
    def reuse(*args, **kwargs):
        steps = int(getattr(self.model, 'flow_matching_steps', 100))
        if steps not in cached:
            if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            cached[steps] = original_forward(*args, **kwargs)
            if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
            generation_seconds[steps] = time.perf_counter() - started
        return cached[steps]
    self.model.lightning_forward = reuse
    paired_dir = OUT / 'paired'
    paired_dir.mkdir(exist_ok=True)
    pair_path = paired_dir / f'{self.diagnostic_sample}.json'
    resume = os.environ.get('MESH_UDF_DIAGNOSTIC_RESUME') == '1'
    records = json.loads(pair_path.read_text())['variants'] if resume and pair_path.exists() else []
    try:
        for name, overrides in VARIANTS.items():
            self.cfg.update(mesh_udf_normal_band_voxels=2., diagnostic_fp32=False,
                            mesh_udf_streamed_grid=True, smooth_borders=False,
                            mesh_udf_reference_resolution=None)
            self.cfg.update(overrides)
            self.model.flow_matching_steps = int(self.cfg.get('test_flow_matching_steps', 100))
            self.mesh_options = MeshOptions.from_config(self.cfg)
            self.diagnostic_dir = str(OUT / name)
            variant_dir = Path(self.diagnostic_dir)
            variant_dir.mkdir(parents=True, exist_ok=True)
            from omegaconf import OmegaConf
            previous_cfg = json.loads((variant_dir/'config.json').read_text()) if (variant_dir/'config.json').exists() else None
            same_config = previous_cfg is not None and all(previous_cfg.get(k) == self.cfg.get(k) for k in (
                'mesh_mc','mesh_bounds_padding','mesh_udf_reference_resolution','mesh_udf_normal_band_voxels',
                'mesh_udf_streamed_grid','smooth_borders','test_flow_matching_steps','resume_checkpoint','test_guidance_scale'))
            previous = next((v for v in records if v['variant']==name), None)
            if resume and same_config and previous and all(
                    r.get('status')=='ok' and Path(r['mesh']).is_file() for r in previous['results']):
                self.test_results.extend(previous['results'])
                print(f'Reusing completed {self.diagnostic_sample}: {name}',flush=True)
                continue
            (variant_dir / 'config.json').write_text(json.dumps(OmegaConf.to_container(self.cfg, resolve=True), indent=2)+'\n')
            print(f'DIAGNOSTIC START {name}', flush=True)
            start = time.perf_counter()
            result = original_step(self, copy.deepcopy(batch), inference_index)
            record = dict(variant=name, steps=self.model.flow_matching_steps,
                          generation_seconds=generation_seconds[self.model.flow_matching_steps],
                          seconds=time.perf_counter()-start, results=result)
            records = [v for v in records if v['variant']!=name] + [record]
            pair_path.write_text(json.dumps(
                dict(sample=self.diagnostic_sample, inference_index=inference_index,
                     conditioning_seed=42+inference_index, diffusion_noise_seed=42,
                     shared_latents_within_step_count=True, variants=records), indent=2) + '\n')
            print(f'DIAGNOSTIC COMPLETE {name}: {json.dumps(result)}', flush=True)
            if any(r.get('status') != 'ok' for r in result):
                raise RuntimeError(f'Diagnostic extraction failed: {name}')
    finally:
        self.model.lightning_forward = original_forward
    return records


Mesh2SDFLightningModule.mesh_from_latents = diagnostic_mesh
Mesh2SDFLightningModule.test_step = controlled
Mesh2SDFLightningModule.current_log_dir = property(lambda self: getattr(self, 'diagnostic_dir', None) or original_dir(self))
from training.test import main
if __name__ == '__main__':
    main()
