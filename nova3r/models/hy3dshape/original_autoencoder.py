"""Compatibility wrapper for the original Hunyuan3D-2.1 ShapeVAE behavior."""

import torch

from nova3r.heads.hunyuan_model.autoencoders import ShapeVAE
from nova3r.models.hy3dshape.utils import synchronize_timer


class OriginalHunyuanShapeVAE(ShapeVAE):
    """Decode the signed Hunyuan field without Nova3R's UDF softplus."""

    def latents2mesh(self, latents: torch.FloatTensor, **kwargs):
        with synchronize_timer("Volume decoding"):
            grid_logits = self.volume_decoder(latents, self.geo_decoder, **kwargs)

        with synchronize_timer("Surface extraction"):
            return self.surface_extractor(grid_logits, **kwargs)
