# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from nova3r.layers.attention import Attention
from nova3r.layers.mlp import Mlp
from nova3r.flow_matching.path.scheduler import LinearScheduler
from nova3r.flow_matching.path import AffineProbPath
from nova3r.flow_matching.solver import ODESolver


def resolve_precomputed_gt_latents(
    batch,
    *,
    expected_shape=None,
    device,
    dtype,
    sample_posterior=True,
    deterministic_seed=None,
):
    """Load a fixed target or sample a precomputed posterior on-device."""
    fixed = batch.get("gt_latents")
    mean = batch.get("gt_latent_posterior_mean")
    logvar = batch.get("gt_latent_posterior_logvar")
    has_posterior = mean is not None or logvar is not None
    if has_posterior:
        from nova3r.heads.hunyuan_model.autoencoders.model import (
            DiagonalGaussianDistribution,
        )

        mean = mean.to(device=device, dtype=dtype, non_blocking=True)
        logvar = logvar.to(device=device, dtype=dtype, non_blocking=True)
        posterior = DiagonalGaussianDistribution([mean, logvar], feat_dim=-1)
        if sample_posterior == "mode":
            latent = posterior.mode()
        else:
            generator = None
            posterior_seed = int(deterministic_seed) + 1 if deterministic_seed is not None else 42 if sample_posterior is False else None
            if posterior_seed is not None:
                generator = torch.Generator(device=device)
                generator.manual_seed(int(posterior_seed))
            latent = posterior.sample(generator=generator)
    elif fixed is not None:
        latent = fixed.to(device=device, dtype=dtype, non_blocking=True)
    else:
        return None

    if expected_shape is not None and (latent.ndim != 3 or tuple(latent.shape[1:]) != tuple(expected_shape)):
        raise ValueError(f"Expected GT latents [B, {expected_shape[0]}, {expected_shape[1]}], " f"got {tuple(latent.shape)}")
    if not torch.isfinite(latent).all():
        raise ValueError("Precomputed GT latents contain non-finite values")
    return latent.contiguous()


def sample_t(
    batch_size: int,
    device,
    dtype=torch.float32,
    mu: float = -0.8,
    sigma: float = 0.8,
    p_zero: float = 0.1,
    generator=None,
):
    """Sample PointDiT-style logit-normal timesteps with exact noise endpoints."""
    z = torch.randn(
        batch_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    t = torch.sigmoid(mu + sigma * z)

    if p_zero > 0:
        zero_mask = (
            torch.rand(
                batch_size,
                device=device,
                generator=generator,
            )
            < p_zero
        )
        t = t.masked_fill(zero_mask, 0.0)
    return t


def sample_linear_flow_path(
    gt_latents,
    deterministic_seed=None,
    *,
    timestep_sampling="uniform",
    timestep_mu=-0.8,
    timestep_sigma=0.8,
    timestep_p_zero=0.1,
):
    """Sample the noise, timestep, and target for the linear flow path."""
    generator = None
    if deterministic_seed is not None:
        generator = torch.Generator(device=gt_latents.device)
        generator.manual_seed(int(deterministic_seed))

    x_0 = torch.randn(
        gt_latents.shape,
        dtype=gt_latents.dtype,
        device=gt_latents.device,
        generator=generator,
    )
    if timestep_sampling == "uniform":
        t = torch.rand(
            gt_latents.shape[0],
            device=gt_latents.device,
            generator=generator,
        )
    elif timestep_sampling == "pointdit":
        t = sample_t(
            gt_latents.shape[0],
            device=gt_latents.device,
            mu=timestep_mu,
            sigma=timestep_sigma,
            p_zero=timestep_p_zero,
            generator=generator,
        )
    sample = AffineProbPath(scheduler=LinearScheduler()).sample(
        x_0=x_0,
        x_1=gt_latents,
        t=t,
    )
    return sample.x_t, sample.dx_t, t


def predict_clean_linear_path_latents(x_t, velocity, t):
    """Convert a linear-path velocity prediction into its clean endpoint."""
    t = t.reshape(t.shape[0], *([1] * (x_t.ndim - 1)))
    return x_t + (1.0 - t) * velocity


def flow_matching_losses(
    v_pred,
    target_velocity,
    x_t,
    t,
    target_latents,
    reconstruction_weight,
):
    """Compute the shared velocity and clean-latent flow-matching losses."""
    velocity_loss = F.mse_loss(v_pred, target_velocity)
    pred_latents = predict_clean_linear_path_latents(x_t, v_pred, t)
    reconstruction_loss = F.mse_loss(pred_latents, target_latents)
    return (
        velocity_loss + reconstruction_weight * reconstruction_loss,
        velocity_loss,
        reconstruction_loss,
        pred_latents,
    )


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core SiT Model                                #
#################################################################################


class SiTBlock(nn.Module):
    """
    A standard DiT adaLN-Zero block with one additional cross-attention layer.

    The latent sequence replaces DiT's image-patch sequence. Conditioning
    tokens are consumed only by cross-attention; the timestep drives adaLN.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        # The seventh vector is the zero-initialized cross-attention gate. It is
        # retained during unconditional pretraining for exact checkpoint
        # compatibility with subsequent conditional training.
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 7 * hidden_size, bias=True))

    def forward(self, x, c, cond=None, use_cross_attention=True):
        modulation = self.adaLN_modulation(c)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            gate_cross,
        ) = modulation.chunk(7, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if use_cross_attention:
            cond = self.norm_cond(cond)
            cross, _ = self.cross_attn(self.norm_cross(x), cond, cond, need_weights=False)
            x = x + cross #* gate_cross.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    DiT-style flow model over latent tokens instead of image patches.

    Relative to standard DiT, ``x_embedder`` is a Linear layer over each
    latent vector and every block has cross-attention over condition tokens.
    A visible-geometry latent can also be injected as a zero-initialized
    residual into the noised latent before that input embedding.
    There is deliberately no patchify/unpatchify path.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        num_latents=512,
        cond_dim=3072,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        latent_cond_dim=64,
        unconditional_training=False,
        timestep_sampling="uniform",
        timestep_mu=-0.8,
        timestep_sigma=0.8,
        timestep_p_zero=0.1,
        visible_geometry_dim=None,
    ):
        super().__init__()

        self.unconditional_training = bool(unconditional_training)
        self.timestep_sampling = str(timestep_sampling)
        self.timestep_mu = float(timestep_mu)
        self.timestep_sigma = float(timestep_sigma)
        self.timestep_p_zero = float(timestep_p_zero)
        self.supports_mesh_decode = False
        self.learn_sigma = learn_sigma
        self.in_channels = input_size
        self.out_channels = input_size * 2 if learn_sigma else input_size
        # Accepted only so older configs continue to load; latents are not patched.
        del patch_size
        self.num_heads = num_heads
        self.num_latents = num_latents

        self.x_embedder = nn.Linear(input_size, hidden_size)
        visible_geometry_dim = input_size if visible_geometry_dim is None else visible_geometry_dim
        self.visible_geometry_embedder = nn.Linear(visible_geometry_dim, input_size)
        self.latent_cond_embedder = nn.Sequential(
            nn.LayerNorm(latent_cond_dim),
            nn.Linear(latent_cond_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Linear(cond_dim, hidden_size)
        # Fixed 1D sin-cos embedding over latent-token index.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_latents, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList(
            [
                SiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.latent_reconstruction_loss_weight = 0.1
        self.initialize_weights()
        if self.unconditional_training:
            self._freeze_conditioning_branch()

    def _freeze_conditioning_branch(self):
        """Keep conditional checkpoint keys without training the unused branch."""
        self.y_embedder.requires_grad_(False)
        self.latent_cond_embedder.requires_grad_(False)
        for block in self.blocks:
            block.norm_cond.requires_grad_(False)
            block.cross_attn.requires_grad_(False)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Standalone DiT Lightning checkpoints may also contain the external,
        # frozen validation decoder under ``first_stage.*``. That decoder is
        # reloaded from vae_ckpt by the Lightning module and is not part of the
        # DiT itself.
        kwargs.pop("stage", None)
        kwargs.pop("aggregator_ckpt", None)
        kwargs.pop("vae_ckpt", None)
        state_dict = {key: value for key, value in state_dict.items() if not key.startswith("first_stage.")}
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Parent modules load children through this hook, bypassing the child's
        # load_state_dict override. Support both standalone and embedded DiTs.
        weight_key = prefix + "visible_geometry_embedder.weight"
        bias_key = prefix + "visible_geometry_embedder.bias"
        if weight_key not in state_dict and bias_key not in state_dict:
            # Legacy checkpoints have no visible-geometry branch. Reset it to
            # zero even when loading into a previously trained model.
            state_dict[weight_key] = torch.zeros_like(self.visible_geometry_embedder.weight)
            state_dict[bias_key] = torch.zeros_like(self.visible_geometry_embedder.bias)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], np.arange(self.num_latents, dtype=np.float32))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Preserve the pretrained noised-latent input exactly until this branch
        # learns a useful residual.
        nn.init.constant_(self.visible_geometry_embedder.weight, 0)
        nn.init.constant_(self.visible_geometry_embedder.bias, 0)

    def _condition_tokens(self, image_tokens, conditioning_latents, x):
        image_tokens = image_tokens.to(device=x.device, dtype=x.dtype)
        if image_tokens.dim() == 4:
            image_tokens = image_tokens.flatten(1, 2)
        elif image_tokens.dim() == 2:
            image_tokens = image_tokens[:, None, :]
        if image_tokens.dim() != 3:
            raise ValueError("condition tokens must have shape [B, M, C]")
        cond = self.y_embedder(image_tokens)
        if conditioning_latents is None:
            return cond

        latent = conditioning_latents.to(device=x.device, dtype=x.dtype)
        if latent.dim() == 4:
            latent = latent.flatten(1, 2)
        elif latent.dim() == 2:
            latent = latent[:, None, :]
        if latent.dim() != 3:
            raise ValueError("conditioning_latents must have shape [B, N, C] (or a " "flattenable 4D equivalent)")
        latent = self.latent_cond_embedder(latent)
        return torch.cat((cond, latent), dim=1)

    def forward(
        self,
        decout=None,
        noised_latents=None,
        timestep=None,
        query_points=None,
        **kwargs,
    ):
        """
        Forward pass of SiT.
        noised_latents: [B, N, C] latent path sample.
        visible_geometry_latents: [B, N, C_visible] visible-geometry latent.
        timestep: [B] or [B, N] flow/diffusion timestep.
        decout[-1]: [B, M, cond_dim] image-condition tokens.
        """
        if noised_latents is None:
            noised_latents = query_points

        x = noised_latents
        visible_geometry_latents = kwargs.get("visible_geometry_latents")
        if visible_geometry_latents is not None:
            visible_geometry_latents = visible_geometry_latents.to(
                device=x.device, dtype=x.dtype
            )
            if visible_geometry_latents.shape[:2] != x.shape[:2]:
                raise ValueError(
                    "visible_geometry_latents must have the same batch and token "
                    f"dimensions as noised_latents; got {tuple(visible_geometry_latents.shape)} "
                    f"and {tuple(x.shape)}"
                )
            x = x + self.visible_geometry_embedder(visible_geometry_latents)
        t = timestep
        if t.dim() == 2:
            t = t[:, 0]
        elif t.dim() == 0:
            t = t.reshape(1).expand(x.shape[0])

        if x.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(f"SiT expected {self.pos_embed.shape[1]} latent tokens, got {x.shape[1]}.")

        x = self.x_embedder(x) + self.pos_embed.to(dtype=x.dtype)  # (B, N, D)

        t = self.t_embedder(t)  # (B, D)
        use_conditioning = not self.unconditional_training
        cond = None
        if use_conditioning:
            if decout is None or len(decout) == 0:
                raise ValueError("Conditioned DiT requires condition tokens in decout.")
            cond = self._condition_tokens(decout[-1], kwargs.get("conditioning_latents", None), x)
        # As in DiT, adaLN is driven by the global conditioning vector. Here
        # that vector is time-only because token conditioning has its own
        # cross-attention path.
        c = t

        for block in self.blocks:
            x = block(
                x,
                c,
                cond,
                use_cross_attention=use_conditioning,
            )  # (B, N, D)
        x = self.final_layer(x, c)  # (B, N, out_channels)

        if self.learn_sigma:
            x, _ = x.chunk(2, dim=-1)
        return x

    def _prepare_gt_latents(self, batch, deterministic_seed=None):
        if not self.unconditional_training:
            raise RuntimeError(
                "Standalone GT-latent training requires "
                "unconditional_training=True. Use Nova3rImgCond.lightning_forward "
                "for normal conditioned training."
            )
        parameter = self.x_embedder.weight
        gt_latents = resolve_precomputed_gt_latents(
            batch,
            expected_shape=(self.num_latents, self.in_channels),
            device=parameter.device,
            dtype=parameter.dtype,
            sample_posterior=getattr(self, "sample_posterior", True),
            deterministic_seed=deterministic_seed,
        )
        if gt_latents is None:
            raise KeyError(
                "Unconditional DiT training requires fixed GT latents or a "
                "precomputed GT posterior. Enable load_gt_latent in the "
                "dataset config."
            )
        return gt_latents

    def _flow_path_sample(self, gt_latents, deterministic_seed):
        return sample_linear_flow_path(
            gt_latents,
            deterministic_seed,
            timestep_sampling=self.timestep_sampling,
            timestep_mu=self.timestep_mu,
            timestep_sigma=self.timestep_sigma,
            timestep_p_zero=self.timestep_p_zero,
        )

    def _sample_unconditional_latents(self, gt_latents, deterministic_seed):
        generator = torch.Generator(device=gt_latents.device)
        generator.manual_seed(42 if deterministic_seed is None else int(deterministic_seed))
        x_init = torch.randn(
            gt_latents.shape,
            dtype=gt_latents.dtype,
            device=gt_latents.device,
            generator=generator,
        )
        solver = ODESolver(velocity_model=_UnconditionalDiTVelocity(self))
        return solver.sample(
            x_init=x_init,
            time_grid=torch.linspace(0, 1, 101, device=gt_latents.device),
            method="euler",
            step_size=0.01,
            return_intermediates=False,
        )

    def lightning_forward(
        self,
        batch,
        criterion=None,
        kl_weight=0.0,
        recon_latents=False,
        deterministic_seed=None,
    ):
        """Match Nova3rImgCond's Lightning contract for GT-latent training."""
        del criterion, kl_weight
        gt_latents = self._prepare_gt_latents(batch, deterministic_seed)
        x_t, dx_t, t = self._flow_path_sample(gt_latents, deterministic_seed)
        v_pred = self(noised_latents=x_t, timestep=t)

        (
            loss,
            velocity_loss,
            latent_reconstruction_loss,
            pred_latents,
        ) = flow_matching_losses(
            v_pred,
            dx_t,
            x_t,
            t,
            gt_latents,
            self.latent_reconstruction_loss_weight,
        )

        # Retain per-sample errors so the Lightning module can report where
        # along the flow path the model is struggling. These are aggregated by
        # sample count across batches/ranks rather than averaged per batch.
        reduce_dims = tuple(range(1, v_pred.ndim))
        velocity_mse_per_sample = (v_pred - dx_t).square().mean(dim=reduce_dims)
        reconstruction_mse_per_sample = (pred_latents - gt_latents).square().mean(dim=reduce_dims)
        details = {
            "velocity_loss": float(velocity_loss.detach()),
            "latent_reconstruction_loss": float(latent_reconstruction_loss.detach()),
            "latent_reconstruction_loss_weight": self.latent_reconstruction_loss_weight,
            "timestep_bucket_samples": {
                "t": t.detach(),
                "velocity_mse": velocity_mse_per_sample.detach(),
                "reconstruction_mse": reconstruction_mse_per_sample.detach(),
            },
        }

        latents = None
        if recon_latents:
            with torch.no_grad():
                latents = self._sample_unconditional_latents(
                    gt_latents,
                    deterministic_seed,
                )
        return loss, details, latents


class _UnconditionalDiTVelocity(nn.Module):
    """Adapt an unconditional DiT to ODESolver's ``forward(x, t)`` API."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, **kwargs):
        del kwargs
        return self.model(noised_latents=x, timestep=t)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, positions):
    return get_1d_sincos_pos_embed_from_grid(embed_dim, positions)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
