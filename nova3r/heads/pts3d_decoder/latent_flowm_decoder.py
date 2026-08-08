# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math

from nova3r.layers.attention import Attention
from nova3r.layers.mlp import Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


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


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################


class SiTBlock(nn.Module):
    """
    A standard DiT adaLN-Zero block with one additional cross-attention layer.

    The latent sequence replaces DiT's image-patch sequence. Conditioning
    tokens are consumed only by cross-attention; the timestep drives adaLN.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
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
        # The first six vectors are exactly DiT's adaLN-Zero parameters.  The
        # seventh is a zero-initialized residual gate for cross-attention.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 7 * hidden_size, bias=True)
        )

    def forward(self, x, c, cond):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_cross = self.adaLN_modulation(c).chunk(7, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        cond = self.norm_cond(cond)
        cross, _ = self.cross_attn(
            self.norm_cross(x), cond, cond, need_weights=False
        )
        x = x + gate_cross.unsqueeze(1) * cross
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
    ):
        super().__init__()

        self.learn_sigma = learn_sigma
        self.in_channels = input_size
        self.out_channels = input_size * 2 if learn_sigma else input_size
        # Retained only so older configs continue to load. Latent tokens are
        # never spatially patched.
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_latents = num_latents

        self.x_embedder = nn.Linear(input_size, hidden_size)
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

        self.blocks = nn.ModuleList([SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

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

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

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
            raise ValueError(
                "conditioning_latents must have shape [B, N, C] (or a "
                "flattenable 4D equivalent)"
            )
        latent = self.latent_cond_embedder(latent)
        return torch.cat((cond, latent), dim=1)

    def forward(self, decout, noised_latents=None, timestep=None, query_points=None, **kwargs):
        """
        Forward pass of SiT.
        noised_latents: [B, N, C] latent path sample.
        timestep: [B] or [B, N] flow/diffusion timestep.
        decout[-1]: [B, M, cond_dim] image-condition tokens.
        """
        if noised_latents is None:
            noised_latents = query_points
        if noised_latents is None:
            raise ValueError("SiT requires noised_latents.")
        if timestep is None:
            raise ValueError("SiT requires timestep.")

        x = noised_latents
        t = timestep
        y = decout[-1]

        if t.dim() == 2:
            t = t[:, 0]
        elif t.dim() == 0:
            t = t.reshape(1).expand(x.shape[0])

        if x.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(f"SiT expected {self.pos_embed.shape[1]} latent tokens, got {x.shape[1]}.")

        x = self.x_embedder(x) + self.pos_embed.to(dtype=x.dtype)  # (B, N, D)

        t = self.t_embedder(t)  # (B, D)
        cond = self._condition_tokens(
            y, kwargs.get("conditioning_latents", None), x
        )
        # As in DiT, adaLN is driven by the global conditioning vector. Here
        # that vector is time-only because token conditioning has its own
        # cross-attention path.
        c = t

        for block in self.blocks:
            x = block(x, c, cond)  # (B, N, D)
        x = self.final_layer(x, c)  # (B, N, out_channels)

        if self.learn_sigma:
            x, _ = x.chunk(2, dim=-1)
        return x


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


#################################################################################
#                                   SiT Configs                                  #
#################################################################################


def SiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def SiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def SiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def SiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def SiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def SiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def SiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def SiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def SiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def SiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def SiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def SiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    "SiT-XL/2": SiT_XL_2,
    "SiT-XL/4": SiT_XL_4,
    "SiT-XL/8": SiT_XL_8,
    "SiT-L/2": SiT_L_2,
    "SiT-L/4": SiT_L_4,
    "SiT-L/8": SiT_L_8,
    "SiT-B/2": SiT_B_2,
    "SiT-B/4": SiT_B_4,
    "SiT-B/8": SiT_B_8,
    "SiT-S/2": SiT_S_2,
    "SiT-S/4": SiT_S_4,
    "SiT-S/8": SiT_S_8,
}
