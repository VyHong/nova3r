# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import math
from typing import Callable, List, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange

from depth_anything_3.utils.logger import logger

from .layers import LayerScale  # noqa: F401
from .layers import Mlp  # noqa: F401
from .layers import (  # noqa: F401
    Block,
    PatchEmbed,
    PositionGetter,
    RotaryPositionEmbedding2D,
    SwiGLUFFNFused,
)
from depth_anything_3.model.reference_view_selector import (
    RefViewStrategy,
    select_reference_view,
    reorder_by_reference,
    restore_original_order,
)
from depth_anything_3.utils.constants import THRESH_FOR_REF_SELECTION
from omegaconf import OmegaConf

from nova3r.layers.patch_embed import Token3DEmbedMLP

# logger = logging.getLogger("dinov2")


def _wrap_cfg(cfg_obj):
    return OmegaConf.create(cfg_obj)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=1.0,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        alt_start=-1,
        qknorm_start=-1,
        rope_start=-1,
        rope_freq=100,
        plus_cam_token=False,
        cat_token=True,
        token_3d=None,
        add_camera_token_3d=False,
        pos_3d_embed_type=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating
                positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating
                positional embeddings
        """
        super().__init__()
        self.patch_start_idx = 1
        norm_layer = nn.LayerNorm
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.alt_start != -1:
            self.camera_token = nn.Parameter(torch.randn(1, 2, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        if self.rope_start != -1:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
            self.position_getter = PositionGetter() if self.rope is not None else None
        else:
            self.rope = None
        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                qk_norm=i >= qknorm_start if qknorm_start != -1 else False,
                rope=self.rope if i >= rope_start and rope_start != -1 else None,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)

        self.__build_3d_tokens__(**_wrap_cfg(token_3d))

        self.pos_3d_embed_type = pos_3d_embed_type
        self.add_camera_token_3d = add_camera_token_3d
        if self.add_camera_token_3d == "new":
            self.camera_token_3d = nn.Parameter(torch.randn(1, 1, embed_dim))
            nn.init.normal_(self.camera_token_3d, std=1e-6)

    def __build_3d_tokens__(self, num_3d_tokens, token_dim_3d, embed_dim, token_3d_embed_config=None):
        self.num_3d_tokens = num_3d_tokens
        self.token_dim_3d = token_dim_3d
        self.embed_dim = embed_dim

        if embed_dim == token_dim_3d:
            self.pts3d_token = nn.Parameter(torch.randn(1, num_3d_tokens, embed_dim))
            nn.init.trunc_normal_(self.pts3d_token, std=0.02)
        else:
            # use a MLP to project the 3D tokens to the same dimension as the patch tokens
            self.pts3d_token = nn.Parameter(torch.randn(1, num_3d_tokens, token_dim_3d))
            nn.init.trunc_normal_(self.pts3d_token, std=0.02)
            self.pts3d_token_proj = Mlp(
                in_features=token_dim_3d,
                hidden_features=embed_dim,
                out_features=embed_dim,
                act_layer=nn.GELU,
                drop=0.0,
                bias=False,
            )

        # Register gradient hook to monitor and normalize gradients on pts3d_token
        def pts3d_token_grad_hook(grad):
            # Check for NaN/Inf in gradients
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                print(f"[CRITICAL] NaN/Inf detected in pts3d_token gradients!")
                print(f"  NaN count: {torch.isnan(grad).sum().item()}")
                print(f"  Inf count: {torch.isinf(grad).sum().item()}")
                # Replace NaN/Inf with zeros to prevent parameter corruption
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"  [WARNING] Replaced NaN/Inf gradients with zeros")

            # Calculate gradient norm and analyze
            grad_norm = torch.norm(grad)

            # ADAPTIVE GRADIENT SCALING
            max_norm = 10.0

            if grad_norm > max_norm:
                # Normalize gradient to max_norm
                scale_factor = max_norm / (grad_norm + 1e-8)
                grad = grad * scale_factor

            return grad

        # Gradient hooks disabled - they can break autograd graph and cause NaN
        # Let the training loop handle gradient clipping instead

        # self.pts3d_token.register_hook(pts3d_token_grad_hook)

        if token_3d_embed_config is not None:
            self.token_3d_embed = eval(token_3d_embed_config.name)(**token_3d_embed_config.params)
        else:
            self.token_3d_embed = None

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the
            # interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using
            # both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_cls_token(self, B, S):
        cls_token = self.cls_token.expand(B, S, -1)
        cls_token = cls_token.reshape(B * S, -1, self.embed_dim)
        return cls_token

    def prepare_tokens_with_masks(self, x, masks=None, cls_token=None, **kwargs):
        B, S, nc, w, h = x.shape
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        cls_token = self.prepare_cls_token(B, S)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        x = rearrange(x, "(b s) n c -> b s n c", b=B, s=S)
        return x

    def _prepare_rope(self, B, S, H, W, device):
        pos = None
        pos_nodiff = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=device)
            pos = rearrange(pos, "(b s) n c -> b s n c", b=B)
            pos_nodiff = torch.zeros_like(pos).to(pos.dtype)
            if self.patch_start_idx > 0:
                pos = pos + 1
                pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(device).to(pos.dtype)
                pos_special = rearrange(pos_special, "(b s) n c -> b s n c", b=B)
                pos = torch.cat([pos_special, pos], dim=2)
                pos_nodiff = pos_nodiff + 1
                pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
        return pos, pos_nodiff

    def _get_3d_tokens(self, cond=None):
        # Check for NaN - if found, abort training (don't try to fix)
        if torch.isnan(self.pts3d_token).any():
            raise ValueError(f"pts3d_token contains NaN! Training should restart from clean checkpoint.")

        if self.embed_dim == self.token_dim_3d:
            pts3d_tokens = self.pts3d_token
        else:
            # Check pts3d_token_proj weights for NaN
            if hasattr(self, "pts3d_token_proj"):
                for name, param in self.pts3d_token_proj.named_parameters():
                    if torch.isnan(param).any():
                        print(f"[CRITICAL] NaN in pts3d_token_proj.{name}")
                        with torch.no_grad():
                            param[torch.isnan(param)] = 0.0

            pts3d_tokens = self.pts3d_token_proj(self.pts3d_token)
            pts3d_tokens = pts3d_tokens.view(1, self.num_3d_tokens, self.embed_dim)

            # Clamp extreme values that could cause downstream NaN
            pts3d_tokens = torch.clamp(pts3d_tokens, min=-10.0, max=10.0)

        if self.token_3d_embed is not None:
            # pts3d_tokens   B, S, C
            # cond:          B, C

            # SAFEGUARD: Check cond (class_tokens) for NaN before processing
            if torch.isnan(cond).any():
                print(f"[ERROR] cond (class_tokens) contains NaN before token_3d_embed processing!")
                print(f"  NaN count: {torch.isnan(cond).sum().item()} / {cond.numel()}")
                # Replace NaN with zeros as emergency fallback
                cond = torch.nan_to_num(cond, nan=0.0)
                print(f"  [WARNING] Replaced NaN in cond with zeros")

            if self.token_3d_embed.use_cond_embed == "first":
                cond = cond[:, 0]  # use the first frame only
            elif self.token_3d_embed.use_cond_embed == "mean":
                cond = cond.mean(dim=1)  # use the mean of all frames

            has_nan_input = torch.isnan(pts3d_tokens).any() or torch.isnan(cond).any()
            if has_nan_input:
                print(f"[ERROR] NaN in inputs BEFORE token_3d_embed!")
                if torch.isnan(pts3d_tokens).any():
                    print(f"  pts3d_tokens NaN count: {torch.isnan(pts3d_tokens).sum().item()}")
                if torch.isnan(cond).any():
                    print(f"  cond NaN count: {torch.isnan(cond).sum().item()}")

            # Check token_3d_embed weights for NaN
            for name, param in self.token_3d_embed.named_parameters():
                if torch.isnan(param).any():
                    print(f"[ERROR] NaN in token_3d_embed weight: {name}")
                    print(f"  NaN count: {torch.isnan(param).sum().item()} / {param.numel()}")

            # Check for extreme values that might cause NaN in forward
            if (torch.abs(pts3d_tokens) > 1e6).any():
                print(f"[WARNING] Extreme values in pts3d_tokens: max_abs={torch.abs(pts3d_tokens).max().item()}")
            if (torch.abs(cond) > 1e6).any():
                print(f"[WARNING] Extreme values in cond: max_abs={torch.abs(cond).max().item()}")

            # Register hook on pts3d_tokens to track gradient BEFORE token_3d_embed
            pts3d_tokens_before_embed = pts3d_tokens.clone().requires_grad_(True)

            def pts3d_tokens_pre_embed_hook(grad):
                if torch.norm(grad) > 10.0:
                    print(f"[GRADIENT FLOW] Gradient entering token_3d_embed: norm={torch.norm(grad).item():.2f}")
                return grad

            # pts3d_tokens_before_embed.register_hook(pts3d_tokens_pre_embed_hook)

            pts3d_tokens = self.token_3d_embed(pts3d_tokens_before_embed, cond=cond)

            if torch.isnan(pts3d_tokens).any():
                print(f"[ERROR] NaN in pts3d_tokens AFTER token_3d_embed!")
                print(f"  NaN count: {torch.isnan(pts3d_tokens).sum().item()}")
                print(
                    f"  Output stats: min={pts3d_tokens[~torch.isnan(pts3d_tokens)].min().item() if (~torch.isnan(pts3d_tokens)).any() else 'all NaN'}, max={pts3d_tokens[~torch.isnan(pts3d_tokens)].max().item() if (~torch.isnan(pts3d_tokens)).any() else 'all NaN'}"
                )
                raise ValueError("NaN detected in token_3d_embed output!")

        if self.add_camera_token_3d:
            # add camera token to 3D tokens
            if self.add_camera_token_3d == "first":
                # add camera token only to the first frame
                camera_token = self.camera_token[:, 0, :].expand(pts3d_tokens.shape[0], -1, -1)
                pts3d_tokens = torch.cat([camera_token, pts3d_tokens], dim=1)
            elif self.add_camera_token_3d == "new":
                # add a new camera token to the 3D tokens
                camera_token = self.camera_token_3d.expand(pts3d_tokens.shape[0], -1, -1)
                pts3d_tokens = torch.cat([camera_token, pts3d_tokens], dim=1)
            else:
                raise NotImplementedError(f"Unsupported add_camera_token_3d: {self.add_camera_token_3d}. Only 'first' is supported.")

        return pts3d_tokens

    def _get_3d_pos_embed(self, B):
        P_3d = self.num_3d_tokens

        if self.pos_3d_embed_type == "3d":
            if P_3d == 512:
                # 3D grid sample 8
                x = torch.arange(0, 8)
                y = torch.arange(0, 8)
                z = torch.arange(0, 8)
                x, y, z = torch.meshgrid(x, y, z)
                w = torch.ones_like(x)
                # concatenate the 3D coordinates
                pos_3d = torch.stack([x, y, z, w], dim=-1).reshape(1, -1, 4)
                pos_3d = pos_3d.expand(B, -1, -1)
            else:
                raise ValueError(f"Unsupported 3D token size: {P_3d}. Only 512 is supported for 3D grid sampling.")
        elif self.pos_3d_embed_type == "1d":
            pos_3d = torch.zeros(B, P_3d, 2).to(self.pts3d_token.device)
            pos_3d[:, :, 0] = torch.arange(0, P_3d)
        else:
            pos_3d = torch.zeros(B, P_3d, 2).to(self.pts3d_token.device)

        # add camera token position if needed
        if self.add_camera_token_3d:
            # add an empty position for the camera token
            pos_3d = torch.cat([torch.zeros(B, 1, 2).to(pos_3d.device).to(pos_3d.dtype), pos_3d], dim=1)
        # pos_3d.shape = (B, P_3d, 2)

        return pos_3d.long()

    def _get_intermediate_layers_not_chunked(self, x, n=1, export_feat_layers=[], **kwargs):
        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len, aux_output = [], len(self.blocks), []
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)

        class_tokens = x[:, :, :1]
        class_tokens = class_tokens.reshape(B, S, -1)
        pts3d_tokens = self._get_3d_tokens(class_tokens)
        pts3d_tokens = pts3d_tokens.expand(B, -1, -1)
        # add camera token to 3d tokens if needed

        # pos_3d = torch.zeros(B, P_3d, 2).to(images.device).to(pos.dtype)
        P_3d = pts3d_tokens.shape[1]
        pos_3d = self._get_3d_pos_embed(B).to(pos.device)
        output_3d = []

        for i, blk in enumerate(self.blocks):
            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos

            if self.alt_start != -1 and (i == self.alt_start - 1) and x.shape[1] >= THRESH_FOR_REF_SELECTION and kwargs.get("cam_token", None) is None:
                # Select reference view using configured strategy
                strategy = kwargs.get("ref_view_strategy", "saddle_balanced")
                logger.info(f"Selecting reference view using strategy: {strategy}")
                b_idx = select_reference_view(x, strategy=strategy)
                # Reorder views to place reference view first
                x = reorder_by_reference(x, b_idx)
                local_x = reorder_by_reference(local_x, b_idx)

            if self.alt_start != -1 and i == self.alt_start:
                if kwargs.get("cam_token", None) is not None:
                    #logger.info("Using camera conditions provided by the user")
                    cam_token = kwargs.get("cam_token")
                else:
                    ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                    src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                    cam_token = torch.cat([ref_token, src_token], dim=1)
                x[:, :, 0] = cam_token

            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x, pts3d_tokens = self.process_attention(x, pts3d_tokens, blk, "global", pos=g_pos, pos_3d=pos_3d, attn_mask=kwargs.get("attn_mask", None))
                global_pts3d = pts3d_tokens
            else:
                x, pts3d_tokens = self.process_attention(x, pts3d_tokens, blk, "local", pos=l_pos, pos_3d=pos_3d)
                local_x = x
                local_pts3d = pts3d_tokens

            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                # Restore original view order if reordering was applied
                if x.shape[1] >= THRESH_FOR_REF_SELECTION and self.alt_start != -1 and "b_idx" in locals():
                    out_x = restore_original_order(out_x, b_idx)
                output.append((out_x[:, :, 0], out_x))
                output_3d.append(torch.cat([local_pts3d, global_pts3d], dim=-1))

            if i in export_feat_layers:
                aux_output.append(x)

        return output, output_3d, aux_output

    def process_attention(
        self,
        x,
        pts3d_tokens,
        block,
        attn_type="global",
        pos=None,
        pos_3d=None,
        attn_mask=None,
    ):
        b, s, n = x.shape[:3]
        if attn_type == "local":
            x = rearrange(x, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")

        elif attn_type == "global":
            x = rearrange(x, "b s n c -> b (s n) c")
            x = torch.cat([x, pts3d_tokens], dim=1)
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")
                pos = torch.cat([pos, pos_3d], dim=1)
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")

        x = block(x, pos=pos, attn_mask=attn_mask)

        if attn_type == "local":
            x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
            pts3d_tokens = block(pts3d_tokens, pos=pos_3d, attn_mask=attn_mask)

        elif attn_type == "global":
            pts3d_tokens = x[:, s * n :, :]
            x = x[:, : s * n, :]
            x = rearrange(x, "b (s n) c -> b s n c", b=b, s=s)
        return x, pts3d_tokens

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        export_feat_layers: List[int] = [],
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        outputs, outputs_3d, aux_outputs = self._get_intermediate_layers_not_chunked(x, n, export_feat_layers=export_feat_layers, **kwargs)

        camera_tokens = [out[0] for out in outputs]
        if outputs[0][1].shape[-1] == self.embed_dim:
            outputs = [self.norm(out[1]) for out in outputs]
        elif outputs[0][1].shape[-1] == (self.embed_dim * 2):
            outputs = [
                torch.cat(
                    [out[1][..., : self.embed_dim], self.norm(out[1][..., self.embed_dim :])],
                    dim=-1,
                )
                for out in outputs
            ]
        else:
            raise ValueError(f"Invalid output shape: {outputs[0][1].shape}")
        aux_outputs = [self.norm(out) for out in aux_outputs]
        outputs = [out[..., 1 + self.num_register_tokens :, :] for out in outputs]
        aux_outputs = [out[..., 1 + self.num_register_tokens :, :] for out in aux_outputs]
        return tuple(zip(outputs, camera_tokens)), outputs_3d, aux_outputs


def vit_small(patch_size=16, num_register_tokens=0, depth=12, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=depth,
        num_heads=6,
        mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, depth=12, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=depth,
        num_heads=12,
        mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, depth=24, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=depth,
        num_heads=16,
        mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, depth=40, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=depth,
        num_heads=24,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
