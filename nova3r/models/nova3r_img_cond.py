# Copyright (c) 2026 Weirong Chen
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from nova3r.heads.hunyuan_model.autoencoders.model import DiagonalGaussianDistribution
from nova3r.models.aggregator_pts3d import AggregatorPts3D
from nova3r.heads.pts3d_decoder import *
from nova3r.heads.triposg_model.autoencoder_kl_triposg import FrequencyPositionalEmbedding
from nova3r.layers.hunyuan_block import FourierEmbedder
import numpy as np
import math
from torch_cluster import fps

from einops import rearrange

from safetensors.torch import load_file
from nova3r.models.aggregator_da3 import DepthAnything3
from nova3r.heads.hunyuan_model.autoencoders.model import ShapeVAE, ShapeVAEDecoder

from nova3r.flow_matching.path.scheduler import LinearScheduler
from nova3r.flow_matching.path import AffineProbPath

from nova3r.flow_matching.solver import ODESolver
from nova3r.models.model_wrapper import BatchModelWrapper

from nova3r.heads.pts3d_decoder.latent_flowm_decoder import SiT


def _load_checkpoint_state_dict(path):
    if path.endswith(".safetensors"):
        state_dict = load_file(path)
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]

    # remove model.
    if state_dict and all(key.startswith("model.") for key in state_dict):
        state_dict = {key[len("model.") :]: value for key, value in state_dict.items()}

    return state_dict


class Nova3rImgCond(nn.Module, PyTorchModelHubMixin):
    """Image-conditioned flow matching model for amodal 3D reconstruction.

    Takes images as input and generates complete 3D point clouds using a VGGT
    backbone encoder and flow matching decoder.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        patch_3d_size=256,
        num_3d_tokens=512,
        cfg=None,
        classifier_free_guidance_drop_pro=0.0,
    ):
        """3D Shape Tokenization Implementation"""
        super().__init__()
        self.cfg = cfg

        # Classifier-free guidance settings
        self.cfg_drop_prob = classifier_free_guidance_drop_pro

        if self.cfg.aggregator.name == "DepthAnything3Net":
            self.da3_aggregator = DepthAnything3(cfg=self.cfg)
        elif self.cfg.aggregator.name == "AggregatorPts3D":
            self.vggt_aggregator = AggregatorPts3D(**self.cfg.aggregator.params)
        elif self.cfg.aggregator.name == "Depth_Predicted":
            pass

        token_dim = self.cfg.aggregator.params.token_dim
        self.detach_vit_token = self.cfg.aggregator.params.detach_vit_token if "detach_vit_token" in self.cfg.aggregator.params else False
        self.detach_vggt_token = self.cfg.aggregator.params.detach_vggt_token if "detach_vggt_token" in self.cfg.aggregator.params else False

        self.embedder = FrequencyPositionalEmbedding(
            num_freqs=8,
            logspace=True,
            input_dim=3,
            include_pi=False,
        )
        # self.embedder.out_dim
        # self.embed_3d_proj = nn.Linear(self.embedder.out_dim, token_dim)

        self.img_token_proj = nn.Linear(embed_dim * 2, token_dim)
        # Initialize img_token_proj with small values to prevent early NaN
        nn.init.trunc_normal_(self.img_token_proj.weight, std=0.02)
        nn.init.zeros_(self.img_token_proj.bias)

        self.num_3d_tokens = num_3d_tokens

        self.use_token_ln = self.cfg.aggregator.params.get("use_token_ln", False)
        self.token_noise_prob = self.cfg.aggregator.params.get("token_noise_prob", 0.0)
        self.token_noise_sigma = self.cfg.aggregator.params.get("token_noise_sigma", 0.0)
        self.sample_posterior = getattr(self.cfg, "sample_posterior", True)
        self.debug_lightning_batch_dir = os.environ.get(
            "NOVA3R_LIGHTNING_DEBUG_BATCH_DIR",
            "debug_points/lightning_forward_batch",
        )
        self.debug_lightning_batch_once = os.environ.get("NOVA3R_LIGHTNING_DEBUG_BATCH_ONCE", "1") != "0"
        self.debug_lightning_batch_max_samples = int(os.environ.get("NOVA3R_LIGHTNING_DEBUG_BATCH_MAX_SAMPLES", "2"))
        self.freeze_target_latents = self._env_flag(
            "NOVA3R_FREEZE_TARGET_LATENTS",
            getattr(self.cfg, "freeze_target_latents", False),
        )
        self.freeze_target_latents_sample_posterior = self._env_flag(
            "NOVA3R_FREEZE_TARGET_LATENTS_SAMPLE_POSTERIOR",
            getattr(self.cfg, "freeze_target_latents_sample_posterior", False),
        )
        self.recon_latent_steps = int(os.environ.get("NOVA3R_RECON_LATENT_STEPS", getattr(self.cfg, "recon_latent_steps", 50)))
        self.recon_latent_step_interval = int(os.environ.get("NOVA3R_RECON_LATENT_STEP_INTERVAL", getattr(self.cfg, "recon_latent_step_interval", 0)))
        self._target_latent_cache = {}
        self._debug_lightning_batch_exported = False

        if self.use_token_ln:
            self.token_norm = nn.LayerNorm(self.cfg.aggregator.params.token_dim)
        else:
            self.token_norm = None

        self.pts3d_head = None
        self.first_stage = None
        if "pts3d_head" in cfg:
            self.pts3d_head = eval(cfg.pts3d_head.name)(**cfg.pts3d_head.params)

        if "first_stage" in cfg:
            self.token_to_moments = nn.Linear(token_dim, self.cfg.first_stage.params.embed_dim * 2)
            self.first_stage = eval(cfg.first_stage.name)(**cfg.first_stage.params)

    def _embed_3d(self, pts3d: torch.Tensor):
        x = self.embedder(pts3d)  # [B, N, d_point]
        x = self.embed_3d_proj(x)  # [B, N, d]
        return x

    def _sample_features(self, x: torch.Tensor, num_tokens: int = 2048, oversample_factor: int = 4, seed: Optional[int] = None):
        """
        Sample points from features of the input point cloud.

        Args:
            x (torch.Tensor): The input point cloud. shape: (B, N, C)
            num_tokens (int, optional): The number of points to sample. Defaults to 2048.
            oversample_factor (int, optional): Factor to oversample before FPS. Defaults to 4.
            seed (Optional[int], optional): The random seed. Defaults to None.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(x.shape[1], num_tokens * oversample_factor, replace=num_tokens * oversample_factor > x.shape[1])
        selected_points = x[:, indices]

        batch_size, num_points, num_channels = selected_points.shape
        flattened_points = selected_points.view(batch_size * num_points, num_channels)
        batch_indices = torch.arange(batch_size).to(x.device).repeat_interleave(num_points)

        # fps sampling
        sampling_ratio = 1.0 / oversample_factor
        sampled_indices = fps(
            flattened_points[:, :3],
            batch_indices,
            ratio=sampling_ratio,
            # random_start=self.training,
            random_start=False,  # for deterministic sampling
        )
        sampled_points = flattened_points[sampled_indices].view(batch_size, -1, num_channels)

        return sampled_points

    def remove_aggregator_weights(self, nova3r_checkpoint):
        for key in list(nova3r_checkpoint.keys()):
            if key.startswith("vggt_aggregator"):
                nova3r_checkpoint.pop(key)
            if key.startswith("img_token_proj"):
                nova3r_checkpoint.pop(key)

        return nova3r_checkpoint

    def remove_decoder_weights(self, nova3r_checkpoint):
        for key in list(nova3r_checkpoint.keys()):
            if key.startswith("pts3d"):
                nova3r_checkpoint.pop(key)

        return nova3r_checkpoint

    def prep_ckpt_for_3d_token(self, ckpt):
        if self.cfg.aggregator.params.token_dim != 128:
            ckpt.pop("token_proj.weight")
            ckpt.pop("token_proj.bias")
            ckpt.pop("pts3d_head.mlp_token.weight")
            ckpt.pop("pts3d_head.mlp_token.bias")
        if self.cfg.aggregator.params.num_3d_tokens != 768:
            ckpt.pop("vggt_aggregator.pts3d_token")
        return ckpt

    def load_state_dict(self, ckpt, **kw):
        new_ckpt = dict(ckpt)
        if self.cfg.aggregator.name == "DepthAnything3Net" and kw.get("stage") != "test":
            state_dict = load_file(kw.get("aggregator_ckpt"))
            missing_keys, unexpected_keys = self.da3_aggregator.load_state_dict(state_dict, strict=False)
            print(f"Loaded DepthAnything3Net aggregator weights")
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            print(f"{"-"*100}")
            new_ckpt = self.remove_aggregator_weights(new_ckpt)

        if "first_stage" in self.cfg and kw.get("stage") != "test":
            state_dict = _load_checkpoint_state_dict(kw.get("vae_ckpt"))
            missing_keys, unexpected_keys = self.first_stage.load_state_dict(state_dict, strict=False)
            print(f"Loaded ShapeVAEDecoder weights")
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            print(f"{"-"*100}")
            # when we use the vae the pts3d head from nova3r is never used
            new_ckpt = self.remove_decoder_weights(new_ckpt)

        if kw.pop("stage", None) != "test" and self.cfg.aggregator.name == "AggregatorPts3D":
            new_ckpt = self.prep_ckpt_for_3d_token(new_ckpt)
        #temp
        # new_ckpt.pop("da3_aggregator.model.backbone.pretrained.pts3d_token")
        missing_keys, unexpected_keys = super().load_state_dict(new_ckpt, strict=False)
        print(f"Loaded Nova3RImgCond weights with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys.")

    def _encode_scene(self, img_tokens, input_pts3d):
        """Encode input images through VGGT backbone to get visual tokens."""
        B = img_tokens.shape[0]

        if self.cfg.aggregator.name == "triposg_point":
            x_kv = self.img_token_proj(img_tokens)  # [B, N, d_point]
            sample_x = self._sample_features(input_pts3d, self.num_3d_tokens)
            x_q = self._embed_3d(sample_x)  # [B, N, d_point]
            x = self.aggregator(x_q, x_kv)

            tokens = self.token_proj(x)
            return tokens

        elif self.cfg.aggregator.name == "triposg_learn":
            # x_kv = self._embed_3d(input_pts3d)  # [B, N, d_point]
            x_kv = self.img_token_proj(img_tokens)

            x_q = self.tokens.unsqueeze(0).expand(B, -1, -1)  # [B, num_tokens, d_point]
            x = self.aggregator(x_q, x_kv)
            tokens = self.token_proj(x)
            return tokens
        elif self.cfg.aggregator.name == "triposg_hybrid":
            # x_kv = self._embed_3d(input_pts3d)
            x_kv = self.img_token_proj(img_tokens)

            # FIX: set input_pts3d to 0
            # input_pts3d = torch.zeros_like(input_pts3d)
            sample_x = self._sample_features(input_pts3d, self.num_3d_tokens)

            x_q_point = self._embed_3d(sample_x)
            x_q_learn = self.tokens.unsqueeze(0).expand(B, -1, -1)  # [B, num_tokens, d_point]

            # [B, num_tokens, d_point * 2]
            x_q = torch.cat([x_q_point, x_q_learn], dim=-1)
            x_q = self.token_merge(x_q)  # [B, num_tokens, d_point]

            x = self.aggregator(x_q, x_kv)
            tokens = self.token_proj(x)
            return tokens
        else:
            raise NotImplementedError(f"Aggregator {self.cfg.aggregator.name} not implemented for encoding.")

    def forward_vggt(
        self,
        images: torch.Tensor,
    ):
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        # aggregated_tokens_list, patch_start_idx = self.vggt_aggregator(images, detach_vit_token=self.detach_vit_token)

        aggregated_tokens_list, aggregated_tokens_3d_list, patch_start_idx, patch_tokens = self.vggt_aggregator(
            images, detach_vit_token=self.detach_vit_token
        )

        pts3d = None
        pts3d_conf = None

        return aggregated_tokens_3d_list, pts3d, pts3d_conf

    def forward_da3(
        self,
        images: torch.Tensor,
        batch: dict,
        ref_view_strategy: str = "first",
    ):
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        aggregated_tokens_list, aggregated_tokens_3d_list, patch_start_idx, patch_tokens = self.da3_aggregator(
            images, intrinsics=batch.get("intrinsics"), extrinsics=batch.get("extrinsics"), ref_view_strategy=ref_view_strategy
        )

        pts3d = None
        pts3d_conf = None

        return aggregated_tokens_3d_list, pts3d, pts3d_conf

    def forward_first_stage(self, batch: dict):

        with torch.no_grad():
            points = batch["aggregator_points"]
            latents = self.first_stage.encode(points, sample_posterior=False)
        return latents

    def _encode(self, images, test=False, **kwargs):
        """Project visual tokens to 3D token space with optional learned token initialization."""

        if self.cfg.aggregator.name == "DepthAnything3Net":
            aggregated_tokens_list, pts3d, pts3d_conf = self.forward_da3(images, batch=kwargs.get("batch"), ref_view_strategy="first")
        elif self.cfg.aggregator.name == "AggregatorPts3D":
            aggregated_tokens_list, pts3d, pts3d_conf = self.forward_vggt(images)
        elif self.cfg.aggregator.name == "Depth_Predicted":
            scene_tokens = kwargs.get("batch")["aggregator_latents"]
            if scene_tokens.dim() == 4:
                scene_tokens = scene_tokens.flatten(1, 2)
            aggregated_tokens_list = [scene_tokens]

        scene_tokens = aggregated_tokens_list[-1]  # [B, S, K, C]
        B, K, C = scene_tokens.shape

        tokens = self.img_token_proj(scene_tokens)

        pointmaps = None

        if self.use_token_ln and self.token_norm is not None:
            tokens = self.token_norm(tokens)

        encode_data = {"pointmaps": pointmaps, "tokens": tokens}
        return encode_data

    def _apply_cfg_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Apply classifier-free guidance dropout to tokens during training.

        Randomly drops (zeros out) tokens with probability cfg_drop_prob.
        This allows the model to learn both conditional and unconditional generation,
        enabling classifier-free guidance at inference time.

        Args:
            tokens (torch.Tensor): Condition tokens with shape [B, N, C]

        Returns:
            torch.Tensor: Tokens with some potentially zeroed out (during training)
        """
        if not self.training or self.cfg_drop_prob <= 0.0:
            return tokens

        B = tokens.shape[0]
        # Create a mask for each sample in the batch
        # With probability cfg_drop_prob, drop all tokens for that sample
        drop_mask = torch.rand(B, device=tokens.device) < self.cfg_drop_prob

        # Expand mask to match token dimensions [B, 1, 1] for broadcasting
        drop_mask = drop_mask.view(B, 1, 1).expand_as(tokens)
        # Zero out tokens where drop_mask is True
        tokens = tokens.masked_fill(drop_mask, 0.0)

        return tokens

    def _encode_with_cfg(self, images: torch.Tensor, cfg_scale: float = 1.0, **kwargs) -> dict:
        """
        Encode images with classifier-free guidance for inference.

        This method runs the encoder and applies CFG interpolation using
        the cfg_scale parameter.

        Args:
            images (torch.Tensor): Input images with shape [B, S, C, H, W]
            cfg_scale (float): Classifier-free guidance scale.
                               1.0 = no guidance (conditional only)
                               >1.0 = stronger guidance toward condition
                               0.0 = unconditional only
            **kwargs: Additional arguments

        Returns:
            dict: Dictionary containing 'tokens' with CFG-interpolated tokens
        """
        # Get conditional tokens (normal encoding, without CFG dropout)
        # Temporarily ensure we're in eval mode
        original_training = self.training
        self.eval()

        aggregated_tokens_list, pts3d, pts3d_conf = self.forward_vggt(images)
        scene_tokens = aggregated_tokens_list[-1]  # [B, S, K, C]
        tokens_cond = self.img_token_proj(scene_tokens)

        # Restore training mode
        if original_training:
            self.train()

        # If cfg_scale is 1.0, just return conditional tokens (no CFG)
        if cfg_scale == 1.0:
            return {"pointmaps": None, "tokens": tokens_cond}

        # Unconditional tokens are zeros
        tokens_uncond = torch.zeros_like(tokens_cond)

        # Apply CFG interpolation: uncond + cfg_scale * (cond - uncond)
        tokens = tokens_uncond + cfg_scale * (tokens_cond - tokens_uncond)

        return {"pointmaps": None, "tokens": tokens}

    def _decode(self, tokens, images, token_mask=None, query_points=None, timestep=None, **kwargs):
        """Decode 3D tokens into point cloud coordinates using flow matching ODE solver."""
        B, S = images.shape[:2]

        predictions = {}

        # [B], assuming each image in the sequence is a different view
        num_views = torch.ones(B, device=images.device) * S
        # with torch.cuda.amp.autocast(enabled=False):

        if self.pts3d_head is not None:
            aggregated_tokens_3d_list = [tokens]
            cond_tokens = tokens

            pts3d_xyz = self.pts3d_head(
                aggregated_tokens_3d_list,
                mask=token_mask,
                query_points=query_points,
                timestep=timestep,
                num_views=num_views,
                conditioning_latents=kwargs.get("conditioning_latents", None),
                aggregator_latents=kwargs.get("aggregator_latents", None),
            )

            predictions["pts3d_xyz"] = pts3d_xyz
            predictions["pts3d_xyz_rel"] = pts3d_xyz
            predictions["pts3d_rgb"] = pts3d_xyz
            predictions["pts3d_conf"] = torch.ones_like(pts3d_xyz[..., [0]])
            predictions["center_xyz"] = pts3d_xyz
            predictions["query_points"] = query_points
            predictions["timestep"] = timestep

        # if self.first_stage is not None:
        #     moments = self.token_to_moments(tokens)
        #     posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
        #     latents = posterior.sample()
        #     latents = self.first_stage.decode(latents)
        #     predictions["latents"] = latents
        #     predictions["posterior"] = posterior

        predictions["images"] = images
        predictions["S"] = S

        return predictions

    def forward(
        self,
        images: torch.Tensor,
        token_mask: torch.Tensor = None,
        query_points: torch.Tensor = None,
        timestep: torch.Tensor = None,
        **kwargs,
    ):
        """Full forward pass: encode images, then decode to 3D point clouds.

        Args:
            images (torch.Tensor): Input images with shape [B, S, C, H, W].
            token_mask (torch.Tensor, optional): Mask for tokens. Default: None.
            query_points (torch.Tensor, optional): Query points for decoding.
            timestep (torch.Tensor, optional): Timestep for flow matching.

        Returns:
            dict: Predictions including 'pts3d_xyz', 'images', and 'S'.
        """
        encode_data = self._encode(images, **kwargs)

        tokens = encode_data["tokens"]

        predictions = self._decode(tokens, images, token_mask=token_mask, query_points=query_points, timestep=timestep, **kwargs)

        return predictions

    # def lightning_forward_sdf_based(self, batch: dict, criterion, kl_weight=0.0):

    #     images = torch.stack(batch["images"], dim=1)
    #     encoder_data = self._encode(images, batch=batch)
    #     tokens = encoder_data["tokens"]
    #     tokens = self._apply_cfg_dropout(tokens)
    #     predictions = self._decode(tokens, images)
    #     latents = predictions["latents"]

    #     geo_points = batch["geo_points"]
    #     geo_points_coords = geo_points[:, :, :3]
    #     geo_points_label = geo_points[:, :, 3:4]

    #     geo_points_label = geo_points_label * 128
    #     geo_points_label = geo_points_label.clamp(-1.0, 1.0)

    #     logits = self.first_stage.geo_decoder(queries=geo_points_coords, latents=latents)
    #     posterior = predictions["posterior"]
    #     kl_loss = posterior.kl(dims=(0, 1, 2))

    #     gt_list = {"sdf_target": geo_points_label}
    #     pred_list = {"sdf_pred": logits}

    #     loss, details = criterion(gt_list, pred_list)

    #     loss += kl_loss * kl_weight
    #     details["kl_loss"] = kl_loss.item()
    #     return loss, details, latents

    @staticmethod
    def _rank_zero():
        try:
            return int(os.environ.get("RANK", "0")) == 0
        except ValueError:
            return True

    @staticmethod
    def _env_flag(name, default=False):
        value = os.environ.get(name)
        if value is None:
            return bool(default)
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _tensor_summary(value):
        if isinstance(value, torch.Tensor):
            summary = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
            if value.numel() > 0 and value.is_floating_point():
                detached = value.detach().float()
                summary.update(
                    {
                        "mean": float(detached.mean().cpu()),
                        "variance": float(detached.var(unbiased=False).cpu()),
                        "min": float(detached.min().cpu()),
                        "max": float(detached.max().cpu()),
                    }
                )
            return summary
        if isinstance(value, list):
            return {
                "type": "list",
                "length": len(value),
                "first": Nova3rImgCond._tensor_summary(value[0]) if value else None,
            }
        return {"type": type(value).__name__, "value": str(value)}

    @staticmethod
    def _batch_value_at(value, index):
        if isinstance(value, torch.Tensor):
            return value[index]
        if isinstance(value, (list, tuple)):
            if value and isinstance(value[0], torch.Tensor):
                return [Nova3rImgCond._batch_value_at(item, index) for item in value]
            if index < len(value):
                return value[index]
        return value

    @staticmethod
    def _debug_name(value):
        return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))

    @staticmethod
    def _to_python_key_part(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.ndim == 0:
                return value.item()
            return tuple(value.reshape(-1).tolist())
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            return tuple(value.reshape(-1).tolist())
        if isinstance(value, (list, tuple)):
            return tuple(Nova3rImgCond._to_python_key_part(item) for item in value)
        return value

    def _target_latent_key(self, batch, sample_idx):
        seq_name = self._to_python_key_part(self._batch_value_at(batch.get("seq_name", "sample"), sample_idx))
        sample_id = self._to_python_key_part(self._batch_value_at(batch.get("id", sample_idx), sample_idx))
        view_ids = self._to_python_key_part(self._batch_value_at(batch.get("subseq_ids", ()), sample_idx))
        return str(seq_name), str(sample_id), str(view_ids)

    def _get_target_latents(self, batch):
        surface = batch["surface"]
        if not self.freeze_target_latents:
            with torch.no_grad():
                return self.first_stage.encode(surface, sample_posterior=self.sample_posterior)

        uncached_indices = []
        cached_latents = []
        keys = [self._target_latent_key(batch, sample_idx) for sample_idx in range(surface.shape[0])]
        for sample_idx, key in enumerate(keys):
            cached = self._target_latent_cache.get(key)
            if cached is None:
                uncached_indices.append(sample_idx)
                cached_latents.append(None)
            else:
                cached_latents.append(cached)

        if uncached_indices:
            with torch.no_grad():
                encoded = self.first_stage.encode(
                    surface[uncached_indices],
                    sample_posterior=self.freeze_target_latents_sample_posterior,
                )
            for local_idx, sample_idx in enumerate(uncached_indices):
                cached = encoded[local_idx].detach().cpu().clone()
                self._target_latent_cache[keys[sample_idx]] = cached
                cached_latents[sample_idx] = cached

        return torch.stack(
            [latent.to(device=surface.device, dtype=surface.dtype) for latent in cached_latents],
            dim=0,
        )

    def _export_lightning_batch_debug(self, batch, images, gt_latents):
        if not self._rank_zero() or not self.debug_lightning_batch_dir:
            return
        if self.debug_lightning_batch_once and self._debug_lightning_batch_exported:
            return

        output_dir = Path(self.debug_lightning_batch_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_size = images.shape[0]
        samples_to_export = batch_size
        if self.debug_lightning_batch_max_samples >= 0:
            samples_to_export = min(batch_size, self.debug_lightning_batch_max_samples)

        for sample_idx in range(samples_to_export):
            seq_name = self._batch_value_at(batch.get("seq_name", "sample"), sample_idx)
            sample_id = self._batch_value_at(batch.get("id", "00000"), sample_idx)
            stem = self._debug_name(f"{sample_idx:03d}_{seq_name}_{sample_id}")

            surface = batch["surface"][sample_idx].detach().cpu()
            geo_points = batch.get("geo_points")
            geo_points = geo_points[sample_idx].detach().cpu() if isinstance(geo_points, torch.Tensor) else None
            gt_latent = gt_latents[sample_idx].detach().float().cpu()

            np.save(output_dir / f"{stem}_gt_latents.npy", gt_latent.numpy())
            if geo_points is not None:
                np.savez_compressed(
                    output_dir / f"{stem}_batch_points.npz",
                    surface=surface.numpy(),
                    geo_points=geo_points.numpy(),
                )

            camera_data = {
                "extrinsics": batch["extrinsics"][sample_idx].detach().cpu().numpy().tolist(),
                "intrinsics": batch["intrinsics"][sample_idx].detach().cpu().numpy().tolist(),
            }
            with open(output_dir / f"{stem}_cameras.json", "w", encoding="utf-8") as f:
                json.dump(camera_data, f, indent=2)

            summary = {key: self._tensor_summary(self._batch_value_at(value, sample_idx)) for key, value in batch.items()}
            summary["gt_latents"] = self._tensor_summary(gt_latent)
            with open(output_dir / f"{stem}_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            try:
                import open3d as o3d

                points = surface[:, :3].numpy().astype(np.float64)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points))
                if surface.shape[1] >= 6:
                    normals = surface[:, 3:6].numpy().astype(np.float64)
                    pcd.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(normals))
                if surface.shape[1] >= 7:
                    labels = surface[:, -1].numpy() > 0.5
                    colors = np.zeros((len(points), 3), dtype=np.float64)
                    colors[~labels] = [0.2, 0.55, 1.0]
                    colors[labels] = [1.0, 0.25, 0.1]
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(str(output_dir / f"{stem}_surface.ply"), pcd)
            except Exception as exc:
                print(f"Skipping lightning batch point cloud export for sample {sample_idx}: {exc}")

            try:
                import cv2

                image_tiles = []
                for image in images[sample_idx].detach().cpu():
                    image_np = image.permute(1, 2, 0).numpy()
                    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
                    image_tiles.append(image_np)
                if image_tiles:
                    image_grid = np.concatenate(image_tiles, axis=1)
                    cv2.imwrite(
                        str(output_dir / f"{stem}_images.png"),
                        cv2.cvtColor(image_grid, cv2.COLOR_RGB2BGR),
                    )
            except Exception as exc:
                print(f"Skipping lightning batch image export for sample {sample_idx}: {exc}")

        self._debug_lightning_batch_exported = True
        print(f"Saved lightning batch debug export: {output_dir} ({samples_to_export}/{batch_size} samples)")

    def lightning_forward(self, batch: dict, criterion, kl_weight=0.0, recon_latents=False):
        gt_latents = self._get_target_latents(batch)

        images = torch.stack(batch["images"], dim=1)
        # self._export_lightning_batch_debug(batch, images, gt_latents)
        encoder_data = self._encode(images, batch=batch)
        tokens = encoder_data["tokens"]

        device = tokens.device
        B, _, _ = tokens.shape
        gt_latents = gt_latents.to(device=device, dtype=tokens.dtype).contiguous()

        x_0 = torch.randn_like(gt_latents)
        t = torch.rand(B, device=device)

        path = AffineProbPath(scheduler=LinearScheduler())
        fm_path = path
        path_sample = fm_path.sample(x_0=x_0, x_1=gt_latents, t=t)

        x_t = path_sample.x_t
        dx_t = path_sample.dx_t
        t_query = t[:, None].expand(B, x_t.shape[1])

        if self.cfg.aggregator.name == "Depth_Predicted":
            conditioning_latents = self.forward_first_stage(batch=batch)
            v_pred = self.pts3d_head(decout=[tokens], noised_latents=x_t, timestep=t_query, conditioning_latents=conditioning_latents)
        else:
            conditioning_latents = None
            v_pred = self.pts3d_head(decout=[tokens], noised_latents=x_t, timestep=t_query)
        loss = F.mse_loss(v_pred, dx_t)

        if recon_latents == True:
            with torch.no_grad():
                step_size = 0.01
                method = "euler"
                token_mask = None
                pts3d_src = None

                num_steps = int(round(1.0 / step_size)) + 1

                wrapped_vf = BatchModelWrapper(model=self)

                wrapped_vf.eval()
                self.eval()

                noise_generator = torch.Generator(device=gt_latents.device)
                noise_generator.manual_seed(42)

                x_init = torch.randn_like(gt_latents, generator=noise_generator)
                T = torch.linspace(0, 1, num_steps).to(device)
                solver = ODESolver(velocity_model=wrapped_vf)
                sol = solver.sample(
                    time_grid=T,
                    x_init=x_init,
                    method=method,
                    step_size=step_size,
                    return_intermediates=False,
                    images=images,
                    token_mask=token_mask,
                    encoder_data=encoder_data,
                    pointmaps=pts3d_src,
                    conditioning_latents=conditioning_latents,
                )

            latents = self.first_stage.decode(sol)
        else:
            latents = None

        details = {}
        return loss, details, latents

