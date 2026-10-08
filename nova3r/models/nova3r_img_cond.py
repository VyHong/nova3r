# Copyright (c) 2026 Weirong Chen
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from nova3r.heads.hunyuan_model.autoencoders.model import ShapeVAE
from nova3r.heads.pts3d_decoder import LatentJointFMDecoder, PointJointFMDecoderV2
from nova3r.heads.pts3d_decoder.latent_flowm_decoder import (
    DiT,
    flow_matching_losses,
    resolve_precomputed_gt_latents,
    sample_linear_flow_path,
)
from nova3r.models.aggregator_pts3d import AggregatorPts3D
from safetensors.torch import load_file
from nova3r.models.aggregator_da3 import DepthAnything3
from nova3r.flow_matching.solver import ODESolver
from nova3r.models.model_wrapper import BatchModelWrapper

from nova3r.utils.batch_debug import save_batch_debug

HEAD_TYPES = {
    "DiT": DiT,
    "LatentJointFMDecoder": LatentJointFMDecoder,
    "PointJointFMDecoderV2": PointJointFMDecoderV2,
}
FIRST_STAGE_TYPES = {"ShapeVAE": ShapeVAE}


def _load_checkpoint_state_dict(path, component=None):
    if path.endswith(".safetensors"):
        checkpoint = load_file(path)
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if (
        component is not None
        and isinstance(checkpoint, dict)
        and component in checkpoint
        and isinstance(checkpoint[component], dict)
    ):
        state_dict = checkpoint[component]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if component is not None:
        prefix = f"{component}."
        component_state = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if component_state:
            state_dict = component_state

    # remove model.
    if state_dict and all(key.startswith("model.") for key in state_dict):
        state_dict = {key[len("model.") :]: value for key, value in state_dict.items()}

    return state_dict


def _without_prefixes(state_dict, prefixes):
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(prefixes)
    }


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

        del (
            img_size,
            patch_size,
            patch_3d_size,
            num_3d_tokens,
            classifier_free_guidance_drop_pro,
        )

        if self.cfg.aggregator.name == "DepthAnything3Net":
            self.da3_aggregator = DepthAnything3(cfg=self.cfg)
        elif self.cfg.aggregator.name == "AggregatorPts3D":
            self.vggt_aggregator = AggregatorPts3D(**self.cfg.aggregator.params)
        elif self.cfg.aggregator.name != "PrecomputedDINOv3":
            raise ValueError(f"Unknown aggregator: {self.cfg.aggregator.name}")

        token_dim = self.cfg.aggregator.params.token_dim
        self.detach_vit_token = self.cfg.aggregator.params.detach_vit_token if "detach_vit_token" in self.cfg.aggregator.params else False

        projection_input_dim = (
            int(self.cfg.aggregator.params.get("feature_dim", embed_dim))
            if self.cfg.aggregator.name == "PrecomputedDINOv3"
            else embed_dim * 2
        )
        self.img_token_proj = nn.Linear(projection_input_dim, token_dim)
        # Initialize img_token_proj with small values to prevent early NaN
        nn.init.trunc_normal_(self.img_token_proj.weight, std=0.02)
        nn.init.zeros_(self.img_token_proj.bias)

        self.use_token_ln = self.cfg.aggregator.params.get("use_token_ln", False)
        self.token_norm = (
            nn.LayerNorm(self.cfg.aggregator.params.token_dim)
            if self.use_token_ln
            else None
        )

        self.pts3d_head = None
        self.first_stage = None
        if "pts3d_head" in cfg:
            self.pts3d_head = HEAD_TYPES[cfg.pts3d_head.name](**cfg.pts3d_head.params)

        if "first_stage" in cfg:
            self.token_to_moments = nn.Linear(token_dim, self.cfg.first_stage.params.embed_dim * 2)
            self.first_stage = FIRST_STAGE_TYPES[cfg.first_stage.name](**cfg.first_stage.params)

    def load_dit_checkpoint(self, checkpoint_path):
        """Overlay a standalone DiT Lightning checkpoint onto pts3d_head."""
        if not isinstance(self.pts3d_head, DiT):
            raise TypeError(
                "experiment.dit_ckpt requires Nova3rImgCond.cfg.pts3d_head "
                f"to be DiT, got {type(self.pts3d_head).__name__}"
            )

        state_dict = _load_checkpoint_state_dict(str(checkpoint_path))
        if any(key.startswith("model.") for key in state_dict):
            # Standalone Mesh2SDFLightningModule checkpoints store the DiT as
            # model.* and may also contain the frozen validation VAE.
            state_dict = {
                key.removeprefix("model."): value
                for key, value in state_dict.items()
                if key.startswith("model.")
            }
        elif any(key.startswith("pts3d_head.") for key in state_dict):
            # Also accept a DiT extracted from a full Nova3rImgCond checkpoint.
            state_dict = {
                key.removeprefix("pts3d_head."): value
                for key, value in state_dict.items()
                if key.startswith("pts3d_head.")
            }

        incompatible = self.pts3d_head.load_state_dict(state_dict, strict=True)
        print(
            f"Loaded standalone DiT checkpoint from {checkpoint_path} into "
            "Nova3rImgCond.pts3d_head"
        )
        return incompatible

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
        stage = kw.pop("stage", None)
        strict = kw.pop("strict", False)
        aggregator_ckpt = kw.pop("aggregator_ckpt", None)
        vae_ckpt = kw.pop("vae_ckpt", None)
        dit_ckpt = kw.pop("dit_ckpt", None)
        if kw:
            unknown = ", ".join(sorted(kw))
            raise TypeError(f"Unexpected load_state_dict keyword argument(s): {unknown}")

        load_aggregator = self.cfg.aggregator.name == "DepthAnything3Net" and stage != "test" and aggregator_ckpt
        load_vae = "first_stage" in self.cfg and stage != "test" and vae_ckpt
        if self.cfg.aggregator.name == "PrecomputedDINOv3" and stage != "test":
            # The precomputed path has no image encoder, and its 1024->token_dim
            # projection does not match the legacy DA3 2048->token_dim layer.
            new_ckpt = _without_prefixes(
                new_ckpt,
                ("vggt_aggregator.", "da3_aggregator.", "img_token_proj."),
            )
        if load_aggregator:
            new_ckpt = _without_prefixes(
                new_ckpt,
                ("vggt_aggregator.", "da3_aggregator.", "img_token_proj."),
            )
        if load_vae:
            # The legacy point decoder is not compatible with the latent-flow
            # head, and any embedded VAE must not overwrite the requested VAE.
            new_ckpt = _without_prefixes(
                new_ckpt,
                ("pts3d_head.", "first_stage."),
            )

        if stage != "test" and self.cfg.aggregator.name == "AggregatorPts3D":
            new_ckpt = self.prep_ckpt_for_3d_token(new_ckpt)

        # Apply the base/full checkpoint first. Optional component checkpoints
        # are overlays and are deliberately applied last, so nothing can replace
        # their initialized weights afterward.
        incompatible = super().load_state_dict(new_ckpt, strict=strict)
        missing_keys, unexpected_keys = incompatible
        print(f"Loaded Nova3RImgCond weights with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys.")

        if load_aggregator:
            state_dict = load_file(aggregator_ckpt)
            component_incompatible = self.da3_aggregator.load_state_dict(state_dict, strict=False)
            print("Loaded DepthAnything3Net aggregator weights")
            print(f"Missing keys: {len(component_incompatible.missing_keys)}")
            print(f"Unexpected keys: {len(component_incompatible.unexpected_keys)}")
            print(f"{'-'*100}")

        if load_vae:
            state_dict = _load_checkpoint_state_dict(vae_ckpt)
            component_missing, component_unexpected = self.first_stage.load_state_dict(
                state_dict,
                strict=True,
            )
            print("Loaded ShapeVAE weights")
            print(f"Missing keys: {len(component_missing)}")
            print(f"Unexpected keys: {len(component_unexpected)}")
            print(f"{'-'*100}")

        # Apply the unconditional DiT last. In particular, this must happen
        # after the VAE overlay, whose legacy checkpoint filtering removes
        # pts3d_head.* from the base checkpoint.
        if stage != "test" and dit_ckpt:
            self.load_dit_checkpoint(dit_ckpt)

        return incompatible

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

        _, aggregated_tokens_3d_list, _, _ = self.da3_aggregator(
            images, intrinsics=batch.get("intrinsics"), extrinsics=batch.get("extrinsics"), ref_view_strategy=ref_view_strategy
        )

        pts3d = None
        pts3d_conf = None

        return aggregated_tokens_3d_list, pts3d, pts3d_conf

    def _encode(self, images, test=False, **kwargs):
        """Project visual tokens to 3D token space with optional learned token initialization."""
        batch = kwargs.get("batch") or {}
        if "image_features" in batch:
            scene_tokens = batch["image_features"]
            scene_tokens = scene_tokens.to(
                device=self.img_token_proj.weight.device,
                dtype=self.img_token_proj.weight.dtype,
                non_blocking=True,
            )
        elif self.cfg.aggregator.name == "PrecomputedDINOv3":
            raise ValueError(
                "PrecomputedDINOv3 requires batch['image_features']; configure the "
                "dataset with modalities: [image_features, gt_latent]"
            )
        elif self.cfg.aggregator.name == "DepthAnything3Net":
            aggregated_tokens_list, _, _ = self.forward_da3(
                images,
                batch=batch,
                ref_view_strategy="first",
            )
            scene_tokens = aggregated_tokens_list[-1]  # [B, S, K, C]
        elif self.cfg.aggregator.name == "AggregatorPts3D":
            aggregated_tokens_list, _, _ = self.forward_vggt(images)
            scene_tokens = aggregated_tokens_list[-1]  # [B, S, K, C]
        tokens = self.img_token_proj(scene_tokens)

        if self.use_token_ln and self.token_norm is not None:
            tokens = self.token_norm(tokens)

        return {
            "pointmaps": None,
            "tokens": tokens,
        }

    def _decode(self, tokens, images, token_mask=None, query_points=None, timestep=None, **kwargs):
        """Decode 3D tokens into point cloud coordinates using flow matching ODE solver."""
        B, S = images.shape[:2]

        predictions = {}

        # [B], assuming each image in the sequence is a different view
        num_views = torch.ones(B, device=images.device) * S
        # with torch.cuda.amp.autocast(enabled=False):

        if self.pts3d_head is not None:
            aggregated_tokens_3d_list = [tokens]

            pts3d_xyz = self.pts3d_head(
                aggregated_tokens_3d_list,
                mask=token_mask,
                query_points=query_points,
                timestep=timestep,
                num_views=num_views,
                conditioning_latents=kwargs.get("conditioning_latents", None),
                visible_geometry_latents=kwargs.get("visible_geometry_latents", None),
            )

            predictions["pts3d_xyz"] = pts3d_xyz
            predictions["pts3d_xyz_rel"] = pts3d_xyz
            predictions["pts3d_rgb"] = pts3d_xyz
            predictions["pts3d_conf"] = torch.ones_like(pts3d_xyz[..., [0]])
            predictions["center_xyz"] = pts3d_xyz
            predictions["query_points"] = query_points
            predictions["timestep"] = timestep

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

    def _prepare_lightning_batch(self, batch, deterministic_seed):
        latent_parameter = self.pts3d_head.x_embedder.weight

        #load or compute gt_latents
        gt_latents = resolve_precomputed_gt_latents(
            batch,
            device=latent_parameter.device,
            dtype=latent_parameter.dtype,
            sample_posterior=self.sample_posterior,
            deterministic_seed=deterministic_seed,
        )
        if gt_latents is None:
            posterior_seed = (
                deterministic_seed
                if deterministic_seed is not None
                else 42 if self.sample_posterior is False else None
            )
            with torch.no_grad():
                gt_latents = self.first_stage.encode(
                    batch["surface"],
                    sample_posterior=self.sample_posterior != "mode",
                    posterior_seed=posterior_seed,
                )

        images = (
            torch.stack(batch["images"], dim=1)
            if "images" in batch
            else None
        )
        #save_batch_debug({**batch, "images": images, "gt_latents": gt_latents}, "debug_points/lightning_batch", max_samples=2, batched=True)
        encoder_data = self._encode(images, batch=batch)
        tokens = encoder_data["tokens"]
        if images is None:
            # ODE plumbing uses images only to recover batch/view counts. Keep
            # this tiny placeholder off the dataset and never invoke an encoder.
            images = tokens.new_zeros((tokens.shape[0], 1, 0, 0, 0))

        conditioning_latents = batch.get("conditioning_latents")
        if conditioning_latents is not None:
            conditioning_latents = conditioning_latents.to(
                device=tokens.device,
                dtype=tokens.dtype,
            ).contiguous()

        visible_geometry_latents = batch.get("visible_geometry_latents")
        if visible_geometry_latents is not None:
            visible_geometry_latents = visible_geometry_latents.to(
                device=tokens.device,
                dtype=tokens.dtype,
            ).contiguous()

        encoder_data["conditioning_latents"] = conditioning_latents
        gt_latents = gt_latents.to(device=tokens.device, dtype=tokens.dtype).contiguous()
        return (
            images,
            encoder_data,
            tokens,
            gt_latents,
            conditioning_latents,
            visible_geometry_latents,
        )

    def _reconstruct_lightning_latents(
        self,
        images,
        encoder_data,
        gt_latents,
        conditioning_latents,
        visible_geometry_latents,
    ):
        with torch.no_grad():
            wrapped_vf = BatchModelWrapper(model=self)
            wrapped_vf.eval()
            self.eval()

            noise_generator = torch.Generator(device=gt_latents.device)
            noise_generator.manual_seed(42)
            x_init = torch.randn_like(gt_latents, generator=noise_generator)
            sol = ODESolver(velocity_model=wrapped_vf).sample(
                time_grid=torch.linspace(0, 1, 101, device=gt_latents.device),
                x_init=x_init,
                method="euler",
                step_size=0.01,
                return_intermediates=False,
                images=images,
                encoder_data=encoder_data,
                conditioning_latents=conditioning_latents,
                visible_geometry_latents=visible_geometry_latents,
            )

        return self.first_stage.decode(sol)

    def lightning_forward(
        self,
        batch: dict,
        criterion,
        kl_weight=0.0,
        recon_latents=False,
        deterministic_seed=None,
    ):
        (
            images,
            encoder_data,
            tokens,
            gt_latents,
            conditioning_latents,
            visible_geometry_latents,
        ) = self._prepare_lightning_batch(batch, deterministic_seed)

        x_t, dx_t, t = sample_linear_flow_path(
            gt_latents,
            deterministic_seed,
            timestep_sampling=self.timestep_sampling,
            timestep_mu=self.timestep_mu,
            timestep_sigma=self.timestep_sigma,
            timestep_p_zero=self.timestep_p_zero,
        )
        v_pred = self.pts3d_head(
            decout=[tokens],
            noised_latents=x_t,
            timestep=t[:, None].expand_as(x_t[..., 0]),
            conditioning_latents=conditioning_latents,
            visible_geometry_latents=visible_geometry_latents,
        )
        loss, velocity_loss, reconstruction_loss, pred_latents = flow_matching_losses(
            v_pred,
            dx_t,
            x_t,
            t,
            gt_latents,
            self.latent_reconstruction_loss_weight,
        )

        # Preserve per-sample errors so the Lightning module can aggregate
        # validation metrics by timestep bucket. Keep these detached: they are
        # diagnostics and must not retain the training graph.
        reduce_dims = tuple(range(1, v_pred.ndim))
        velocity_mse_per_sample = (v_pred - dx_t).square().mean(dim=reduce_dims)
        reconstruction_mse_per_sample = (
            (pred_latents - gt_latents).square().mean(dim=reduce_dims)
        )
        details = {
            "velocity_loss": float(velocity_loss.detach()),
            "latent_reconstruction_loss": float(reconstruction_loss.detach()),
            "timestep_bucket_samples": {
                "t": t.detach(),
                "velocity_mse": velocity_mse_per_sample.detach(),
                "reconstruction_mse": reconstruction_mse_per_sample.detach(),
            },
        }

        latents = None
        if recon_latents:
            latents = self._reconstruct_lightning_latents(
                images,
                encoder_data,
                gt_latents,
                conditioning_latents,
                visible_geometry_latents,
            )

        return loss, details, latents
