"""Small, typed setup helpers shared by Lightning modules."""

from dataclasses import dataclass
from typing import Any

from nova3r.heads.hunyuan_model.autoencoders.model import ShapeVAEDecoder
from nova3r.models.nova3r_img_cond import _load_checkpoint_state_dict


@dataclass(frozen=True)
class OptimizerOptions:
    learning_rate: float
    scheduler: str
    warmup_epochs: int
    eta_min: float

    @classmethod
    def from_config(cls, cfg):
        return cls(
            learning_rate=float(cfg.lr),
            scheduler=str(cfg.get("lr_scheduler", "exponential")),
            warmup_epochs=int(cfg.get("warmup_epochs", 0)),
            eta_min=float(cfg.get("lr_scheduler_eta_min", 1e-6)),
        )


@dataclass(frozen=True)
class FlowTrainingOptions:
    latent_reconstruction_loss_weight: float
    sample_posterior: Any
    timestep_sampling: str
    timestep_mu: float
    timestep_sigma: float
    timestep_p_zero: float

    @classmethod
    def from_config(cls, cfg):
        sample_posterior = cfg.get("sample_posterior", True)
        if sample_posterior not in (True, False, "mode"):
            raise ValueError(
                "sample_posterior must be true, false, or 'mode', "
                f"got {sample_posterior!r}"
            )
        return cls(
            latent_reconstruction_loss_weight=float(
                cfg.get("latent_reconstruction_loss_weight", 0.1)
            ),
            sample_posterior=sample_posterior,
            timestep_sampling=str(cfg.get("timestep_sampling", "uniform")),
            timestep_mu=float(cfg.get("timestep_mu", -0.8)),
            timestep_sigma=float(cfg.get("timestep_sigma", 0.8)),
            timestep_p_zero=float(cfg.get("timestep_p_zero", 0.1)),
        )

    def apply_to(self, model):
        for name, value in vars(self).items():
            setattr(model, name, value)


def build_frozen_shape_decoder(cfg, model):
    """Build the optional validation decoder used by standalone latent models."""
    if hasattr(model, "first_stage") or "vae" not in cfg:
        return None

    checkpoint = cfg.get("vae_ckpt")
    if not checkpoint:
        raise ValueError("vae_ckpt is required when experiment.vae is configured")

    decoder = ShapeVAEDecoder(**cfg.vae.params)
    decoder.load_state_dict(
        _load_checkpoint_state_dict(str(checkpoint), component="vae"),
        strict=True,
    )
    decoder.requires_grad_(False)
    decoder.eval()
    print("Loaded frozen ShapeVAE decoder")
    return decoder


def freeze_layers(model, patterns):
    """Freeze parameters whose names contain one of the configured patterns."""
    patterns = tuple(patterns or ())
    matched = set()
    for name, parameter in model.named_parameters():
        pattern = next((pattern for pattern in patterns if pattern in name), None)
        if pattern is not None:
            parameter.requires_grad_(False)
            matched.add(pattern)

    for pattern in patterns:
        if pattern in matched:
            print(f"Froze: {pattern}")
        else:
            print(f"Freeze pattern matched no parameters: {pattern}")
    return matched
