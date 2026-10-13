"""Checkpoint size reductions that preserve resumable training state."""

import torch


FROZEN_VAE_PREFIX = "first_stage."
ADAM_MOMENT_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


def extract_model_state_dict(checkpoint):
    """Extract the model, excluding separate Lightning-module components."""
    if "model" in checkpoint:
        return checkpoint["model"]
    if "state_dict" not in checkpoint:
        return checkpoint
    state_dict = checkpoint["state_dict"]
    if any(key.startswith("model.") for key in state_dict):
        return {
            key[len("model."):]: value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
    return state_dict


def remove_frozen_vae_state(checkpoint):
    """Remove the externally loaded, frozen ShapeVAE from a checkpoint."""
    # Do not mutate a state-dict container that the checkpoint connector may
    # share with live training state.
    state_dict = checkpoint.get("state_dict", {}).copy()
    removed = [key for key in state_dict if key.startswith(FROZEN_VAE_PREFIX)]
    for key in removed:
        del state_dict[key]
    checkpoint["state_dict"] = state_dict
    return removed


def restore_frozen_vae_state(checkpoint, module_state_dict):
    """Supply ShapeVAE state from the already initialized Lightning module."""
    state_dict = checkpoint.setdefault("state_dict", {})
    restored = []
    for key, value in module_state_dict.items():
        if key.startswith(FROZEN_VAE_PREFIX) and key not in state_dict:
            state_dict[key] = value
            restored.append(key)
    return restored


def cast_adam_moments(checkpoint, dtype):
    """Cast Adam moments in an isolated checkpoint copy.

    Optimizer ``state_dict`` containers can share their nested dictionaries
    with the live optimizer. Copy every container that we modify so saving a
    compact checkpoint never changes the optimizer used by training.
    """
    converted = 0
    converted_optimizers = []
    for optimizer in checkpoint.get("optimizer_states", ()):
        converted_optimizer = optimizer.copy()
        converted_states = {}
        for parameter_id, parameter_state in optimizer.get("state", {}).items():
            converted_parameter_state = parameter_state.copy()
            for key in ADAM_MOMENT_KEYS:
                value = parameter_state.get(key)
                if torch.is_tensor(value) and value.is_floating_point() and value.dtype != dtype:
                    converted_parameter_state[key] = value.to(dtype=dtype)
                    converted += 1
            converted_states[parameter_id] = converted_parameter_state
        converted_optimizer["state"] = converted_states
        converted_optimizers.append(converted_optimizer)
    checkpoint["optimizer_states"] = converted_optimizers
    return converted
