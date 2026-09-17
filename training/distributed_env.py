"""Early CUDA/DDP environment setup shared by training entry points.

This module intentionally imports no Torch packages. Call ``configure`` before
anything that may import torch so CUDA reads the final environment.
"""

import argparse
import os


def configure() -> int | None:
    """Configure CUDA visibility and the validated NCCL transport workaround."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--devices", type=int, default=None)
    args, _ = parser.parse_known_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.devices is not None:
        if args.devices < 1:
            parser.error("--devices must be at least 1")
        # An explicit --devices value is authoritative for direct runs. This
        # also repairs stale inherited masks such as CUDA_VISIBLE_DEVICES=0
        # when two GPUs are available and --devices 2 was requested.
        requested_mapping = ",".join(str(index) for index in range(args.devices))
        current_mapping = os.environ.get("CUDA_VISIBLE_DEVICES")
        if current_mapping != requested_mapping:
            if current_mapping is not None:
                print(
                    f"Setting CUDA_VISIBLE_DEVICES={requested_mapping!r} "
                    f"for --devices {args.devices} (was {current_mapping!r})",
                    flush=True,
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = requested_mapping
        if args.devices > 1:
            # The local NCCL smoke test proved P2P/CUMEM hangs on this setup,
            # while NCCL shared-memory transport completes successfully.
            os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    return args.devices


def validate(torch_module, devices: int, accelerator: str) -> None:
    """Fail early when the requested CUDA device count is not actually visible."""
    if accelerator not in {"gpu", "cuda"}:
        return
    if devices > 1:
        # Also covers runs where the device count came from the config rather
        # than an early --devices argument. NCCL reads this at communicator init.
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    visible = torch_module.cuda.device_count()
    if visible < devices:
        mapping = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        raise RuntimeError(
            f"Requested {devices} CUDA devices, but PyTorch sees {visible}. "
            f"CUDA_VISIBLE_DEVICES={mapping!r}. Request enough GPUs from the "
            "scheduler or correct the container GPU mapping."
        )
    print(
        f"CUDA setup: requested={devices}, visible={visible}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
        f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', '0')}",
        flush=True,
    )
