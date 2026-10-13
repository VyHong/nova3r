#!/usr/bin/env python3
"""Hold GPU memory and run a periodic CUDA workload until terminated."""

import argparse
import math
import signal
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--busy-seconds", type=float, default=1.0)
    parser.add_argument("--idle-seconds", type=float, default=4.0)
    parser.add_argument("--memory-fraction", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.memory_fraction < 1:
        parser.error("--memory-fraction must be greater than 0 and less than 1")
    return args


def main():
    args = parse_args()
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for the GPU keepalive")

    device = torch.device(args.device)
    left = torch.randn(
        args.matrix_size, args.matrix_size, device=device, dtype=torch.float16
    )
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    target_memory = math.ceil(total_memory * args.memory_fraction)
    padding_bytes = max(0, target_memory - torch.cuda.memory_allocated(device))
    # Keep this tensor alive throughout the workload, and touch every byte so
    # the allocation is backed by GPU memory even during idle intervals.
    memory_padding = torch.empty(padding_bytes, device=device, dtype=torch.uint8)
    memory_padding.zero_()
    torch.cuda.synchronize(device)
    print(
        f"GPU keepalive started on {device} "
        f"({args.busy_seconds}s busy / {args.idle_seconds}s idle; "
        f"holding {torch.cuda.memory_allocated(device) / total_memory:.1%} "
        f"of GPU memory)",
        flush=True,
    )

    with torch.inference_mode():
        while running:
            busy_until = time.monotonic() + args.busy_seconds
            while running and time.monotonic() < busy_until:
                torch.mm(left, right, out=output)
            torch.cuda.synchronize(device)
            if running:
                time.sleep(args.idle_seconds)

    print("GPU keepalive stopped", flush=True)


if __name__ == "__main__":
    main()
