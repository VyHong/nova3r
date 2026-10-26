#!/usr/bin/env python3
"""Warm the OS page cache for assets referenced by a 3D-FRONT manifest."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import threading
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--future-root", type=Path,
                        default=Path("/tmp/datasets/3dfront/future_model"))
    parser.add_argument("--texture-root", type=Path,
                        default=Path("/tmp/datasets/3dfront/3D-FRONT-texture"))
    parser.add_argument("--hdri-root", type=Path,
                        default=Path("/tmp/datasets/3dfront/hdris"))
    parser.add_argument("--cc-root", type=Path,
                        default=Path("/tmp/datasets/3dfront/cctextures"))
    parser.add_argument("--include-cc", action="store_true",
                        help="Also read the large CC-material collection")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-scenes", type=int, default=0,
                        help="Warm only the first N unique dwellings; 0 means all")
    return parser.parse_args()


def files_below(path):
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from (item for item in path.rglob("*") if item.is_file())


def main():
    args = parse_args()
    jobs = json.loads(args.manifest.read_text())
    scenes = list(dict.fromkeys(Path(job["scene"]) for job in jobs))
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    paths = set()
    for scene in scenes:
        paths.add(scene)
        data = json.loads(scene.read_text())
        for furniture in data.get("furniture", []):
            paths.update(files_below(args.future_root / furniture["jid"]))
        for material in data.get("material", []):
            jid = material.get("jid")
            if jid:
                paths.update(files_below(args.texture_root / jid))
    paths.update(files_below(args.hdri_root))
    if args.include_cc:
        paths.update(files_below(args.cc_root))

    paths = sorted(paths)
    sizes = {path: path.stat().st_size for path in paths}
    total = sum(sizes.values())
    completed = 0
    lock = threading.Lock()
    started = time.perf_counter()

    def warm(path):
        with path.open("rb", buffering=0) as stream:
            while stream.read(8 * 1024 * 1024):
                pass
        return sizes[path]

    print(f"Warming {len(paths):,} files ({total / 2**30:.1f} GiB) from "
          f"{len(scenes):,} dwellings", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(warm, path) for path in paths]
        for future in as_completed(futures):
            amount = future.result()
            with lock:
                completed += amount
                elapsed = max(time.perf_counter() - started, 1e-9)
                rate = completed / elapsed
                fraction = completed / total if total else 1.0
                eta = (total - completed) / rate if rate else 0.0
                print(f"\rCache {fraction:6.2%} | {completed / 2**30:.1f}/"
                      f"{total / 2**30:.1f} GiB | {rate / 2**30:.2f} GiB/s | "
                      f"ETA {eta:.0f}s", end="", file=sys.stderr, flush=True)
    print("\nPage-cache warm-up complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
