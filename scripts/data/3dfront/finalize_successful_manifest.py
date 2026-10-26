#!/usr/bin/env python3
"""Write a manifest containing exactly N fully rendered 3D-FRONT rooms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def complete(folder: Path, room: str, views: int, width: int, samples: int) -> bool:
    required = [folder / "room.glb", folder / "metadata.json", folder / "extrinsics.json"]
    required.extend(folder / f"{room}_panorama_{index:03d}.png" for index in range(views))
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        metadata = json.loads((folder / "metadata.json").read_text())
    except (OSError, ValueError):
        return False
    return (metadata.get("view_count") == views
            and metadata.get("resolution") == [width, width // 2]
            and metadata.get("render_samples") == samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panorama-list", type=Path)
    parser.add_argument("--target", type=int, default=2100)
    parser.add_argument("--views", type=int, default=10)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--width", type=int, default=1036)
    args = parser.parse_args()

    selected = []
    seen = set()
    for manifest_path in args.manifests:
        for job in json.loads(manifest_path.read_text()):
            key = (str(Path(job["scene"]).resolve()), job["room"])
            if key in seen:
                continue
            seen.add(key)
            folder = args.render_root / Path(job["scene"]).stem / job["room"]
            if complete(folder, job["room"], args.views, args.width, args.samples):
                selected.append(job)
                if len(selected) == args.target:
                    break
        if len(selected) == args.target:
            break

    if len(selected) != args.target:
        raise RuntimeError(
            f"Found only {len(selected)} complete rooms; requested {args.target}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2) + "\n")

    panorama_path = args.panorama_list
    if panorama_path is not None:
        panoramas = []
        for job in selected:
            relative = Path(Path(job["scene"]).stem) / job["room"]
            panoramas.extend(
                (relative / f"{job['room']}_panorama_{index:03d}.png").as_posix()
                for index in range(args.views)
            )
        panorama_path.parent.mkdir(parents=True, exist_ok=True)
        panorama_path.write_text(json.dumps(panoramas, indent=2) + "\n")

    print(f"Wrote {len(selected)} complete rooms to {args.output}")
    if panorama_path is not None:
        print(f"Wrote {len(selected) * args.views} panoramas to {panorama_path}")


if __name__ == "__main__":
    main()
