#!/usr/bin/env python3
"""Build a deterministic manifest of furnished, windowed 3D-FRONT rooms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SUPPORTED_ROOM_TYPES = {
    "LivingDiningRoom", "LivingRoom", "MasterBedroom", "SecondBedroom",
    "Bedroom", "DiningRoom", "Study", "KidsRoom", "Kitchen",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path,
                        default=Path("/tmp/datasets/3dfront/3D-FRONT"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=2100)
    parser.add_argument("--min-furniture", type=int, default=4)
    parser.add_argument("--rooms-per-scene", type=int, default=1,
                        help="Maximum eligible rooms selected from each dwelling")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[],
                        help=("Exclude scene/room pairs from an existing renderer manifest; "
                              "may be supplied more than once"))
    args = parser.parse_args()
    if args.target < 1 or args.min_furniture < 0 or args.rooms_per_scene < 1:
        raise SystemExit(
            "target and rooms-per-scene must be positive; min-furniture must be non-negative"
        )

    excluded = set()
    for manifest_path in args.exclude_manifest:
        for job in json.loads(manifest_path.read_text()):
            excluded.add((str(Path(job["scene"]).resolve()), job["room"]))

    jobs = []
    for scene_path in sorted(args.scenes.glob("*.json")):
        data = json.loads(scene_path.read_text())
        meshes = {mesh["uid"]: mesh for mesh in data.get("mesh", [])}
        scene_jobs = []
        for room in data.get("scene", {}).get("room", []):
            children = room.get("children", [])
            mesh_types = {
                meshes[child["ref"]].get("type", "").lower()
                for child in children if child.get("ref") in meshes
            }
            furniture_count = sum(child.get("ref") not in meshes for child in children)
            if (room.get("type") in SUPPORTED_ROOM_TYPES
                    and {"floor", "window"} <= mesh_types
                    and furniture_count >= args.min_furniture):
                job = {
                    "scene": str(scene_path.resolve()),
                    "room": room["instanceid"],
                    "room_type": room.get("type"),
                    "furniture_count": furniture_count,
                }
                if (job["scene"], job["room"]) not in excluded:
                    scene_jobs.append(job)
        # Prefer the more richly furnished room when limiting each dwelling.
        scene_jobs.sort(key=lambda job: (-job["furniture_count"], job["room"]))
        jobs.extend(scene_jobs[:args.rooms_per_scene])
        if len(jobs) >= args.target:
            break

    jobs = jobs[:args.target]

    if len(jobs) != args.target:
        raise RuntimeError(
            f"Found only {len(jobs)} eligible rooms; requested {args.target}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jobs, indent=2) + "\n")
    print(f"Wrote {len(jobs)} rooms to {args.output}")


if __name__ == "__main__":
    main()
