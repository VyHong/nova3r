import argparse
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys


def process_sequence(scene, scene_name, seq_entry, args):
    print(f"Found sequence: {seq_entry.name} in scene {scene_name}")
    sequence_dir = scene / scene_name / "Scene_Info" / seq_entry.name
    if args.output_root:
        output_sequence_dir = (
            Path(args.output_root)
            / scene_name
            / scene_name
            / "Scene_Info"
            / seq_entry.name
        )
    else:
        output_sequence_dir = sequence_dir
    command = [
        sys.executable,
        str(Path(__file__).with_name("generate.py")),
        "--depth",
        str(sequence_dir / args.depth_name),
        "--depth_out",
        str(output_sequence_dir / "depth"),
        "--shape",
        args.shape,
        "--fov",
        str(args.fov),
        "--res",
        str(args.res),
    ]
    if not args.depth_only:
        command.extend(
            [
                "--image",
                str(sequence_dir / args.image_name),
                "--image_out",
                str(output_sequence_dir / "rgb"),
            ]
        )
    subprocess.run(command, check=True)

def main():
    parser = argparse.ArgumentParser(
        description="Generate matching RGB/depth cubemap faces for ReplicaPano."
    )
    parser.add_argument("--data-root", default="/tmp/datasets/replica_pano")
    parser.add_argument(
        "--output-root",
        help="Writable output root (default: write alongside source panoramas)",
    )
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, help="Process at most this many sequences")
    parser.add_argument("--depth-only", action="store_true")
    parser.add_argument("--image-name", default="rgb.png")
    parser.add_argument("--depth-name", default="depth.png")
    parser.add_argument("--shape", default="cube")
    parser.add_argument("--fov", type=float, default=95.0)
    parser.add_argument("--res", type=int, default=518)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    data_root = Path(args.data_root)
    print(f"Data root: {data_root}")
    
    futures = []
    failures = 0
    submitted = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for scene in data_root.glob("*"):
            if args.limit is not None and submitted >= args.limit:
                break

            if scene.is_dir():
                print(f"Processing scene: {scene.name}")
                scene_info_dir = scene / scene.name / "Scene_Info"
                if scene_info_dir.exists():
                    for seq_entry in scene_info_dir.glob("*"):
                        if args.limit is not None and submitted >= args.limit:
                            break
                        if seq_entry.is_dir():
                            futures.append(
                                executor.submit(
                                    process_sequence, scene, scene.name, seq_entry, args
                                )
                            )
                            submitted += 1
        
        # Wait for completion
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except subprocess.CalledProcessError as e:
                failures += 1
                print(f"Process failed: {e}")
            except Exception as e:
                failures += 1
                print(f"An error occurred: {e}")
    if failures:
        raise SystemExit(f"{failures} sequence(s) failed")

if __name__ == '__main__':
    main()
