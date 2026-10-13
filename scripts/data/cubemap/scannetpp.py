import os
from pathlib import Path
import subprocess
import concurrent.futures
import argparse
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR = Path(__file__).with_name("generate.py")


def find_matching_mask(mask_dir, pano_entry):
    matches = sorted(path for path in mask_dir.glob(f"{pano_entry.stem}.*") if path.is_file())
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple anonymization masks match {pano_entry.name}: {matches}")
    return matches[0]


def find_matching_depth(panocam_dir, images_dir_name, pano_entry, depth_dir_name):
    if depth_dir_name:
        candidate_names = [depth_dir_name]
    elif images_dir_name == "resized_images":
        candidate_names = ["resized_depth", "resized_depths", "depth", "depths"]
    else:
        candidate_names = ["depth", "depths"]

    matches = []
    for directory_name in candidate_names:
        directory = panocam_dir / directory_name
        matches.extend(
            path
            for path in directory.glob(f"{pano_entry.stem}.*")
            if path.is_file()
        )
    matches = sorted(set(matches))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple depth maps match {pano_entry.name}: {matches}")
    return matches[0]


def process_panorama(
    scene_name,
    panocam_dir,
    images_dir_name,
    pano_entry,
    output_root,
    args,
):
    print(f"Processing panorama: {pano_entry.name} in scene: {scene_name}")

    # 1. Define input image path
    image_path = str(pano_entry)

    # 2. Replicate the directory structure directly inside cubemap_data/<scene_name>
    # Result: cubemap_data/{scene_name}/{panocam_dir_name}/{images_dir_name}/
    output_dir = os.path.join(output_root, scene_name, panocam_dir.name, images_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Define output path base filename
    image_output_path = os.path.join(output_dir, pano_entry.stem)

    mask_dir_name = "resized_anon_mask" if images_dir_name == "resized_images" else "anon_mask"
    mask_path = find_matching_mask(panocam_dir / mask_dir_name, pano_entry)
    depth_path = find_matching_depth(
        panocam_dir, images_dir_name, pano_entry, args.depth_dir_name
    )
    if args.require_depth and depth_path is None:
        raise FileNotFoundError(f"No depth map found for {pano_entry}")

    command = [sys.executable, str(GENERATOR)]
    if not args.depth_only:
        if mask_path is None:
            raise FileNotFoundError(
                f"No mask found for {pano_entry} in {panocam_dir / mask_dir_name}"
            )
        mask_output_path = os.path.join(
            output_root, scene_name, panocam_dir.name, mask_dir_name, pano_entry.stem
        )
        command.extend(
            [
                "--image",
                image_path,
                "--anon_mask",
                str(mask_path),
                "--image_out",
                image_output_path,
                "--anon_mask_out",
                mask_output_path,
            ]
        )
    if depth_path is not None:
        depth_output_path = os.path.join(
            output_root,
            scene_name,
            panocam_dir.name,
            depth_path.parent.name,
            pano_entry.stem,
        )
        command.extend(["--depth", str(depth_path), "--depth_out", depth_output_path])
    command.extend(
        ["--shape", args.shape, "--fov", str(args.fov), "--res", str(args.res)]
    )

    # 4. Run the cubemap generation script
    try:
        subprocess.run(
            command,
            check=True,
            cwd=REPO_ROOT,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Cubemap generation failed for {pano_entry.name}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Could not process {pano_entry.name}: {e}") from e


def main(data_root, output_root, workers, args):
    if not data_root.is_dir():
        raise ValueError(f"Data root does not exist: {data_root}")
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")

    futures = []
    max_workers = workers

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for scene in Path(data_root).glob("*"):
            if scene.is_dir():
                scene_name = scene.name

                # Look for the panocam folder
                for panocam_dir in scene.glob("panocam*"):
                    if panocam_dir.is_dir():

                        # Look for the images folder
                        for images_dir in panocam_dir.glob("images*"):
                            if images_dir.is_dir():

                                # Loop through individual panoramas
                                for pano_entry in images_dir.glob("*"):
                                    if pano_entry.is_file() and pano_entry.suffix.lower() in [".png", ".jpg", ".jpeg"]:

                                        futures.append(
                                            executor.submit(
                                                process_panorama,
                                                scene_name,
                                                panocam_dir,
                                                images_dir.name,
                                                pano_entry,
                                                output_root,
                                                args,
                                            )
                                        )

        print(f"Submitted {len(futures)} panoramas for processing. Waiting for completion...")
        failed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failed += 1
                print(f"Failed: {error}")
        print(f"All processes completed ({len(futures) - failed} succeeded, {failed} failed).")
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ScanNet++ RGB, depth, and anonymization-mask cubemaps."
    )
    parser.add_argument("--data_root", type=Path, default=Path("/tmp/scannetpp_data/data"))
    parser.add_argument("--output_root", type=Path, default=Path("/tmp/scannetpp_data/cubemap_data"))
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument(
        "--depth-dir-name",
        help="Sibling depth panorama directory (default: auto-detect depth/depths variants)",
    )
    parser.add_argument("--require-depth", action="store_true")
    parser.add_argument("--depth-only", action="store_true")
    parser.add_argument("--shape", default="cube")
    parser.add_argument("--fov", type=float, default=95.0)
    parser.add_argument("--res", type=int, default=518)
    args = parser.parse_args()
    if args.depth_only:
        args.require_depth = True
    main(args.data_root, args.output_root, args.workers, args)
