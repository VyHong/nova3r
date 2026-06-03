import os
from pathlib import Path
import subprocess
import concurrent.futures

# Define your input root and output root
data_root = "/tmp/scannetpp_data/data"  # Update this to your actual input path
output_root = "/tmp/scannetpp_data/cubemap_data"


def process_panorama(scene_name, panocam_dir_name, images_dir_name, pano_entry):
    print(f"Processing panorama: {pano_entry.name} in scene: {scene_name}")

    # 1. Define input image path
    image_path = str(pano_entry)

    # 2. Replicate the directory structure directly inside cubemap_data/<scene_name>
    # Result: cubemap_data/{scene_name}/{panocam_dir_name}/{images_dir_name}/
    output_dir = os.path.join(output_root, scene_name, panocam_dir_name, images_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Define output path base filename
    image_output_path = os.path.join(output_dir, pano_entry.stem)

    # 4. Run the cubemap generation script
    try:
        subprocess.run(["python", "experimentation/image_generation.py", "--image", image_path, "--image_out", image_output_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Process failed for {pano_entry.name}: {e}")
    except Exception as e:
        print(f"An error occurred while processing {pano_entry.name}: {e}")


def main():
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")

    futures = []
    max_workers = min(16, os.cpu_count())

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

                                        futures.append(executor.submit(process_panorama, scene_name, panocam_dir.name, images_dir.name, pano_entry))

        print(f"Submitted {len(futures)} panoramas for processing. Waiting for completion...")
        concurrent.futures.wait(futures)
        print("All processes completed.")


if __name__ == "__main__":
    main()
