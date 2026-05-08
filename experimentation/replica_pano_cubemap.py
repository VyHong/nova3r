import os
from pathlib import Path
import subprocess
import concurrent.futures

data_root = "/tmp/datasets/replica_pano/"

def process_sequence(scene, scene_name, seq_entry):
    print(f"Found sequence: {seq_entry.name} in scene {scene_name}")
    image_path = os.path.join(scene, scene_name, "Scene_Info", seq_entry, "rgb.png")
    image_output_path = os.path.join(scene, scene_name, "Scene_Info", seq_entry, "rgb")
    subprocess.run(["python", "experimentation/image_generation.py", "--image", image_path, "--image_out", image_output_path], check=True)

def main():
    print(f"Data root: {data_root}")
    
    futures = []
    # Using ThreadPoolExecutor since the tasks are mainly waiting for subprocesses
    max_workers = min(16, os.cpu_count()) # reasonable default
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for scene in Path(data_root).glob("*"):

            if scene.is_dir():  # Process only office_0 for now:
                print(f"Processing scene: {scene.name}")
                scene_info_dir = Path(os.path.join(scene, scene.name, "Scene_Info"))
                if scene_info_dir.exists():
                    for seq_entry in scene_info_dir.glob("*"):
                        futures.append(executor.submit(process_sequence, scene, scene.name, seq_entry))
        
        # Wait for completion
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except subprocess.CalledProcessError as e:
                print(f"Process failed: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()