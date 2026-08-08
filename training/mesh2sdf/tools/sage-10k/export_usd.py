from isaacsim_utils import (
    start_simulation_app,
    get_room_layout_scene_usd_separate_from_layout
)
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export USD file")
    parser.add_argument("layout_json_path", type=str, help="Layout JSON file path")
    parser.add_argument("output_dir", type=str, help="Output directory")
    args = parser.parse_args()

    start_simulation_app()
    result = get_room_layout_scene_usd_separate_from_layout(args.layout_json_path, args.output_dir)
    if result["status"] == "success":
        print(f"USD file exported to {args.output_dir}")
    else:
        print(f"Failed to export USD file: {result['message']}")