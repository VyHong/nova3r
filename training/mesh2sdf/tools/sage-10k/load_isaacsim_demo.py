from isaacsim_utils import (
    start_simulation_app,
    get_layout_scene_loaded,
    simulate_the_scene,
    generate_physics_statistics
)
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export USD file")
    parser.add_argument("layout_json_path", type=str, help="Layout JSON file path")
    args = parser.parse_args()

    start_simulation_app()
    track_ids = get_layout_scene_loaded(args.layout_json_path)
    result = simulate_the_scene(track_ids)

    # print(result)
    unstable_objects = result["unstable_objects"]
    traced_data_all = result["traced_data_all"]

    # Generate physics statistics with pre-computed collision record
    physics_statistics = generate_physics_statistics(traced_data_all, track_ids)
    
    # Print summary
    print(f"\n=== Physics Simulation Results ===")
    print(f"Total objects: {physics_statistics['total_objects']}")
    print(f"Stable objects: {physics_statistics['stable_objects']}")
    print(f"Unstable objects: {physics_statistics['unstable_objects']}")
    print(f"Stability ratio: {physics_statistics['stability_ratio']:.2%}")
    