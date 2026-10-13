import asyncio
import json
import sys
import os
import argparse
import numpy as np
from PIL import Image

# Add the server directory to the Python path to import from layout.py
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, server_dir)

# from tex_utils import export_layout_to_mesh_dict_list
from tex_utils_local import export_layout_to_mesh_dict_list_v2
from glb_utils import (
    create_glb_scene,
    add_textured_mesh_to_glb_scene,
    save_glb_scene
)
from utils import (
    dict_to_floor_plan,
)


def get_material_pbr_factors(texture_info):
    """Read PBR factors from current SAGE metadata and legacy flat keys."""
    pbr_parameters = texture_info.get("pbr_parameters") or {}
    metallic = texture_info.get(
        "metallic_factor", pbr_parameters.get("metallic", 0.0)
    )
    roughness = texture_info.get(
        "roughness_factor", pbr_parameters.get("roughness", 1.0)
    )
    return float(metallic), float(roughness)


def export_glb(layout_file_path: str, output_dir: str, output_filename: str | None = None):
    """Test loading layout from JSON file"""
    

    with open(layout_file_path, 'r') as f:
        layout_data = json.load(f)
    current_layout = dict_to_floor_plan(layout_data)

    os.makedirs(output_dir, exist_ok=True)
    export_glb_path = os.path.join(
        output_dir,
        output_filename or os.path.basename(layout_file_path).replace(".json", ".glb"),
    )
    mesh_dict_list = export_layout_to_mesh_dict_list_v2(current_layout, os.path.dirname(layout_file_path))
    scene = create_glb_scene()
    for mesh_id, mesh_data in mesh_dict_list.items():
        metallic_factor, roughness_factor = get_material_pbr_factors(
            mesh_data['texture']
        )
        mesh_data_dict = {
            'vertices': mesh_data['mesh'].vertices,
            'faces': mesh_data['mesh'].faces,
            'vts': mesh_data['texture']['vts'],
            'fts': mesh_data['texture']['fts'],
            'texture_image': np.array(Image.open(mesh_data['texture']['texture_map_path'])),
            'metallic_factor': metallic_factor,
            'roughness_factor': roughness_factor
        }
        add_textured_mesh_to_glb_scene(
            mesh_data_dict, 
            scene, 
            material_name=f"material_{mesh_id}", 
            mesh_name=f"mesh_{mesh_id}", 
            preserve_coordinate_system=True,
        )
    save_glb_scene(export_glb_path, scene)
    print(f"GLB exported to: {os.path.abspath(export_glb_path)}")
    return export_glb_path

        

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Export layout to GLB file")
    parser.add_argument("layout_file_path", type=str, help="Layout file path")
    parser.add_argument("output_dir", type=str, help="Output directory")
    args = parser.parse_args()
    
    # Run the test
    export_glb(args.layout_file_path, args.output_dir)
