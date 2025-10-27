#!/usr/bin/env python3
"""
Batch processing script for MorphIt sphere packing on multiple mesh files.
Processes all mesh files in a directory and saves results with panda_ prefix.
"""


import os
import sys
from pathlib import Path
from config import get_config, update_config_from_dict
from morphit import MorphIt
from scripts.create_box_urdf import modify_box_urdf


def create_spheres(input_dir, output_dir, num_spheres_per_joint):
    # MorphIt configuration
    config_updates = {
        "model.num_spheres": num_spheres_per_joint,
        "training.iterations": 50,
        "training.verbose_frequency": 10,
        "training.logging_enabled": False,
        "training.density_control_min_interval": 25,
        "visualization.enabled": False,
        "visualization.off_screen": True,
        "visualization.save_video": False,
    }

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get input directory path
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)

    # Find all mesh files (common mesh formats)
    mesh_extensions = [".obj", ".stl", ".ply", ".dae", ".mesh"]
    mesh_files = []

    for ext in mesh_extensions:
        mesh_files.extend(input_path.glob(f"*{ext}"))

    if not mesh_files:
        print(f"No mesh files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(mesh_files)} mesh files to process")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    # Process each mesh file
    for mesh_file in sorted(mesh_files):
        mesh_name = mesh_file.stem  # Get filename without extension
        output_name = f"{mesh_name}.json"
        output_path = Path(output_dir) / output_name

        print(f"Processing: {mesh_file.name} -> {output_name}")

        try:
            # Get base configuration
            config = get_config("MorphIt-B")

            # Update mesh path and output filename
            config_updates["model.mesh_path"] = str(mesh_file)
            config_updates["output_filename"] = output_name
            config_updates["results_dir"] = output_dir

            # Apply configuration updates
            config = update_config_from_dict(config, config_updates)

            # Create and train MorphIt model
            model = MorphIt(config)

            print(f"  - Training MorphIt model...")
            tracker = model.train()

            # Save results
            print(f"  - Saving results to {output_name}")
            model.save_results()

            print(f"  - Completed successfully!")

        except Exception as e:
            print(f"  - Error processing {mesh_file.name}: {str(e)}")
            continue

        print()

    print(f"Spheres saved in: {output_dir}")


if __name__ == "__main__":

    num_spheres_per_joint = 20
    asset_name = 'pink_box'

    mesh_dir = "../assets/urdfs/pink_box/" 
    spheres_dir = f"results/spheres/{asset_name}_{num_spheres_per_joint}"

    create_spheres(input_dir=mesh_dir, output_dir=spheres_dir, num_spheres_per_joint=num_spheres_per_joint)

    orig_urdf = "../assets/urdfs/pink_box/pink_box.urdf"
    output_urdf = f"results/urdfs/{asset_name}_{num_spheres_per_joint}.urdf"

    modify_box_urdf(input_urdf_path=orig_urdf, spheres_dir=spheres_dir, output_urdf_path=output_urdf)