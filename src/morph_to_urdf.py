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
from scripts.create_panda_urdf import read_json_spheres, create_fixed_joint, create_sphere_link


def create_spheres(input_dir, output_dir, num_spheres_per_joint, asset_name):
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
        output_name = f"{asset_name}_{mesh_name}.json"
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

import os
import xml.etree.ElementTree as ET

def modify_urdf(input_urdf_path, spheres_dir, output_urdf_path):
    """Modify URDF to give each mesh link its own sphere-based sublinks."""
    # Parse URDF preserving comments
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(input_urdf_path, parser=parser)
    root = tree.getroot()

    # Register namespaces
    ET.register_namespace("", "")
    ET.register_namespace("drake", "http://drake.mit.edu")

    # Add world link and fixed joint if missing
    world_link = root.find(".//link[@name='world']")
    if world_link is None:
        world_link = ET.Element("link", name="world")
        root.insert(0, world_link)

        # Connect world to the first existing link
        all_links = root.findall(".//link")
        if len(all_links) > 1:
            base_link_name = all_links[1].get("name")  # e.g. "part_1"
        else:
            base_link_name = "part_1"

        world_joint = ET.Element("joint", name="world_to_base", type="fixed")
        ET.SubElement(world_joint, "parent", link="world")
        ET.SubElement(world_joint, "child", link=base_link_name)
        ET.SubElement(world_joint, "origin", xyz="0 0 0", rpy="0 0 0")
        root.insert(1, world_joint)

    # Process each link (skip world)
    for link in root.findall(".//link"):
        link_name = link.get("name")
        if link_name == "world":
            continue

        print(f"Processing link: {link_name}")

        # JSON filename for sphere data
        json_filename = f"{link_name}.json"
        json_path = os.path.join(spheres_dir, json_filename)

        # Save inertial info (optional)
        inertial_elem = link.find("inertial")

        # Remove original visual and collision
        for elem in link.findall("visual"):
            link.remove(elem)
        for elem in link.findall("collision"):
            link.remove(elem)

        if not os.path.exists(json_path):
            print(f"No sphere data for {link_name}, skipping sphere creation.")
            continue

        try:
            centers, radii = read_json_spheres(json_path)
            print(f"  Found {len(centers)} spheres for {link_name}")

            for idx, (center, radius) in enumerate(zip(centers, radii), 1):
                sphere_link, sphere_link_name = create_sphere_link(center, radius, link_name, idx)
                root.append(sphere_link)

                joint = create_fixed_joint(link_name, sphere_link_name, center, idx)
                root.append(joint)

        except Exception as e:
            print(f"Error processing {link_name}: {str(e)}")
            continue

    # Save updated URDF
    tree.write(output_urdf_path, encoding="utf-8", xml_declaration=True)
    print(f"Successfully created URDF file: {output_urdf_path}")


if __name__ == "__main__":
    num_spheres_per_joint = 10
    asset_name = 'blue_box'

    mesh_dir = "/home/ava/Research/Codes/Fall25/Genesis/genesis/assets/urdf/blue_box/assets" 
    spheres_dir = f"results/spheres/{asset_name}_{num_spheres_per_joint}"

    create_spheres(input_dir=mesh_dir, output_dir=spheres_dir, num_spheres_per_joint=num_spheres_per_joint, asset_name=asset_name)

    orig_urdf = "/home/ava/Research/Codes/Fall25/Genesis/genesis/assets/urdf/blue_box/model.urdf"
    output_urdf = f"results/urdfs/{asset_name}.urdf"

    modify_urdf(input_urdf_path=orig_urdf, spheres_dir=spheres_dir, output_urdf_path=output_urdf)