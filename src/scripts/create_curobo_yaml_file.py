import yaml
import json
import os
import glob


def json_files_to_yaml(
    json_dir="../results/batch_output",
    output_yaml_file="franka.yml",
):
    """
    Convert JSON files containing collision sphere data to a single YAML file
    with proper formatting for the Franka Panda robot.

    Special handling for panda_finger to create a YAML anchor for panda_leftfinger
    and reference it in panda_rightfinger.
    """
    # Read all JSON files first to collect data
    collision_spheres = {}
    finger_spheres = None

    # Collect all JSON files that match the pattern panda_*.json
    json_files = glob.glob(os.path.join(json_dir, "panda_*.json"))
    print(f"Found JSON files: {json_files}")

    # Process each JSON file
    for json_file in json_files:
        # Extract link name from filename
        link_name = os.path.basename(json_file).split(".")[0]
        print(f"Processing file: {json_file}, link_name: {link_name}")

        # Read JSON file
        try:
            with open(json_file, "r") as file:
                data = json.load(file)

            # Extract centers and radii
            centers = data.get("centers", [])
            radii = data.get("radii", [])

            # Create spheres list
            spheres = []
            for i in range(min(len(centers), len(radii))):
                sphere = {"center": centers[i], "radius": radii[i]}
                spheres.append(sphere)

            # Special handling for finger data
            if link_name == "panda_finger":
                finger_spheres = spheres  # Save for later
            else:
                collision_spheres[link_name] = spheres

        except Exception as e:
            print(f"Error processing file {json_file}: {e}")
            continue

    # Now manually construct the YAML file with the desired format
    with open(output_yaml_file, "w") as file:
        # Write header
        # Write robot and urdf path
        file.write("robot: Franka Panda\n")
        file.write("urdf_path: urdf/franka_description/franka_panda_dyn.urdf\n")

        # Start collision_spheres section
        file.write("collision_spheres:\n")

        # Write collision spheres for all links
        for link_name, spheres in collision_spheres.items():
            file.write(f"  {link_name}:\n")
            for sphere in spheres:
                file.write("    - center:\n")
                file.write(f"        - {sphere['center'][0]}\n")
                file.write(f"        - {sphere['center'][1]}\n")
                file.write(f"        - {sphere['center'][2]}\n")
                file.write(f"      radius: {sphere['radius']}\n")

        # Add finger data with anchor if available
        if finger_spheres:
            # Add leftfinger with anchor
            file.write("  panda_leftfinger:\n")
            for sphere in finger_spheres:
                file.write("    - center:\n")
                file.write(f"        - {sphere['center'][0]}\n")
                file.write(f"        - {sphere['center'][1]}\n")
                file.write(f"        - {sphere['center'][2]}\n")
                file.write(f"      radius: {sphere['radius']}\n")

            # Add rightfinger with reference to leftfinger
            file.write("  panda_rightfinger:\n")
            for sphere in finger_spheres:
                file.write("    - center:\n")
                file.write(f"        - {sphere['center'][0]}\n")
                file.write(f"        - {sphere['center'][1]}\n")
                file.write(f"        - {sphere['center'][2]}\n")
                file.write(f"      radius: {sphere['radius']}\n")

    print(f"Created YAML file: {output_yaml_file}")


if __name__ == "__main__":
    # Convert JSON files to YAML
    json_files_to_yaml()
    print("Conversion completed.")
