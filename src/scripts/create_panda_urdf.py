import json
import xml.etree.ElementTree as ET
import os
from copy import deepcopy

# ======== COLOR CONFIGURATION ========
# Set USE_SINGLE_COLOR to True to use a single color for all links, False to use the multi-color scheme
USE_SINGLE_COLOR = False

# Specify a single color in hex format (e.g., "#7E57C2" for purple)
# This will be used for all links if USE_SINGLE_COLOR is True
SINGLE_COLOR_HEX = "#7E57C2"


# Function to convert hex color to RGBA (used for SINGLE_COLOR_HEX)
def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip("#")
    return {
        "r": int(hex_color[0:2], 16) / 255.0,
        "g": int(hex_color[2:4], 16) / 255.0,
        "b": int(hex_color[4:6], 16) / 255.0,
        "a": alpha,
    }


# Define colors for each link (used when USE_SINGLE_COLOR is False)
LINK_COLORS = {
    "panda_link0": {"r": 0.9, "g": 0.9, "b": 0.9, "a": 1.0},  # Light gray
    "panda_link1": {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0},  # Dark gray
    "panda_link2": {"r": 0.9, "g": 0.9, "b": 0.9, "a": 1.0},  # Light gray
    "panda_link3": {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0},  # Dark gray
    "panda_link4": {"r": 0.9, "g": 0.9, "b": 0.9, "a": 1.0},  # Light gray
    "panda_link5": {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0},  # Dark gray
    "panda_link6": {"r": 0.9, "g": 0.9, "b": 0.9, "a": 1.0},  # Light gray
    "panda_link7": {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0},  # Dark gray
    "panda_hand": {"r": 0.9, "g": 0.9, "b": 0.9, "a": 1.0},  # Light gray
    "panda_leftfinger": {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0},  # Dark gray
    "panda_rightfinger": {"r": 0.9, "g": 0.9, "b": 0.9, "a": 1.0},  # Light gray
}


def read_json_spheres(json_path):
    """Read sphere data from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["centers"], data["radii"]


def create_material_element(link_name, sphere_idx):
    """Create a material element with unique name and specified color."""
    material = ET.Element("material")
    material_name = f"color_{link_name}_{sphere_idx}"
    material.set("name", material_name)
    color_elem = ET.SubElement(material, "color")

    if USE_SINGLE_COLOR:
        # Use the single color specified at the top
        color = hex_to_rgba(SINGLE_COLOR_HEX)
    else:
        # Determine color based on the parent link name (without _sphereX suffix)
        parent_link = link_name.split("_sphere")[0]
        color = LINK_COLORS.get(parent_link, {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0})

    color_elem.set("rgba", f"{color['r']} {color['g']} {color['b']} {color['a']}")

    return material, material_name


def create_sphere_link(center, radius, parent_link_name, sphere_idx):
    """Create a new link containing a single sphere visual and collision."""
    # Create new link element
    link_name = f"{parent_link_name}_sphere{sphere_idx}"
    link = ET.Element("link")
    link.set("name", link_name)

    # Add inertial properties
    inertial = ET.SubElement(link, "inertial")

    origin = ET.SubElement(inertial, "origin")
    origin.set("xyz", "0 0 0")
    origin.set("rpy", "0 0 0")

    mass = ET.SubElement(inertial, "mass")
    mass.set("value", "0.1")

    inertia = ET.SubElement(inertial, "inertia")
    inertia.set("ixx", "0.01")
    inertia.set("ixy", "0")
    inertia.set("ixz", "0")
    inertia.set("iyy", "0.01")
    inertia.set("iyz", "0")
    inertia.set("izz", "0.01")

    # Create visual element
    visual = ET.SubElement(link, "visual")
    visual_origin = ET.SubElement(visual, "origin")
    visual_origin.set("rpy", "0 0 0")
    visual_origin.set("xyz", "0 0 0")

    visual_geometry = ET.SubElement(visual, "geometry")
    visual_sphere = ET.SubElement(visual_geometry, "sphere")
    visual_sphere.set("radius", str(radius))

    # Add material reference
    material, material_name = create_material_element(link_name, 0)
    visual.append(material)

    # Create collision element
    collision = ET.SubElement(link, "collision")
    collision_origin = ET.SubElement(collision, "origin")
    collision_origin.set("rpy", "0 0 0")
    collision_origin.set("xyz", "0 0 0")

    collision_geometry = ET.SubElement(collision, "geometry")
    collision_sphere = ET.SubElement(collision_geometry, "sphere")
    collision_sphere.set("radius", str(radius))

    # Add Drake properties
    props = ET.SubElement(collision, "drake:proximity_properties")
    ET.SubElement(props, "drake:rigid_hydroelastic")
    res_hint = ET.SubElement(props, "drake:mesh_resolution_hint")
    res_hint.set("value", "1.5")
    dissipation = ET.SubElement(props, "drake:hunt_crossley_dissipation")
    dissipation.set("value", "1.25")

    return link, link_name


def create_fixed_joint(parent_link, child_link, center, joint_idx=0):
    """Create a fixed joint connecting parent link to a sphere link."""
    joint = ET.Element("joint")
    joint.set("name", f"{parent_link}_to_{child_link}")
    joint.set("type", "fixed")

    parent_elem = ET.SubElement(joint, "parent")
    parent_elem.set("link", parent_link)

    child_elem = ET.SubElement(joint, "child")
    child_elem.set("link", child_link)

    origin = ET.SubElement(joint, "origin")
    origin.set("xyz", f"{center[0]} {center[1]} {center[2]}")
    origin.set("rpy", "0 0 0")

    return joint


def modify_urdf(input_urdf_path, spheres_dir, output_urdf_path):
    """Modify URDF to give each sphere its own link while preserving structure."""
    # Parse URDF preserving comments
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(input_urdf_path, parser=parser)
    root = tree.getroot()

    # Register namespaces
    ET.register_namespace("", "")
    ET.register_namespace("drake", "http://drake.mit.edu")

    # Add world link and joint to connect to panda_link0 if not already present
    world_link = root.find(".//link[@name='world']")
    if world_link is None:
        world_link = ET.Element("link")
        world_link.set("name", "world")
        root.insert(0, world_link)

        # Only add the joint if we had to add the world link
        world_joint = ET.Element("joint")
        world_joint.set("name", "world_to_base")
        world_joint.set("type", "fixed")

        parent = ET.SubElement(world_joint, "parent")
        parent.set("link", "world")

        child = ET.SubElement(world_joint, "child")
        child.set("link", "panda_link0")

        origin = ET.SubElement(world_joint, "origin")
        origin.set("xyz", "0 0 0")
        origin.set("rpy", "0 0 0")

        root.insert(1, world_joint)

    # Process each main link
    for link in root.findall(".//link"):
        link_name = link.get("name")
        if link_name == "world" or not (
            link_name.startswith("panda_link")
            or link_name in ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
        ):
            continue

        print(f"Processing link {link_name}")
        json_filename = f"{link_name}.json"

        if link_name == "panda_leftfinger" or link_name == "panda_rightfinger":
            json_filename = "panda_finger.json"

        # Construct the json path using the link number
        json_path = os.path.join(spheres_dir, json_filename)

        # Store original inertial data before removing existing visual/collision
        inertial_elem = link.find("inertial")

        # Remove existing visual and collision elements from the main link
        for elem in link.findall("visual"):
            link.remove(elem)
        for elem in link.findall("collision"):
            link.remove(elem)

        # IGNORE PANDA FINGER LINKS!
        if link_name == "panda_leftfinger" or link_name == "panda_rightfinger":
            continue

        if not os.path.exists(json_path):
            print(
                f"No sphere data for {link_name} at {json_path}, creating a default sphere"
            )
            raise Exception("")

        try:
            centers, radii = read_json_spheres(json_path)

            # Create individual links for each sphere and connect them to the main link
            for idx, (center, radius) in enumerate(zip(centers, radii), 1):
                # Create a new link for the sphere
                sphere_link, sphere_link_name = create_sphere_link(
                    center, radius, link_name, idx
                )
                root.append(sphere_link)

                # Create a fixed joint connecting the main link to the sphere link
                joint = create_fixed_joint(link_name, sphere_link_name, center, idx)
                root.append(joint)

        except Exception as e:
            print(f"Error processing {link_name}: {str(e)}")
            continue

    # Write modified URDF preserving XML declaration and formatting
    tree.write(output_urdf_path, encoding="utf-8", xml_declaration=True)
    print(f"Successfully created URDF file: {output_urdf_path}")


if __name__ == "__main__":
    input_urdf = "panda.urdf"
    spheres_dir = "../results/batch_output"
    output_urdf = "panda_multi_sphere.urdf"

    modify_urdf(input_urdf, spheres_dir, output_urdf)
