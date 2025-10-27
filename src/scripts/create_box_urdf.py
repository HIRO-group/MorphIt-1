#!/usr/bin/env python3
import os
import json
import xml.etree.ElementTree as ET

from scripts.create_panda_urdf import USE_SINGLE_COLOR, hex_to_rgba, LINK_COLORS, SINGLE_COLOR_HEX

# --- Your existing helpers (unchanged) ---

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

    # Assumes you already define these globals in your project:
    # USE_SINGLE_COLOR, SINGLE_COLOR_HEX, hex_to_rgba, LINK_COLORS
    if USE_SINGLE_COLOR:
        color = hex_to_rgba(SINGLE_COLOR_HEX)
    else:
        parent_link = link_name.split("_sphere")[0]
        color = LINK_COLORS.get(parent_link, {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0})

    color_elem.set("rgba", f"{color['r']} {color['g']} {color['b']} {color['a']}")
    return material, material_name

def create_sphere_link(center, radius, parent_link_name, sphere_idx):
    """Create a new link containing a single sphere visual and collision."""
    link_name = f"{parent_link_name}_sphere{sphere_idx}"
    link = ET.Element("link")
    link.set("name", link_name)

    # Inertial (simple default values, same as your Panda code)
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

    # Visual
    visual = ET.SubElement(link, "visual")
    visual_origin = ET.SubElement(visual, "origin")
    visual_origin.set("rpy", "0 0 0")
    visual_origin.set("xyz", "0 0 0")

    visual_geometry = ET.SubElement(visual, "geometry")
    visual_sphere = ET.SubElement(visual_geometry, "sphere")
    visual_sphere.set("radius", str(radius))

    material, material_name = create_material_element(link_name, 0)
    visual.append(material)

    # Collision
    collision = ET.SubElement(link, "collision")
    collision_origin = ET.SubElement(collision, "origin")
    collision_origin.set("rpy", "0 0 0")
    collision_origin.set("xyz", "0 0 0")

    collision_geometry = ET.SubElement(collision, "geometry")
    collision_sphere = ET.SubElement(collision_geometry, "sphere")
    collision_sphere.set("radius", str(radius))

    # Drake proximity props (same as your Panda code)
    props = ET.SubElement(collision, "drake:proximity_properties")
    ET.SubElement(props, "drake:rigid_hydroelastic")
    res_hint = ET.SubElement(props, "drake:mesh_resolution_hint")
    res_hint.set("value", "1.5")
    dissipation = ET.SubElement(props, "drake:hunt_crossley_dissipation")
    dissipation.set("value", "1.25")

    # Fixed joint to parent
    joint = ET.Element("joint")
    joint.set("name", f"{parent_link_name}_to_{link_name}")
    joint.set("type", "fixed")

    parent_elem = ET.SubElement(joint, "parent")
    parent_elem.set("link", parent_link_name)

    child_elem = ET.SubElement(joint, "child")
    child_elem.set("link", link_name)

    joint_origin = ET.SubElement(joint, "origin")
    joint_origin.set("xyz", f"{center[0]} {center[1]} {center[2]}")
    joint_origin.set("rpy", "0 0 0")

    return link, joint, link_name

# --- Minimal URDF modifier for the box ---

def modify_box_urdf(input_urdf_path, spheres_dir, output_urdf_path):
    """Add per-sphere links (visual+collision) to the pink_box_link, like the Panda script."""
    # Parse URDF preserving comments
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(input_urdf_path, parser=parser)
    root = tree.getroot()
    ET.register_namespace("drake", "http://drake.mit.edu")
    root.set("xmlns:drake", "http://drake.mit.edu")


    # Register namespaces (for Drake)
    ET.register_namespace("drake", "http://drake.mit.edu")

    # Ensure world link and fixed joint to pink_box_link (like your Panda code)
    world_link = root.find(".//link[@name='world']")
    if world_link is None:
        world_link = ET.Element("link")
        world_link.set("name", "world")
        root.insert(0, world_link)

        world_joint = ET.Element("joint")
        world_joint.set("name", "world_to_box")
        world_joint.set("type", "fixed")
        parent = ET.SubElement(world_joint, "parent")
        parent.set("link", "world")
        child = ET.SubElement(world_joint, "child")
        child.set("link", "pink_box_link")
        origin = ET.SubElement(world_joint, "origin")
        origin.set("xyz", "0 0 0")
        origin.set("rpy", "0 0 0")
        root.insert(1, world_joint)

    # Find the single target link
    link = root.find(".//link[@name='pink_box_link']")
    if link is None:
        raise RuntimeError("Link 'pink_box_link' not found in URDF")

    # Remove existing visual/collision from the main link (same behavior as your Panda script)
    for elem in link.findall("visual"):
        link.remove(elem)
    for elem in link.findall("collision"):
        link.remove(elem)

    # Read sphere data: use a single JSON file named 'pink_box_link.json'

    json_path = os.path.join(spheres_dir, 'pink_box.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No sphere data found: {json_path}")

    centers, radii = read_json_spheres(json_path)

    # Create a child link + fixed joint per sphere
    for idx, (center, radius) in enumerate(zip(centers, radii), 1):
        sphere_link, joint, _ = create_sphere_link(center, radius, "pink_box_link", idx)
        root.append(sphere_link)
        root.append(joint)

    # Write result
    tree.write(output_urdf_path, encoding="utf-8", xml_declaration=True)
    print(f"Successfully created URDF file: {output_urdf_path}")
