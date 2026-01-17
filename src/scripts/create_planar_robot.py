"""
Planar Rotating Robot Model Generator
======================================

Generates a URDF robot model with:
- 2 prismatic joints (x, y) for planar translation
- 1 revolute joint (theta) for rotation
- N sphere links rigidly attached to the rotating base

This creates a planar mobile manipulator where spheres represent the object geometry.

Input:
    JSON file with 'centers' and 'radii' arrays from sphere decomposition

Output:
    URDF (.xml) file with kinematic chain: world → x → y → theta → spheres

Usage:
    1. Configure the CONFIG dictionary below
    2. Run: python write_planar_rotating_spheres_xml.py
    3. Load in Genesis or other URDF-compatible simulators

Kinematic Chain:
    world --[prismatic X]--> x_link --[prismatic Y]--> y_link --[revolute θ]--> rot_base --[fixed]--> spheres
"""

import json
from pathlib import Path

# =========================
# Configuration
# =========================

CONFIG = {
    # -------------------------
    # Input/Output
    # -------------------------
    # Path to MorphIt JSON results
    "input_json": "../results/output/morphit_results.json",
    "output_xml": "planar_robot.xml",  # Output URDF filename
    "robot_name": "planar_spheres",    # Robot name in URDF

    # -------------------------
    # Visual Appearance
    # -------------------------
    # RGBA color for all spheres (red)
    "default_color_rgba": (0.8, 0.0, 0.0, 1.0),

    # -------------------------
    # Numerical Precision
    # -------------------------
    "decimals": 6,  # Decimal places for floating point values in XML

    # -------------------------
    # Mass Distribution
    # -------------------------
    # Total mass distributed across spheres proportional to r³ (volume)
    "total_mass": 0.1,  # kg - total mass of all spheres combined

    # -------------------------
    # Rotation Center
    # -------------------------
    # use_centroid: Choose the point where theta rotation axis passes through
    #   - True: Use geometric centroid of all sphere centers
    #   - False: Use rotation_center_xyz as the rotation axis location
    "use_centroid": True,
    "rotation_center_xyz": (0.0, 0.0, 0.0),  # Used when use_centroid=False

    # -------------------------
    # Rotation Axis
    # -------------------------
    # theta_axis: Direction of rotation axis for theta joint
    #   - "x": Rotate about X-axis
    #   - "y": Rotate about Y-axis
    #   - "z": Rotate about Z-axis (typical for planar robots)
    "theta_axis": "z",

    # -------------------------
    # Joint Limits and Dynamics
    # -------------------------
    # Limits for prismatic joints (meters)
    "x_limit": (-500.0, 500.0),      # (lower, upper) bounds for X translation
    "y_limit": (-500.0, 500.0),      # (lower, upper) bounds for Y translation

    # Limits for revolute joint (radians)
    # ±π for full rotation
    "theta_limit": (-3.141592653589793, 3.141592653589793),

    # Joint effort and velocity limits (applied to all joints)
    # Maximum torque/force (N⋅m for revolute, N for prismatic)
    "effort": 5000.0,
    # Maximum velocity (rad/s for revolute, m/s for prismatic)
    "velocity": 5000.0,

    # -------------------------
    # Carrier Link Properties
    # -------------------------
    # Small but nonzero masses/inertias for intermediate carrier links
    # (x_link, y_link, rot_base) to keep URDF parsers happy
    "carrier_mass": 0.001,         # kg - negligible mass for carrier links
    "carrier_inertia_diag": 1e-5,  # kg⋅m² - minimal inertia tensor diagonal

    # -------------------------
    # Coordinate Transform
    # -------------------------
    # Optional global offset applied to input centers BEFORE computing rotation center
    # Useful for aligning input coordinates with desired world frame
    "global_offset_xyz": (0.0, 0.0, 0.0),  # (x, y, z) offset in meters
}


# =========================
# Utility Functions
# =========================

def fnum(v, d):
    """
    Format a number to fixed decimal precision.

    Args:
        v: Number to format
        d: Number of decimal places

    Returns:
        String representation with d decimal places
    """
    return f"{float(v):.{d}f}"


def inertia_solid_sphere(m, r):
    """
    Calculate moment of inertia for a solid sphere about its center.

    Formula: I = (2/5) * m * r²  (for each principal axis)

    Args:
        m: Mass of sphere (kg)
        r: Radius of sphere (m)

    Returns:
        Tuple (Ixx, Iyy, Izz) - all equal for a sphere
    """
    I = (2.0 / 5.0) * m * (r ** 2)
    return I, I, I


def axis_vec(name: str):
    """
    Convert axis name to unit vector string for URDF.

    Args:
        name: Axis name - "x", "y", or "z"

    Returns:
        String representation of unit vector (e.g., "1 0 0" for x-axis)

    Raises:
        ValueError: If axis name is not x, y, or z
    """
    n = name.lower().strip()
    if n == "x":
        return "1 0 0"
    if n == "y":
        return "0 1 0"
    if n == "z":
        return "0 0 1"
    raise ValueError(f"theta_axis must be 'x', 'y', or 'z', got '{name}'")


def load_and_process_spheres(cfg):
    """
    Load sphere data from JSON and compute derived quantities.

    Processing steps:
    1. Load centers and radii from JSON
    2. Apply global offset to centers
    3. Compute rotation center (centroid or specified)
    4. Calculate positions relative to rotation center
    5. Distribute mass proportional to r³

    Args:
        cfg: Configuration dictionary

    Returns:
        Tuple of:
        - centers: Original sphere centers (after global offset)
        - radii: Sphere radii
        - rel_centers: Positions relative to rotation center
        - masses: Individual sphere masses
        - rotation_center: (cx, cy, cz) point where rotation axis passes

    Raises:
        AssertionError: If centers and radii arrays have different lengths
        FileNotFoundError: If input JSON file doesn't exist
    """
    # Load JSON data
    data = json.loads(Path(cfg["input_json"]).read_text())
    centers = data["centers"]
    radii = data["radii"]
    assert len(centers) == len(radii), \
        f"centers/radii length mismatch: {len(centers)} vs {len(radii)}"

    # Apply optional global offset to all centers
    gx, gy, gz = cfg["global_offset_xyz"]
    centers = [[c[0] + gx, c[1] + gy, c[2] + gz] for c in centers]

    # Determine rotation center (where theta axis passes through)
    if cfg["use_centroid"]:
        cx = sum(c[0] for c in centers) / len(centers)
        cy = sum(c[1] for c in centers) / len(centers)
        cz = sum(c[2] for c in centers) / len(centers)
    else:
        cx, cy, cz = cfg["rotation_center_xyz"]

    # Calculate positions relative to rotation center
    # These will be used for fixed joints from rot_base to sphere links
    rel_centers = [[c[0] - cx, c[1] - cy, c[2] - cz] for c in centers]

    # Distribute total mass by volume (r³)
    vols = [r ** 3 for r in radii]
    vtot = sum(vols) if vols else 1.0
    masses = [cfg["total_mass"] * v / vtot for v in vols]

    return centers, radii, rel_centers, masses, (cx, cy, cz)


# =========================
# URDF Generation Functions
# =========================

def write_urdf_header(cfg):
    """
    Generate URDF header with XML declaration, robot tag, and material.

    Args:
        cfg: Configuration dictionary

    Returns:
        List of XML lines
    """
    d = cfg["decimals"]
    rgba = " ".join(fnum(x, d) for x in cfg["default_color_rgba"])

    xml = []
    xml.append('<?xml version="1.0"?>')
    xml.append(f'<robot name="{cfg["robot_name"]}">')
    xml.append('  <material name="default_color">')
    xml.append(f'    <color rgba="{rgba}"/>')
    xml.append('  </material>')

    return xml


def write_carrier_links(cfg):
    """
    Generate carrier link definitions (x_link, y_link, rot_base).

    These are lightweight links that form the kinematic chain but carry
    minimal mass. The actual object mass is in the sphere links.

    Args:
        cfg: Configuration dictionary

    Returns:
        List of XML lines
    """
    d = cfg["decimals"]
    carr_m = fnum(cfg["carrier_mass"], d)
    carr_I = fnum(cfg["carrier_inertia_diag"], d)

    xml = []
    xml.append('  <link name="world"/>')

    for link_name in ["x_link", "y_link", "rot_base"]:
        xml.append(f'  <link name="{link_name}">')
        xml.append('    <inertial>')
        xml.append(f'      <mass value="{carr_m}"/>')
        xml.append(
            f'      <inertia ixx="{carr_I}" ixy="0.0" ixz="0.0" '
            f'iyy="{carr_I}" iyz="0.0" izz="{carr_I}"/>'
        )
        xml.append('    </inertial>')
        xml.append('  </link>')

    return xml


def write_kinematic_joints(cfg, rotation_center):
    """
    Generate the three kinematic joints: x (prismatic), y (prismatic), theta (revolute).

    Joint chain: world → x_link → y_link → rot_base

    Args:
        cfg: Configuration dictionary
        rotation_center: (cx, cy, cz) tuple - point where rotation axis intersects

    Returns:
        List of XML lines
    """
    d = cfg["decimals"]
    xlo, xhi = cfg["x_limit"]
    ylo, yhi = cfg["y_limit"]
    thlo, thhi = cfg["theta_limit"]
    axis = axis_vec(cfg["theta_axis"])
    cx, cy, cz = rotation_center

    eff = fnum(cfg["effort"], d)
    vel = fnum(cfg["velocity"], d)

    xml = []

    # X-axis prismatic joint (world → x_link)
    xml.append('  <joint name="x_joint" type="prismatic">')
    xml.append('    <parent link="world"/>')
    xml.append('    <child link="x_link"/>')
    xml.append('    <axis xyz="1 0 0"/>')
    xml.append(
        f'    <limit lower="{fnum(xlo, d)}" upper="{fnum(xhi, d)}" '
        f'effort="{eff}" velocity="{vel}"/>'
    )
    xml.append('  </joint>')

    # Y-axis prismatic joint (x_link → y_link)
    xml.append('  <joint name="y_joint" type="prismatic">')
    xml.append('    <parent link="x_link"/>')
    xml.append('    <child link="y_link"/>')
    xml.append('    <axis xyz="0 1 0"/>')
    xml.append(
        f'    <limit lower="{fnum(ylo, d)}" upper="{fnum(yhi, d)}" '
        f'effort="{eff}" velocity="{vel}"/>'
    )
    xml.append('  </joint>')

    # Theta revolute joint (y_link → rot_base) at rotation center
    xml.append('  <joint name="theta_joint" type="revolute">')
    xml.append('    <parent link="y_link"/>')
    xml.append('    <child link="rot_base"/>')
    xml.append(
        f'    <origin xyz="{fnum(cx, d)} {fnum(cy, d)} {fnum(cz, d)}" rpy="0 0 0"/>'
    )
    xml.append(f'    <axis xyz="{axis}"/>')
    xml.append(
        f'    <limit lower="{fnum(thlo, d)}" upper="{fnum(thhi, d)}" '
        f'effort="{eff}" velocity="{vel}"/>'
    )
    xml.append('  </joint>')

    return xml


def write_sphere_links(cfg, rel_centers, radii, masses):
    """
    Generate sphere link definitions and their fixed joints to rot_base.

    Each sphere has:
    - Visual geometry (sphere with radius)
    - Collision geometry (same sphere)
    - Inertial properties (mass and moment of inertia)
    - Fixed joint to rot_base at its relative position

    Args:
        cfg: Configuration dictionary
        rel_centers: List of [x, y, z] positions relative to rotation center
        radii: List of sphere radii
        masses: List of sphere masses

    Returns:
        List of XML lines
    """
    d = cfg["decimals"]
    xml = []

    for i, (ctr_rel, r, m) in enumerate(zip(rel_centers, radii, masses)):
        sx, sy, sz = (fnum(v, d) for v in ctr_rel)
        rad = fnum(r, d)
        ixx, iyy, izz = inertia_solid_sphere(m, r)
        link_name = f"sphere_{i}"

        # Sphere link definition
        xml.append(f'  <link name="{link_name}">')

        # Visual representation
        xml.append('    <visual>')
        xml.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        xml.append(f'      <geometry><sphere radius="{rad}"/></geometry>')
        xml.append('      <material name="default_color"/>')
        xml.append('    </visual>')

        # Collision geometry
        xml.append('    <collision>')
        xml.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        xml.append(f'      <geometry><sphere radius="{rad}"/></geometry>')
        xml.append('    </collision>')

        # Inertial properties
        xml.append('    <inertial>')
        xml.append(f'      <mass value="{fnum(m, d)}"/>')
        xml.append(
            f'      <inertia ixx="{fnum(ixx, d)}" ixy="0.0" ixz="0.0" '
            f'iyy="{fnum(iyy, d)}" iyz="0.0" izz="{fnum(izz, d)}"/>'
        )
        xml.append('    </inertial>')
        xml.append('  </link>')

        # Fixed joint connecting sphere to rotating base
        xml.append(f'  <joint name="{link_name}_fixed" type="fixed">')
        xml.append('    <parent link="rot_base"/>')
        xml.append(f'    <child link="{link_name}"/>')
        xml.append(f'    <origin xyz="{sx} {sy} {sz}" rpy="0 0 0"/>')
        xml.append('  </joint>')

    return xml


# =========================
# Main Execution
# =========================

def main():
    """
    Main execution function.

    Workflow:
    1. Load and process sphere data from JSON
    2. Generate URDF XML sections:
       - Header with material definition
       - Carrier links (world, x_link, y_link, rot_base)
       - Kinematic joints (x, y, theta)
       - Sphere links with fixed attachments
    3. Write complete URDF to file
    4. Print summary information
    """
    cfg = CONFIG
    d = cfg["decimals"]

    # Load and process input data
    centers, radii, rel_centers, masses, rotation_center = load_and_process_spheres(
        cfg)
    cx, cy, cz = rotation_center

    # Build URDF XML
    xml = []
    xml.extend(write_urdf_header(cfg))
    xml.extend(write_carrier_links(cfg))
    xml.extend(write_kinematic_joints(cfg, rotation_center))
    xml.extend(write_sphere_links(cfg, rel_centers, radii, masses))
    xml.append('</robot>\n')

    # Write to file
    output_path = Path(cfg["output_xml"])
    output_path.write_text("\n".join(xml), encoding="utf-8")

    # Print summary
    print(f"✓ Wrote planar robot URDF to: {output_path}")
    print(f"\nKinematic Structure:")
    print(
        f"  world → [x_joint] → x_link → [y_joint] → y_link → [theta_joint] → rot_base → spheres")
    print(f"\nJoint Configuration:")
    print(
        f"  X translation: {cfg['x_limit'][0]:.2f} to {cfg['x_limit'][1]:.2f} m")
    print(
        f"  Y translation: {cfg['y_limit'][0]:.2f} to {cfg['y_limit'][1]:.2f} m")
    print(
        f"  θ rotation:    {cfg['theta_limit'][0]:.4f} to {cfg['theta_limit'][1]:.4f} rad (about {cfg['theta_axis'].upper()}-axis)")
    print(f"\nObject Properties:")
    print(f"  Rotation center: ({cx:.{d}f}, {cy:.{d}f}, {cz:.{d}f})")
    print(f"  Total mass: {cfg['total_mass']} kg")
    print(f"  Sphere count: {len(radii)}")
    print(f"  Radius range: [{min(radii):.{d}f}, {max(radii):.{d}f}] m")
    print(f"\nDynamics:")
    print(f"  Max effort: {cfg['effort']} N⋅m / N")
    print(f"  Max velocity: {cfg['velocity']} rad/s / m/s")


if __name__ == "__main__":
    main()
