"""
Sphere-Based Robot Model Generator
===================================

Converts MorphIt sphere decomposition results (JSON) into URDF or MJCF robot models
for use with Genesis physics simulator.

Input:
    JSON file with 'centers' and 'radii' arrays from sphere decomposition

Output:
    URDF (.urdf) or MJCF (.xml) file representing the object as rigidly-connected spheres

Usage:
    1. Configure the CONFIG dictionary below
    2. Run: python sphere_to_model.py
    3. Load in Genesis with:
       - URDF: gs.morphs.URDF(file="path/to/output.urdf")
       - MJCF: gs.morphs.MJCF(file="path/to/output.xml")
"""

import json
from pathlib import Path
import math

# =========================
# Configuration
# =========================

CONFIG = {
    # -------------------------
    # Input/Output
    # -------------------------
    # Path to MorphIt JSON results
    "input_json": "../results/output/morphit_results.json",
    # Output filename (extension auto-added based on format)
    "output_path": "planar_object",
    "robot_name": "planar_object",   # Name attribute in URDF/MJCF model

    # -------------------------
    # Format Selection
    # -------------------------
    # format: "urdf" or "mjcf"
    #   - "urdf": Unified Robot Description Format (auto extension: .urdf)
    #   - "mjcf": MuJoCo XML format (auto extension: .xml)
    "format": "mjcf",

    # -------------------------
    # Anchoring Mode
    # -------------------------
    # anchored: Controls if object is static or dynamic
    #   - False: Dynamic object under gravity
    #       * URDF: floating root (no parent link)
    #       * MJCF: includes <freejoint/> for 6-DOF motion
    #   - True: Static object welded to world
    #       * URDF: fixed joint connecting to world link
    #       * MJCF: no freejoint (body is fixed)
    "anchored": False,

    # -------------------------
    # Visual Appearance
    # -------------------------
    "default_color_rgba": (0.2, 0.6, 1.0, 1.0),  # RGBA color for all spheres

    # -------------------------
    # Numerical Precision
    # -------------------------
    "decimals": 6,  # Decimal places for floating point values in XML

    # -------------------------
    # Mass Distribution
    # -------------------------
    # Total mass distributed across spheres proportional to r³ (volume)
    "total_mass": 1.0,  # kg - distributed by volume across all spheres

    # -------------------------
    # Coordinate Frame Setup
    # -------------------------
    # use_centroid: Choose origin for the object's base frame
    #   - True: Use geometric centroid of all sphere centers
    #   - False: Use rotation_center_xyz as origin
    "use_centroid": True,
    "rotation_center_xyz": (0.0, 0.0, 0.0),  # Used when use_centroid=False

    # global_offset_xyz: Shift applied to input centers BEFORE computing centroid
    # Useful for aligning input coordinates with desired world frame
    "global_offset_xyz": (0.0, 0.0, 0.0),  # (x, y, z) offset in meters

    # -------------------------
    # URDF-Specific Parameters
    # -------------------------
    # Small nonzero values for base link to keep URDF parsers happy
    "base_mass": 0.001,       # kg - negligible mass for root link
    "base_inertia_diag": 1e-5,  # kg⋅m² - minimal inertia tensor diagonal

    # -------------------------
    # Safety Constraints
    # -------------------------
    # Clamp radii to avoid numerical issues with zero/negative values
    "min_radius": 1e-6,  # meters - minimum sphere radius
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


def load_inputs(cfg):
    """
    Load and preprocess sphere data from JSON file.

    Processing steps:
    1. Load centers and radii from JSON
    2. Match array lengths (pad/trim radii if needed)
    3. Clamp radii to minimum value
    4. Apply global offset to centers
    5. Compute origin (centroid or specified point)
    6. Calculate relative positions from origin
    7. Distribute mass proportional to r³

    Args:
        cfg: Configuration dictionary

    Returns:
        Tuple of:
        - centers: Original sphere centers (after global offset)
        - radii: Sphere radii (clamped to min_radius)
        - rel_centers: Positions relative to chosen origin
        - masses: Individual sphere masses
        - world_origin: (cx, cy, cz) chosen as base frame origin

    Raises:
        ValueError: If JSON missing required fields or arrays are empty
    """
    # Load JSON data
    data = json.loads(Path(cfg["input_json"]).read_text())
    centers = list(map(list, data.get("centers", [])))
    radii = list(map(float, data.get("radii", [])))

    if not centers or not radii:
        raise ValueError(
            "JSON must contain non-empty 'centers' and 'radii' arrays.")

    # Match array lengths (if radii shorter, repeat last; if longer, truncate)
    if len(radii) < len(centers):
        radii = radii + [radii[-1]] * (len(centers) - len(radii))
    else:
        radii = radii[:len(centers)]

    # Clamp tiny/negative radii to avoid numerical issues
    rmin = cfg["min_radius"]
    radii = [max(rmin, r) for r in radii]

    # Apply optional global offset to all centers
    gx, gy, gz = cfg["global_offset_xyz"]
    centers = [[c[0] + gx, c[1] + gy, c[2] + gz] for c in centers]

    # Determine origin for base coordinate frame
    if cfg["use_centroid"]:
        # Use geometric centroid of sphere centers
        cx = sum(c[0] for c in centers) / len(centers)
        cy = sum(c[1] for c in centers) / len(centers)
        cz = sum(c[2] for c in centers) / len(centers)
    else:
        # Use specified rotation center
        cx, cy, cz = cfg["rotation_center_xyz"]

    # Calculate positions relative to base frame
    rel_centers = [[c[0] - cx, c[1] - cy, c[2] - cz] for c in centers]

    # Distribute total mass by volume (r³)
    vols = [(r ** 3) for r in radii]
    vtot = sum(vols) if vols else 1.0
    masses = [cfg["total_mass"] * v / vtot for v in vols]

    return centers, radii, rel_centers, masses, (cx, cy, cz)


# =========================
# URDF Writer
# =========================

def write_urdf(cfg, rel_centers, radii, masses, world_origin):
    """
    Generate URDF XML string for sphere-based robot model.

    URDF Structure:
    - If anchored=True:
        world (link) --[fixed joint]--> base (link) --[fixed joints]--> sphere links
    - If anchored=False:
        base (floating root) --[fixed joints]--> sphere links

    Each sphere is a separate link with:
    - Visual geometry (sphere)
    - Collision geometry (sphere)
    - Inertial properties (mass, moment of inertia)

    Load in Genesis with: gs.morphs.URDF(file="output.urdf")

    Args:
        cfg: Configuration dictionary
        rel_centers: Sphere positions relative to base frame
        radii: Sphere radii
        masses: Sphere masses
        world_origin: (cx, cy, cz) base frame position in world

    Returns:
        String containing complete URDF XML
    """
    d = cfg["decimals"]
    rgba = " ".join(fnum(x, d) for x in cfg["default_color_rgba"])
    base_m = fnum(cfg["base_mass"], d)
    base_I = fnum(cfg["base_inertia_diag"], d)
    cx, cy, cz = world_origin

    xml = []
    xml.append('<?xml version="1.0"?>')
    xml.append(f'<robot name="{cfg["robot_name"]}">')

    # Define shared material for all spheres
    xml.append('  <material name="default_color">')
    xml.append(f'    <color rgba="{rgba}"/>')
    xml.append('  </material>')

    # If anchored, create explicit world link for fixed attachment point
    if cfg["anchored"]:
        xml.append('  <link name="world"/>')

    # Root base link (has minimal mass/inertia to keep parsers happy)
    xml.append('  <link name="base">')
    xml.append('    <inertial>')
    xml.append(f'      <mass value="{base_m}"/>')
    xml.append(
        f'      <inertia ixx="{base_I}" ixy="0.0" ixz="0.0" iyy="{base_I}" iyz="0.0" izz="{base_I}"/>'
    )
    xml.append('    </inertial>')
    xml.append('  </link>')

    # Anchor base to world if static, otherwise leave as floating root
    if cfg["anchored"]:
        xml.append('  <joint name="world_to_base" type="fixed">')
        xml.append('    <parent link="world"/>')
        xml.append('    <child link="base"/>')
        xml.append(
            f'    <origin xyz="{fnum(cx,d)} {fnum(cy,d)} {fnum(cz,d)}" rpy="0 0 0"/>')
        xml.append('  </joint>')

    # Create sphere links, each rigidly attached to base
    for i, ((sx, sy, sz), r, m) in enumerate(zip(rel_centers, radii, masses)):
        sx, sy, sz = (fnum(sx, d), fnum(sy, d), fnum(sz, d))
        rad = fnum(float(r), d)
        ixx, iyy, izz = inertia_solid_sphere(m, r)
        link_name = f"sphere_{i}"

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
            f'      <inertia ixx="{fnum(ixx, d)}" ixy="0.0" ixz="0.0" iyy="{fnum(iyy, d)}" iyz="0.0" izz="{fnum(izz, d)}"/>'
        )
        xml.append('    </inertial>')
        xml.append('  </link>')

        # Fixed joint connecting sphere to base at relative position
        xml.append(f'  <joint name="{link_name}_fixed" type="fixed">')
        xml.append('    <parent link="base"/>')
        xml.append(f'    <child link="{link_name}"/>')
        xml.append(f'    <origin xyz="{sx} {sy} {sz}" rpy="0 0 0"/>')
        xml.append('  </joint>')

    xml.append('</robot>\n')
    return "\n".join(xml)


# =========================
# MJCF Writer
# =========================

def write_mjcf(cfg, rel_centers, radii, masses, world_origin):
    """
    Generate MuJoCo XML (MJCF) string for sphere-based robot model.

    MJCF Structure:
    - worldbody contains a base body at world_origin
    - If anchored=False: base has <freejoint/> for 6-DOF motion
    - If anchored=True: base has no freejoint (fixed to world)
    - Child bodies represent spheres at relative positions

    Each sphere is a body with:
    - Geom element (type="sphere") with computed density
    - Mass/inertia automatically computed by MuJoCo from density

    Load in Genesis with: gs.morphs.MJCF(file="output.xml")

    Args:
        cfg: Configuration dictionary
        rel_centers: Sphere positions relative to base frame
        radii: Sphere radii
        masses: Sphere masses
        world_origin: (cx, cy, cz) base frame position in world

    Returns:
        String containing complete MJCF XML
    """
    d = cfg["decimals"]
    rgba = " ".join(fnum(x, d) for x in cfg["default_color_rgba"])
    cx, cy, cz = world_origin

    xml = []
    xml.append('<?xml version="1.0"?>')
    xml.append(f'<mujoco model="{cfg["robot_name"]}">')
    xml.append('  <option gravity="0 0 -9.8"/>')
    xml.append('  <compiler inertiafromgeom="true"/>')

    xml.append('  <worldbody>')
    xml.append(
        f'    <body name="base" pos="{fnum(cx,d)} {fnum(cy,d)} {fnum(cz,d)}">')

    # Add freejoint for dynamic objects (required by Genesis parser to have name)
    if not cfg["anchored"]:
        xml.append('      <freejoint name="base_free"/>')

    # Create sphere bodies as children of base
    for i, ((sx, sy, sz), r, m) in enumerate(zip(rel_centers, radii, masses)):
        sx, sy, sz = (fnum(sx, d), fnum(sy, d), fnum(sz, d))
        rad = float(r)
        rad_s = fnum(rad, d)

        # Calculate density from mass and volume
        # Volume = (4/3)πr³
        vol = (4.0 / 3.0) * math.pi * (rad ** 3)
        dens = m / vol if vol > 0 else 0.0
        dens_s = fnum(dens, d)

        xml.append(f'      <body name="sphere_{i}" pos="{sx} {sy} {sz}">')
        # Named geom for better debugging/visualization
        xml.append(
            f'        <geom name="sphere_{i}_geom" type="sphere" '
            f'size="{rad_s}" density="{dens_s}" rgba="{rgba}"/>'
        )
        xml.append('      </body>')

    xml.append('    </body>')
    xml.append('  </worldbody>')
    xml.append('</mujoco>\n')
    return "\n".join(xml)


# =========================
# Main Execution
# =========================

def main():
    """
    Main execution function.

    Workflow:
    1. Load and preprocess sphere data from JSON
    2. Generate URDF or MJCF XML based on format setting
    3. Auto-determine file extension from format
    4. Write XML to file
    5. Print summary information
    """
    cfg = CONFIG

    # Load and process input data
    centers, radii, rel_centers, masses, world_origin = load_inputs(cfg)

    # Validate format selection
    fmt = cfg["format"].lower().strip()
    if fmt not in ("urdf", "mjcf"):
        raise ValueError("CONFIG['format'] must be 'urdf' or 'mjcf'.")

    # Generate XML and determine output path
    out = Path(cfg["output_path"])

    if fmt == "urdf":
        xml_text = write_urdf(cfg, rel_centers, radii, masses, world_origin)
        # Auto-assign .urdf extension
        if out.suffix.lower() not in (".urdf", ".xml"):
            out = out.with_suffix(".urdf")
        out.write_text(xml_text, encoding="utf-8")

        print(f"✓ Wrote URDF to: {out}")
        print(f"  Load with: gs.morphs.URDF(file='{out}')")
        print(
            f"  Mode: {'Static (welded to world)' if cfg['anchored'] else 'Dynamic (floating root)'}")

    else:  # mjcf
        xml_text = write_mjcf(cfg, rel_centers, radii, masses, world_origin)
        # Auto-assign .xml extension for MJCF
        if out.suffix.lower() != ".xml":
            out = out.with_suffix(".xml")
        out.write_text(xml_text, encoding="utf-8")

        print(f"✓ Wrote MJCF to: {out}")
        print(f"  Load with: gs.morphs.MJCF(file='{out}')")
        print(
            f"  Mode: {'Static (no freejoint)' if cfg['anchored'] else 'Dynamic (freejoint)'}")

    # Print summary statistics
    d = cfg["decimals"]
    cx, cy, cz = world_origin
    print(f"\nModel Summary:")
    print(f"  Origin: ({cx:.{d}f}, {cy:.{d}f}, {cz:.{d}f})")
    print(f"  Total mass: {cfg['total_mass']} kg")
    print(f"  Sphere count: {len(radii)}")
    print(f"  Radius range: [{min(radii):.{d}f}, {max(radii):.{d}f}] m")


if __name__ == "__main__":
    main()
