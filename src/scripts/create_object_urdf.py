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
import trimesh
import numpy as np

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
    "output_path": "sphere_box",
    "robot_name": "sphere_box",  # Name attribute in URDF/MJCF model
    # -------------------------
    # Format Selection
    # -------------------------
    # format: "urdf" or "mjcf"
    #   - "urdf": Unified Robot Description Format (auto extension: .urdf)
    #   - "mjcf": MuJoCo XML format (auto extension: .xml)
    "format": "mjcf",
    # -------------------------
    # Mass Properties Source
    # -------------------------
    # mass_properties_source: "spheres" or "mesh"
    #   - "spheres": Calculate mass, COM, inertia from sphere decomposition
    #   - "mesh": Use original mesh OBJ file (requires mesh_obj_path)
    # Note: Mesh properties are always computed as ground truth for comparison
    "mass_properties_source": "spheres",
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
    # material_density: Density for mass calculations (kg/m³)
    #   - Used for both mesh and sphere mass calculations
    #   - Common values: 1000 (water), 7850 (steel), 2700 (aluminum)
    "material_density": 1000.0,  # kg/m³
    # -------------------------
    # Coordinate Frame Setup
    # -------------------------
    # use_mass_weighted_com: Choose origin for the object's base frame
    #   - True: Use mass-weighted COM of spheres (recommended)
    #   - False: Use rotation_center_xyz as origin
    "use_mass_weighted_com": True,
    "rotation_center_xyz": (0.0, 0.0, 0.0),  # Used when use_mass_weighted_com=False
    # global_offset_xyz: Shift applied to input centers BEFORE computing COM
    # Useful for aligning input coordinates with desired world frame
    "global_offset_xyz": (0.0, 0.0, 0.0),  # (x, y, z) offset in meters
    # -------------------------
    # URDF-Specific Parameters
    # -------------------------
    # Small nonzero values for base link to keep URDF parsers happy
    "base_mass": 0.001,  # kg - negligible mass for root link
    "base_inertia_diag": 1e-5,  # kg⋅m² - minimal inertia tensor diagonal
    # -------------------------
    # Safety Constraints
    # -------------------------
    # Clamp radii to avoid numerical issues with zero/negative values
    "min_radius": 1e-6,  # meters - minimum sphere radius
    # -------------------------
    # Mesh Reference (for ground truth comparison)
    # -------------------------
    "mesh_obj_path": "../../mesh_models/box.obj",  # original mesh (for comparison)
    # -------------------------
    # Physics Parameters
    # -------------------------
    "friction": (1.0, 0.005, 0.0001),  # (sliding, torsional, rolling)
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
    I = (2.0 / 5.0) * m * (r**2)
    return I, I, I


def compute_massprops_from_spheres(centers, radii, density):
    """
    Compute mass, COM, and inertia tensor from sphere decomposition.

    Assumes uniform density for all spheres.

    Args:
        centers: List of sphere center positions [[x,y,z], ...]
        radii: List of sphere radii [r1, r2, ...]
        density: Material density (kg/m³)

    Returns:
        mass (float): Total mass
        com (np.ndarray): Center of mass position (3,)
        fullinertia (tuple): (Ixx, Iyy, Izz, Ixy, Ixz, Iyz) about COM
    """
    centers = np.array(centers, dtype=float)
    radii = np.array(radii, dtype=float)

    # Calculate individual sphere masses: m = ρ * V = ρ * (4/3) * π * r³
    volumes = (4.0 / 3.0) * np.pi * (radii**3)
    masses = density * volumes
    total_mass = masses.sum()

    # Calculate mass-weighted center of mass
    com = (masses[:, np.newaxis] * centers).sum(axis=0) / total_mass
    print("Computed COM from spheres:", com)

    # Calculate inertia tensor about COM
    I_total = np.zeros((3, 3), dtype=float)
    eye3 = np.eye(3)

    for i in range(len(centers)):
        m_i = masses[i]
        r_i = radii[i]
        c_i = centers[i]

        # Inertia of sphere about its own center
        I_sphere = (2.0 / 5.0) * m_i * (r_i**2) * eye3

        # Position relative to COM
        c_rel = c_i - com
        c_sq = np.dot(c_rel, c_rel)

        # Parallel axis theorem: shift to COM
        I_parallel = m_i * (c_sq * eye3 - np.outer(c_rel, c_rel))

        I_total += I_sphere + I_parallel

    # Extract fullinertia components for MuJoCo
    Ixx, Iyy, Izz = I_total[0, 0], I_total[1, 1], I_total[2, 2]
    Ixy, Ixz, Iyz = I_total[0, 1], I_total[0, 2], I_total[1, 2]
    fullinertia = (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)

    return total_mass, com, fullinertia


def compute_massprops_from_obj(cfg):
    mesh_path = Path(cfg["mesh_obj_path"])
    if not mesh_path.exists():
        raise FileNotFoundError(f"OBJ not found: {mesh_path.resolve()}")

    mesh = trimesh.load_mesh(mesh_path, force="mesh")

    if not mesh.is_watertight:
        print(
            "WARNING: mesh is not watertight; volume/mass properties may be unreliable."
        )

    rho = float(cfg["material_density"])

    # Compute COM
    com = mesh.center_mass
    mesh = mesh.copy()
    mesh.apply_translation(-com)

    # Mass
    volume = float(mesh.volume)
    mass = rho * volume

    # Inertia tensor - MUST SCALE BY DENSITY!
    I = np.array(mesh.moment_inertia * rho, dtype=float)  # ← ADD: * rho

    Ixx, Iyy, Izz = I[0, 0], I[1, 1], I[2, 2]
    Ixy, Ixz, Iyz = I[0, 1], I[0, 2], I[1, 2]

    fullinertia = (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)

    return mass, np.asarray(com, dtype=float), fullinertia


def print_mass_properties_comparison(mesh_props, sphere_props):
    """
    Print comparison between mesh and sphere mass properties.

    Args:
        mesh_props: Tuple (mass, com, fullinertia) from mesh
        sphere_props: Tuple (mass, com, fullinertia) from spheres
    """
    mesh_mass, mesh_com, mesh_inertia = mesh_props
    sphere_mass, sphere_com, sphere_inertia = sphere_props

    print("\n" + "=" * 60)
    print("MASS PROPERTIES COMPARISON")
    print("=" * 60)

    # Mass comparison
    mass_error = abs(sphere_mass - mesh_mass)
    mass_rel_error = (mass_error / mesh_mass * 100) if mesh_mass > 0 else 0
    print(f"\nMass:")
    print(f"  Mesh (ground truth): {mesh_mass:.6f} kg")
    print(f"  Spheres:             {sphere_mass:.6f} kg")
    print(f"  Absolute error:      {mass_error:.6e} kg")
    print(f"  Relative error:      {mass_rel_error:.3f}%")

    # COM comparison
    com_error = np.linalg.norm(sphere_com - mesh_com)
    print(f"\nCenter of Mass:")
    print(
        f"  Mesh (ground truth): [{mesh_com[0]:.6f}, {mesh_com[1]:.6f}, {mesh_com[2]:.6f}]"
    )
    print(
        f"  Spheres:             [{sphere_com[0]:.6f}, {sphere_com[1]:.6f}, {sphere_com[2]:.6f}]"
    )
    print(f"  Distance error:      {com_error:.6e} m")

    # Inertia comparison
    mesh_I = np.array(mesh_inertia)
    sphere_I = np.array(sphere_inertia)
    inertia_error = np.abs(sphere_I - mesh_I)

    print(f"\nInertia Tensor (about COM):")
    print(f"  Mesh (ground truth):")
    print(f"    Ixx={mesh_I[0]:.6e}, Iyy={mesh_I[1]:.6e}, Izz={mesh_I[2]:.6e}")
    print(f"    Ixy={mesh_I[3]:.6e}, Ixz={mesh_I[4]:.6e}, Iyz={mesh_I[5]:.6e}")
    print(f"  Spheres:")
    print(f"    Ixx={sphere_I[0]:.6e}, Iyy={sphere_I[1]:.6e}, Izz={sphere_I[2]:.6e}")
    print(f"    Ixy={sphere_I[3]:.6e}, Ixz={sphere_I[4]:.6e}, Iyz={sphere_I[5]:.6e}")
    print(f"  Absolute errors:")
    print(
        f"    ΔIxx={inertia_error[0]:.6e}, ΔIyy={inertia_error[1]:.6e}, ΔIzz={inertia_error[2]:.6e}"
    )
    print(
        f"    ΔIxy={inertia_error[3]:.6e}, ΔIxz={inertia_error[4]:.6e}, ΔIyz={inertia_error[5]:.6e}"
    )

    # Frobenius norm for overall inertia error
    I_mesh_matrix = np.array(
        [
            [mesh_I[0], mesh_I[3], mesh_I[4]],
            [mesh_I[3], mesh_I[1], mesh_I[5]],
            [mesh_I[4], mesh_I[5], mesh_I[2]],
        ]
    )
    I_sphere_matrix = np.array(
        [
            [sphere_I[0], sphere_I[3], sphere_I[4]],
            [sphere_I[3], sphere_I[1], sphere_I[5]],
            [sphere_I[4], sphere_I[5], sphere_I[2]],
        ]
    )
    frobenius_error = np.linalg.norm(I_sphere_matrix - I_mesh_matrix, "fro")
    print(f"  Frobenius norm error: {frobenius_error:.6e}")

    print("=" * 60 + "\n")


def load_inputs(cfg):
    """
    Load and preprocess sphere data from JSON file.

    Processing steps:
    1. Load centers and radii from JSON
    2. Match array lengths (pad/trim radii if needed)
    3. Clamp radii to minimum value
    4. Apply global offset to centers

    Args:
        cfg: Configuration dictionary

    Returns:
        Tuple of:
        - centers: Original sphere centers (after global offset)
        - radii: Sphere radii (clamped to min_radius)

    Raises:
        ValueError: If JSON missing required fields or arrays are empty
    """
    # Load JSON data
    data = json.loads(Path(cfg["input_json"]).read_text())
    centers = list(map(list, data.get("centers", [])))
    radii = list(map(float, data.get("radii", [])))

    if not centers or not radii:
        raise ValueError("JSON must contain non-empty 'centers' and 'radii' arrays.")

    # Match array lengths (if radii shorter, repeat last; if longer, truncate)
    if len(radii) < len(centers):
        radii = radii + [radii[-1]] * (len(centers) - len(radii))
    else:
        radii = radii[: len(centers)]

    # Clamp tiny/negative radii to avoid numerical issues
    rmin = cfg["min_radius"]
    radii = [max(rmin, r) for r in radii]

    # Apply optional global offset to all centers
    gx, gy, gz = cfg["global_offset_xyz"]
    centers = [[c[0] + gx, c[1] + gy, c[2] + gz] for c in centers]

    return centers, radii


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
    xml.append("  </material>")

    # If anchored, create explicit world link for fixed attachment point
    if cfg["anchored"]:
        xml.append('  <link name="world"/>')

    # Root base link (has minimal mass/inertia to keep parsers happy)
    xml.append('  <link name="base">')
    xml.append("    <inertial>")
    xml.append(f'      <mass value="{base_m}"/>')
    xml.append(
        f'      <inertia ixx="{base_I}" ixy="0.0" ixz="0.0" iyy="{base_I}" iyz="0.0" izz="{base_I}"/>'
    )
    xml.append("    </inertial>")
    xml.append("  </link>")

    # Anchor base to world if static, otherwise leave as floating root
    if cfg["anchored"]:
        xml.append('  <joint name="world_to_base" type="fixed">')
        xml.append('    <parent link="world"/>')
        xml.append('    <child link="base"/>')
        xml.append(
            f'    <origin xyz="{fnum(cx,d)} {fnum(cy,d)} {fnum(cz,d)}" rpy="0 0 0"/>'
        )
        xml.append("  </joint>")

    # Create sphere links, each rigidly attached to base
    for i, ((sx, sy, sz), r, m) in enumerate(zip(rel_centers, radii, masses)):
        sx, sy, sz = (fnum(sx, d), fnum(sy, d), fnum(sz, d))
        rad = fnum(float(r), d)
        ixx, iyy, izz = inertia_solid_sphere(m, r)
        link_name = f"sphere_{i}"

        xml.append(f'  <link name="{link_name}">')

        # Visual representation
        xml.append("    <visual>")
        xml.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        xml.append(f'      <geometry><sphere radius="{rad}"/></geometry>')
        xml.append('      <material name="default_color"/>')
        xml.append("    </visual>")

        # Collision geometry
        xml.append("    <collision>")
        xml.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        xml.append(f'      <geometry><sphere radius="{rad}"/></geometry>')
        xml.append("    </collision>")

        # Inertial properties
        xml.append("    <inertial>")
        xml.append(f'      <mass value="{fnum(m, d)}"/>')
        xml.append(
            f'      <inertia ixx="{fnum(ixx, d)}" ixy="0.0" ixz="0.0" iyy="{fnum(iyy, d)}" iyz="0.0" izz="{fnum(izz, d)}"/>'
        )
        xml.append("    </inertial>")
        xml.append("  </link>")

        # Fixed joint connecting sphere to base at relative position
        xml.append(f'  <joint name="{link_name}_fixed" type="fixed">')
        xml.append('    <parent link="base"/>')
        xml.append(f'    <child link="{link_name}"/>')
        xml.append(f'    <origin xyz="{sx} {sy} {sz}" rpy="0 0 0"/>')
        xml.append("  </joint>")

    xml.append("</robot>\n")
    return "\n".join(xml)


# =========================
# MJCF Writer
# =========================


def write_mjcf(cfg, rel_centers, radii, target_mass, target_fullinertia):
    """
    Generate MJCF XML string for sphere-based robot model.

    Args:
        cfg: Configuration dictionary
        rel_centers: Sphere positions relative to COM
        radii: Sphere radii
        target_mass: Total mass to use
        target_fullinertia: Inertia tensor (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)

    Returns:
        String containing complete MJCF XML
    """
    d = cfg["decimals"]
    rgba = " ".join(fnum(x, d) for x in cfg["default_color_rgba"])
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = target_fullinertia

    fric = cfg.get("friction", (1.0, 0.005, 0.0001))
    fric_s = " ".join(fnum(x, d) for x in fric)

    xml = []
    xml.append('<?xml version="1.0"?>')
    xml.append(f'<mujoco model="{cfg["robot_name"]}">')
    xml.append(
        '  <compiler angle="degree" coordinate="local" inertiafromgeom="false" meshdir="assets"/>'
    )
    xml.append('  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>')
    xml.append('  <size njmax="1000" nconmax="200"/>')
    xml.append("  <worldbody>")
    xml.append(
        '    <geom name="floor" type="plane" pos="0 0 0" size="5 5 0.1" friction="1 0.005 0.0001"/>'
    )
    xml.append('    <body name="box" pos="0 0 1">')

    if not cfg["anchored"]:
        xml.append('      <freejoint name="box_free"/>')

    xml.append(
        f'      <inertial pos="0 0 0" '
        f'mass="{fnum(target_mass,d)}" '
        f'fullinertia="{fnum(Ixx,d)} {fnum(Iyy,d)} {fnum(Izz,d)} {fnum(Ixy,d)} {fnum(Ixz,d)} {fnum(Ixz,d)}"/>'
    )

    # massless collision spheres (so they don't alter inertia)
    for i, ((sx, sy, sz), r) in enumerate(zip(rel_centers, radii)):
        xml.append(
            f'      <geom name="s{i}" type="sphere" '
            f'pos="{fnum(sx,d)} {fnum(sy,d)} {fnum(sz,d)}" '
            f'size="{fnum(r,d)}" mass="0" '
            f'friction="{fric_s}" rgba="{rgba}"/>'
        )

    xml.append("    </body>")
    xml.append("  </worldbody>")
    xml.append("</mujoco>\n")
    return "\n".join(xml)


# =========================
# Main Execution
# =========================


def main():
    cfg = CONFIG

    # Load sphere data
    centers, radii = load_inputs(cfg)

    # Always compute mesh properties as ground truth
    print("Computing mesh properties (ground truth)...")
    mesh_mass, mesh_com, mesh_fullinertia = compute_massprops_from_obj(cfg)

    # Compute sphere properties
    print("Computing sphere properties...")
    sphere_mass, sphere_com, sphere_fullinertia = compute_massprops_from_spheres(
        centers, radii, cfg["material_density"]
    )

    # Print comparison
    print_mass_properties_comparison(
        (mesh_mass, mesh_com, mesh_fullinertia),
        (sphere_mass, sphere_com, sphere_fullinertia),
    )

    # Choose which properties to use based on config
    source = cfg["mass_properties_source"].lower().strip()
    if source == "spheres":
        print(f"Using SPHERE mass properties for MJCF generation")
        target_mass = sphere_mass
        target_com = sphere_com
        target_fullinertia = sphere_fullinertia
    elif source == "mesh":
        print(f"Using MESH mass properties for MJCF generation")
        target_mass = mesh_mass
        target_com = mesh_com
        target_fullinertia = mesh_fullinertia
    else:
        raise ValueError(
            f"Invalid mass_properties_source: {source}. Must be 'spheres' or 'mesh'"
        )

    # Determine origin for coordinate frame
    if cfg["use_mass_weighted_com"]:
        # Use mass-weighted COM from spheres
        world_origin = tuple(sphere_com)
    else:
        # Use specified rotation center
        world_origin = cfg["rotation_center_xyz"]

    # Calculate sphere positions relative to chosen origin
    cx, cy, cz = world_origin
    rel_centers = [[c[0] - cx, c[1] - cy, c[2] - cz] for c in centers]

    # Calculate individual sphere masses for URDF (not used in MJCF)
    radii_array = np.array(radii)
    volumes = (4.0 / 3.0) * np.pi * (radii_array**3)
    masses = cfg["material_density"] * volumes

    # Generate output file
    fmt = cfg["format"].lower().strip()
    out = Path(cfg["output_path"])

    if fmt == "mjcf":
        xml_text = write_mjcf(cfg, rel_centers, radii, target_mass, target_fullinertia)
        if out.suffix.lower() != ".xml":
            out = out.with_suffix(".xml")
        out.write_text(xml_text, encoding="utf-8")
        print(f"\nWrote MJCF: {out}")
    elif fmt == "urdf":
        xml_text = write_urdf(cfg, rel_centers, radii, masses, world_origin)
        if out.suffix.lower() != ".urdf":
            out = out.with_suffix(".urdf")
        out.write_text(xml_text, encoding="utf-8")
        print(f"\nWrote URDF: {out}")
    else:
        raise ValueError(f"Invalid format: {fmt}. Must be 'urdf' or 'mjcf'")


if __name__ == "__main__":
    main()
