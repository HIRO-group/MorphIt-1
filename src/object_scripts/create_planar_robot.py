# write_planar_rotating_spheres_xml.py
# Builds a URDF-style XML with planar x/y joints + a theta revolute, and N sphere links fixed to the rotating base.

import json
from pathlib import Path

CONFIG = {
    "input_json": "pusher_stick.json",  # hardcoded input
    "output_xml": "../urdfs/planar_robot.xml",
    "robot_name": "planar_spheres",

    # Visuals
    "default_color_rgba": (0.8, 0.0, 0.0, 1.0),

    # Precision
    "decimals": 6,

    # Mass: distributed to spheres by r^3
    "total_mass": 1.0,  # kg

    # Where the rotation axis passes through; can instead use centroid
    "use_centroid": True,
    "rotation_center_xyz": (0.0, 0.0, 0.0),

    # Axis for theta (usually z for planar)
    "theta_axis": "z",

    # Joint limits/effort
    "x_limit": (-500.0, 500.0),
    "y_limit": (-500.0, 500.0),
    "theta_limit": (-3.141592653589793, 3.141592653589793),
    "effort": 5000.0,
    "velocity": 5000.0,

    # Small but nonzero inertias for x_link and y_link carriers
    "carrier_mass": 0.001,
    "carrier_inertia_diag": 1e-5,

    # Optional global shift before placing around theta center
    "global_offset_xyz": (0.0, 0.0, 0.0),
}


def fnum(v, d): return f"{float(v):.{d}f}"


def inertia_solid_sphere(m, r):
    I = (2.0/5.0) * m * (r**2)
    return I, I, I


def axis_vec(name: str):
    n = name.lower().strip()
    if n == "x":
        return "1 0 0"
    if n == "y":
        return "0 1 0"
    if n == "z":
        return "0 0 1"
    raise ValueError("theta_axis must be 'x','y', or 'z'")


def main():
    cfg = CONFIG
    d = cfg["decimals"]

    data = json.loads(Path(cfg["input_json"]).read_text())
    centers = data["centers"]
    radii = data["radii"]
    assert len(centers) == len(radii), "centers/radii length mismatch"

    # Optional global shift
    gx, gy, gz = cfg["global_offset_xyz"]
    centers = [[c[0]+gx, c[1]+gy, c[2]+gz] for c in centers]

    # Rotation center
    if cfg["use_centroid"]:
        cx = sum(c[0] for c in centers)/len(centers)
        cy = sum(c[1] for c in centers)/len(centers)
        cz = sum(c[2] for c in centers)/len(centers)
    else:
        cx, cy, cz = cfg["rotation_center_xyz"]

    # Positions relative to rotation center (for fixed joints to robot_base)
    rel_centers = [[c[0]-cx, c[1]-cy, c[2]-cz] for c in centers]

    # Mass distribution
    vols = [r**3 for r in radii]
    vtot = sum(vols) if vols else 1.0
    masses = [cfg["total_mass"] * v / vtot for v in vols]

    # XML begin
    rgba = " ".join(fnum(x, d) for x in cfg["default_color_rgba"])
    xlo, xhi = cfg["x_limit"]
    ylo, yhi = cfg["y_limit"]
    thlo, thhi = cfg["theta_limit"]
    axis = axis_vec(cfg["theta_axis"])
    carr_m = fnum(cfg["carrier_mass"], d)
    carr_I = fnum(cfg["carrier_inertia_diag"], d)

    xml = []
    xml.append('<?xml version="1.0"?>')
    xml.append(f'<robot name="{cfg["robot_name"]}">')
    xml.append('  <material name="default_color">')
    xml.append(f'    <color rgba="{rgba}"/>')
    xml.append('  </material>')

    # Links: world, x_link, y_link, robot_base
    xml.append('  <link name="world"/>')

    xml.append('  <link name="x_link">')
    xml.append('    <inertial>')
    xml.append(f'      <mass value="{carr_m}"/>')
    xml.append(
        f'      <inertia ixx="{carr_I}" ixy="0.0" ixz="0.0" iyy="{carr_I}" iyz="0.0" izz="{carr_I}"/>')
    xml.append('    </inertial>')
    xml.append('  </link>')

    xml.append('  <link name="y_link">')
    xml.append('    <inertial>')
    xml.append(f'      <mass value="{carr_m}"/>')
    xml.append(
        f'      <inertia ixx="{carr_I}" ixy="0.0" ixz="0.0" iyy="{carr_I}" iyz="0.0" izz="{carr_I}"/>')
    xml.append('    </inertial>')
    xml.append('  </link>')

    xml.append('  <link name="robot_base">')
    xml.append('    <inertial>')
    xml.append(f'      <mass value="{carr_m}"/>')
    xml.append(
        f'      <inertia ixx="{carr_I}" ixy="0.0" ixz="0.0" iyy="{carr_I}" iyz="0.0" izz="{carr_I}"/>')
    xml.append('    </inertial>')
    xml.append('  </link>')

    # Joints: x (prismatic X), y (prismatic Y), theta (revolute at rotation center)
    xml.append('  <joint name="x_joint" type="prismatic">')
    xml.append('    <parent link="world"/>')
    xml.append('    <child link="x_link"/>')
    xml.append('    <axis xyz="1 0 0"/>')
    xml.append(
        f'    <limit lower="{fnum(xlo,d)}" upper="{fnum(xhi,d)}" effort="{fnum(cfg["effort"],d)}" velocity="{fnum(cfg["velocity"],d)}"/>')
    xml.append('  </joint>')

    xml.append('  <joint name="y_joint" type="prismatic">')
    xml.append('    <parent link="x_link"/>')
    xml.append('    <child link="y_link"/>')
    xml.append('    <axis xyz="0 1 0"/>')
    xml.append(
        f'    <limit lower="{fnum(ylo,d)}" upper="{fnum(yhi,d)}" effort="{fnum(cfg["effort"],d)}" velocity="{fnum(cfg["velocity"],d)}"/>')
    xml.append('  </joint>')

    xml.append('  <joint name="theta_joint" type="revolute">')
    xml.append('    <parent link="y_link"/>')
    xml.append('    <child link="robot_base"/>')
    xml.append(
        f'    <origin xyz="{fnum(cx,d)} {fnum(cy,d)} {fnum(cz,d)}" rpy="0 0 0"/>')
    xml.append(f'    <axis xyz="{axis}"/>')
    xml.append(
        f'    <limit lower="{fnum(thlo,d)}" upper="{fnum(thhi,d)}" effort="{fnum(cfg["effort"],d)}" velocity="{fnum(cfg["velocity"],d)}"/>')
    xml.append('  </joint>')

    # Sphere links fixed to robot_base
    for i, (ctr_rel, r, m) in enumerate(zip([[c[0]-cx, c[1]-cy, c[2]-cz] for c in centers], radii, masses)):
        sx, sy, sz = (fnum(v, d) for v in ctr_rel)
        rad = fnum(r, d)
        ixx, iyy, izz = inertia_solid_sphere(m, r)
        link_name = f"sphere_{i}"

        xml.append(f'  <link name="{link_name}">')
        xml.append('    <visual>')
        xml.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        xml.append(f'      <geometry><sphere radius="{rad}"/></geometry>')
        xml.append('      <material name="default_color"/>')
        xml.append('    </visual>')
        xml.append('    <collision>')
        xml.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        xml.append(f'      <geometry><sphere radius="{rad}"/></geometry>')
        xml.append('    </collision>')
        xml.append('    <inertial>')
        xml.append(f'      <mass value="{fnum(m,d)}"/>')
        xml.append(
            f'      <inertia ixx="{fnum(ixx,d)}" ixy="0.0" ixz="0.0" iyy="{fnum(iyy,d)}" iyz="0.0" izz="{fnum(izz,d)}"/>')
        xml.append('    </inertial>')
        xml.append('  </link>')

        xml.append(f'  <joint name="{link_name}_fixed" type="fixed">')
        xml.append('    <parent link="robot_base"/>')
        xml.append(f'    <child link="{link_name}"/>')
        xml.append(f'    <origin xyz="{sx} {sy} {sz}" rpy="0 0 0"/>')
        xml.append('  </joint>')

    xml.append('</robot>\n')

    Path(cfg["output_xml"]).write_text("\n".join(xml), encoding="utf-8")
    print(f"Wrote {cfg['output_xml']} with {len(radii)} spheres.")
    print(
        f"Planar joints: x_joint, y_joint; rotation: theta_joint about {cfg['theta_axis'].upper()}")
    print(
        f"Rotation center: ({cx:.6f}, {cy:.6f}, {cz:.6f}); total mass: {cfg['total_mass']} kg")


if __name__ == "__main__":
    main()
