import json
from pathlib import Path
import math
# =========================
# Config
# =========================
CONFIG = {
    # IO
    "input_json": "t_shape.json",
    # extension updated by format if you leave .xml
    "output_path": "../urdfs/planar_object.xml",
    "robot_name": "planar_object",

    # Output format + anchoring
    #   format: "urdf" or "mjcf"
    #   anchored=False -> dynamic under gravity (URDF floating root OR MJCF <freejoint/>)
    #   anchored=True  -> welded to world (URDF fixed joint OR MJCF no freejoint)
    "format": "mjcf",
    "anchored": False,

    # Visuals
    "default_color_rgba": (0.2, 0.6, 1.0, 1.0),

    # Precision
    "decimals": 6,

    # Mass distribution
    "total_mass": 1.0,  # kg, distributed by r^3

    # Centering
    "use_centroid": True,
    "rotation_center_xyz": (0.0, 0.0, 0.0),

    # Optional global shift applied to input centers BEFORE centering
    "global_offset_xyz": (0.0, 0.0, 0.0),

    # Base inertial (URDF only; small nonzero keeps parsers happy)
    "base_mass": 0.001,
    "base_inertia_diag": 1e-5,

    # Minimum radius clamp to avoid zero/negative radii
    "min_radius": 1e-6,
}

# =========================
# Helpers
# =========================


def fnum(v, d):  # float formatting
    return f"{float(v):.{d}f}"


def inertia_solid_sphere(m, r):
    # Solid sphere about own center: I = 2/5 m r^2 (diagonal)
    I = (2.0 / 5.0) * m * (r ** 2)
    return I, I, I


def load_inputs(cfg):
    data = json.loads(Path(cfg["input_json"]).read_text())
    centers = list(map(list, data.get("centers", [])))
    radii = list(map(float, data.get("radii", [])))

    if not centers or not radii:
        raise ValueError(
            "JSON must contain non-empty 'centers' and 'radii' arrays.")

    # Match lengths (trim/pad radii)
    if len(radii) < len(centers):
        radii = radii + [radii[-1]] * (len(centers) - len(radii))
    else:
        radii = radii[:len(centers)]

    # Clamp tiny/negative radii
    rmin = cfg["min_radius"]
    radii = [max(rmin, r) for r in radii]

    # Optional pre-centering shift
    gx, gy, gz = cfg["global_offset_xyz"]
    centers = [[c[0] + gx, c[1] + gy, c[2] + gz] for c in centers]

    # Choose anchor/rotation center
    if cfg["use_centroid"]:
        cx = sum(c[0] for c in centers) / len(centers)
        cy = sum(c[1] for c in centers) / len(centers)
        cz = sum(c[2] for c in centers) / len(centers)
    else:
        cx, cy, cz = cfg["rotation_center_xyz"]

    # Positions relative to base
    rel_centers = [[c[0] - cx, c[1] - cy, c[2] - cz] for c in centers]

    # Mass by r^3
    vols = [(r ** 3) for r in radii]
    vtot = sum(vols) if vols else 1.0
    masses = [cfg["total_mass"] * v / vtot for v in vols]

    return centers, radii, rel_centers, masses, (cx, cy, cz)

# =========================
# URDF writer (Genesis: load with gs.morphs.URDF)
# =========================


def write_urdf(cfg, rel_centers, radii, masses, world_origin):
    d = cfg["decimals"]
    rgba = " ".join(fnum(x, d) for x in cfg["default_color_rgba"])
    base_m = fnum(cfg["base_mass"], d)
    base_I = fnum(cfg["base_inertia_diag"], d)
    cx, cy, cz = world_origin

    xml = []
    xml.append('<?xml version="1.0"?>')
    xml.append(f'<robot name="{cfg["robot_name"]}">')

    # Shared material
    xml.append('  <material name="default_color">')
    xml.append(f'    <color rgba="{rgba}"/>')
    xml.append('  </material>')

    # If anchored, create an explicit world link and a fixed joint. If not anchored, leave base root free.
    if cfg["anchored"]:
        xml.append('  <link name="world"/>')

    # Root base
    xml.append('  <link name="object_base">')
    xml.append('    <inertial>')
    xml.append(f'      <mass value="{base_m}"/>')
    xml.append(
        f'      <inertia ixx="{base_I}" ixy="0.0" ixz="0.0" iyy="{base_I}" iyz="0.0" izz="{base_I}"/>'
    )
    xml.append('    </inertial>')
    xml.append('  </link>')

    # If anchored: weld base to world at computed origin; else: no parent -> floating root (dynamic).
    if cfg["anchored"]:
        xml.append('  <joint name="world_to_base" type="fixed">')
        xml.append('    <parent link="world"/>')
        xml.append('    <child link="object_base"/>')
        xml.append(
            f'    <origin xyz="{fnum(cx,d)} {fnum(cy,d)} {fnum(cz,d)}" rpy="0 0 0"/>')
        xml.append('  </joint>')

    # Spheres fixed to base
    for i, ((sx, sy, sz), r, m) in enumerate(zip(rel_centers, radii, masses)):
        sx, sy, sz = (fnum(sx, d), fnum(sy, d), fnum(sz, d))
        rad = fnum(float(r), d)
        ixx, iyy, izz = inertia_solid_sphere(m, r)
        link_name = f"sphere_{i}"

        xml.append(f'  <link name="{link_name}">')
        # visual
        xml.append('    <visual>')
        xml.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        xml.append(f'      <geometry><sphere radius="{rad}"/></geometry>')
        xml.append('      <material name="default_color"/>')
        xml.append('    </visual>')
        # collision
        xml.append('    <collision>')
        xml.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        xml.append(f'      <geometry><sphere radius="{rad}"/></geometry>')
        xml.append('    </collision>')
        # inertial
        xml.append('    <inertial>')
        xml.append(f'      <mass value="{fnum(m, d)}"/>')
        xml.append(
            f'      <inertia ixx="{fnum(ixx, d)}" ixy="0.0" ixz="0.0" iyy="{fnum(iyy, d)}" iyz="0.0" izz="{fnum(izz, d)}"/>'
        )
        xml.append('    </inertial>')
        xml.append('  </link>')

        xml.append(f'  <joint name="{link_name}_fixed" type="fixed">')
        xml.append('    <parent link="object_base"/>')
        xml.append(f'    <child link="{link_name}"/>')
        xml.append(f'    <origin xyz="{sx} {sy} {sz}" rpy="0 0 0"/>')
        xml.append('  </joint>')

    xml.append('</robot>\n')
    return "\n".join(xml)

# =========================
# MJCF writer (Genesis: load with gs.morphs.MJCF)
# =========================


def cf(cfg, rel_centers, radii, masses, world_origin):
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
        f'    <body name="object_base" pos="{fnum(cx,d)} {fnum(cy,d)} {fnum(cz,d)}">')
    if not cfg["anchored"]:
        # ✅ Name the free joint to satisfy Genesis parser
        xml.append('      <freejoint name="base_free"/>')

    # ✅ FIX: Add small invisible geom to give object_base its own mass
    # This ensures proper inertia without relying on child mass propagation
    xml.append(
        '      <geom type="sphere" size="0.001" density="1000" rgba="0 0 0 0"/>')

    for i, ((sx, sy, sz), r, m) in enumerate(zip(rel_centers, radii, masses)):
        sx, sy, sz = (fnum(sx, d), fnum(sy, d), fnum(sz, d))
        rad = float(r)
        rad_s = fnum(rad, d)

        vol = (4.0 / 3.0) * math.pi * (rad ** 3)
        dens = m / vol if vol > 0 else 0.0
        dens_s = fnum(dens, d)

        # Keep nested body structure for Genesis TF tracking
        xml.append(f'      <body name="sphere_{i}" pos="{sx} {sy} {sz}">')
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
# Main
# =========================


def main():
    cfg = CONFIG
    centers, radii, rel_centers, masses, world_origin = load_inputs(cfg)

    fmt = cfg["format"].lower().strip()
    if fmt not in ("urdf", "mjcf"):
        raise ValueError("CONFIG['format'] must be 'urdf' or 'mjcf'.")

    # Build text
    if fmt == "urdf":
        xml_text = write_urdf(cfg, rel_centers, radii, masses, world_origin)
        out = Path(cfg["output_path"])
        if out.suffix.lower() not in (".urdf", ".xml"):
            out = out.with_suffix(".urdf")
        out.write_text(xml_text, encoding="utf-8")
        print(f"Wrote URDF to: {out}")
        print("Load with: gs.morphs.URDF(file=...)")
        print("Anchored =", cfg['anchored'],
              "=>", "static (welded to world)" if cfg['anchored'] else "dynamic (floating root)")

    else:  # mjcf
        xml_text = cf(cfg, rel_centers, radii, masses, world_origin)
        out = Path(cfg["output_path"])
        if out.suffix.lower() != ".xml":
            out = out.with_suffix(".xml")
        out.write_text(xml_text, encoding="utf-8")
        print(f"Wrote MJCF to: {out}")
        print("Load with: gs.morphs.MJCF(file=...)")
        print("Anchored =", cfg['anchored'],
              "=>", "static (no <freejoint/>)" if cfg['anchored'] else "dynamic (<freejoint/>)")

    d = cfg["decimals"]
    cx, cy, cz = world_origin
    print(f"Origin used: ({cx:.{d}f}, {cy:.{d}f}, {cz:.{d}f})")
    print(f"Total sphere mass: {cfg['total_mass']} kg")
    print(f"Spheres: {len(radii)}")


if __name__ == "__main__":
    main()
