"""
Stage 3 of the robot pipeline: rewrite the URDF.

Reads the original URDF + the per-link sphere JSONs produced by stage 2
and emits a new URDF where every PACK collision element has been:

    <collision><geometry><mesh .../></geometry></collision>

replaced by N sphere child-links (one per packed sphere), each connected
to the original link via a fixed joint at the appropriate position. The
original collision's <origin xyz= rpy=> is composed onto the sphere
center so the spheres land where the original mesh sat.

Visuals, SKIP-PRIMITIVE / SKIP-ALREADY-SPHERE collisions, joints,
transmissions, gazebo blocks, drake:* extensions, and every other
top-level URDF element pass through untouched. Only <collision>
elements with action == "pack" are replaced.

Two entry points:

  * ``rewrite_urdf(report, spheres_dir, output_path)`` — library form
    used by ``run_pipeline.py`` and the FastAPI assemble endpoint.

  * CLI: ``python -m scripts.robot.create_robot_urdf
                --inspection inspection.json
                --spheres-dir <dir>
                --output <path>``
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Path setup mirrors pack_robot_meshes.py.
HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from discover import (  # noqa: E402
    ACTION_PACK,
    ACTION_REMOVE_ALREADY_SPHERE,
    ACTION_REMOVE_PRIMITIVE,
    CollisionItem,
    InspectionReport,
)


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def _rpy_to_matrix(rpy: Tuple[float, float, float]) -> List[List[float]]:
    """Build a 3x3 rotation matrix from URDF roll/pitch/yaw (XYZ extrinsic).

    URDF convention: rotation = Rz(yaw) * Ry(pitch) * Rx(roll).
    We compute the composed matrix in pure Python to avoid pulling in
    numpy at this layer (the overall script is otherwise dependency-free).
    """
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    # Rz * Ry * Rx
    return [
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [-sp,      cp * sr,                 cp * cr],
    ]


def _apply_collision_transform(
    sphere_center: Tuple[float, float, float],
    collision_xyz: Tuple[float, float, float],
    collision_rpy: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Lift a sphere center from mesh frame into link frame.

    The URDF says: this collision's mesh is positioned at
    (collision_xyz, collision_rpy) within the parent link's frame.
    MorphIt produces sphere centers in mesh frame. To position the
    sphere as a child of the link, we apply the same transform:

        c_link = R(collision_rpy) @ c_mesh + collision_xyz

    For Franka and most ROS robots collision_xyz is zero and
    collision_rpy is identity, so this reduces to c_mesh — but doing
    it correctly here means we don't break for less tidy URDFs.
    """
    R = _rpy_to_matrix(collision_rpy)
    cx, cy, cz = sphere_center
    rotated = (
        R[0][0] * cx + R[0][1] * cy + R[0][2] * cz,
        R[1][0] * cx + R[1][1] * cy + R[1][2] * cz,
        R[2][0] * cx + R[2][1] * cy + R[2][2] * cz,
    )
    return (
        rotated[0] + collision_xyz[0],
        rotated[1] + collision_xyz[1],
        rotated[2] + collision_xyz[2],
    )


# ---------------------------------------------------------------------
# Default sphere-link templates
# ---------------------------------------------------------------------

# Each sphere child needs a small inertial block to satisfy strict URDF
# parsers. Drake / Gazebo accept the values below for the panda script
# without complaint; they're tiny but nonzero so the parser doesn't
# refuse the link.
SPHERE_INERTIAL_MASS = 0.001
SPHERE_INERTIAL_DIAG = 1e-5

# Default sphere RGBA when the original collision had no material info
# we can copy. Picked to match the existing object pipeline.
DEFAULT_SPHERE_RGBA = (0.2, 0.6, 1.0, 1.0)

# Maximum hue swing (in fractions of the full color circle) when the
# variation slider is at 1.0. 0.15 ≈ 54° — wide enough to be obvious,
# narrow enough to keep neighbouring links visually related.
COLOR_VARIATION_MAX_HUE_SWING = 0.15


def _fmt(v: float, decimals: int = 6) -> str:
    return f"{float(v):.{decimals}f}"


def hex_to_rgba(hex_str: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    """Parse "#rrggbb" / "rrggbb" -> (r, g, b, alpha) in 0..1."""
    h = hex_str.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected #rrggbb hex color, got {hex_str!r}")
    return (
        int(h[0:2], 16) / 255.0,
        int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0,
        float(alpha),
    )


def _vary_color(
    base_rgba: Tuple[float, float, float, float],
    t: float,
    variation: float,
) -> Tuple[float, float, float, float]:
    """Shift the hue of `base_rgba` by `t * variation * MAX_SWING`.

    `t` is the link's normalized position in [-1, +1] across the pack;
    `variation` in [0, 1] scales the spread (0 = no variation, all links
    share the base color; 1 = full ±MAX_SWING swing).
    """
    if variation <= 0:
        return base_rgba
    r, g, b, a = base_rgba
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    new_h = (h + t * variation * COLOR_VARIATION_MAX_HUE_SWING) % 1.0
    nr, ng, nb = colorsys.hls_to_rgb(new_h, l, s)
    return (nr, ng, nb, a)


def _make_sphere_link(
    *,
    parent_link_name: str,
    sphere_idx: int,
    radius: float,
    color_rgba: Tuple[float, float, float, float],
    drake_proximity: Optional[ET.Element] = None,
) -> Tuple[ET.Element, str]:
    """Build a sphere child link with visual + collision + minimal inertia.

    Returns (link element, link name).
    """
    link_name = f"{parent_link_name}_sphere{sphere_idx}"
    link = ET.Element("link", {"name": link_name})

    # Inertial — tiny but nonzero
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": _fmt(SPHERE_INERTIAL_MASS)})
    ET.SubElement(inertial, "inertia", {
        "ixx": _fmt(SPHERE_INERTIAL_DIAG),
        "ixy": "0", "ixz": "0",
        "iyy": _fmt(SPHERE_INERTIAL_DIAG),
        "iyz": "0",
        "izz": _fmt(SPHERE_INERTIAL_DIAG),
    })

    # Visual
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    visual_geom = ET.SubElement(visual, "geometry")
    ET.SubElement(visual_geom, "sphere", {"radius": _fmt(radius)})
    material = ET.SubElement(visual, "material",
                             {"name": f"color_{link_name}"})
    rgba_str = " ".join(_fmt(c) for c in color_rgba)
    ET.SubElement(material, "color", {"rgba": rgba_str})

    # Collision — also a sphere of the same radius
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    coll_geom = ET.SubElement(collision, "geometry")
    ET.SubElement(coll_geom, "sphere", {"radius": _fmt(radius)})

    # Preserve any drake:proximity_properties from the original
    # collision so simulator-side hydroelastic / dissipation settings
    # carry across to the spheres.
    if drake_proximity is not None:
        collision.append(deepcopy(drake_proximity))

    return link, link_name


def _make_fixed_joint(
    *,
    parent_link: str,
    child_link: str,
    xyz: Tuple[float, float, float],
) -> ET.Element:
    """Fixed joint placing the sphere child at xyz in the parent link."""
    joint = ET.Element("joint", {
        "name": f"{parent_link}_to_{child_link}",
        "type": "fixed",
    })
    ET.SubElement(joint, "parent", {"link": parent_link})
    ET.SubElement(joint, "child", {"link": child_link})
    ET.SubElement(joint, "origin", {
        "xyz": " ".join(_fmt(c) for c in xyz),
        "rpy": "0 0 0",
    })
    return joint


# ---------------------------------------------------------------------
# Sphere-JSON loading
# ---------------------------------------------------------------------

def _spheres_json_path(spheres_dir: Path, item: CollisionItem) -> Path:
    return spheres_dir / f"{item.link_name}_{item.collision_index}.json"


def _load_spheres(json_path: Path) -> Tuple[List[List[float]], List[float]]:
    """Read a stage-2 JSON and return (centers, radii)."""
    with json_path.open() as f:
        data = json.load(f)
    centers = [list(map(float, c)) for c in data["centers"]]
    radii = [float(r) for r in data["radii"]]
    if len(centers) != len(radii):
        raise ValueError(
            f"{json_path}: centers ({len(centers)}) and radii "
            f"({len(radii)}) length mismatch"
        )
    return centers, radii


# ---------------------------------------------------------------------
# Main rewriter
# ---------------------------------------------------------------------

@dataclass
class RewriteStats:
    """Summary of what the rewriter changed."""

    links_with_collisions_replaced: int
    mesh_collisions_replaced: int          # PACK items: mesh -> sphere children
    primitive_collisions_removed: int      # box / cylinder / capsule
    sphere_collisions_removed: int         # existing <sphere> stripped
    sphere_children_added: int
    skipped_pack_items: List[Tuple[str, int, str]]  # (link, idx, reason)

    @property
    def collisions_stripped(self) -> int:
        """Total <collision> elements removed (any reason)."""
        return (self.mesh_collisions_replaced
                + self.primitive_collisions_removed
                + self.sphere_collisions_removed)


def _find_link_element(root: ET.Element, link_name: str) -> Optional[ET.Element]:
    for link in root.findall(".//link"):
        if link.get("name") == link_name:
            return link
    return None


def rewrite_urdf(
    report: InspectionReport,
    spheres_dir: Path,
    output_path: Path,
    *,
    base_color_rgba: Tuple[float, float, float, float] = DEFAULT_SPHERE_RGBA,
    color_variation: float = 0.0,
) -> RewriteStats:
    """Strip pack collisions and append sphere children + joints.

    Args:
        report:           the inspection report from stage 1 (reads
                          ``urdf_path`` for the original URDF + the
                          per-collision metadata it needs).
        spheres_dir:      where stage 2 wrote ``<link>_<idx>.json`` files.
        output_path:      where to write the new URDF.
        base_color_rgba:  RGBA tuple in 0..1 for sphere visuals. Default
                          is a soft blue that matches the object pipeline.
        color_variation:  0..1, hue spread across packed links. 0 = every
                          link shares the base color; higher = each link
                          gets a slightly different shade for visual
                          differentiation. Has no effect on object mode
                          (single packed link).

    Returns:
        RewriteStats describing the diff.
    """
    urdf_path = Path(report.urdf_path)

    # Parse, preserving comments. ET.write() will reformat whitespace
    # but the structural output is what matters for downstream tools.
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(urdf_path, parser=parser)
    root = tree.getroot()

    # Keep drake namespace on the root so drake:* elements survive.
    ET.register_namespace("", "")
    ET.register_namespace("drake", "http://drake.mit.edu")

    # We process ALL collisions the report classified — pack, remove-
    # primitive, remove-already-sphere — in one pass. Mesh items get
    # replaced with sphere children; primitive / sphere items are just
    # stripped (the user-supplied meshes are the source of truth, so
    # any other collision representation in the original URDF is
    # redundant once we've packed the meshes).
    REMOVE_ACTIONS = (ACTION_REMOVE_PRIMITIVE, ACTION_REMOVE_ALREADY_SPHERE)
    actionable = [
        c for c in report.collisions
        if c.action == ACTION_PACK or c.action in REMOVE_ACTIONS
    ]

    # Per-link color: spread the base hue across the packed links so each
    # one is visually distinguishable. `t` runs from -1 at the first link
    # to +1 at the last; the helper centers the variation around the base.
    pack_link_names: List[str] = []
    seen = set()
    for c in actionable:
        if c.action == ACTION_PACK and c.link_name not in seen:
            seen.add(c.link_name)
            pack_link_names.append(c.link_name)
    n_pack_links = len(pack_link_names)
    link_color: Dict[str, Tuple[float, float, float, float]] = {}
    for i, name in enumerate(pack_link_names):
        if n_pack_links > 1 and color_variation > 0:
            t = (i / (n_pack_links - 1)) * 2 - 1
            link_color[name] = _vary_color(base_color_rgba, t, color_variation)
        else:
            link_color[name] = base_color_rgba

    # Group by link so we can resolve collision_index against the
    # *current* DOM in document order. Stripping collisions in reverse
    # index order preserves the indices of un-touched siblings.
    by_link: Dict[str, List[CollisionItem]] = {}
    for item in actionable:
        by_link.setdefault(item.link_name, []).append(item)

    stats = RewriteStats(
        links_with_collisions_replaced=0,
        mesh_collisions_replaced=0,
        primitive_collisions_removed=0,
        sphere_collisions_removed=0,
        sphere_children_added=0,
        skipped_pack_items=[],
    )

    # Track new sphere links + joints to append at end (so we don't
    # disturb index ordering during the strip phase).
    new_links: List[ET.Element] = []
    new_joints: List[ET.Element] = []

    for link_name, items in by_link.items():
        link_elem = _find_link_element(root, link_name)
        if link_elem is None:
            for it in items:
                if it.action == ACTION_PACK:
                    stats.skipped_pack_items.append(
                        (link_name, it.collision_index,
                         f"link <{link_name}> not found in URDF")
                    )
            continue

        items_sorted = sorted(items, key=lambda c: c.collision_index,
                              reverse=True)
        link_changed = False

        for item in items_sorted:
            collisions = link_elem.findall("collision")
            if item.collision_index >= len(collisions):
                if item.action == ACTION_PACK:
                    stats.skipped_pack_items.append(
                        (link_name, item.collision_index,
                         f"collision_index out of range "
                         f"(link has {len(collisions)} collisions)")
                    )
                continue
            target_coll = collisions[item.collision_index]

            if item.action == ACTION_PACK:
                json_path = _spheres_json_path(spheres_dir, item)
                if not json_path.exists():
                    stats.skipped_pack_items.append(
                        (link_name, item.collision_index,
                         f"missing JSON: {json_path}")
                    )
                    continue

                # Hold a deep-copyable handle to drake props before we
                # remove the parent — ElementTree losses subtree on remove.
                drake_props = target_coll.find(
                    "{http://drake.mit.edu}proximity_properties"
                )
                if drake_props is None:
                    drake_props = target_coll.find("drake:proximity_properties")

                link_elem.remove(target_coll)
                stats.mesh_collisions_replaced += 1
                link_changed = True

                centers, radii = _load_spheres(json_path)

                for sphere_idx, (c_mesh, radius) in enumerate(
                    zip(centers, radii), start=1
                ):
                    c_link = _apply_collision_transform(
                        tuple(c_mesh),
                        tuple(item.origin_xyz),
                        tuple(item.origin_rpy),
                    )

                    sphere_link, sphere_link_name = _make_sphere_link(
                        parent_link_name=link_name,
                        sphere_idx=sphere_idx,
                        radius=radius,
                        color_rgba=link_color.get(link_name, base_color_rgba),
                        drake_proximity=drake_props,
                    )
                    joint = _make_fixed_joint(
                        parent_link=link_name,
                        child_link=sphere_link_name,
                        xyz=c_link,
                    )
                    new_links.append(sphere_link)
                    new_joints.append(joint)
                    stats.sphere_children_added += 1

            elif item.action == ACTION_REMOVE_PRIMITIVE:
                link_elem.remove(target_coll)
                stats.primitive_collisions_removed += 1
                link_changed = True

            elif item.action == ACTION_REMOVE_ALREADY_SPHERE:
                link_elem.remove(target_coll)
                stats.sphere_collisions_removed += 1
                link_changed = True

        if link_changed:
            stats.links_with_collisions_replaced += 1

    for elem in new_links:
        root.append(elem)
    for elem in new_joints:
        root.append(elem)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return stats


# ---------------------------------------------------------------------
# Inspection report loader
# ---------------------------------------------------------------------

def load_inspection(path: Path) -> InspectionReport:
    """Re-hydrate a saved inspection.json into an InspectionReport."""
    with path.open() as f:
        d = json.load(f)
    items = [
        CollisionItem(
            link_name=c["link_name"],
            collision_index=c["collision_index"],
            geometry_type=c["geometry_type"],
            action=c["action"],
            mesh_path=c.get("mesh_path"),
            mesh_filename=c.get("mesh_filename"),
            mesh_scale=tuple(c.get("mesh_scale", (1.0, 1.0, 1.0))),
            origin_xyz=tuple(c.get("origin_xyz", (0.0, 0.0, 0.0))),
            origin_rpy=tuple(c.get("origin_rpy", (0.0, 0.0, 0.0))),
            warning=c.get("warning"),
        )
        for c in d.get("collisions", [])
    ]
    return InspectionReport(
        robot_dir=d["robot_dir"],
        urdf_path=d["urdf_path"],
        urdfs_in_folder=d.get("urdfs_in_folder", []),
        collisions=items,
        visual_count=d.get("visual_count", 0),
        warnings=d.get("warnings", []),
        errors=d.get("errors", []),
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="create_robot_urdf",
        description=(
            "Rewrite a robot URDF to replace mesh-based collisions with "
            "sphere child-links produced by MorphIt (stage 2 of the "
            "robot pipeline)."
        ),
    )
    parser.add_argument("--inspection", type=Path, required=True,
                        help="Path to inspection.json from stage 2 "
                             "(e.g. <output-dir>/inspection.json).")
    parser.add_argument("--spheres-dir", type=Path, required=True,
                        help="Directory with <link>_<idx>.json files.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output URDF path.")
    args = parser.parse_args(argv)

    if not args.inspection.is_file():
        print(f"error: --inspection {args.inspection} not found",
              file=sys.stderr)
        return 2
    if not args.spheres_dir.is_dir():
        print(f"error: --spheres-dir {args.spheres_dir} not found",
              file=sys.stderr)
        return 2

    report = load_inspection(args.inspection)
    if report.errors:
        for err in report.errors:
            print(f"error: {err}", file=sys.stderr)
        return 2

    stats = rewrite_urdf(report, args.spheres_dir, args.output)

    print(f"\nRewrite complete.")
    print(f"  Original URDF       : {report.urdf_path}")
    print(f"  Output URDF         : {args.output}")
    print(f"  Links touched       : {stats.links_with_collisions_replaced}")
    print(f"  Mesh -> spheres     : {stats.mesh_collisions_replaced}")
    print(f"  Primitives removed  : {stats.primitive_collisions_removed}")
    print(f"  Existing spheres rm : {stats.sphere_collisions_removed}")
    print(f"  Sphere children     : {stats.sphere_children_added}")
    if stats.skipped_pack_items:
        print(f"  Skipped pack items  : {len(stats.skipped_pack_items)}")
        for link, idx, reason in stats.skipped_pack_items:
            print(f"    - {link}[{idx}]: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
