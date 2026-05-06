"""
Stage 1 of the robot pipeline: discovery & inspection.

Walks a robot folder, finds the URDF(s), parses the chosen URDF, and
returns an `InspectionReport` describing what the packing stage will do
to each `<collision>` element it finds.

The report is the contract between every later stage:
  - The CLI prints it (humans).
  - The HTTP API returns it as JSON (UI inspection step).
  - The packing stage iterates the `pack` items.
  - The URDF-rewriter step strips `pack` collisions and inserts spheres.

This file does NOT touch any meshes or run any optimization; it is a
pure metadata pass that should complete in milliseconds even on large
URDFs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------

# Mesh extensions the API + packing stage know how to handle. .dae
# (COLLADA) loads via trimesh's collada extension and is concatenated
# into a single Mesh by `trimesh.load(force="mesh")` — works for all
# the .dae collisions we've seen so far in the urdf_files dataset.
# Anything outside this list is flagged at inspection time so we don't
# discover the failure three minutes into a long-running pack.
SUPPORTED_MESH_EXTENSIONS: Tuple[str, ...] = (".obj", ".stl", ".ply", ".dae")

# action values for a single <collision> element.
ACTION_PACK = "pack"                                  # mesh, will be sphere-packed
ACTION_REMOVE_PRIMITIVE = "remove-primitive"          # box / cylinder / capsule, stripped
ACTION_REMOVE_ALREADY_SPHERE = "remove-already-sphere"  # existing <sphere>, stripped
ACTION_ERROR = "error"                                # mesh not found / unsupported / malformed


@dataclass
class CollisionItem:
    """One <collision> element inside a <link>.

    `collision_index` is the 0-based index of this element among the
    sibling <collision> children of its <link>; it identifies the exact
    element the rewriter must strip in stage 3 when action == "pack".
    """

    link_name: str
    collision_index: int
    geometry_type: str               # "mesh" | "box" | "sphere" | "cylinder" | "capsule" | "unknown"
    action: str                      # ACTION_*
    mesh_path: Optional[str] = None  # absolute resolved path (string for JSON)
    mesh_filename: Optional[str] = None  # raw URDF filename attribute
    mesh_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    warning: Optional[str] = None


@dataclass
class InspectionReport:
    """Everything stage 2 needs to start packing."""

    robot_dir: str
    urdf_path: str
    urdfs_in_folder: List[str] = field(default_factory=list)
    collisions: List[CollisionItem] = field(default_factory=list)
    visual_count: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Convenience views ------------------------------------------------

    def to_pack(self) -> List[CollisionItem]:
        return [c for c in self.collisions if c.action == ACTION_PACK]

    def removed_primitives(self) -> List[CollisionItem]:
        return [c for c in self.collisions if c.action == ACTION_REMOVE_PRIMITIVE]

    def errored(self) -> List[CollisionItem]:
        return [c for c in self.collisions if c.action == ACTION_ERROR]

    def to_dict(self) -> dict:
        """JSON-serializable form used by the UI."""
        d = asdict(self)
        # tuples render as lists in JSON; that's fine — UI doesn't care.
        return d


# ---------------------------------------------------------------------
# URDF discovery
# ---------------------------------------------------------------------

def find_urdfs(robot_dir: Path) -> List[Path]:
    """Recursively list `.urdf` files under `robot_dir`."""
    return sorted(robot_dir.rglob("*.urdf"))


def select_urdf(urdfs: List[Path], requested: Optional[str]) -> Path:
    """Pick which URDF to process.

    - If `requested` is given, match by basename (case-sensitive).
    - Else if exactly one URDF was found, return it.
    - Else raise — the caller (CLI / API) must surface the choice.

    Errors carry the available filenames so the UI can re-prompt.
    """
    if requested:
        matches = [u for u in urdfs if u.name == requested]
        if not matches:
            available = sorted({u.name for u in urdfs})
            raise ValueError(
                f"--urdf {requested!r} not found. Available: {available}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"--urdf {requested!r} matches multiple paths: "
                f"{[str(p) for p in matches]}"
            )
        return matches[0]

    if not urdfs:
        raise ValueError("No .urdf files found in folder.")
    if len(urdfs) == 1:
        return urdfs[0]

    available = sorted({u.name for u in urdfs})
    raise ValueError(
        f"Multiple URDFs found; specify with --urdf NAME.urdf. "
        f"Available: {available}"
    )


# ---------------------------------------------------------------------
# Mesh path resolution
# ---------------------------------------------------------------------

def resolve_mesh_path(
    filename: str, robot_dir: Path, urdf_path: Path
) -> Optional[Path]:
    """Resolve a URDF `<mesh filename=...>` value to an absolute path.

    Resolution order, first hit wins:
      1. ``package://X/path``:
         a. ``robot_dir / X / path``           (user dropped the package parent)
         b. ``robot_dir / path``               (user dropped the package itself)
         c. recursive search for the basename  (last resort, only if unique)
      2. ``file://path`` -> strip the prefix.
      3. Absolute path -> as-is if it exists.
      4. Relative path -> resolved against the URDF's directory.

    Returns ``None`` if the file cannot be located.
    """
    if filename.startswith("package://"):
        rest = filename[len("package://"):]
        if "/" not in rest:
            return None
        pkg_name, rel = rest.split("/", 1)

        candidate = robot_dir / pkg_name / rel
        if candidate.exists():
            return candidate.resolve()

        candidate = robot_dir / rel
        if candidate.exists():
            return candidate.resolve()

        # Recursive fallback: search by basename. Only commit if exactly
        # one match — multiple matches would be ambiguous.
        basename = Path(rel).name
        matches = list(robot_dir.rglob(basename))
        if len(matches) == 1:
            return matches[0].resolve()
        return None

    if filename.startswith("file://"):
        return Path(filename[len("file://"):]).resolve()

    p = Path(filename)
    if p.is_absolute():
        return p.resolve() if p.exists() else None

    candidate = urdf_path.parent / p
    return candidate.resolve() if candidate.exists() else None


# ---------------------------------------------------------------------
# URDF parsing helpers
# ---------------------------------------------------------------------

_XACRO_ELEMENT = re.compile(r"<xacro:")        # <xacro:include>, <xacro:macro>, ...
_XACRO_SUBST = re.compile(r"\$\{[^}]+\}")      # ${prop} substitutions


def _looks_like_xacro(text: str) -> bool:
    """Detect *unresolved* xacro source.

    A bare ``xmlns:xacro="..."`` declaration on <robot> is harmless and
    survives in many fully-resolved URDFs (the converter often forgets
    to strip it). What actually matters is whether the file contains
    any unresolved xacro markers: a ``<xacro:*>`` element or a
    ``${...}`` substitution. Either of those means xacro processing
    hasn't been run yet.
    """
    return bool(_XACRO_ELEMENT.search(text) or _XACRO_SUBST.search(text))


def _parse_origin(origin_elem) -> Tuple[
    Tuple[float, float, float], Tuple[float, float, float]
]:
    """Return ((x,y,z), (r,p,y)) from a URDF <origin> element.

    URDF default for both is the zero vector when the element is
    absent or a particular attribute is missing.
    """
    if origin_elem is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xyz_str = origin_elem.get("xyz", "0 0 0")
    rpy_str = origin_elem.get("rpy", "0 0 0")
    xyz = tuple(float(v) for v in xyz_str.split())
    rpy = tuple(float(v) for v in rpy_str.split())
    if len(xyz) != 3:
        xyz = (0.0, 0.0, 0.0)
    if len(rpy) != 3:
        rpy = (0.0, 0.0, 0.0)
    return xyz, rpy


def _parse_scale(scale_str: Optional[str]) -> Tuple[float, float, float]:
    """Return the (sx, sy, sz) scale from a <mesh scale="..."> attribute."""
    if not scale_str:
        return (1.0, 1.0, 1.0)
    parts = scale_str.split()
    if len(parts) == 1:
        v = float(parts[0])
        return (v, v, v)
    if len(parts) == 3:
        return tuple(float(p) for p in parts)
    return (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------

def inspect_urdf(robot_dir: Path, urdf_path: Path) -> InspectionReport:
    """Build the inspection report for one URDF.

    Walks every <link>'s <collision> children and classifies each into
    one of the ACTION_* buckets. Visuals are counted but not enumerated
    — they're left untouched by the rewriter.
    """
    robot_dir = robot_dir.resolve()
    urdf_path = urdf_path.resolve()

    report = InspectionReport(
        robot_dir=str(robot_dir),
        urdf_path=str(urdf_path),
        urdfs_in_folder=[str(p) for p in find_urdfs(robot_dir)],
    )

    text = urdf_path.read_text()
    if _looks_like_xacro(text):
        report.errors.append(
            "This file looks like unresolved xacro source: it contains "
            "<xacro:...> elements or ${...} substitutions that have not "
            "been expanded. Run `xacro foo.xacro > foo.urdf` first and "
            "drop the resolved URDF instead."
        )
        return report

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        report.errors.append(f"URDF is not valid XML: {exc}")
        return report

    if root.tag != "robot":
        report.errors.append(
            f"Top-level element is <{root.tag}>, expected <robot>. "
            "Is this actually a URDF?"
        )
        return report

    # Iterate every <link> and classify its collisions.
    for link in root.findall(".//link"):
        link_name = link.get("name", "<unnamed>")

        # Visuals: count meshed visuals so the user can see "we leave 10 visuals untouched"
        for visual in link.findall("visual"):
            if visual.find("geometry/mesh") is not None:
                report.visual_count += 1

        for ci, coll in enumerate(link.findall("collision")):
            geom = coll.find("geometry")
            origin_xyz, origin_rpy = _parse_origin(coll.find("origin"))

            if geom is None:
                report.collisions.append(CollisionItem(
                    link_name=link_name,
                    collision_index=ci,
                    geometry_type="unknown",
                    action=ACTION_ERROR,
                    origin_xyz=origin_xyz,
                    origin_rpy=origin_rpy,
                    warning="<collision> has no <geometry> child.",
                ))
                continue

            mesh_elem = geom.find("mesh")
            if mesh_elem is not None:
                filename = mesh_elem.get("filename", "")
                scale = _parse_scale(mesh_elem.get("scale"))
                resolved = resolve_mesh_path(filename, robot_dir, urdf_path)

                if resolved is None:
                    report.collisions.append(CollisionItem(
                        link_name=link_name,
                        collision_index=ci,
                        geometry_type="mesh",
                        action=ACTION_ERROR,
                        mesh_filename=filename,
                        mesh_scale=scale,
                        origin_xyz=origin_xyz,
                        origin_rpy=origin_rpy,
                        warning=(
                            f"Mesh not found: {filename}. "
                            "Drop the package root so package:// URIs resolve."
                        ),
                    ))
                    continue

                suffix = resolved.suffix.lower()
                if suffix not in SUPPORTED_MESH_EXTENSIONS:
                    report.collisions.append(CollisionItem(
                        link_name=link_name,
                        collision_index=ci,
                        geometry_type="mesh",
                        action=ACTION_ERROR,
                        mesh_path=str(resolved),
                        mesh_filename=filename,
                        mesh_scale=scale,
                        origin_xyz=origin_xyz,
                        origin_rpy=origin_rpy,
                        warning=(
                            f"Unsupported mesh extension {suffix!r}; "
                            f"only {SUPPORTED_MESH_EXTENSIONS} are packed."
                        ),
                    ))
                    continue

                report.collisions.append(CollisionItem(
                    link_name=link_name,
                    collision_index=ci,
                    geometry_type="mesh",
                    action=ACTION_PACK,
                    mesh_path=str(resolved),
                    mesh_filename=filename,
                    mesh_scale=scale,
                    origin_xyz=origin_xyz,
                    origin_rpy=origin_rpy,
                ))
                continue

            # Primitive geometries. We treat the meshes as the source of
            # truth: any non-mesh <collision> is removed by the rewriter,
            # because keeping it alongside the new sphere children would
            # double up the link's collision representation.
            for prim in ("sphere", "box", "cylinder", "capsule"):
                if geom.find(prim) is not None:
                    if prim == "sphere":
                        action = ACTION_REMOVE_ALREADY_SPHERE
                        warning = (
                            "Existing <sphere> collision will be removed; "
                            "the mesh-derived spheres are the canonical "
                            "collision in the output URDF."
                        )
                    else:
                        action = ACTION_REMOVE_PRIMITIVE
                        warning = (
                            f"<{prim}> collision will be removed. "
                            "Mesh collisions are the source of truth; "
                            "redundant primitives are stripped."
                        )
                    report.collisions.append(CollisionItem(
                        link_name=link_name,
                        collision_index=ci,
                        geometry_type=prim,
                        action=action,
                        origin_xyz=origin_xyz,
                        origin_rpy=origin_rpy,
                        warning=warning,
                    ))
                    break
            else:
                report.collisions.append(CollisionItem(
                    link_name=link_name,
                    collision_index=ci,
                    geometry_type="unknown",
                    action=ACTION_ERROR,
                    origin_xyz=origin_xyz,
                    origin_rpy=origin_rpy,
                    warning=(
                        "Unknown collision geometry (not mesh / box / sphere "
                        "/ cylinder / capsule)."
                    ),
                ))

    return report


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _format_report(report: InspectionReport) -> str:
    """Human-readable summary used by the CLI."""
    lines: List[str] = []
    lines.append(f"Robot folder : {report.robot_dir}")
    if len(report.urdfs_in_folder) <= 5:
        lines.append(f"URDFs found  : {[Path(p).name for p in report.urdfs_in_folder]}")
    else:
        lines.append(f"URDFs found  : {len(report.urdfs_in_folder)} files")
    lines.append(f"Selected URDF: {Path(report.urdf_path).name}")
    lines.append("")

    if report.errors:
        lines.append("ERRORS:")
        for e in report.errors:
            lines.append(f"  - {e}")
        lines.append("")

    pack = report.to_pack()
    rm_prim = report.removed_primitives()
    rm_sph = [c for c in report.collisions if c.action == ACTION_REMOVE_ALREADY_SPHERE]
    errs = report.errored()

    lines.append(
        f"Collision elements: {len(report.collisions)} total "
        f"({len(pack)} pack, {len(rm_prim)} remove-primitive, "
        f"{len(rm_sph)} remove-already-sphere, {len(errs)} error)"
    )
    lines.append(f"Visual elements (untouched): {report.visual_count}")
    lines.append("")

    def _show(label: str, items: List[CollisionItem]):
        if not items:
            return
        lines.append(f"{label} ({len(items)}):")
        for c in items:
            tag = f"{c.link_name}[{c.collision_index}]"
            if c.action == ACTION_PACK and c.mesh_path:
                lines.append(f"  {tag}  {c.geometry_type}  →  {c.mesh_path}")
            else:
                msg = c.warning or ""
                lines.append(f"  {tag}  {c.geometry_type}  {msg}")
        lines.append("")

    _show("PACK", pack)
    _show("REMOVE-PRIMITIVE", rm_prim)
    _show("REMOVE-ALREADY-SPHERE", rm_sph)
    _show("ERROR", errs)

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="discover",
        description=(
            "Inspect a robot folder and print a plan of which collisions will "
            "be sphere-packed, skipped, or errored. Read-only — runs no MorphIt."
        ),
    )
    parser.add_argument(
        "--robot-dir",
        type=Path,
        required=True,
        help="Folder containing the URDF and meshes (recursive search).",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="URDF filename (basename) to inspect. Required if multiple "
        "URDFs are in the folder.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the inspection report as JSON instead of human text.",
    )
    args = parser.parse_args(argv)

    if not args.robot_dir.is_dir():
        print(f"error: --robot-dir {args.robot_dir} is not a directory",
              file=sys.stderr)
        return 2

    try:
        urdfs = find_urdfs(args.robot_dir)
        urdf_path = select_urdf(urdfs, args.urdf)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = inspect_urdf(args.robot_dir, urdf_path)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(_format_report(report))

    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
