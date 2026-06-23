"""Stage a dataset robot into web/examples/ for bundling.

Copies just the URDF + the collision meshes the pipeline resolves
(preserving each file's path relative to the source robot dir, so the
URDF's relative mesh refs keep resolving). Visual meshes are skipped —
the web UI never renders them for example robots (the packed pane shows
spheres only; the original pane parses collision geometry).

Usage:
    venv-morph/bin/python scripts/stage_robot_example.py <src_robot_dir> <dest_name> [urdf_basename]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "src/scripts", REPO / "src/scripts/robot"):
    sys.path.insert(0, str(p))

from discover import find_urdfs, select_urdf, inspect_urdf  # noqa: E402


def main() -> int:
    src = Path(sys.argv[1]).resolve()
    dest_name = sys.argv[2]
    requested = sys.argv[3] if len(sys.argv) > 3 else None
    dest = REPO / "web" / "examples" / dest_name

    urdf = select_urdf(find_urdfs(src), requested)
    report = inspect_urdf(src, urdf)

    if dest.exists():
        shutil.rmtree(dest)

    # Files to copy: the URDF itself + every resolved collision mesh.
    files = {urdf}
    for c in report.collisions:
        if c.mesh_path:
            files.add(Path(c.mesh_path))

    copied = 0
    for f in sorted(files):
        rel = f.resolve().relative_to(src)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        copied += 1

    total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"staged {dest_name}: {copied} file(s), {total/1024:.0f} KB -> {dest}")
    print(f"  urdf basename: {urdf.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
