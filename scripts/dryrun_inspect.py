"""Fast inspect-only dry run for candidate robots (no training).

Validates that a robot's URDF parses, its collision meshes resolve, and
reports how many collision elements would be sphere-packed. Use before
committing to the (expensive) bundling step.

Usage:
    venv-morph/bin/python scripts/dryrun_inspect.py <robot_dir> [urdf_basename]
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "src/scripts", REPO / "src/scripts/robot"):
    sys.path.insert(0, str(p))

from discover import find_urdfs, select_urdf, inspect_urdf, ACTION_PACK, ACTION_ERROR  # noqa: E402


def main() -> int:
    robot_dir = Path(sys.argv[1]).resolve()
    requested = sys.argv[2] if len(sys.argv) > 2 else None
    urdfs = find_urdfs(robot_dir)
    urdf = select_urdf(urdfs, requested)
    report = inspect_urdf(robot_dir, urdf)

    actions = Counter(c.action for c in report.collisions)
    print(f"  urdf       : {urdf.relative_to(robot_dir)}")
    print(f"  collisions : {len(report.collisions)}  -> {dict(actions)}")
    errors = [c for c in report.collisions if c.action == ACTION_ERROR]
    for c in errors:
        print(f"    ERROR {c.link_name}[{c.collision_index}]: {getattr(c, 'detail', '')}")
    n_pack = actions.get(ACTION_PACK, 0)
    print(f"  VERDICT    : {'OK' if n_pack and not errors else 'CHECK'} "
          f"({n_pack} link(s) would pack, {len(errors)} error(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
