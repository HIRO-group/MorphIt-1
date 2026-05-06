"""
Stage 4 of the robot pipeline: end-to-end orchestrator.

Discover -> pack -> rewrite, in one CLI invocation. Used by the FastAPI
service when the user hits "go" from the UI; also handy from the shell
for full offline runs.

For finer-grained control (per-link progress bars, separate retries),
call the per-stage modules directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Path setup mirrors the per-stage modules.
HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from discover import find_urdfs, inspect_urdf, select_urdf  # noqa: E402
from pack_robot_meshes import VARIANT_CHOICES, pack_all  # noqa: E402
from create_robot_urdf import rewrite_urdf  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description=(
            "Drop-in robot pipeline: discover URDF + meshes, pack each "
            "collision mesh with MorphIt, rewrite URDF with sphere "
            "children. Outputs <output-dir>/<robot>_spherical.urdf."
        ),
    )
    parser.add_argument("--robot-dir", type=Path, required=True,
                        help="Folder containing the URDF and meshes.")
    parser.add_argument("--urdf", type=str, default=None,
                        help="URDF basename. Required if multiple "
                             "URDFs are present in the folder.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write the spheres/ dir, "
                             "inspection.json, and final URDF.")
    parser.add_argument("--variant", type=str, default="MorphIt-B",
                        choices=VARIANT_CHOICES)
    parser.add_argument("--num-spheres", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    if not args.robot_dir.is_dir():
        print(f"error: --robot-dir {args.robot_dir} is not a directory",
              file=sys.stderr)
        return 2

    # 1. Discover + select URDF.
    try:
        urdfs = find_urdfs(args.robot_dir)
        urdf_path = select_urdf(urdfs, args.urdf)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # 2. Inspect.
    report = inspect_urdf(args.robot_dir, urdf_path)
    if report.errors:
        for err in report.errors:
            print(f"error: {err}", file=sys.stderr)
        return 2

    pack_count = len(report.to_pack())
    if pack_count == 0:
        print("error: no collision meshes to pack in this URDF.",
              file=sys.stderr)
        return 1

    # 3. Pack every collision mesh.
    pack_all(
        report,
        variant=args.variant,
        num_spheres=args.num_spheres,
        iterations=args.iterations,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # 4. Rewrite URDF. Output filename derives from the input URDF stem
    # so it's obvious which robot it represents.
    output_urdf = args.output_dir / f"{urdf_path.stem}_spherical.urdf"
    stats = rewrite_urdf(
        report,
        spheres_dir=args.output_dir / "spheres",
        output_path=output_urdf,
    )

    print(f"\n=== Pipeline complete ===")
    print(f"  Robot URDF      : {urdf_path}")
    print(f"  Variant         : {args.variant}")
    print(f"  Spheres / link  : {args.num_spheres}")
    print(f"  Output URDF     : {output_urdf}")
    print(f"  Spheres added   : {stats.sphere_children_added}")
    if stats.skipped_pack_items:
        print(f"  Skipped         : {len(stats.skipped_pack_items)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
