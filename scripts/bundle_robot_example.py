"""
Run the full robot sphere-packing pipeline for one bundled robot and
write the resulting spherical URDF + per-link sphere JSONs into
web/examples/.

Usage:
    venv-morph/bin/python scripts/bundle_robot_example.py --name kinova

Args mirror the UI's Run MorphIt defaults; seed pinned for
reproducibility.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = SRC_DIR / "scripts"
ROBOT_SCRIPTS_DIR = SCRIPTS_DIR / "robot"
WEB_API_DIR = REPO_ROOT / "web" / "api"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ROBOT_SCRIPTS_DIR))
sys.path.insert(0, str(WEB_API_DIR))

from discover import ACTION_PACK, find_urdfs, inspect_urdf, select_urdf  # noqa: E402
from pack_robot_meshes import pack_one_link  # noqa: E402
from create_robot_urdf import DEFAULT_SPHERE_RGBA, rewrite_urdf  # noqa: E402
from main import EXAMPLE_ROBOTS  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="kinova",
                   help="Registry key from EXAMPLE_ROBOTS.")
    p.add_argument("--variant", default="MorphIt-B")
    p.add_argument("--num-spheres", type=int, default=20)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    robot = EXAMPLE_ROBOTS.get(args.name)
    if robot is None:
        raise SystemExit(
            f"unknown robot {args.name!r}; known: {sorted(EXAMPLE_ROBOTS)}"
        )
    folder: Path = robot["folder"]
    if not folder.exists():
        raise SystemExit(f"robot folder missing: {folder}")

    # Convention from the existing robot-mode API path: pack_one_link
    # writes each link's JSON into <output_dir>/spheres/, and
    # rewrite_urdf reads from the same spheres/ subdir. Mirror that.
    out_dir = REPO_ROOT / "web" / "examples" / f"{args.name}.spheres"
    spheres_dir = out_dir / "spheres"
    out_urdf = robot["spherical_urdf"]

    # Wipe any previous output so a re-run is clean.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    spheres_dir.mkdir(parents=True)

    # Step 1: inspect the URDF -> report of PACK / SKIP / etc. items.
    urdfs = find_urdfs(folder)
    urdf_path = select_urdf(urdfs, robot["urdf"])
    report = inspect_urdf(folder, urdf_path)

    pack_items = [c for c in report.collisions if c.action == ACTION_PACK]
    print(f"packing {len(pack_items)} collision element(s) for {args.name}...")

    # Step 2: pack each link.
    for i, item in enumerate(pack_items, 1):
        print(f"  [{i}/{len(pack_items)}] {item.link_name}[{item.collision_index}]")
        pack_one_link(
            item,
            variant=args.variant,
            num_spheres=args.num_spheres,
            iterations=args.iterations,
            output_dir=out_dir,
            seed=args.seed,
            config_overrides=None,
        )

    # Step 3: assemble the spherical URDF.
    print(f"assembling -> {out_urdf}")
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    stats = rewrite_urdf(
        report,
        spheres_dir=spheres_dir,
        output_path=out_urdf,
        base_color_rgba=DEFAULT_SPHERE_RGBA,
        color_variation=0.3,
    )
    if stats.skipped_pack_items:
        raise SystemExit(
            "skipped pack items: " + "; ".join(
                f"{l}[{i}]: {r}" for l, i, r in stats.skipped_pack_items
            )
        )
    print(f"done: {out_urdf}  ({out_urdf.stat().st_size:,} bytes)")
    print(f"sphere JSONs in {spheres_dir}")


if __name__ == "__main__":
    main()
