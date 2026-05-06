"""
Stage 2 of the robot pipeline: sphere-pack each collision mesh.

Iterates the PACK items from stage 1's inspection report and runs
MorphIt on each. Writes one JSON per packed collision element to:

    <output_dir>/spheres/<link>_<collision_index>.json

The collision_index suffix lets us safely handle the (rare) case of a
single link that has multiple <mesh>-collision children.

Two entry points:

  * ``pack_one_link(item, ...)`` — packs one collision element. Used by
    the FastAPI per-link endpoint so the UI can show a progress bar
    (one HTTP call per link).

  * ``pack_all(report, ...)`` — drives the entire report sequentially.
    Used by the CLI and by ``run_pipeline.py``.

Both write the same files; the only difference is who calls them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

# Path setup: src/scripts/robot/x.py
#                              ^         ^
#                              parents[0] parents[1] = src/
HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))      # `import discover`
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))   # `import morphit`, `import config`

from config import get_config, update_config_from_dict  # noqa: E402
from morphit import MorphIt  # noqa: E402
from training import train_morphit  # noqa: E402

from discover import (  # noqa: E402
    ACTION_PACK,
    CollisionItem,
    InspectionReport,
    find_urdfs,
    inspect_urdf,
    select_urdf,
)

VARIANT_CHOICES = ("MorphIt-V", "MorphIt-S", "MorphIt-B")


def _spheres_dir(output_dir: Path) -> Path:
    return output_dir / "spheres"


def _json_filename(link_name: str, collision_index: int) -> str:
    """Per-collision filename: ``<link>_<idx>.json``.

    The suffix avoids overwriting siblings when a single link has more
    than one <mesh> collision child. For the common case of one mesh
    per link the index is 0 and the filename reads cleanly.
    """
    return f"{link_name}_{collision_index}.json"


@dataclass
class PackResult:
    """Outcome of packing one collision element."""

    link_name: str
    collision_index: int
    num_spheres: int
    json_path: str
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------
# Single-link packing
# ---------------------------------------------------------------------

def pack_one_link(
    item: CollisionItem,
    *,
    variant: str = "MorphIt-B",
    num_spheres: int = 20,
    iterations: int = 200,
    output_dir: Path,
    seed: Optional[int] = None,
    config_overrides: Optional[dict] = None,
) -> PackResult:
    """Run MorphIt on one collision mesh and write its sphere JSON.

    Args:
        item:        a CollisionItem with action == ACTION_PACK and a
                     resolved ``mesh_path``.
        variant:     "MorphIt-V" | "MorphIt-S" | "MorphIt-B".
        num_spheres: target sphere count.
        iterations:  gradient steps.
        output_dir:  parent directory; the JSON lands in ``<dir>/spheres/``.
        seed:        optional random seed for reproducibility.

    Returns:
        ``PackResult`` describing what was written.

    Raises:
        ValueError: if ``item.action != "pack"`` or ``mesh_path`` is None.
    """
    if item.action != ACTION_PACK:
        raise ValueError(
            f"pack_one_link called on non-PACK item: "
            f"{item.link_name}[{item.collision_index}] action={item.action}"
        )
    if not item.mesh_path:
        raise ValueError(
            f"pack_one_link called on item with no resolved mesh_path: "
            f"{item.link_name}[{item.collision_index}]"
        )

    # Non-identity <mesh scale=...> isn't yet honored. Most robotics
    # URDFs (Franka, FR3, UR, Spot) use identity scale, so this only
    # matters for atypical models. Warn loudly so the user knows the
    # output may be in unscaled mesh coordinates.
    if tuple(item.mesh_scale) != (1.0, 1.0, 1.0):
        print(
            f"  warning: {item.link_name}[{item.collision_index}] uses "
            f"<mesh scale={item.mesh_scale}>; we pack in unscaled mesh "
            f"coordinates and stage 3 will need to re-apply the scale."
        )

    spheres_dir = _spheres_dir(output_dir)
    spheres_dir.mkdir(parents=True, exist_ok=True)

    json_name = _json_filename(item.link_name, item.collision_index)

    updates = {
        "model.mesh_path": item.mesh_path,
        "model.num_spheres": num_spheres,
        "training.iterations": iterations,
        "training.logging_enabled": False,
        "visualization.enabled": False,
        "visualization.off_screen": True,
        "visualization.save_video": False,
        "results_dir": str(spheres_dir),
        "output_filename": json_name,
    }
    if seed is not None:
        updates["random_seed"] = seed
    if config_overrides:
        # Caller-supplied keys are dotted config paths (e.g.
        # "training.coverage_weight"). They override anything above.
        updates.update(config_overrides)

    config = update_config_from_dict(get_config(variant), updates)

    t0 = time.perf_counter()
    model = MorphIt(config)
    train_morphit(model)
    model.save_results()
    elapsed = time.perf_counter() - t0

    json_path = spheres_dir / json_name
    return PackResult(
        link_name=item.link_name,
        collision_index=item.collision_index,
        num_spheres=num_spheres,
        json_path=str(json_path),
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------
# Whole-robot driver
# ---------------------------------------------------------------------

def pack_all(
    report: InspectionReport,
    *,
    variant: str = "MorphIt-B",
    num_spheres: int = 20,
    iterations: int = 200,
    output_dir: Path,
    seed: Optional[int] = None,
) -> List[PackResult]:
    """Pack every PACK item in ``report`` sequentially.

    Saves the inspection report alongside the sphere JSONs so stage 3
    can be re-run later without redoing discovery.
    """
    pack_items = report.to_pack()
    if not pack_items:
        print("No collision meshes to pack.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPacking {len(pack_items)} collision mesh(es) "
          f"(variant={variant}, num_spheres={num_spheres}, "
          f"iterations={iterations})")
    print(f"Output -> {_spheres_dir(output_dir)}\n")

    results: List[PackResult] = []
    for i, item in enumerate(pack_items, 1):
        mesh_basename = Path(item.mesh_path).name if item.mesh_path else "?"
        print(f"[{i}/{len(pack_items)}] {item.link_name}[{item.collision_index}] "
              f"<- {mesh_basename}")
        result = pack_one_link(
            item,
            variant=variant,
            num_spheres=num_spheres,
            iterations=iterations,
            output_dir=output_dir,
            seed=seed,
        )
        results.append(result)
        print(f"  done in {result.elapsed_seconds:.1f}s -> {result.json_path}\n")

    # Persist the inspection report so stage 3 (URDF rewriter) can find
    # the original URDF path and the per-collision metadata it needs.
    report_path = output_dir / "inspection.json"
    with report_path.open("w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"All {len(results)} link(s) packed. Report saved to {report_path}")
    return results


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pack_robot_meshes",
        description=(
            "Sphere-pack every collision mesh in a robot's URDF. "
            "Writes one JSON per collision into <output-dir>/spheres/."
        ),
    )
    parser.add_argument("--robot-dir", type=Path, required=True,
                        help="Folder with the URDF and meshes.")
    parser.add_argument("--urdf", type=str, default=None,
                        help="URDF filename (basename). Required if "
                             "multiple URDFs are in the folder.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where spheres/ and inspection.json land.")
    parser.add_argument("--variant", type=str, default="MorphIt-B",
                        choices=VARIANT_CHOICES)
    parser.add_argument("--num-spheres", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility "
                             "(applied to every link).")
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
    if report.errors:
        for err in report.errors:
            print(f"error: {err}", file=sys.stderr)
        return 2

    pack_all(
        report,
        variant=args.variant,
        num_spheres=args.num_spheres,
        iterations=args.iterations,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
