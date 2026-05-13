"""
Pre-pack meshes with default UI settings and bundle the result into
web/examples/ so the page can display packed objects immediately on load.

Writes two files into web/examples/{name}/:
  - {name}.obj         (copy of the source mesh, used for wireframe overlay)
  - {name}.urdf        (sphere-decomposition URDF with an embedded
                        `morphit:centroid` XML comment — the centroid
                        travels inside the URDF so copy-pasting it
                        also carries the alignment metadata)

Re-run whenever the UI defaults change, or to add a new object:

    venv-morph/bin/python scripts/bundle_object_example.py --name <object>
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = SRC_DIR / "scripts"
WEB_API_DIR = REPO_ROOT / "web" / "api"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(WEB_API_DIR))

from config import get_config, update_config_from_dict  # noqa: E402
from morphit import MorphIt  # noqa: E402
from training import train_morphit  # noqa: E402
from create_object_urdf import load_inputs as urdf_load_inputs, write_urdf  # noqa: E402
from main import inject_centroid_comment  # noqa: E402

# Where the *source* mesh for each named example lives. Keys must match
# entries in EXAMPLE_OBJECTS in web/api/main.py (the API's registry).
SOURCES: dict[str, Path] = {
    "bunny": REPO_ROOT / "mesh_models" / "objects" / "bunny.obj",
    # Add a new entry here when introducing a new pre-baked object.
}

OUT_DIR = REPO_ROOT / "web" / "examples"

# Match the UI's "Run MorphIt" defaults so the pre-baked result is
# what a user would get clicking Run themselves with no advanced
# overrides. Seed pinned for reproducibility.
DEFAULTS = {
    "variant": "MorphIt-B",
    "num_spheres": 20,
    "iterations": 200,
    "seed": 0,
    "base_color_rgba": (0.2, 0.6, 1.0, 1.0),
}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="bunny",
                   help="Registry key from SOURCES (default: bunny).")
    args = p.parse_args(argv)

    if args.name not in SOURCES:
        raise SystemExit(
            f"unknown object {args.name!r}; known: {sorted(SOURCES)}"
        )
    source_mesh = SOURCES[args.name]
    if not source_mesh.exists():
        raise SystemExit(f"source mesh not found: {source_mesh}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_mesh = OUT_DIR / f"{args.name}.obj"
    out_urdf = OUT_DIR / f"{args.name}.urdf"

    shutil.copy2(source_mesh, out_mesh)
    print(f"copied  {source_mesh} -> {out_mesh}")

    # Retire any pre-existing .meta.json sidecar: the centroid now
    # lives inside the URDF, so the sidecar would just go stale.
    stale_meta = OUT_DIR / f"{args.name}.meta.json"
    if stale_meta.exists():
        stale_meta.unlink()
        print(f"removed stale sidecar {stale_meta}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out_name = "morphit_result.json"
        config = update_config_from_dict(get_config(DEFAULTS["variant"]), {
            "model.mesh_path": str(source_mesh),
            "model.num_spheres": DEFAULTS["num_spheres"],
            "training.iterations": DEFAULTS["iterations"],
            "training.logging_enabled": False,
            "visualization.enabled": False,
            "visualization.off_screen": True,
            "visualization.save_video": False,
            "results_dir": str(tmp),
            "output_filename": out_name,
            "random_seed": DEFAULTS["seed"],
        })

        print(f"training MorphIt for {args.name}...")
        model = MorphIt(config)
        train_morphit(model)
        model.save_results()

        urdf_cfg = {
            "format": "urdf",
            "anchored": False,
            "default_color_rgba": DEFAULTS["base_color_rgba"],
            "decimals": 6,
            "total_mass": 1.0,
            "use_centroid": True,
            "rotation_center_xyz": (0.0, 0.0, 0.0),
            "global_offset_xyz": (0.0, 0.0, 0.0),
            "base_mass": 0.001,
            "base_inertia_diag": 1e-5,
            "min_radius": 1e-6,
            "input_json": str(tmp / out_name),
            "robot_name": args.name,
        }
        _centers, radii, rel_centers, masses, world_origin = urdf_load_inputs(urdf_cfg)
        urdf_text = write_urdf(urdf_cfg, rel_centers, radii, masses, world_origin)
        urdf_text = inject_centroid_comment(urdf_text, world_origin)

    out_urdf.write_text(urdf_text)
    print(f"wrote   {out_urdf}  ({len(urdf_text):,} bytes)")
    print(f"centroid: {tuple(float(v) for v in world_origin)}")


if __name__ == "__main__":
    main()
