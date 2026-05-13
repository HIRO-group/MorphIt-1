"""
Render a small PNG thumbnail per bundled example mesh.

Reads the EXAMPLE_OBJECTS registry from web/api/main.py and writes
<name>.png next to each mesh under web/examples/. Idempotent: skips
entries whose PNG mtime is newer than the source mesh unless --force
is passed.

Usage:
    venv-morph/bin/python scripts/generate_thumbnails.py
    venv-morph/bin/python scripts/generate_thumbnails.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyvista as pv

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_API_DIR = REPO_ROOT / "web" / "api"
sys.path.insert(0, str(WEB_API_DIR))

from main import EXAMPLE_OBJECTS  # noqa: E402

THUMB_SIZE = (256, 256)
# Cyber-dark palette matching the library modal: near-black surface,
# cool grey mesh, faint cyan-tinted edges.
BG = (0.024, 0.039, 0.075)      # #060a13 — matches tile thumb well
MESH_COLOR = (0.62, 0.66, 0.74) # cool grey, visible against dark bg
EDGE_COLOR = (0.27, 0.42, 0.55) # cyan-leaning slate for edges


def _needs_rebuild(src: Path, out: Path, force: bool) -> bool:
    if force or not out.exists():
        return True
    return out.stat().st_mtime < src.stat().st_mtime


def render_thumbnail(src_mesh: Path, out_png: Path) -> None:
    """Write a 256x256 PNG of the mesh in a 3/4 view."""
    mesh = pv.read(str(src_mesh))
    p = pv.Plotter(off_screen=True, window_size=THUMB_SIZE)
    p.set_background(BG)
    p.add_mesh(
        mesh,
        color=MESH_COLOR,
        show_edges=True,
        edge_color=EDGE_COLOR,
        line_width=0.5,
        lighting=True,
        smooth_shading=True,
    )
    # 3/4 isometric-ish view, framed to mesh bounds.
    p.view_isometric()
    p.camera.zoom(1.1)
    p.screenshot(str(out_png))
    p.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Re-render even if the PNG is up to date.")
    args = ap.parse_args(argv)

    built = 0
    skipped = 0
    for name, obj in EXAMPLE_OBJECTS.items():
        src = Path(obj["path"])
        if not src.exists():
            print(f"  [skip] {name}: source mesh missing ({src})")
            continue
        out = src.with_suffix(".png")
        if not _needs_rebuild(src, out, args.force):
            print(f"  [up-to-date] {name} -> {out}")
            skipped += 1
            continue
        render_thumbnail(src, out)
        print(f"  [built] {name} -> {out}  ({out.stat().st_size:,} bytes)")
        built += 1

    print(f"\n{built} built, {skipped} up-to-date")


if __name__ == "__main__":
    main()
