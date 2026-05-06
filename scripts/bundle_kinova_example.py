"""
One-off helper: bundle a minimal kinova_description into web/examples/
for use as the robot-mode default example.

Reads the source URDF from the urdf_files_dataset, copies every mesh it
references into web/examples/kinova_description/, and writes a URDF
that references .stl meshes (one format across the whole bundle).

Per-mesh logic:
  1. If a hand-resized `<stem>_resized.STL` exists next to the source
     .dae, prefer it (the user provides decimated/convexified versions
     for the heavy meshes).
  2. Otherwise convert the source .dae to .stl losslessly via trimesh.

The output URDF is identical to the source except the `package://`
mesh references are rewritten to `.stl`.

Re-run any time the source URDF or the *_resized.STL files change:

    venv-morph/bin/python scripts/bundle_kinova_example.py [--urdf PATH]
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import trimesh

DEFAULT_URDF = (
    "/Users/nn/devenv/urdf_files_dataset/urdf_files/oems/xacro_generated/"
    "kinova_robotics/kinova_description/urdf/m1n4s200_standalone.urdf"
)
PACKAGE_NAME = "kinova_description"

OUT_ROOT = (
    Path(__file__).resolve().parents[1] / "web" / "examples" / PACKAGE_NAME
)


def _resolve_mesh_root(urdf_path: Path) -> Path:
    """Locate the meshes/ directory the URDF references."""
    candidates = [
        urdf_path.parent.parent / "meshes",
        urdf_path.parents[1] / "meshes",
        urdf_path.parents[2] / "meshes",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"could not locate meshes/ siblings of {urdf_path}")


def _resized_variant(mesh_root: Path, stem: str) -> Path | None:
    """Return the path to <stem>_resized.STL if it exists, else None.

    Filename is case-sensitive on most filesystems but the dataset's
    files use uppercase ``.STL``; we glob both extensions to be safe.
    """
    for cand in (mesh_root / f"{stem}_resized.STL",
                 mesh_root / f"{stem}_resized.stl"):
        if cand.exists():
            return cand
    return None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", default=DEFAULT_URDF,
                   help="Source URDF path inside the urdf_files_dataset.")
    args = p.parse_args(argv)

    src_urdf = Path(args.urdf)
    if not src_urdf.exists():
        raise SystemExit(f"source URDF not found: {src_urdf}")

    out_urdf = OUT_ROOT / "urdf" / src_urdf.name
    out_meshes = OUT_ROOT / "meshes"
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    out_meshes.mkdir(parents=True, exist_ok=True)

    text = src_urdf.read_text()
    refs = sorted(set(re.findall(
        rf"package://{PACKAGE_NAME}/(meshes/[^\"']+)", text,
    )))
    print(f"Found {len(refs)} unique mesh refs in {src_urdf.name}")

    mesh_root = _resolve_mesh_root(src_urdf)

    # Wipe stale bundled meshes first so a renamed .dae -> .stl doesn't
    # leave both files behind.
    for stale in out_meshes.iterdir():
        if stale.is_file():
            stale.unlink()

    rename_map: dict[str, str] = {}  # source rel -> bundled rel
    total_faces_in = total_faces_out = 0

    for rel in refs:
        rel_path = Path(rel)            # e.g. meshes/hand_2finger.dae
        stem = rel_path.stem            # e.g. hand_2finger
        new_rel = f"meshes/{stem}.stl"  # always .stl in the bundle
        out_path = OUT_ROOT / new_rel

        # Prefer a hand-decimated `<stem>_resized.STL` if the user has
        # placed one alongside the source meshes — that's the path we
        # use for the heavy meshes (hand_2finger, base, etc.).
        resized = _resized_variant(mesh_root, stem)
        if resized is not None:
            shutil.copy2(resized, out_path)
            n_in = -1  # don't bother loading the source
            try:
                n_out = len(trimesh.load(out_path, force="mesh").faces)
            except Exception:
                n_out = -1
            note = f"resized STL: {n_out:,} faces" if n_out >= 0 else "resized STL"
            note = f"{note} (from {resized.name})"
        else:
            # Fall back to source .dae (or whatever extension the URDF
            # refers to). Load with trimesh and re-export as .stl so
            # the bundle is single-format. No decimation.
            src_mesh = mesh_root / rel_path.name
            if not src_mesh.exists():
                matches = list(mesh_root.rglob(rel_path.name))
                if not matches:
                    print(f"  [skip] not found on disk: {rel}")
                    continue
                src_mesh = matches[0]
            m = trimesh.load(src_mesh, force="mesh")
            m.export(out_path)
            n_in = len(m.faces)
            n_out = n_in
            note = f"converted to stl: {n_out:,} faces (from {src_mesh.name})"

        rename_map[rel] = new_rel
        if n_in > 0:
            total_faces_in += n_in
        total_faces_out += max(0, n_out)
        print(f"  {rel:40s} -> {new_rel:30s} {note}  "
              f"({out_path.stat().st_size:>9,} bytes)")

    # Rewrite the URDF: every package://X/meshes/Y.* becomes package://X/meshes/Y.stl
    new_text = text
    for old_rel, new_rel in rename_map.items():
        new_text = new_text.replace(
            f"package://{PACKAGE_NAME}/{old_rel}",
            f"package://{PACKAGE_NAME}/{new_rel}",
        )
    out_urdf.write_text(new_text)
    print(f"\nWrote {out_urdf}")

    bundle_size = sum(p.stat().st_size for p in OUT_ROOT.rglob("*") if p.is_file())
    print(f"Bundle size: {bundle_size:,} bytes")
    if total_faces_in:
        print(f"Face totals (from sources we loaded): "
              f"{total_faces_in:,} -> {total_faces_out:,}")


if __name__ == "__main__":
    main()
