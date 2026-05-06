"""
MorphIt HTTP API.

Single FastAPI service. Endpoints:
  GET  /                     -> static UI (index.html)
  GET  /healthz              -> liveness/readiness
  GET  /api/example/link0    -> serve the bundled link0.obj demo mesh
  POST /api/morph            -> object mode: upload mesh, run MorphIt, return URDF

Robot mode (3-step flow, stateful sessions, 1-hour TTL):
  POST /api/robot/inspect    -> upload folder, return inspection report + session_id
  POST /api/robot/pack-link  -> pack one collision element (called per-link by UI)
  POST /api/robot/assemble   -> stitch packed JSONs into the final spherical URDF

Runs synchronously: each individual request blocks for its training run.
Suitable for small meshes / short iteration counts; upgrade to a job queue
if request durations get long enough to hit ingress/proxy timeouts.
"""

import json as _json
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse

# /app/web/api/main.py -> /app/src      (and /app/src/scripts for URDF gen)
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
SCRIPTS_DIR = SRC_DIR / "scripts"
ROBOT_SCRIPTS_DIR = SCRIPTS_DIR / "robot"
WEB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ROBOT_SCRIPTS_DIR))

# Imports from MorphIt source (src/ on sys.path).
from config import get_config, update_config_from_dict  # noqa: E402
from morphit import MorphIt  # noqa: E402
from training import train_morphit  # noqa: E402

# URDF generation (object mode): reuse the canonical implementation from
# src/scripts/create_object_urdf.py rather than duplicating its logic here.
from create_object_urdf import load_inputs as urdf_load_inputs, write_urdf  # noqa: E402

# Robot pipeline modules — same code path the CLI uses, just driven from
# HTTP handlers instead of argparse.
from discover import (  # noqa: E402
    ACTION_PACK,
    InspectionReport,
    find_urdfs,
    inspect_urdf,
    resolve_mesh_path,
    select_urdf,
)
from pack_robot_meshes import VARIANT_CHOICES as ROBOT_VARIANT_CHOICES, pack_one_link  # noqa: E402
from create_robot_urdf import (  # noqa: E402
    DEFAULT_SPHERE_RGBA,
    hex_to_rgba,
    rewrite_urdf,
)

ALLOWED_VARIANTS = ("MorphIt-V", "MorphIt-S", "MorphIt-B")
# trimesh handles these natively without extra system deps.
ALLOWED_EXTENSIONS = (".obj", ".stl", ".ply")

# Upload caps. These are the line between "reasonable robotics file" and
# "user clearly dragged in the wrong directory". A typical robot
# description package is ~5-50 MB; even Spot-class with all visuals lands
# under 100 MB. We err generous on the file totals so legitimate URDFs
# don't get cut off.
MB = 1024 * 1024
MAX_OBJECT_MESH_BYTES = 100 * MB           # single-mesh upload (object mode)
MAX_ROBOT_FOLDER_BYTES = 200 * MB          # total bytes across all files in robot folder
MAX_ROBOT_PER_FILE_BYTES = 50 * MB         # any single file inside the folder


def _too_large(actual_bytes: int, limit_bytes: int, kind: str) -> HTTPException:
    """Build a 413 with an actionable message.

    The hint points the user at the two standard ways to shrink heavy
    collision geometry: face-count reduction (decimation) and convex
    decomposition. Both are appropriate for collision use because
    spheres approximate the *shape*, not the surface microstructure.
    """
    return HTTPException(
        status_code=413,
        detail=(
            f"{kind} too large: {actual_bytes / MB:.1f} MB exceeds the "
            f"{limit_bytes / MB:.0f} MB limit. Reduce the file size of "
            "the heaviest mesh by decimating it (e.g. trimesh's "
            "simplify_quadric_decimation, MeshLab's Quadric Edge "
            "Collapse, or Blender's Decimate modifier) or by replacing "
            "it with a convex decomposition (V-HACD, CoACD)."
        ),
    )

# Map of UI-flat advanced keys -> dotted config paths used by
# update_config_from_dict. Anything outside this allow-list is ignored
# silently — keeps the API resilient to UI experiments without letting
# arbitrary config keys be pushed in.
ADVANCED_OVERRIDE_MAP: Dict[str, str] = {
    "num_inside_samples":              "model.num_inside_samples",
    "num_surface_samples":             "model.num_surface_samples",
    "density_control_enabled":         "training.density_control_enabled",
    "density_control_min_interval":    "training.density_control_min_interval",
    "density_control_cooling_factor":  "training.density_control_cooling_factor",
    "coverage_weight":                 "training.coverage_weight",
    "overlap_weight":                  "training.overlap_weight",
    "boundary_weight":                 "training.boundary_weight",
    "surface_weight":                  "training.surface_weight",
    "containment_weight":              "training.containment_weight",
    "sqem_weight":                     "training.sqem_weight",
    "hausdorff_weight":                "training.hausdorff_weight",
    "mesh_containment_weight":         "training.mesh_containment_weight",
}


def _parse_advanced(advanced_json: str) -> Dict[str, Any]:
    """Decode the `advanced` JSON form-field into a dotted-path overrides dict.

    Quietly drops keys not in the allow-list and skips Nones; raises a
    400 if the payload is malformed.
    """
    try:
        raw = _json.loads(advanced_json or "{}") if advanced_json else {}
    except _json.JSONDecodeError as exc:
        raise HTTPException(400, f"advanced must be valid JSON: {exc}")
    if not isinstance(raw, dict):
        raise HTTPException(400, "advanced must be a JSON object")
    return {
        ADVANCED_OVERRIDE_MAP[k]: v
        for k, v in raw.items()
        if k in ADVANCED_OVERRIDE_MAP and v is not None
    }


def _safe_color_rgba(hex_color: Optional[str]) -> tuple:
    """Hex string -> RGBA, falling back to the default sphere blue."""
    if not hex_color:
        return DEFAULT_SPHERE_RGBA
    try:
        return hex_to_rgba(hex_color)
    except ValueError:
        return DEFAULT_SPHERE_RGBA

# Defaults for create_object_urdf.write_urdf — chosen to match the script's
# CONFIG dict so server output matches what the user would get running the
# script locally with default settings.
URDF_DEFAULTS = {
    "format": "urdf",
    "anchored": False,
    "default_color_rgba": (0.2, 0.6, 1.0, 1.0),
    "decimals": 6,
    "total_mass": 1.0,
    "use_centroid": True,
    "rotation_center_xyz": (0.0, 0.0, 0.0),
    "global_offset_xyz": (0.0, 0.0, 0.0),
    "base_mass": 0.001,
    "base_inertia_diag": 1e-5,
    "min_radius": 1e-6,
}

app = FastAPI(title="MorphIt API", version="0.1.0")


@app.get("/")
def index():
    # no-store so edits to index.html show up on a normal refresh during
    # local dev. The page is tiny, so no cdn/caching benefit is lost in prod.
    return FileResponse(
        WEB_DIR / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}


# Demo mesh shipped with the image so first-time visitors can run MorphIt
# without uploading their own file.
EXAMPLE_MESH = WEB_DIR / "examples" / "link0.obj"


@app.get("/api/example/link0")
def example_link0():
    return FileResponse(
        EXAMPLE_MESH,
        media_type="model/obj",
        filename="link0.obj",
    )


@app.post("/api/morph")
async def morph(
    mesh: UploadFile = File(...),
    variant: str = Form("MorphIt-B"),
    num_spheres: int = Form(20),
    iterations: int = Form(200),
    seed: Optional[int] = Form(None),
    advanced: str = Form("{}"),
    base_color: str = Form("#3399ff"),
):
    """
    Object mode: run MorphIt on an uploaded mesh and return a URDF describing
    the spherical approximation.

    Form fields:
      mesh         file upload (.obj/.stl/.ply)
      variant      one of MorphIt-V | MorphIt-S | MorphIt-B (default: MorphIt-B)
      num_spheres  target sphere count (default: 20)
      iterations   gradient iterations (default: 200)
      seed         optional random seed (omit for non-deterministic)
      advanced     JSON object of advanced overrides; see ADVANCED_OVERRIDE_MAP
      base_color   "#rrggbb" sphere color (default soft blue)
    """
    if variant not in ALLOWED_VARIANTS:
        raise HTTPException(400, f"variant must be one of {ALLOWED_VARIANTS}")
    if not 1 <= num_spheres <= 200:
        raise HTTPException(400, "num_spheres must be in [1, 200]")
    if not 1 <= iterations <= 1000:
        raise HTTPException(400, "iterations must be in [1, 1000]")

    suffix = Path(mesh.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"mesh extension must be one of {ALLOWED_EXTENSIONS}")

    # Robot name = mesh filename stem; falls back to "object" for blank/odd inputs.
    robot_name = Path(mesh.filename or "object").stem or "object"

    advanced_overrides = _parse_advanced(advanced)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh_path = tmp / f"input{suffix}"
        data = await mesh.read()
        if len(data) > MAX_OBJECT_MESH_BYTES:
            raise _too_large(len(data), MAX_OBJECT_MESH_BYTES, "Mesh file")
        mesh_path.write_bytes(data)

        out_name = "morphit_result.json"
        updates = {
            "model.mesh_path": str(mesh_path),
            "model.num_spheres": num_spheres,
            "training.iterations": iterations,
            "training.logging_enabled": False,
            "visualization.enabled": False,
            "visualization.off_screen": True,
            "visualization.save_video": False,
            "results_dir": str(tmp),
            "output_filename": out_name,
            **advanced_overrides,
        }
        if seed is not None:
            updates["random_seed"] = seed
        config = update_config_from_dict(get_config(variant), updates)

        model = MorphIt(config)
        train_morphit(model)
        model.save_results()

        # Hand the centers/radii JSON to the URDF generator. load_inputs
        # reads from disk, computes the centroid-relative positions, and
        # distributes total_mass across spheres by volume.
        urdf_cfg = {
            **URDF_DEFAULTS,
            "input_json": str(tmp / out_name),
            "robot_name": robot_name,
            "default_color_rgba": _safe_color_rgba(base_color),
        }
        _centers, radii, rel_centers, masses, world_origin = urdf_load_inputs(urdf_cfg)
        urdf_text = write_urdf(urdf_cfg, rel_centers, radii, masses, world_origin)

    # Sphere centers in the URDF are emitted relative to `world_origin`
    # (the centroid). The web UI overlays the *original* mesh as a
    # wireframe; to place it in the same frame as the URDF spheres it
    # needs to translate the mesh by -world_origin. That offset isn't
    # otherwise recoverable from the URDF, so we pass it via header.
    cx, cy, cz = world_origin
    return Response(
        content=urdf_text,
        media_type="application/xml",
        headers={"X-Morphit-Centroid": f"{cx:.6f},{cy:.6f},{cz:.6f}"},
    )


# =====================================================================
# Robot mode (3-step stateful pipeline)
# =====================================================================

# In-memory session store. Keyed by uuid string. Single-process / single-
# pod deploy, so this is fine; if we ever shard the API across replicas
# this needs to become Redis-backed.
ROBOT_SESSIONS_ROOT = Path(tempfile.gettempdir()) / "morphit-robot-sessions"
ROBOT_SESSION_TTL_SECONDS = 60 * 60  # 1 hour


@dataclass
class RobotSession:
    """One user's robot-mode session state.

    `work_dir` holds the unzipped folder (the user's upload). `output_dir`
    holds stage-2 sphere JSONs and the stage-3 final URDF. Both live
    under ROBOT_SESSIONS_ROOT/<id>/ and are recursively deleted at TTL.
    """

    session_id: str
    work_dir: Path
    output_dir: Path
    report: Optional[InspectionReport] = None
    last_access: float = field(default_factory=time.time)


_robot_sessions: Dict[str, RobotSession] = {}


def _gc_robot_sessions() -> None:
    """Drop expired sessions and their tempdirs.

    Cheap O(N) sweep run at the start of every robot endpoint; not a
    background task because uvicorn workers don't share state and we'd
    rather not invent one for a low-traffic research demo.
    """
    now = time.time()
    expired = [
        sid for sid, s in _robot_sessions.items()
        if now - s.last_access > ROBOT_SESSION_TTL_SECONDS
    ]
    for sid in expired:
        s = _robot_sessions.pop(sid)
        shutil.rmtree(s.work_dir, ignore_errors=True)
        shutil.rmtree(s.output_dir, ignore_errors=True)


def _new_robot_session() -> RobotSession:
    sid = uuid.uuid4().hex
    base = ROBOT_SESSIONS_ROOT / sid
    work_dir = base / "robot"
    output_dir = base / "out"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    s = RobotSession(session_id=sid, work_dir=work_dir, output_dir=output_dir)
    _robot_sessions[sid] = s
    return s


def _get_robot_session(session_id: str) -> RobotSession:
    if session_id not in _robot_sessions:
        raise HTTPException(404, f"session {session_id!r} not found or expired")
    s = _robot_sessions[session_id]
    s.last_access = time.time()
    return s


def _safe_join(base: Path, rel: str) -> Path:
    """Join a relative upload path to `base` without escaping it.

    Defends against path-traversal in the user-supplied filename. Also
    rejects absolute paths and Windows drive letters.
    """
    rel = rel.replace("\\", "/").lstrip("/")
    if not rel:
        raise HTTPException(400, "Empty upload filename.")
    p = Path(rel)
    if p.is_absolute() or any(part in ("..", "") for part in p.parts):
        raise HTTPException(400, f"Bad upload path: {rel!r}")
    target = (base / p).resolve()
    if base.resolve() not in target.parents and target != base.resolve():
        raise HTTPException(400, f"Upload path escapes work dir: {rel!r}")
    return target


EXAMPLE_KINOVA_DIR = WEB_DIR / "examples" / "kinova_description"
EXAMPLE_KINOVA_URDF = "m1n4s200_standalone.urdf"


@app.post("/api/robot/example/kinova")
def robot_example_kinova():
    """Bootstrap a robot-mode session from the bundled kinova_description.

    Copies web/examples/kinova_description/ into a fresh session work_dir
    so the rest of the robot pipeline (inspect / pack-link / assemble /
    file) operates on it identically to a user-uploaded folder. Returns
    the same shape `/api/robot/inspect` does, so the UI can drop-in
    replace its inspect call with this for the "use example" path.
    """
    if not EXAMPLE_KINOVA_DIR.exists():
        raise HTTPException(
            500,
            "kinova example not bundled; run scripts/bundle_kinova_example.py",
        )

    _gc_robot_sessions()
    session = _new_robot_session()

    # Mirror the upload pattern: drop everything under <work_dir>/<pkg>/...
    # so package://kinova_description/Y resolves naturally.
    pkg_root = session.work_dir / "kinova_description"
    shutil.copytree(EXAMPLE_KINOVA_DIR, pkg_root)

    try:
        urdfs = find_urdfs(session.work_dir)
        urdf_path = select_urdf(urdfs, EXAMPLE_KINOVA_URDF)
    except ValueError as exc:
        raise HTTPException(500, f"kinova example malformed: {exc}")

    report = inspect_urdf(session.work_dir, urdf_path)
    session.report = report

    return {"session_id": session.session_id, **report.to_dict()}


@app.get("/api/robot/file")
def robot_file(session_id: str, path: str):
    """Serve a file from the session's uploaded folder.

    Used by the side-by-side "original collision" viewer to fetch the
    URDF text and the meshes it references. Path can be either:

      * a package URI (``package://X/path``) — resolved via the same
        helper the inspection step uses, so the viewer sees the same
        files inspect did;
      * a relative path under the session work_dir — joined safely so
        the client can't escape the session sandbox.

    Always read-only; the endpoint never writes.
    """
    _gc_robot_sessions()
    session = _get_robot_session(session_id)

    work_root = session.work_dir.resolve()

    if path.startswith("package://"):
        urdf_path = Path(session.report.urdf_path) if session.report else session.work_dir
        resolved = resolve_mesh_path(path, session.work_dir, urdf_path)
        if resolved is None:
            raise HTTPException(404, f"could not resolve {path!r}")
    else:
        p = Path(path)
        if p.is_absolute():
            # Server-absolute paths (e.g. report.urdf_path) — accept but
            # sandbox-check below.
            resolved = p
        else:
            resolved = _safe_join(session.work_dir, path)

    if not resolved.exists():
        raise HTTPException(404, f"file not found: {path!r}")

    # Defensive: every served file must live under the session's work_dir.
    try:
        resolved.resolve().relative_to(work_root)
    except ValueError:
        raise HTTPException(403, "path escapes session sandbox")

    return FileResponse(resolved)


@app.post("/api/robot/inspect")
async def robot_inspect(
    files: List[UploadFile] = File(...),
    urdf: Optional[str] = Form(None),
):
    """Step 1: receive folder upload, extract, return inspection plan.

    Files are sent multipart with their relative paths preserved as the
    `filename` (UI uses `formData.append('files', f, f.webkitRelativePath)`).
    We reconstruct the directory tree under the session's work_dir and
    delegate to ``inspect_urdf`` for the actual analysis.
    """
    _gc_robot_sessions()
    session = _new_robot_session()

    total_bytes = 0
    for f in files:
        rel = f.filename or ""
        target = _safe_join(session.work_dir, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = await f.read()

        # Per-file cap: catches a single oversized mesh (e.g. user
        # accidentally dragged a 200 MB visual asset).
        if len(data) > MAX_ROBOT_PER_FILE_BYTES:
            shutil.rmtree(session.work_dir.parent, ignore_errors=True)
            _robot_sessions.pop(session.session_id, None)
            raise _too_large(
                len(data), MAX_ROBOT_PER_FILE_BYTES,
                f"File {rel!r}",
            )

        # Total-folder cap: catches "user dropped ~/Downloads or the
        # whole repo" — sized to fit even Spot-class URDFs comfortably.
        total_bytes += len(data)
        if total_bytes > MAX_ROBOT_FOLDER_BYTES:
            shutil.rmtree(session.work_dir.parent, ignore_errors=True)
            _robot_sessions.pop(session.session_id, None)
            raise _too_large(
                total_bytes, MAX_ROBOT_FOLDER_BYTES, "Folder",
            )

        target.write_bytes(data)

    try:
        urdfs = find_urdfs(session.work_dir)
        urdf_path = select_urdf(urdfs, urdf)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    report = inspect_urdf(session.work_dir, urdf_path)
    session.report = report

    return {"session_id": session.session_id, **report.to_dict()}


@app.post("/api/robot/pack-link")
async def robot_pack_link(
    session_id: str = Form(...),
    link_name: str = Form(...),
    collision_index: int = Form(...),
    variant: str = Form("MorphIt-B"),
    num_spheres: int = Form(20),
    iterations: int = Form(200),
    seed: Optional[int] = Form(None),
    advanced: str = Form("{}"),
):
    """Step 2: pack a single collision element. Called once per PACK item.

    The UI loops over `report.collisions` (filtered to action="pack"),
    calling this endpoint sequentially so a progress bar can advance one
    link at a time without holding a single multi-minute HTTP connection.
    """
    _gc_robot_sessions()
    session = _get_robot_session(session_id)
    if session.report is None:
        raise HTTPException(400, "session has no inspection report; call /inspect first")

    if variant not in ROBOT_VARIANT_CHOICES:
        raise HTTPException(400, f"variant must be one of {ROBOT_VARIANT_CHOICES}")
    if not 1 <= num_spheres <= 200:
        raise HTTPException(400, "num_spheres must be in [1, 200]")
    if not 1 <= iterations <= 1000:
        raise HTTPException(400, "iterations must be in [1, 1000]")

    item = next(
        (c for c in session.report.collisions
         if c.link_name == link_name and c.collision_index == collision_index),
        None,
    )
    if item is None:
        raise HTTPException(
            404, f"no collision item: {link_name}[{collision_index}]"
        )
    if item.action != ACTION_PACK:
        raise HTTPException(
            400,
            f"item {link_name}[{collision_index}] has action {item.action!r}, "
            f"not {ACTION_PACK!r}",
        )

    advanced_overrides = _parse_advanced(advanced)

    result = pack_one_link(
        item,
        variant=variant,
        num_spheres=num_spheres,
        iterations=iterations,
        output_dir=session.output_dir,
        seed=seed,
        config_overrides=advanced_overrides or None,
    )
    return result.to_dict()


@app.post("/api/robot/assemble")
async def robot_assemble(
    session_id: str = Form(...),
    base_color: str = Form("#3399ff"),
    color_variation: float = Form(0.3),
):
    """Step 3: stitch packed JSONs into the final spherical URDF.

    Returns the URDF as application/xml so the frontend's existing
    URDF viewer / copy / download flow works without modification.
    """
    _gc_robot_sessions()
    session = _get_robot_session(session_id)
    if session.report is None:
        raise HTTPException(400, "session has no inspection report")

    if not 0.0 <= color_variation <= 1.0:
        raise HTTPException(400, "color_variation must be in [0, 1]")

    urdf_stem = Path(session.report.urdf_path).stem or "robot"
    output_urdf = session.output_dir / f"{urdf_stem}_spherical.urdf"
    spheres_dir = session.output_dir / "spheres"

    stats = rewrite_urdf(
        session.report,
        spheres_dir=spheres_dir,
        output_path=output_urdf,
        base_color_rgba=_safe_color_rgba(base_color),
        color_variation=float(color_variation),
    )

    if stats.skipped_pack_items:
        # The UI should have packed every PACK item before calling this;
        # if it didn't, surface which links are missing so the UI can
        # retry just those.
        missing = [f"{l}[{i}]: {r}" for l, i, r in stats.skipped_pack_items]
        raise HTTPException(
            400,
            "Some collisions weren't packed: " + "; ".join(missing),
        )

    urdf_text = output_urdf.read_text()
    return Response(content=urdf_text, media_type="application/xml")
