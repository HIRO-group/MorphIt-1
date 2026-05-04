"""
MorphIt HTTP API.

Single FastAPI service. Endpoints:
  GET  /              -> static UI (index.html)
  GET  /healthz       -> liveness/readiness
  POST /api/morph     -> upload a mesh, run MorphIt, return packing JSON

Runs synchronously: a request blocks for the duration of one training run.
Suitable for small meshes / short iteration counts; upgrade to a job queue
if request durations get long enough to hit ingress/proxy timeouts.
"""

import json
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

# /app/web/api/main.py -> /app/src
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
WEB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

# Imports from MorphIt source (src/ on sys.path).
from config import get_config, update_config_from_dict  # noqa: E402
from morphit import MorphIt  # noqa: E402
from training import train_morphit  # noqa: E402

ALLOWED_VARIANTS = ("MorphIt-V", "MorphIt-S", "MorphIt-B")
# trimesh handles these natively without extra system deps.
ALLOWED_EXTENSIONS = (".obj", ".stl", ".ply")

app = FastAPI(title="MorphIt API", version="0.1.0")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html", media_type="text/html")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/morph")
async def morph(
    mesh: UploadFile = File(...),
    variant: str = Form("MorphIt-B"),
    num_spheres: int = Form(20),
    iterations: int = Form(200),
):
    """
    Run MorphIt on the uploaded mesh and return the resulting sphere packing.

    Form fields:
      mesh         file upload (.obj/.stl/.ply)
      variant      one of MorphIt-V | MorphIt-S | MorphIt-B (default: MorphIt-B)
      num_spheres  target sphere count (default: 20)
      iterations   gradient iterations (default: 200)
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

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh_path = tmp / f"input{suffix}"
        mesh_path.write_bytes(await mesh.read())

        out_name = "morphit_result.json"
        config = update_config_from_dict(
            get_config(variant),
            {
                "model.mesh_path": str(mesh_path),
                "model.num_spheres": num_spheres,
                "training.iterations": iterations,
                "training.logging_enabled": False,
                "visualization.enabled": False,
                "visualization.off_screen": True,
                "visualization.save_video": False,
                "results_dir": str(tmp),
                "output_filename": out_name,
            },
        )

        model = MorphIt(config)
        train_morphit(model)
        model.save_results()

        with (tmp / out_name).open() as f:
            payload = json.load(f)

    return JSONResponse(payload)
