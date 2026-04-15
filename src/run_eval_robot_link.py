from config import get_config, update_config_from_dict
from morphit import MorphIt
from training import train_morphit

import json
import numpy as np
from pathlib import Path

# ── Parameters ────────────────────────────────────────────────────────────────

MESH_PATH = "../mesh_models/fr3/collision/link0.obj"
MORPHIT_VARIANT = "MorphIt-B"  # "MorphIt-B", "MorphIt-S", or "MorphIt-V"
OUTPUT_DIR = Path("output_spheres_morphit")
N_MIN = 1
N_MAX = 100

TRAINING_CONFIG = {
    "training.iterations": 800,
    "training.verbose_frequency": 50,
    "training.logging_enabled": False,
    "training.density_control_min_interval": 250,
    "visualization.enabled": False,
    "visualization.save_video": False,
}

# ── Helpers ───────────────────────────────────────────────────────────────────


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_spheres_to_json(centers, radii, output_file, mesh_path=None):
    """Write sphere centers and radii to a JSON file."""
    data = {
        "centers": [[float(v) for v in c] for c in centers],
        "radii": [float(r) for r in radii],
    }
    if mesh_path:
        data["mesh_path"] = str(mesh_path)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=1, cls=NumpyEncoder)


def run_morphit(n_spheres):
    """Run MorphIt for a given sphere count and return (centers, radii)."""
    config = get_config(MORPHIT_VARIANT)
    updates = {
        "model.num_spheres": n_spheres,
        "model.mesh_path": MESH_PATH,
        **TRAINING_CONFIG,
    }
    config = update_config_from_dict(config, updates)

    model = MorphIt(config)
    model.pv_init(enabled=False, off_screen=True, save_video=False, filename="")
    train_morphit(model)

    # Extract sphere parameters from trained model
    # model.centers and model.radii are nn.Parameter properties (see morphit.py)
    centers = model.centers.detach().cpu().numpy()
    radii = model.radii.detach().cpu().numpy()
    return centers, radii


# ── Main ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)

for n in range(N_MIN, N_MAX + 1):
    print(
        f"[{n:>3}/{N_MAX}] Running {MORPHIT_VARIANT} with {n} sphere(s)...",
        end=" ",
        flush=True,
    )

    centers, radii = run_morphit(n)

    output_file = OUTPUT_DIR / f"link0_ns{n}.json"
    write_spheres_to_json(centers, radii, output_file, mesh_path=MESH_PATH)

    print(f"saved {len(radii)} sphere(s) → {output_file}")

print(f"\nDone. {N_MAX - N_MIN + 1} files written to '{OUTPUT_DIR}/'.")
