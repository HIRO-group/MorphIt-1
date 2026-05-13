"""
Cross-library evaluation harness for MorphIt-1.

Runs the four canonical variants (V, S, B, Obj) on a small mesh grid and
emits both a human-readable metrics table and a structured metrics.json
that compare_libraries.py can diff against the matching run from
morphit-plus-plus.

Intended use:

    cd src
    python scripts/debug_quick_eval.py

Then in plus-plus, run optimization/scripts/debug_quick_eval.py, then
python scripts/compare_libraries.py to diff the two metrics.json files.

The seed is fixed inside this script only — config defaults stay
random_seed=None so general MorphIt runs remain stochastic.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

# Path setup — this file lives in src/scripts/
SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from config import get_config, update_config_from_dict  # noqa: E402
from morphit import MorphIt  # noqa: E402
from training import train_morphit  # noqa: E402


# ─── Grid ────────────────────────────────────────────────────────────
# (mesh_label, mesh_path_relative_to_mesh_models). link0 covers Panda
# kinematics, bunny covers small organic objects, vase covers large
# axis-asymmetric COM-offset objects (the inertia / COM stress case).
MESHES = (
    ("link0", "fr3/collision/link0.obj"),
    ("bunny", "objects/bunny.obj"),
    ("vase", "objects/vase.obj"),
)
N_SPHERES = (64,)
VARIANTS = ("morphit-v", "morphit-s", "morphit-b", "morphit-obj")

# ─── Training knobs ──────────────────────────────────────────────────
SEED = 0
ITERATIONS = 300
DENSITY = 1000.0
DENSITY_CONTROL = True
PER_SPHERE_MASS = False  # flip True to exercise the learnable-mass path

# Base params shared across all variants.
COMMON_PARAMS = {
    "model.num_inside_samples": 5000,
    "model.num_surface_samples": 5000,
    "model.density": DENSITY,
    "training.verbose_frequency": 500,
    "training.density_control_enabled": DENSITY_CONTROL,
    "training.density_control_min_interval": 160,
    # flatness_loss has a Python for-loop that doubles per-iter cost;
    # leave off for the geometry comparison. mass/com/inertia stay at
    # the config-default values per variant (zero for V/S/B, weighted
    # for Obj).
    "training.flatness_weight": 0.0,
    "training.logging_enabled": False,
    "training.early_stopping": False,
    "visualization.enabled": False,
    # Optimizer lrs MUST be pinned explicitly — the two libraries'
    # config defaults disagree (m1: center=0.001 radius=0.002;
    # pp: center=0.0002 radius=0.0001), so any cross-library run
    # that relies on defaults silently produces 5× / 20× different
    # Adam steps and diverges from iter 1. radius_lr is in raw
    # (pre-softplus) space; effective step is sigmoid(raw_r) * lr.
    "training.center_lr": 0.001,
    "training.radius_lr": 0.1,
}

# Per-variant overrides on top of the config.py defaults. Weights for
# V/S/B match plus-plus's debug_loop.py (paper-tuned 6-loss regime).
# Obj inherits its weights from config.py — no override needed.
VARIANT_OVERRIDES = {
    "morphit-v": {
        "base_config": "MorphIt-V",
        "weights": {
            "coverage_weight": 1000.0, "overlap_weight": 0.05,
            "boundary_weight": 1.0, "surface_weight": 7.0,
            "containment_weight": 1.0, "sqem_weight": 10.0,
            "hausdorff_weight": 0.0,
            "mass_weight": 0.0, "com_weight": 0.0, "inertia_weight": 0.0,
        },
    },
    "morphit-s": {
        "base_config": "MorphIt-S",
        "weights": {
            "coverage_weight": 0.01, "overlap_weight": 0.01,
            "boundary_weight": 5000.0, "surface_weight": 100.0,
            "containment_weight": 1.0, "sqem_weight": 1000.0,
            "hausdorff_weight": 0.0,
            "mass_weight": 0.0, "com_weight": 0.0, "inertia_weight": 0.0,
        },
    },
    "morphit-b": {
        "base_config": "MorphIt-B",
        "weights": {
            "coverage_weight": 1500.0, "overlap_weight": 0.3,
            "boundary_weight": 1.0, "surface_weight": 50.0,
            "containment_weight": 1.0, "sqem_weight": 3000.0,
            "hausdorff_weight": 0.0,
            "mass_weight": 0.0, "com_weight": 0.0, "inertia_weight": 0.0,
        },
    },
    "morphit-obj": {
        "base_config": "MorphIt-Obj-mass" if PER_SPHERE_MASS else "MorphIt-Obj",
        "weights": {},  # use config-default Obj weights verbatim
    },
}

MESH_ROOT = SRC.parent / "mesh_models"
DEBUG_RESULTS = SRC / "results" / "debug_quick_eval"
METRICS_JSON = DEBUG_RESULTS / "metrics.json"

# ─── Eval config ─────────────────────────────────────────────────────
NUM_SURFACE = 50_000
NUM_VOLUME = 50_000
BOUNDS_EXPAND = 1.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Training ────────────────────────────────────────────────────────

def _sync_clock() -> float:
    """Wall time after CUDA sync — apples-to-apples timings across runs."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def train_one(variant_name: str, mesh_label: str, mesh_path: Path,
              n_spheres: int, out_dir: Path):
    """Train one (variant, mesh, n) and dump result JSON. Returns
    (init_time_s, optimize_time_s) or None if a cached result exists."""
    out_file = out_dir / f"{mesh_label}.json"
    if out_file.exists():
        return None

    variant_cfg = VARIANT_OVERRIDES[variant_name]
    config = get_config(variant_cfg["base_config"])
    updates = {
        "model.num_spheres": n_spheres,
        "model.mesh_path": str(mesh_path),
        "model.per_sphere_mass": (
            PER_SPHERE_MASS and variant_name == "morphit-obj"
        ),
        "training.iterations": ITERATIONS,
        **{f"training.{k}": v for k, v in variant_cfg["weights"].items()},
        **COMMON_PARAMS,
    }
    config = update_config_from_dict(config, updates)
    config.random_seed = SEED
    config.results_dir = str(out_dir)
    config.output_filename = f"{mesh_label}.json"

    t0 = _sync_clock()
    model = MorphIt(config)
    init_time_s = _sync_clock() - t0

    t1 = _sync_clock()
    train_morphit(model)
    optimize_time_s = _sync_clock() - t1

    model.save_results()
    return init_time_s, optimize_time_s


# ─── Evaluation ──────────────────────────────────────────────────────

def load_mesh_artifacts(mesh_path: Path):
    """Sample held-out surface + volume points + physics ground truth.
    RNG is fixed (seed=0) so re-running gives identical eval points
    independent of any training-time stochasticity."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    lo, hi = mesh.bounds
    c = 0.5 * (lo + hi)
    he = 0.5 * (hi - lo) * BOUNDS_EXPAND
    bounds = np.stack([c - he, c + he])
    sample_volume = float(np.prod(bounds[1] - bounds[0]))

    surf_rng = np.random.default_rng(0)
    # trimesh.sample.sample_surface accepts seed kwarg in modern versions;
    # fall back to np.random.seed for the call if the kwarg is missing.
    try:
        surf, _ = trimesh.sample.sample_surface(
            mesh, NUM_SURFACE, seed=0)
    except TypeError:
        np.random.seed(0)
        surf, _ = trimesh.sample.sample_surface(mesh, NUM_SURFACE)
    surf = np.asarray(surf, dtype=np.float64)

    vol = surf_rng.uniform(bounds[0], bounds[1], size=(NUM_VOLUME, 3))
    inside = np.zeros(len(vol), dtype=bool)
    for s in range(0, len(vol), 5000):
        e = min(s + 5000, len(vol))
        inside[s:e] = mesh.contains(vol[s:e])

    return {
        "mesh": mesh,
        "mesh_volume": float(mesh.volume),
        "mesh_scale": float(mesh.scale),
        "sample_volume": sample_volume,
        "surface": surf,
        "volume": vol,
        "is_inside_mesh": inside,
        "mesh_mass": float(mesh.volume) * DENSITY,
        "mesh_com": np.asarray(mesh.center_mass, dtype=np.float64),
        "mesh_inertia": np.asarray(
            mesh.moment_inertia, dtype=np.float64) * DENSITY,
    }


def load_run(results_root: Path, variant: str, n: int, mesh_label: str):
    p = results_root / variant / str(n) / f"{mesh_label}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    entry = {
        "centers": np.asarray(d["centers"], dtype=np.float64),
        "radii": np.asarray(d["radii"], dtype=np.float64),
        "actual_n": len(d["radii"]),
    }
    if "masses" in d:
        entry["masses"] = np.asarray(d["masses"], dtype=np.float64)
    return entry


def surface_distance_metrics(surf_pts, centers, radii):
    tc = torch.as_tensor(centers, dtype=torch.float32, device=DEVICE)
    tr = torch.as_tensor(radii, dtype=torch.float32, device=DEVICE)
    n = len(surf_pts)
    out = np.empty(n, dtype=np.float64)
    bs = 50_000
    for s in range(0, n, bs):
        e = min(s + bs, n)
        pts = torch.as_tensor(
            surf_pts[s:e], dtype=torch.float32, device=DEVICE)
        signed = torch.linalg.norm(
            pts.unsqueeze(1) - tc.unsqueeze(0), dim=2) - tr.unsqueeze(0)
        nearest = signed.min(dim=1)[0]
        out[s:e] = nearest.abs().cpu().numpy()
    return float(out.max()), float(out.mean())


def volume_overlap_metrics(art, centers, radii):
    tc = torch.as_tensor(centers, dtype=torch.float32, device=DEVICE)
    tr2 = torch.as_tensor(radii ** 2, dtype=torch.float32, device=DEVICE)
    n = len(art["volume"])
    inside_sph = np.zeros(n, dtype=bool)
    bs = 50_000
    for s in range(0, n, bs):
        e = min(s + bs, n)
        pts = torch.as_tensor(
            art["volume"][s:e], dtype=torch.float32, device=DEVICE)
        d2 = ((pts.unsqueeze(1) - tc.unsqueeze(0)) ** 2).sum(dim=2)
        inside_sph[s:e] = (d2 <= tr2.unsqueeze(0)).any(dim=1).cpu().numpy()
    sphere_vol = inside_sph.sum() / n * art["sample_volume"]
    both_vol = (inside_sph & art["is_inside_mesh"]).sum() / \
        n * art["sample_volume"]
    out_vol = sphere_vol - both_vol
    mv = art["mesh_volume"] if art["mesh_volume"] > 0 else 1e-12
    return both_vol / mv, out_vol / mv, sphere_vol / mv


def escape_metrics(art, centers, radii):
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    inside = art["mesh"].contains(centers)
    n_out = int((~inside).sum())
    min_radius = 0.001 * art["mesh_scale"]
    n_tiny = int((radii < min_radius).sum())
    return n_out, n_tiny


def physics_metrics(art, centers, radii, masses=None):
    """Frame-correct (sphere COM) relative mass / COM / inertia errors.
    Same formulation as plus-plus's debug_loop_physics.py:physics_metrics
    and matches the normalized losses in losses.py.

    If ``masses`` is provided (per_sphere_mass run), uses it directly;
    otherwise derives masses from DENSITY × (4/3)π·r³.
    """
    if masses is None:
        masses = DENSITY * (4.0 / 3.0) * np.pi * radii ** 3
    total_mass = float(masses.sum())
    sphere_com = (masses[:, None] * centers).sum(axis=0) / total_mass

    centered = centers - sphere_com
    eye3 = np.eye(3)
    I = ((2.0 / 5.0) * masses * radii ** 2).sum() * eye3
    for m_i, c_i in zip(masses, centered):
        I += m_i * (np.dot(c_i, c_i) * eye3 - np.outer(c_i, c_i))

    mass_abs = abs(total_mass - art["mesh_mass"])
    com_abs = float(np.linalg.norm(sphere_com - art["mesh_com"]))
    I_abs = float(np.linalg.norm(I - art["mesh_inertia"], ord="fro"))
    return {
        "mass_abs": mass_abs,
        "mass_rel": mass_abs / max(art["mesh_mass"], 1e-20),
        "com_abs": com_abs,
        "com_rel": com_abs / max(art["mesh_scale"], 1e-20),
        "I_abs": I_abs,
        "I_rel": I_abs / max(
            np.linalg.norm(art["mesh_inertia"], ord="fro"), 1e-20),
    }


def eval_run(results_root, variant, n, mesh_label, art):
    entry = load_run(results_root, variant, n, mesh_label)
    if entry is None:
        return None
    max_d, avg_d = surface_distance_metrics(
        art["surface"], entry["centers"], entry["radii"])
    r_in, r_out, r_uni = volume_overlap_metrics(
        art, entry["centers"], entry["radii"])
    n_out, n_tiny = escape_metrics(art, entry["centers"], entry["radii"])
    phys = physics_metrics(
        art, entry["centers"], entry["radii"], entry.get("masses"))
    return {
        "actual_n": entry["actual_n"],
        "n_out": n_out, "n_tiny": n_tiny,
        "r_in": r_in, "r_out": r_out, "r_uni": r_uni,
        "d_avg_mm": avg_d * 1000, "d_max_mm": max_d * 1000,
        **phys,
    }


# ─── Reporting ───────────────────────────────────────────────────────

HEADER = (f"{'n':>4} {'mesh':>6} {'variant':<12}"
          f" {'actual':>6} {'n_out':>5} {'n_tiny':>6}"
          f" {'r_in':>6} {'r_out':>6} {'r_uni':>6}"
          f" {'d_avg_mm':>9} {'d_max_mm':>9}"
          f" {'mass_rel':>9} {'com_rel':>9} {'I_rel':>9}")


def _fmt_row(n, mesh_label, variant, r):
    return (f"{n:>4} {mesh_label:>6} {variant:<12}"
            f" {r['actual_n']:>6} {r['n_out']:>5} {r['n_tiny']:>6}"
            f" {r['r_in']:>6.3f} {r['r_out']:>6.3f} {r['r_uni']:>6.3f}"
            f" {r['d_avg_mm']:>9.3f} {r['d_max_mm']:>9.3f}"
            f" {r['mass_rel']:>9.3f} {r['com_rel']:>9.3f} {r['I_rel']:>9.3f}")


def print_table(rows, mesh_labels):
    print()
    print(HEADER)
    print("-" * len(HEADER))
    for n in N_SPHERES:
        for mesh_label in mesh_labels:
            for variant in VARIANTS:
                r = next((r for r in rows if r["n"] == n
                          and r["mesh"] == mesh_label
                          and r["variant"] == variant), None)
                if r is None:
                    print(f"{n:>4} {mesh_label:>6} {variant:<12}  MISSING")
                    continue
                print(_fmt_row(n, mesh_label, variant, r))
            print()


# ─── Main ────────────────────────────────────────────────────────────

def _library_label():
    return "morphit-1"


def _git_sha():
    try:
        import subprocess
        sha = subprocess.check_output(
            ["git", "-C", str(SRC.parent), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha or None
    except Exception:
        return None


def main():
    if DEBUG_RESULTS.exists():
        shutil.rmtree(DEBUG_RESULTS)
    DEBUG_RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"[debug] library = {_library_label()}  seed = {SEED}")
    print(f"[debug] cleaned {DEBUG_RESULTS}")
    print(f"[debug] iterations={ITERATIONS}  density_control={DENSITY_CONTROL}"
          f"  per_sphere_mass={PER_SPHERE_MASS}")

    train_start = time.time()
    timings: list[dict] = []

    for variant in VARIANTS:
        for n in N_SPHERES:
            for mesh_label, mesh_rel in MESHES:
                out_dir = DEBUG_RESULTS / variant / str(n)
                out_dir.mkdir(parents=True, exist_ok=True)
                mesh_path = MESH_ROOT / mesh_rel
                t0 = time.time()
                result = train_one(
                    variant, mesh_label, mesh_path, n, out_dir)
                wall = time.time() - t0
                if result is None:
                    print(f"  {variant:12} n={n:>3} {mesh_label:6} — "
                          f"cached ({wall:.1f}s)")
                    continue
                init_s, opt_s = result
                iter_ms = 1000.0 * opt_s / max(1, ITERATIONS)
                timings.append({
                    "variant": variant, "n": n, "mesh": mesh_label,
                    "init_time_s": init_s, "optimize_time_s": opt_s,
                    "iter_ms": iter_ms,
                })
                print(f"  {variant:12} n={n:>3} {mesh_label:6} — "
                      f"init={init_s:.2f}s opt={opt_s:.2f}s "
                      f"({iter_ms:.1f} ms/iter)")

    print(f"[debug] training done in {time.time()-train_start:.1f}s")

    arts = {label: load_mesh_artifacts(MESH_ROOT / rel)
            for label, rel in MESHES}
    mesh_labels = [label for label, _ in MESHES]

    rows = []
    for n in N_SPHERES:
        for mesh_label in mesh_labels:
            for variant in VARIANTS:
                r = eval_run(DEBUG_RESULTS, variant, n,
                             mesh_label, arts[mesh_label])
                if r is None:
                    continue
                t = next((t for t in timings
                          if t["variant"] == variant
                          and t["n"] == n
                          and t["mesh"] == mesh_label), {})
                rows.append({
                    "variant": variant, "mesh": mesh_label, "n": n,
                    "init_time_s": t.get("init_time_s"),
                    "optimize_time_s": t.get("optimize_time_s"),
                    "iter_ms": t.get("iter_ms"),
                    **r,
                })

    print_table(rows, mesh_labels)

    metrics_doc = {
        "library": _library_label(),
        "git_sha": _git_sha(),
        "seed": SEED,
        "iterations": ITERATIONS,
        "density_control_enabled": DENSITY_CONTROL,
        "per_sphere_mass": PER_SPHERE_MASS,
        "density": DENSITY,
        "device": DEVICE,
        "grid": {
            "meshes": [label for label, _ in MESHES],
            "n_spheres": list(N_SPHERES),
            "variants": list(VARIANTS),
        },
        "rows": rows,
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics_doc, f, indent=2)
    print(f"[debug] wrote {METRICS_JSON}")


if __name__ == "__main__":
    main()
