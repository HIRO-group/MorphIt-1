"""
Quick evaluation loop: train MorphIt on a small grid of (link, num_spheres,
variant), then evaluate each result against held-out surface and volume
samples. Prints a metrics table and the paper-ordering checks.

Intended use: change a knob in src/, then:

    cd src
    python scripts/debug_quick_eval.py

Outputs go to a sibling results dir (DEBUG_RESULTS) so the main user-facing
panda_output is left untouched. The grid (LINKS / N_SPHERES / VARIANTS) is
small by design — the goal is fast iteration, not exhaustive evaluation.
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
# Each LINK entry is (label, path_relative_to_mesh_models). Labels feed
# the metrics table; paths point at the actual .obj used for training
# and held-out eval.
LINKS = (
    ("link0", "fr3/collision/link0.obj"),
    ("bunny", "bunny.obj"),
)
N_SPHERES = (100, 200)
# Pick from {"morphit-v", "morphit-s", "morphit-b"} or any subset.
VARIANTS = ("morphit-v", "morphit-s")

# ─── Training knobs ──────────────────────────────────────────────────
ITERATIONS = 300
DENSITY_CONTROL = True  # flip False to time pure Adam without re-packing

# Per-iteration cost is ~5 ms on a recent GPU at n=64 — keep in mind when
# growing the grid. flatness_loss has a per-iter Python loop so it is left
# disabled here; mass/com/inertia and hausdorff are zeroed so the loss
# matches the paper's geometry-only formulation. mesh_containment is
# intentionally NOT zeroed — V's containment relies on it (see
# debug_v_escape.py for the sweep that pins this).
COMMON_PARAMS = {
    "model.num_inside_samples": 5000,
    "model.num_surface_samples": 5000,
    "training.verbose_frequency": 500,
    "training.density_control_enabled": DENSITY_CONTROL,
    "training.density_control_min_interval": 160,
    "training.flatness_weight": 0.0,
    "training.hausdorff_weight": 0.0,
    "training.mass_weight": 0.0,
    "training.com_weight": 0.0,
    "training.inertia_weight": 0.0,
    "training.logging_enabled": False,
    "training.early_stopping": False,
    "visualization.enabled": False,
    # radius_lr in raw (pre-softplus) space; effective step is
    # sigmoid(raw_r) * lr.
    "training.radius_lr": 0.1,
}

# Loss-weight overrides per variant. These match the paper's tuning under
# the 6-loss regime. Keep them here (not in get_config) so the debug
# script can be self-documenting about what each variant emphasises.
VARIANT_WEIGHTS = {
    "morphit-v": {  # volume-dominant
        "base_config": "MorphIt-V",
        "weights": {
            "coverage_weight": 1000.0, "overlap_weight": 0.05,
            "boundary_weight": 1.0, "surface_weight": 7.0,
            "containment_weight": 1.0, "sqem_weight": 10.0,
        },
    },
    "morphit-s": {  # surface-dominant
        "base_config": "MorphIt-S",
        "weights": {
            "coverage_weight": 0.01, "overlap_weight": 0.01,
            "boundary_weight": 5000.0, "surface_weight": 100.0,
            "containment_weight": 1.0, "sqem_weight": 1000.0,
        },
    },
    "morphit-b": {  # balanced
        "base_config": "MorphIt-B",
        "weights": {
            "coverage_weight": 1500.0, "overlap_weight": 0.3,
            "boundary_weight": 1.0, "surface_weight": 50.0,
            "containment_weight": 1.0, "sqem_weight": 3000.0,
        },
    },
}

MESH_ROOT = SRC.parent / "mesh_models"
DEBUG_RESULTS = SRC / "results" / "debug_quick_eval"

# ─── Eval config ─────────────────────────────────────────────────────
NUM_SURFACE = 50_000
NUM_VOLUME = 50_000
BOUNDS_EXPAND = 1.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Training ────────────────────────────────────────────────────────

def _sync_clock() -> float:
    """Wall time after a CUDA sync — gives apples-to-apples timings."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def train_one(variant_name: str, link_label: str, mesh_path: Path,
              n_spheres: int, out_dir: Path):
    """Train one (variant, mesh, n) configuration and dump result JSON.

    The output filename uses ``link_label`` (not ``mesh_path.stem``) so
    meshes that share a stem but live in different folders don't collide.
    Returns (init_time_s, optimize_time_s) or None if cached.
    """
    out_file = out_dir / f"{link_label}.json"
    if out_file.exists():
        return None

    variant_cfg = VARIANT_WEIGHTS[variant_name]
    config = get_config(variant_cfg["base_config"])
    updates = {
        "model.num_spheres": n_spheres,
        "model.mesh_path": str(mesh_path),
        "training.iterations": ITERATIONS,
        **{f"training.{k}": v for k, v in variant_cfg["weights"].items()},
        **COMMON_PARAMS,
    }
    config = update_config_from_dict(config, updates)
    config.results_dir = str(out_dir)
    config.output_filename = f"{link_label}.json"

    t0 = _sync_clock()
    model = MorphIt(config)
    init_time_s = _sync_clock() - t0

    t1 = _sync_clock()
    train_morphit(model)
    optimize_time_s = _sync_clock() - t1

    model.save_results()
    return init_time_s, optimize_time_s


# ─── Evaluation ──────────────────────────────────────────────────────

def load_link_artifacts(mesh_path: Path):
    """Sample held-out surface + volume points for evaluation. Sampling
    seed is fixed so re-running gives identical eval points (only the
    training pass is stochastic). The held-out mesh handle is kept on
    the artifact dict so the escape-count metric can call ``.contains``
    on it directly."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    lo, hi = mesh.bounds
    c = 0.5 * (lo + hi)
    he = 0.5 * (hi - lo) * BOUNDS_EXPAND
    bounds = np.stack([c - he, c + he])
    sample_volume = float(np.prod(bounds[1] - bounds[0]))

    surf, _ = trimesh.sample.sample_surface(mesh, NUM_SURFACE)
    surf = np.asarray(surf, dtype=np.float64)

    rng = np.random.default_rng(0)
    vol = rng.uniform(bounds[0], bounds[1], size=(NUM_VOLUME, 3))
    inside = np.zeros(len(vol), dtype=bool)
    for s in range(0, len(vol), 5000):
        e = min(s + 5000, len(vol))
        inside[s:e] = mesh.contains(vol[s:e])

    return {"mesh": mesh, "mesh_volume": float(mesh.volume),
            "mesh_scale": float(mesh.scale),
            "sample_volume": sample_volume,
            "surface": surf, "volume": vol, "is_inside_mesh": inside}


def load_run(results_root: Path, method: str, n: int, link: str):
    p = results_root / method / str(n) / f"{link}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return {
        "centers": np.asarray(d["centers"], dtype=np.float64),
        "radii": np.asarray(d["radii"], dtype=np.float64),
        "actual_n": len(d["radii"]),
    }


def surface_distance_metrics(surf_pts, centers, radii):
    """Max / mean unsigned surface error in metres."""
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
    """r_in: inside-mesh fraction of sphere union covered by mesh.
    r_out: sphere volume outside mesh.
    r_uni: total sphere union volume normalised to mesh volume.
    All as ratios of mesh_volume."""
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
    """Count packing pathologies that the density-control bad-sphere
    rule targets:

      • ``n_out``  — sphere centers strictly outside the mesh. These
        are the spheres ``_prune_escaped_spheres`` would strip at end
        of training (and ``_identify_bad_spheres`` would reseed mid-
        training). A nonzero count after training implies a gradient
        path is still pushing centers out faster than density control
        can pull them back in.
      • ``n_tiny`` — radii below 0.001 × mesh.scale (matching the
        ``density_control_min_radius_fraction`` default). Tiny spheres
        contribute essentially zero coverage and indicate radius
        collapse from overlap / boundary pressure.
    """
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    inside = art["mesh"].contains(centers)
    n_out = int((~inside).sum())
    min_radius = 0.001 * art["mesh_scale"]
    n_tiny = int((radii < min_radius).sum())
    return n_out, n_tiny


def eval_run(results_root, method, n, link, art):
    entry = load_run(results_root, method, n, link)
    if entry is None:
        return None
    max_d, avg_d = surface_distance_metrics(
        art["surface"], entry["centers"], entry["radii"])
    r_in, r_out, r_uni = volume_overlap_metrics(
        art, entry["centers"], entry["radii"])
    n_out, n_tiny = escape_metrics(art, entry["centers"], entry["radii"])
    return {"actual_n": entry["actual_n"], "n_out": n_out, "n_tiny": n_tiny,
            "r_in": r_in, "r_out": r_out,
            "r_uni": r_uni, "d_avg_mm": avg_d * 1000, "d_max_mm": max_d * 1000}


# ─── Reporting ───────────────────────────────────────────────────────

HEADER = (f"{'n':>4}  {'link':>6}  {'method':<11}"
          f"  {'actual':>6}  {'n_out':>5}  {'n_tiny':>6}"
          f"  {'r_in':>7}  {'r_out':>7}  {'r_uni':>7}"
          f"  {'d_avg_mm':>9}  {'d_max_mm':>9}")


def _fmt_row(n, link, variant, r):
    return (f"{n:>4}  {link:>6}  {variant:<11}"
            f"  {r['actual_n']:>6}  {r['n_out']:>5}  {r['n_tiny']:>6}"
            f"  {r['r_in']:>7.3f}  {r['r_out']:>7.3f}  {r['r_uni']:>7.3f}"
            f"  {r['d_avg_mm']:>9.3f}  {r['d_max_mm']:>9.3f}")


def print_per_link_table(rows, link_labels):
    print()
    print(HEADER)
    print("-" * len(HEADER))
    for n in N_SPHERES:
        for link in link_labels:
            for variant in VARIANTS:
                r = next((r for r in rows if r["n"] == n
                          and r["link"] == link
                          and r["method"] == variant), None)
                if r is None:
                    print(f"{n:>4}  {link:>6}  {variant:<11}  MISSING")
                    continue
                print(_fmt_row(n, link, variant, r))
            print()


def print_averaged_table(rows):
    print("=" * 110)
    print("  Averaged across links. Paper ordering targets:")
    print("     r_in :  V > B > S   (V ~ 1)")
    print("     r_out:  V > B > S   (S ~ 0)")
    print("     r_uni:  V > B > S   (B ~ 1)")
    print("     d_avg:  V > S > B   (B lowest)")
    print("=" * 110)
    print(HEADER)
    for n in N_SPHERES:
        for variant in VARIANTS:
            sel = [r for r in rows if r["n"] == n and r["method"] == variant]
            if not sel:
                continue
            avg = {k: np.mean([r[k] for r in sel])
                   for k in ("actual_n", "n_out", "n_tiny",
                             "r_in", "r_out", "r_uni",
                             "d_avg_mm", "d_max_mm")}
            print(f"{n:>4}  {'avg':>6}  {variant:<11}"
                  f"  {avg['actual_n']:>6.1f}"
                  f"  {avg['n_out']:>5.1f}  {avg['n_tiny']:>6.1f}"
                  f"  {avg['r_in']:>7.3f}  {avg['r_out']:>7.3f}"
                  f"  {avg['r_uni']:>7.3f}"
                  f"  {avg['d_avg_mm']:>9.3f}  {avg['d_max_mm']:>9.3f}")
        print()


# ─── Main ────────────────────────────────────────────────────────────

def main():
    if DEBUG_RESULTS.exists():
        shutil.rmtree(DEBUG_RESULTS)
    print(f"[debug] cleaned {DEBUG_RESULTS}")

    print(f"[debug] Training into {DEBUG_RESULTS}")
    print(f"[debug] ITERATIONS={ITERATIONS}  "
          f"DENSITY_CONTROL={DENSITY_CONTROL}  "
          f"density_control_min_interval="
          f"{COMMON_PARAMS['training.density_control_min_interval']}")

    train_start = time.time()
    timings: list[dict] = []

    for variant in VARIANTS:
        for n in N_SPHERES:
            for link_label, link_rel in LINKS:
                out_dir = DEBUG_RESULTS / variant / str(n)
                out_dir.mkdir(parents=True, exist_ok=True)
                mesh_path = MESH_ROOT / link_rel
                t0 = time.time()
                result = train_one(variant, link_label, mesh_path,
                                   n, out_dir)
                wall = time.time() - t0
                if result is None:
                    print(f"  {variant:10} n={n:>3} {link_label} — "
                          f"cached ({wall:.1f}s)")
                    continue
                init_time_s, optimize_time_s = result
                iter_ms = 1000.0 * optimize_time_s / max(1, ITERATIONS)
                timings.append({"variant": variant, "n": n,
                                "link": link_label,
                                "init_time_s": init_time_s,
                                "optimize_time_s": optimize_time_s,
                                "iter_ms": iter_ms})
                print(f"  {variant:10} n={n:>3} {link_label} — "
                      f"init={init_time_s:.2f}s  "
                      f"optimize={optimize_time_s:.2f}s  "
                      f"({iter_ms:.1f} ms/iter)")

    total_wall = time.time() - train_start
    print(f"[debug] training done in {total_wall:.1f}s")

    if timings:
        init_mean = np.mean([t["init_time_s"] for t in timings])
        opt_mean = np.mean([t["optimize_time_s"] for t in timings])
        iter_mean = np.mean([t["iter_ms"] for t in timings])
        print(f"[timing] mean: init={init_mean:.2f}s  "
              f"optimize={opt_mean:.2f}s  {iter_mean:.1f} ms/iter")

    arts = {label: load_link_artifacts(MESH_ROOT / rel)
            for label, rel in LINKS}
    link_labels = [label for label, _ in LINKS]

    rows = []
    for n in N_SPHERES:
        for link_label in link_labels:
            for variant in VARIANTS:
                r = eval_run(DEBUG_RESULTS, variant, n,
                             link_label, arts[link_label])
                if r is None:
                    continue
                rows.append({"n": n, "link": link_label,
                             "method": variant, **r})

    print_per_link_table(rows, link_labels)
    print_averaged_table(rows)


if __name__ == "__main__":
    main()
