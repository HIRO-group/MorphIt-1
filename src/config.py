"""
Configuration file for MorphIt sphere packing system.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    num_spheres: int = 25
    mesh_path: str = "../mesh_models/fr3/collision/link0.obj"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Sphere initialization parameters
    # Controls size variation (log-normal sigma)
    initial_radius_variation: float = 0.1
    num_inside_samples: int = 5000  # Points inside mesh for coverage computation
    num_surface_samples: int = 5000  # Points on mesh surface for surface loss

    max_spheres: int = num_spheres  # Maximum number of spheres allowed

    density: float = 1000.0  # Material density (kg/m³). Used by mass/com/inertia losses.

    # Per-sphere learnable mass. When False (default), model.masses is
    # derived analytically from radii × uniform density. When True, an
    # nn.Parameter _log_masses is registered (softplus-positive, like
    # radii) and the optimizer learns per-sphere masses independently
    # of radii — useful for objects with non-uniform internal density.
    per_sphere_mass: bool = False


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    iterations: int = 300
    verbose_frequency: int = 50

    # Logging the packing evolution
    logging_enabled: bool = False

    # Optimizer parameters. radius_lr is in raw (pre-softplus) space — the
    # effective step on the real radius is sigmoid(raw_r) * lr.
    center_lr: float = 0.002
    radius_lr: float = 0.005
    # mass_lr: only used when model.per_sphere_mass=True. Per-sphere
    # mass is softplus-reparameterized like radii; lr operates in raw
    # space. Default tuned so per-iter step is meaningful for typical
    # m ~ M_mesh/N (the init value).
    mass_lr: float = 0.01
    grad_clip_norm: float = 1.0

    # Loss weights - can be overridden by get_config()
    coverage_weight: float = 10.0
    overlap_weight: float = 0.01
    boundary_weight: float = 5.0
    surface_weight: float = 5.0
    containment_weight: float = 5.0
    sqem_weight: float = 800.0
    mass_weight: float = 0.0
    com_weight: float = 0.0
    inertia_weight: float = 0.0
    # flatness_loss has a Python for-loop over face groups that forces
    # per-iteration GPU→CPU sync, roughly doubling per-iter cost. It is
    # not in the paper's 6-loss formulation — disabled by default.
    flatness_weight: float = 0.0
    hausdorff_weight: float = 0.0
    mesh_containment_weight: float = 0.0

    # Early stopping parameters
    early_stopping: bool = False
    convergence_patience: int = 50
    convergence_threshold: float = 0.001

    # Density control parameters
    density_control_enabled: bool = True
    density_control_min_interval: int = 160
    density_control_patience: int = 1
    density_control_grad_threshold: float = 1e-4
    # Warmup: mini-optimization over only the freshly-placed spheres,
    # with survivors frozen, so new spheres settle before the main
    # optimizer sees them. Set to 0 to disable.
    density_control_warmup_steps: int = 10
    # Cooling: each pass replaces a smaller fraction (temperature * cooling).
    # Set to 1.0 to hold temperature constant across passes.
    density_control_cooling_factor: float = 0.85
    # Force-remove spheres during density control whose radius falls below
    # this fraction of the mesh scale, OR whose center has drifted outside
    # the mesh. Catches collapsed / escaped spheres that the marginal-value
    # scoring on its own can leave in place if they happen to sit on a
    # locally-unique gap.
    density_control_min_radius_fraction: float = 0.001


@dataclass
class VisualizationConfig:
    """Visualization configuration parameters."""

    enabled: bool = False
    off_screen: bool = True
    save_video: bool = True
    video_filename: str = "sphere_filling.mp4"
    render_interval: int = 5

    # Visualization appearance
    sphere_color: str = "blue"
    sphere_opacity: float = 0.3
    mesh_color: str = "white"
    mesh_line_width: float = 1.5
    mesh_opacity: float = 0.8

    # Sample points visualization
    show_sample_points: bool = True
    show_surface_points: bool = True
    sample_points_subsample: int = 100000
    surface_points_subsample: int = 100000
    sample_point_color: str = "red"
    surface_point_color: str = "green"
    point_size: int = 5

    # Camera parameters
    camera_position: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    camera_focal_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_view_up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    camera_azimuth: float = 80
    camera_elevation: float = 120
    camera_roll: float = 120
    camera_zoom: float = 1.5


@dataclass
class MorphItConfig:
    """Main configuration class combining all sub-configurations."""

    # ``field(default_factory=...)`` is required: using plain ``ModelConfig()``
    # as a default makes every MorphItConfig instance share the SAME
    # ModelConfig object, which makes ``update_config_from_dict`` mutate
    # state visible to every other config created by ``get_config``.
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Output configuration
    results_dir: str = "results/output"
    output_filename: str = "morphit_results.json"

    # Random seed (None means no seed)
    random_seed: int = None


# Alternative loss weight configurations
LOSS_WEIGHT_CONFIGS = {
    "MorphIt-V": {
        "coverage_weight": 5000.0,
        "overlap_weight": 0.1,
        "boundary_weight": 1.0,
        "surface_weight": 0.1,
        "containment_weight": 1.0,
        "sqem_weight": 10.0,
        "mass_weight": 0.0,
        "com_weight": 0.0,
        "inertia_weight": 0.0,
        "flatness_weight": 0.0,
        "hausdorff_weight": 0.0,
        # mesh_containment is the only loss whose gradient acts on a
        # sphere whose center has fully escaped the mesh (boundary +
        # coverage + sqem all decay to zero in that regime). Weight 100
        # eliminates ~70% of escapes on bunny n=64; the hard projection
        # in training._project_centers_inside_mesh handles the rest.
        # Bumped from 10 → 100 alongside removing _prune_escaped_spheres.
        "mesh_containment_weight": 100.0,
    },
    "MorphIt-S": {
        "coverage_weight": 0.01,
        "overlap_weight": 0.01,
        "boundary_weight": 1000.0,
        "surface_weight": 10.0,
        "containment_weight": 1.0,
        "sqem_weight": 1000.0,
        "mass_weight": 0.0,
        "com_weight": 0.0,
        "inertia_weight": 0.0,
        "flatness_weight": 0.0,
        "hausdorff_weight": 10.0,
        "mesh_containment_weight": 500.0,
    },
    "MorphIt-B": {
        "coverage_weight": 1500.0,
        "overlap_weight": 0.3,
        "boundary_weight": 1.0,
        "surface_weight": 50.0,
        "containment_weight": 10.0,
        "sqem_weight": 3000.0,
        "mass_weight": 0.0,
        "com_weight": 0.0,
        "inertia_weight": 0.0,
        "flatness_weight": 0.0,
        "hausdorff_weight": 0.0,
        "mesh_containment_weight": 10.0,
    },
    # MorphIt-Obj: physics-aware variant — mass/com/inertia losses on
    # in addition to geometry. Weights match plus-plus's MorphIt-Obj
    # exactly so cross-library packings are comparable.
    "MorphIt-Obj": {
        "coverage_weight": 100.0,
        "overlap_weight": 0.3,
        "boundary_weight": 0.0,
        "surface_weight": 50.0,
        "containment_weight": 1.0,
        "sqem_weight": 50.0,
        "mass_weight": 30.0,
        "com_weight": 10.0,
        "inertia_weight": 30.0,
        "flatness_weight": 0.0,
        "hausdorff_weight": 10.0,
        "mesh_containment_weight": 500.0,
    },
    # MorphIt-Obj-mass: identical weights to MorphIt-Obj — only the
    # parameterization changes (per-sphere mass becomes a learnable
    # parameter). Set model.per_sphere_mass=True alongside this.
    "MorphIt-Obj-mass": {
        "coverage_weight": 100.0,
        "overlap_weight": 0.3,
        "boundary_weight": 0.0,
        "surface_weight": 50.0,
        "containment_weight": 1.0,
        "sqem_weight": 50.0,
        "mass_weight": 30.0,
        "com_weight": 10.0,
        "inertia_weight": 30.0,
        "flatness_weight": 0.0,
        "hausdorff_weight": 10.0,
        "mesh_containment_weight": 500.0,
    },
}


def get_config(loss_config: str = "MorphIt-B") -> MorphItConfig:
    """
    Get configuration with specified loss weight configuration.

    Args:
        loss_config: One of "MorphIt-V", "MorphIt-S", or "MorphIt-B"

    Returns:
        MorphItConfig instance with specified loss weights
    """
    config = MorphItConfig()

    if loss_config in LOSS_WEIGHT_CONFIGS:
        weights = LOSS_WEIGHT_CONFIGS[loss_config]
        for key, value in weights.items():
            setattr(config.training, key, value)
    else:
        raise ValueError(
            f"Unknown loss config: {loss_config}. Available: {list(LOSS_WEIGHT_CONFIGS.keys())}"
        )

    return config


def update_config_from_dict(
    config: MorphItConfig, updates: Dict[str, Any]
) -> MorphItConfig:
    """
    Update configuration from a dictionary of updates.

    Args:
        config: Base configuration
        updates: Dictionary with nested updates (e.g., {"training.iterations": 100})

    Returns:
        Updated configuration
    """
    for key, value in updates.items():
        if "." in key:
            section, param = key.split(".", 1)
            if hasattr(config, section):
                section_config = getattr(config, section)
                if hasattr(section_config, param):
                    setattr(section_config, param, value)
                else:
                    raise ValueError(
                        f"Unknown parameter: {param} in section {section}")
            else:
                raise ValueError(f"Unknown section: {section}")
        else:
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

    return config
