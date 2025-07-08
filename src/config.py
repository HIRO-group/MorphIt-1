"""
Configuration file for MorphIt sphere packing system.
All parameters are centralized here for easy modification.
"""

import torch
from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    num_spheres: int = 15
    mesh_path: str = "../mesh_models/fr3/collision/link0.obj"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Sphere initialization parameters
    initial_radius_variation: float = 0.4  # Controls size variation (log-normal sigma)
    num_inside_samples: int = 5000  # Points inside mesh for coverage computation
    num_surface_samples: int = 1000  # Points on mesh surface for surface loss

    # Density control parameters
    radius_threshold: float = 0.02  # Threshold for pruning small spheres
    coverage_threshold: float = (
        0.01  # Threshold for adding spheres to poor coverage areas
    )
    max_spheres: int = 50  # Maximum number of spheres allowed


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    iterations: int = 50
    verbose_frequency: int = 50

    # Optimizer parameters
    center_lr: float = 0.005
    radius_lr: float = 0.0005
    grad_clip_norm: float = 1.0

    # Loss weights - MorphIt-B configuration
    coverage_weight: float = 100.0
    overlap_weight: float = 1.1
    boundary_weight: float = 5.0
    surface_weight: float = 5.0
    containment_weight: float = 5.0
    sqem_weight: float = 800.0

    # Early stopping parameters
    early_stopping: bool = True
    convergence_patience: int = 50
    convergence_threshold: float = 0.001

    # Density control parameters
    density_control_min_interval: int = 100
    density_control_patience: int = 100
    density_control_grad_threshold: float = 1e-4


@dataclass
class VisualizationConfig:
    """Visualization configuration parameters."""

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

    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    visualization: VisualizationConfig = VisualizationConfig()

    # Output configuration
    results_dir: str = "results/output"
    output_filename: str = "morphit_results.json"

    # Random seed (None means no seed)
    random_seed: int = None


# Alternative loss weight configurations
LOSS_WEIGHT_CONFIGS = {
    "MorphIt-V": {
        "coverage_weight": 4000.0,
        "overlap_weight": 0.1,
        "boundary_weight": 10.0,
        "surface_weight": 0.1,
        "containment_weight": 50.0,
        "sqem_weight": 100.0,
    },
    "MorphIt-S": {
        "coverage_weight": 0.01,
        "overlap_weight": 0.01,
        "boundary_weight": 5000.0,
        "surface_weight": 100.0,
        "containment_weight": 1.0,
        "sqem_weight": 1000.0,
    },
    "MorphIt-B": {
        "coverage_weight": 100.0,
        "overlap_weight": 1.1,
        "boundary_weight": 5.0,
        "surface_weight": 5.0,
        "containment_weight": 5.0,
        "sqem_weight": 800.0,
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
                    raise ValueError(f"Unknown parameter: {param} in section {section}")
            else:
                raise ValueError(f"Unknown section: {section}")
        else:
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

    return config
