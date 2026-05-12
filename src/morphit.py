"""
MorphIt - Main sphere packing class.
Refactored from SpherePacker with improved modularity and configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
import time
from typing import Tuple, List, Dict, Optional, Any
import json
from pathlib import Path

from config import MorphItConfig
from inside_mesh import check_mesh_contains


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus: log(exp(x) - 1). Numerically stable."""
    return x + torch.log(-torch.expm1(-x))


class MorphIt(nn.Module):
    """
    MorphIt sphere packing system.

    A neural network-based approach for packing spheres inside 3D meshes
    with adaptive density control and multiple loss functions.
    """

    def __init__(self, config: Optional[MorphItConfig] = None):
        """
        Initialize MorphIt system.

        Args:
            config: Configuration object. If None, uses default config.
        """
        super(MorphIt, self).__init__()

        # Set configuration
        if config is None:
            from config import get_config

            config = get_config()
        self.config = config

        # Set random seed if specified
        if config.random_seed is not None:
            torch.manual_seed(config.random_seed)
            np.random.seed(config.random_seed)

        # Initialize system
        self._initialize_system()

    def _initialize_system(self):
        """Initialize the MorphIt system components."""

        # Set device
        self.device = torch.device(self.config.model.device)
        print(f"Using device: {self.device}")

        # Store configuration for easy access
        self.num_spheres = self.config.model.num_spheres
        self.mesh_path = self.config.model.mesh_path

        # Load and store mesh
        self.query_mesh = trimesh.load(self.mesh_path, force="mesh")
        self.mesh_volume = self.query_mesh.volume

        # Mass / center-of-mass / inertia of the target mesh. Pre-computed
        # once because the mass/com/inertia losses reference them every
        # iteration. trimesh.moment_inertia is volume-units * 1; we scale
        # by density so the residual is in physically meaningful kg·m²
        # (matching mesh_mass = volume * density).
        self.density = self.config.model.density
        self.mesh_mass = self.query_mesh.volume * self.density
        self.mesh_com = torch.tensor(
            self.query_mesh.center_mass, dtype=torch.float32, device=self.device
        )
        self.mesh_inertia = torch.tensor(
            self.query_mesh.moment_inertia * self.density,
            dtype=torch.float32,
            device=self.device,
        )

        # Initialize components
        self._initialize_spheres()
        self._initialize_sample_points()

        # Initialize additional components
        self.evolution_logger = None
        self.pl = None  # PyVista plotter placeholder

    def _initialize_spheres(self):
        """Initialize sphere centers and radii.

        Centers: voxel-grid placement (paper default). Voxel size is
        auto-tuned so the set of inside cells has at least num_spheres
        members; num_spheres of them are then randomly selected. This
        yields a well-spread, structured seed.
        Radii:   volume-preserving log-normal around the paper's mean
                 radius target (mesh_volume / num_spheres).
        """
        centers = self._voxel_sample_centers(self.num_spheres)
        radii = self._initialize_radii_with_variation(self.num_spheres)

        self._centers = nn.Parameter(centers)
        # Softplus reparameterization keeps radii strictly positive and
        # avoids numerical pathologies when a sphere briefly gets zero
        # radius during training. We compensate for the softplus gradient
        # attenuation by using a higher radius_lr (see TrainingConfig).
        self._radii = nn.Parameter(_inverse_softplus(radii))
        self.num_spheres = len(radii)
        self._print_initialization_stats(self.radii)

    def _voxel_sample_centers(self, num_spheres: int,
                              safety: float = 1.5,
                              max_iters: int = 10) -> torch.Tensor:
        """Place sphere centers at cell centers of a uniform voxel grid.

        Voxel size is auto-tuned so the set of cells whose centers lie
        inside the mesh has at least ``num_spheres`` members (typically
        ~safety × num_spheres). ``num_spheres`` of those cells are then
        randomly selected. This yields a well-spread, grid-like seed
        (so spheres start in distinct parts of the volume) with enough
        random variation that repeat runs don't get trapped in the same
        local minimum.

        Target occupancy: #inside_voxels ≈ safety × num_spheres.
        Assuming the inside-volume fraction of the AABB is roughly
        stable over voxel size changes, #inside_voxels ≈ V_mesh / s**3,
        so the initial estimate is s = (V_mesh / (safety*N))**(1/3).
        The iteration corrects if that estimate is off (thin meshes,
        concavities, etc.).
        """
        mesh = self.query_mesh
        lo, hi = mesh.bounds
        extent = hi - lo

        voxel_size = float((mesh.volume / (num_spheres * safety)) ** (1.0 / 3.0))
        # Clamp initial guess to a sane range relative to the mesh scale.
        voxel_size = min(max(voxel_size, mesh.scale * 1e-3), mesh.scale)

        inside = np.empty((0, 3), dtype=np.float64)
        for _ in range(max_iters):
            n_axis = np.maximum(1, np.ceil(extent / voxel_size).astype(int))
            axes = [
                lo[i] + (np.arange(n_axis[i]) + 0.5) * voxel_size
                for i in range(3)
            ]
            grid = np.stack(
                np.meshgrid(*axes, indexing="ij"), axis=-1
            ).reshape(-1, 3)

            mask = check_mesh_contains(mesh, grid)
            inside = grid[mask]

            if len(inside) >= num_spheres:
                break
            # Too few inside cells — shrink voxel and retry.
            voxel_size *= 0.75

        if len(inside) < num_spheres:
            # Mesh is too thin / concave for voxels to fit; top up with
            # uniform volume samples so the caller always gets N centers.
            needed = num_spheres - len(inside)
            fallback = trimesh.sample.volume_mesh(
                mesh, count=max(needed * 3, 100))
            if len(fallback) < needed:
                surf, _ = trimesh.sample.sample_surface(
                    mesh, count=needed * 2)
                fallback = np.vstack([fallback, surf + 0.05 *
                                      (mesh.center_mass - surf)])
            inside = np.vstack([inside, fallback[:needed]])

        n_candidates = len(inside)
        if n_candidates > num_spheres:
            idx = np.random.choice(n_candidates, num_spheres, replace=False)
            inside = inside[idx]

        print(f"Voxel init: size={voxel_size:.5f}m  "
              f"candidates={n_candidates}  selected={num_spheres}")
        return torch.tensor(inside, dtype=torch.float32, device=self.device)

    def _initialize_radii_with_variation(self, num_spheres: int) -> torch.Tensor:
        """Initialize radii with non-uniform distribution preserving target volume."""
        # Calculate target sphere volume
        target_sphere_volume = self.mesh_volume / num_spheres
        mean_radius = (3 * target_sphere_volume / (4 * np.pi)) ** (1 / 3)

        # Generate log-normal variation
        variation = self.config.model.initial_radius_variation
        log_normal_samples = (
            torch.from_numpy(
                np.random.lognormal(mean=0.0, sigma=variation, size=num_spheres)
            )
            .float()
            .to(self.device)
        )

        # Normalize to preserve total volume
        volume_weights = log_normal_samples**3
        volume_scale_factor = (num_spheres / volume_weights.sum()) ** (1 / 3)
        radius_factors = log_normal_samples * volume_scale_factor

        return mean_radius * radius_factors

    def _print_initialization_stats(self, radii: torch.Tensor):
        """Print initialization statistics."""
        with torch.no_grad():
            min_radius = radii.min().item()
            max_radius = radii.max().item()
            mean_radius = radii.mean().item()

            print(f"Initial radius distribution:")
            print(f"  - Min: {min_radius:.4f}")
            print(f"  - Mean: {mean_radius:.4f}")
            print(f"  - Max: {max_radius:.4f}")

            # Verify volume preservation
            SPHERE_VOLUME_CONSTANT = 4 * np.pi / 3
            total_volume = (SPHERE_VOLUME_CONSTANT * (radii**3)).sum().item()
            print(f"  - Target volume: {self.mesh_volume:.4f}")
            print(f"  - Initial volume: {total_volume:.4f}")

    def _initialize_sample_points(self):
        """Initialize sample points for loss computation."""
        # Initialize inside samples for coverage loss
        self._initialize_inside_samples()

        # Initialize surface samples for surface loss
        self._initialize_surface_samples()

    def _initialize_inside_samples(self):
        """Pre-compute sample points inside mesh for coverage computation."""
        num_points = self.config.model.num_inside_samples

        points = np.zeros((0, 3))
        batch_size = min(num_points * 2, 10000)
        mesh_bounds = self.query_mesh.bounds

        while len(points) < num_points:
            # Generate samples in batches
            samples = np.random.uniform(
                low=mesh_bounds[0], high=mesh_bounds[1], size=(batch_size, 3)
            )

            # Check which samples are inside the mesh
            inside = check_mesh_contains(self.query_mesh, samples)
            points = (
                np.vstack([points, samples[inside]])
                if len(points) > 0
                else samples[inside]
            )

            if len(points) >= num_points:
                points = points[:num_points]
                break

        self.inside_samples = torch.tensor(
            points, dtype=torch.float32, device=self.device
        )

    def _initialize_surface_samples(self):
        """Pre-sample points on mesh surface for surface loss computation."""
        num_samples = self.config.model.num_surface_samples

        # Sample points on mesh surface
        samples, face_ids = trimesh.sample.sample_surface(self.query_mesh, num_samples)

        # Convert to tensors
        self.surface_samples = torch.tensor(
            samples, dtype=torch.float32, device=self.device
        )
        self.surface_face_ids = torch.tensor(
            face_ids, dtype=torch.long, device=self.device
        )

        # Pre-compute face normals
        self.face_normals = torch.tensor(
            self.query_mesh.face_normals, dtype=torch.float32, device=self.device
        )
        self.surface_normals = self.face_normals[self.surface_face_ids]

    @property
    def centers(self) -> torch.Tensor:
        """Get sphere centers."""
        return self._centers

    @centers.setter
    def centers(self, value: torch.Tensor):
        """Set sphere centers."""
        self._centers = nn.Parameter(torch.tensor(value, device=self.device))

    @property
    def radii(self) -> torch.Tensor:
        """Get sphere radii (always positive via softplus reparameterization)."""
        return F.softplus(self._radii)

    @radii.setter
    def radii(self, value: torch.Tensor):
        """Set sphere radii (stores inverse-softplus internally)."""
        self._radii = nn.Parameter(
            _inverse_softplus(torch.tensor(value, device=self.device))
        )

    @property
    def masses(self) -> torch.Tensor:
        """Per-sphere mass (kg), derived from uniform density × (4/3)π·r³.

        All physics losses (mass_loss, com_loss, inertia_loss) read from
        here so the per-sphere mass formulation can be swapped at the
        model layer without touching the loss code.
        """
        FOUR_THIRDS_PI = (4.0 / 3.0) * 3.141592653589793
        return self.config.model.density * FOUR_THIRDS_PI * (self.radii ** 3)

    def save_results(self, filename: Optional[str] = None) -> None:
        """
        Save sphere centers and radii to JSON file.

        Args:
            filename: Output filename. If None, uses config default.
        """
        if filename is None:
            filename = self.config.output_filename

        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "centers": self.centers.detach().cpu().numpy().tolist(),
            "radii": self.radii.detach().cpu().numpy().tolist(),
            "mesh_path": self.mesh_path,
            "num_spheres": self.num_spheres,
            "config": self._config_to_dict(),
        }

        filepath = results_dir / filename
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to: {filepath}")

    def _config_to_dict(self) -> Dict:
        """Convert configuration to dictionary for serialization."""
        return {
            "model": vars(self.config.model),
            "training": vars(self.config.training),
            "visualization": vars(self.config.visualization),
            "results_dir": self.config.results_dir,
            "output_filename": self.config.output_filename,
            "random_seed": self.config.random_seed,
        }

    def get_sphere_statistics(self) -> Dict:
        """Get statistics about current sphere configuration."""
        with torch.no_grad():
            centers_np = self.centers.detach().cpu().numpy()
            radii_np = self.radii.detach().cpu().numpy()

            return {
                "num_spheres": self.num_spheres,
                "radius_stats": {
                    "min": float(radii_np.min()),
                    "max": float(radii_np.max()),
                    "mean": float(radii_np.mean()),
                    "std": float(radii_np.std()),
                },
                "total_sphere_volume": float((4 / 3 * np.pi * (radii_np**3)).sum()),
                "mesh_volume": float(self.mesh_volume),
                "volume_ratio": float(
                    (4 / 3 * np.pi * (radii_np**3)).sum() / self.mesh_volume
                ),
                "center_bounds": {
                    "min": centers_np.min(axis=0).tolist(),
                    "max": centers_np.max(axis=0).tolist(),
                },
            }

    # Visualization methods
    def pv_init(
        self,
        enabled: bool = False,
        off_screen: bool = False,
        save_video: bool = False,
        filename: str = "morphit.mp4",
    ):
        """
        Initialize PyVista plotter for visualization.

        Args:
            off_screen: Whether to run in off-screen mode
            save_video: Whether to save video
            filename: Video filename
        """
        if enabled == False:
            print(f"Disabled pyvista visualization.")
            return
        from visualization import MorphItVisualizer

        self.visualizer = MorphItVisualizer(self, self.config.visualization)
        self.visualizer.pv_init(enabled, off_screen, save_video, filename)

        # For backward compatibility
        self.pl = self.visualizer.plotter
        self.off_screen = off_screen
        self.save_video = save_video

    def pv_render(self):
        """Render current sphere state."""
        if hasattr(self, "visualizer"):
            self.visualizer.pv_render()

    def pv_close(self):
        """Close PyVista plotter."""
        if hasattr(self, "visualizer"):
            self.visualizer.pv_close()

    def pv_screenshot(self, filename: str):
        """Take screenshot."""
        if hasattr(self, "visualizer"):
            self.visualizer.pv_screenshot(filename)

    def initialize_render_thread(self, render_interval: int = 5):
        """
        Initialize rendering thread.

        Args:
            render_interval: Interval between renders
        """
        from logger import RenderThread

        self.render_thread = RenderThread(self, render_interval)
        self.render_thread.start()

    def stop_render_thread(self):
        """Stop rendering thread."""
        if hasattr(self, "render_thread"):
            self.render_thread.stop()

    def train(self, config_updates: Optional[Dict[str, Any]] = None):
        """
        Train the MorphIt model.

        Args:
            config_updates: Optional configuration updates

        Returns:
            Convergence tracker with training history
        """
        from training import train_morphit

        # Apply config updates if provided
        if config_updates is not None:
            from config import update_config_from_dict

            self.config = update_config_from_dict(self.config, config_updates)

        return train_morphit(self)
