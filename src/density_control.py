"""
Adaptive density control module for MorphIt sphere packing.
Handles sphere pruning and addition based on optimization progress.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Any
from inside_mesh import check_mesh_contains


class DensityController:
    """
    Adaptive density control for sphere packing optimization.

    Follows concepts from 3D Gaussian Splatting for dynamic sphere management.
    """

    def __init__(self, model, config):
        """
        Initialize density controller.

        Args:
            model: MorphIt model instance
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.device = model.device

        # Track when density control was last performed
        self.last_density_control_iter = 0

    def should_perform_density_control(
        self,
        loss_history: List[float],
        position_grad_history: List[float],
        radius_grad_history: List[float],
        iteration: int,
    ) -> bool:
        """
        Determine if density control should be performed.

        Args:
            loss_history: List of recent loss values
            position_grad_history: History of position gradient magnitudes
            radius_grad_history: History of radius gradient magnitudes
            iteration: Current iteration number

        Returns:
            Whether to perform density control
        """
        min_interval = self.config.training.density_control_min_interval
        patience = self.config.training.density_control_patience
        grad_threshold = self.config.training.density_control_grad_threshold

        # Always wait for minimum interval
        if iteration - self.last_density_control_iter < min_interval:
            return False

        # Force density control after longer intervals
        if iteration - self.last_density_control_iter > min_interval * 2:
            return True

        # Need enough history
        if len(loss_history) < patience or len(position_grad_history) < patience:
            return False

        # Check if loss has plateaued
        recent_losses = loss_history[-patience:]
        loss_change = abs(recent_losses[0] - recent_losses[-1]) / max(
            abs(recent_losses[0]), 1e-5
        )
        loss_plateaued = loss_change < 0.01  # Less than 1% change

        # Check if gradients are small
        recent_pos_grads = position_grad_history[-patience:]
        recent_rad_grads = radius_grad_history[-patience:]

        grads_small = (
            sum(g < grad_threshold for g in recent_pos_grads) > patience // 2
            and sum(g < grad_threshold for g in recent_rad_grads) > patience // 2
        )

        should_densify = loss_plateaued or grads_small

        if should_densify:
            print("\n--- Density Control Trigger ---")
            print(f"Loss plateau: {loss_plateaued} (change: {loss_change:.6f})")
            print(f"Small gradients: {grads_small}")

        return should_densify

    def adaptive_density_control(self) -> Tuple[int, int]:
        """
        Perform adaptive density control.

        Returns:
            Tuple of (spheres_added, spheres_removed)
        """
        radius_threshold = self.config.model.radius_threshold
        coverage_threshold = self.config.model.coverage_threshold
        max_spheres = self.config.model.max_spheres

        print("\n--- Starting Adaptive Density Control ---")
        initial_count = self.model.num_spheres
        print(f"Initial sphere count: {initial_count}")

        # 1. Prune ineffective spheres
        spheres_removed = self._prune_spheres(radius_threshold)

        # 2. Add spheres to poorly covered areas
        spheres_added = self._add_spheres_to_poor_coverage(
            coverage_threshold, max_spheres
        )

        # Update sphere count
        self.model.num_spheres = len(self.model._radii)

        # Print summary
        print(f"Density control summary:")
        print(f"  - Initial count: {initial_count}")
        print(f"  - Removed: {spheres_removed}")
        print(f"  - Added: {spheres_added}")
        print(f"  - Final count: {self.model.num_spheres}")

        return spheres_added, spheres_removed

    def _prune_spheres(self, radius_threshold: float) -> int:
        """
        Prune ineffective spheres.

        Args:
            radius_threshold: Minimum radius threshold

        Returns:
            Number of spheres removed
        """
        # Find spheres with small radius
        small_radius_mask = self.model.radii < radius_threshold

        # Find spheres with centers outside mesh
        with torch.no_grad():
            centers_np = self.model.centers.detach().cpu().numpy()
            outside_mesh_mask = torch.tensor(
                ~check_mesh_contains(self.model.query_mesh, centers_np),
                device=self.device,
                dtype=torch.bool,
            )

        # Combine pruning criteria
        prune_mask = small_radius_mask | outside_mesh_mask
        spheres_to_remove = prune_mask.sum().item()

        print(f"Spheres to prune: {spheres_to_remove}")
        print(f"  - Small radius: {small_radius_mask.sum().item()}")
        print(f"  - Center outside mesh: {outside_mesh_mask.sum().item()}")

        if spheres_to_remove > 0:
            # Keep valid spheres
            valid_indices = ~prune_mask
            self.model._centers = nn.Parameter(self.model._centers[valid_indices])
            self.model._radii = nn.Parameter(self.model._radii[valid_indices])
            print(f"After pruning: {len(self.model._radii)} spheres remaining")

        return spheres_to_remove

    def _add_spheres_to_poor_coverage(
        self, coverage_threshold: float, max_spheres: int
    ) -> int:
        """
        Add spheres to poorly covered areas.

        Args:
            coverage_threshold: Threshold for poor coverage
            max_spheres: Maximum number of spheres allowed

        Returns:
            Number of spheres added
        """
        # Calculate coverage of inside samples
        centers = self.model.centers
        radii = self.model.radii

        dists = torch.norm(
            self.model.inside_samples.unsqueeze(1) - centers.unsqueeze(0), dim=2
        )
        sphere_coverage = dists - radii.unsqueeze(0)
        min_distances, _ = torch.min(sphere_coverage, dim=1)

        # Find poorly covered regions
        poorly_covered = min_distances > coverage_threshold
        poor_regions = self.model.inside_samples[poorly_covered]

        # Limit number of new spheres
        space_available = max_spheres - len(self.model._radii)
        spheres_to_add = min(len(poor_regions), space_available)

        if spheres_to_add > 0:
            print(f"Poorly covered regions: {len(poor_regions)}")
            print(f"Space available: {space_available}")
            print(f"Adding {spheres_to_add} new spheres")

            # Select positions for new spheres
            if len(poor_regions) > spheres_to_add:
                # Prioritize worst coverage areas
                sorted_indices = torch.argsort(
                    min_distances[poorly_covered], descending=True
                )
                selected_indices = sorted_indices[:spheres_to_add]
                new_centers = poor_regions[selected_indices]
            else:
                new_centers = poor_regions

            # Set appropriate radii for new spheres
            new_radii = (
                torch.ones(len(new_centers), device=self.device)
                * self.model.radii.mean()
                * 0.5
            )

            # Add new spheres
            self.model._centers = nn.Parameter(
                torch.cat([self.model._centers, new_centers], dim=0)
            )
            self.model._radii = nn.Parameter(
                torch.cat([self.model._radii, new_radii], dim=0)
            )

            return len(new_centers)

        return 0

    def prune_spheres(self, radius_threshold: float = 0.001) -> int:
        """
        Simple sphere pruning function.

        Args:
            radius_threshold: Minimum radius threshold

        Returns:
            Number of spheres removed
        """
        print("\nPruning spheres...")
        initial_count = len(self.model._radii)
        print(f"Initial sphere count: {initial_count}")

        # Find spheres to remove
        small_radius_mask = self.model._radii < radius_threshold

        # Find spheres outside mesh
        with torch.no_grad():
            centers_np = self.model._centers.detach().cpu().numpy()
            outside_mesh_mask = torch.tensor(
                ~check_mesh_contains(self.model.query_mesh, centers_np),
                device=self.device,
                dtype=torch.bool,
            )

        # Combine criteria
        prune_mask = small_radius_mask | outside_mesh_mask
        valid_indices = ~prune_mask

        # Report statistics
        spheres_removed = prune_mask.sum().item()
        print(f"Removing {spheres_removed} spheres:")
        print(f"  - Small radius: {small_radius_mask.sum().item()}")
        print(f"  - Outside mesh: {outside_mesh_mask.sum().item()}")

        # Update parameters
        self.model._centers = nn.Parameter(self.model._centers[valid_indices])
        self.model._radii = nn.Parameter(self.model._radii[valid_indices])
        self.model.num_spheres = len(self.model._radii)

        print(f"After pruning: {self.model.num_spheres} spheres remaining")

        return spheres_removed

    def update_last_density_control_iter(self, iteration: int):
        """Update the last density control iteration."""
        self.last_density_control_iter = iteration
