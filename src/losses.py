"""
Loss functions for MorphIt sphere packing system.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any


class MorphItLosses:
    """Collection of loss functions for MorphIt sphere packing."""

    def __init__(self, model):
        """
        Initialize loss functions.

        Args:
            model: MorphIt model instance
        """
        self.model = model
        self.device = model.device

    def _compute_coverage_loss(self, inside_dists: torch.Tensor) -> torch.Tensor:
        """Efficient coverage loss using pre-computed distances."""
        sphere_coverage = inside_dists - self.model.radii.unsqueeze(0)
        min_coverage = torch.min(sphere_coverage, dim=1)[0]
        return torch.mean(torch.relu(min_coverage))

    def _compute_overlap_penalty(self, pairwise_dists: torch.Tensor) -> torch.Tensor:
        """Efficient overlap penalty using pre-computed distances."""
        # Mask diagonal
        n = self.model.centers.shape[0]
        eye_mask = torch.eye(n, device=self.device)
        dists = pairwise_dists + eye_mask * 1000.0

        radii_sum = self.model.radii.unsqueeze(1) + self.model.radii.unsqueeze(0)
        overlap = torch.relu(radii_sum - dists)
        return torch.mean(overlap)

    def _compute_boundary_penalty(self, surface_dists: torch.Tensor) -> torch.Tensor:
        """Efficient boundary penalty using pre-computed distances."""
        sphere_coverage = surface_dists - self.model.radii.unsqueeze(0)
        return torch.mean(torch.relu(-sphere_coverage))

    def _compute_surface_loss(self, surface_dists: torch.Tensor) -> torch.Tensor:
        """Efficient surface loss using pre-computed distances."""
        sphere_coverage = surface_dists - self.model.radii.unsqueeze(0)
        closest_dists = torch.min(sphere_coverage, dim=1)[0]
        return torch.mean(torch.abs(closest_dists))

    def _compute_containment_loss(self, pairwise_dists: torch.Tensor) -> torch.Tensor:
        """Efficient containment loss using pre-computed distances."""
        # Mask diagonal
        dists = (
            pairwise_dists
            + torch.eye(len(self.model.centers), device=self.device) * 1000
        )

        containment_depth = self.model.radii.unsqueeze(0) - (
            dists + self.model.radii.unsqueeze(1)
        )
        containment = torch.relu(containment_depth)
        return torch.mean(containment**2)

    def _compute_sqem_loss(self, surface_dists: torch.Tensor) -> torch.Tensor:
        """Efficient SQEM loss using pre-computed distances."""
        # Direction vectors from samples to sphere centers
        diff_vec = self.model.surface_samples.unsqueeze(
            1
        ) - self.model.centers.unsqueeze(0)

        # Compute signed distance using normal projection
        signed_dist = torch.sum(
            diff_vec * self.model.surface_normals.unsqueeze(1), dim=2
        ) - self.model.radii.unsqueeze(0)

        # Find closest sphere for each surface sample
        closest_sphere_idx = torch.argmin(
            surface_dists - self.model.radii.unsqueeze(0), dim=1
        )

        # Get signed distance to closest sphere
        closest_dist = torch.gather(
            signed_dist, 1, closest_sphere_idx.unsqueeze(1)
        ).squeeze(1)

        return torch.mean(closest_dist**2)

    def _compute_mass_loss(self) -> torch.Tensor:
        sphere_masses = (4.0 / 3.0) * np.pi * (self.model.radii**3)
        total_mass = sphere_masses.sum()
        return (total_mass - self.model.mesh_mass) ** 2

    def _compute_com_loss(self) -> torch.Tensor:
        sphere_masses = (4.0 / 3.0) * np.pi * (self.model.radii**3)
        total_mass = sphere_masses.sum()
        com = (sphere_masses.unsqueeze(1) * self.model.centers).sum(dim=0) / total_mass
        return torch.sum((com - self.model.mesh_com) ** 2)

    def _compute_inertia_loss(self) -> torch.Tensor:
        radii = self.model.radii
        centers = self.model.centers
        m = (4.0 / 3.0) * np.pi * (radii**3)

        I = torch.zeros(3, 3, device=self.device)
        eye3 = torch.eye(3, device=self.device)
        I_diag = (2.0 / 5.0) * m * (radii**2)

        for i in range(len(centers)):
            ci = centers[i]
            I_center = eye3 * I_diag[i]
            ci_sq = (ci * ci).sum()
            I_parallel = m[i] * (ci_sq * eye3 - torch.outer(ci, ci))
            I += I_center + I_parallel

        return torch.sum((I - self.model.mesh_inertia) ** 2)

    def _compute_distance_matrices(self):
        """Pre-compute distance matrices used across multiple loss functions."""
        centers = self.model.centers

        # Distance from inside samples to all sphere centers [num_inside, num_spheres]
        inside_to_centers = torch.norm(
            self.model.inside_samples.unsqueeze(1) - centers.unsqueeze(0), dim=2
        )

        # Distance from surface samples to all sphere centers [num_surface, num_spheres]
        surface_to_centers = torch.norm(
            self.model.surface_samples.unsqueeze(1) - centers.unsqueeze(0), dim=2
        )

        # Pairwise distances between sphere centers [num_spheres, num_spheres]
        center_pairwise = torch.norm(centers.unsqueeze(1) - centers.unsqueeze(0), dim=2)

        return inside_to_centers, surface_to_centers, center_pairwise

    def compute_all_losses(self) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components efficiently using pre-computed distances.
        """
        # Pre-compute all distance matrices once
        inside_dists, surface_dists, pairwise_dists = self._compute_distance_matrices()

        losses = {
            "coverage_loss": self._compute_coverage_loss(inside_dists),
            "overlap_penalty": self._compute_overlap_penalty(pairwise_dists),
            "boundary_penalty": self._compute_boundary_penalty(surface_dists),
            "surface_loss": self._compute_surface_loss(surface_dists),
            "containment_loss": self._compute_containment_loss(pairwise_dists),
            "sqem_loss": self._compute_sqem_loss(surface_dists),
            "mass_loss": self._compute_mass_loss(),
            "com_loss": self._compute_com_loss(),
            "inertia_loss": self._compute_inertia_loss(),
        }

        return losses

    def compute_weighted_total_loss(self, weights: Dict[str, float]) -> torch.Tensor:
        """
        Compute weighted total loss.

        Args:
            weights: Dictionary of loss weights

        Returns:
            Weighted total loss tensor
        """
        losses = self.compute_all_losses()

        total_loss = torch.tensor(0.0, device=self.device)
        for loss_name, loss_value in losses.items():
            if loss_name in weights:
                total_loss += weights[loss_name] * loss_value

        return total_loss

    def get_loss_weights_from_config(self, config) -> Dict[str, float]:
        """
        Extract loss weights from configuration.

        Args:
            config: Training configuration

        Returns:
            Dictionary of loss weights
        """
        return {
            "coverage_loss": config.coverage_weight,
            "overlap_penalty": config.overlap_weight,
            "boundary_penalty": config.boundary_weight,
            "surface_loss": config.surface_weight,
            "containment_loss": config.containment_weight,
            "sqem_loss": config.sqem_weight,
            "mass_loss": config.mass_weight,
            "com_loss": config.com_weight,
            "inertia_loss": config.inertia_weight,
        }
