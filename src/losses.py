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

    def compute_coverage_loss(self) -> torch.Tensor:
        """
        Compute coverage loss - penalizes areas inside mesh not covered by spheres.

        Returns:
            Coverage loss tensor
        """
        centers = self.model.centers
        radii = self.model.radii

        # Compute distances from inside samples to sphere centers
        dists = torch.norm(
            self.model.inside_samples.unsqueeze(1) - centers.unsqueeze(0), dim=2
        )

        # Check sphere coverage (negative means inside sphere)
        sphere_coverage = dists - radii.unsqueeze(0)
        min_coverage = torch.min(sphere_coverage, dim=1)[0]

        # Penalize uncovered areas (positive distances)
        return torch.mean(torch.relu(min_coverage))

    def compute_overlap_penalty(self) -> torch.Tensor:
        """
        Compute penalty for overlapping spheres.

        Returns:
            Overlap penalty tensor
        """
        centers = self.model.centers
        radii = self.model.radii

        # Compute pairwise distances between sphere centers
        center_diffs = centers.unsqueeze(1) - centers.unsqueeze(0)  # [n, n, 3]
        dists = torch.norm(center_diffs, dim=2)  # [n, n]

        # Mask diagonal to ignore self-overlaps
        n = centers.shape[0]
        eye_mask = torch.eye(n, device=self.device)
        dists = dists + eye_mask * 1000.0  # Add large value to diagonal

        # Compute overlap (positive when spheres overlap)
        radii_sum = radii.unsqueeze(1) + radii.unsqueeze(0)  # [n, n]
        overlap = torch.relu(radii_sum - dists)

        return torch.mean(overlap)

    def compute_boundary_penalty(self) -> torch.Tensor:
        """
        Compute penalty for spheres extending beyond mesh boundary.

        Returns:
            Boundary penalty tensor
        """
        # Use point-based boundary loss for efficiency
        dists = torch.norm(
            self.model.surface_samples.unsqueeze(1) - self.model.centers.unsqueeze(0),
            dim=2,
        )
        sphere_coverage = dists - self.model.radii.unsqueeze(0)

        # Penalize spheres that extend beyond surface (negative coverage)
        return torch.mean(torch.relu(-sphere_coverage))

    def compute_surface_loss(self) -> torch.Tensor:
        """
        Compute surface loss using pre-sampled points on mesh surface.

        Returns:
            Surface loss tensor
        """
        centers = self.model.centers
        radii = self.model.radii
        samples = self.model.surface_samples

        # Compute distances from surface samples to all spheres
        dists = torch.norm(
            samples.unsqueeze(1) - centers.unsqueeze(0), dim=2
        ) - radii.unsqueeze(0)

        # Find closest sphere for each sample
        closest_sphere_idx = torch.argmin(dists, dim=1)
        closest_dists = torch.gather(dists, 1, closest_sphere_idx.unsqueeze(1)).squeeze(
            1
        )

        # Return mean absolute distance to closest sphere
        return torch.mean(torch.abs(closest_dists))

    def compute_containment_loss(self) -> torch.Tensor:
        """
        Compute containment loss - penalizes spheres contained within other spheres.

        Returns:
            Containment loss tensor
        """
        centers = self.model.centers
        radii = self.model.radii

        # Calculate pairwise distances between sphere centers
        dists = torch.norm(centers.unsqueeze(1) - centers.unsqueeze(0), dim=2)

        # Mask diagonal to avoid self-containment
        dists = dists + torch.eye(len(centers), device=self.device) * 1000

        # Check containment: sphere i is contained in sphere j if
        # distance(i,j) + radius(i) <= radius(j)
        containment_depth = radii.unsqueeze(0) - (dists + radii.unsqueeze(1))

        # Apply ReLU to only count positive containment
        containment = torch.relu(containment_depth)

        # Square the containment to strongly penalize deeper containment
        return torch.mean(containment**2)

    def compute_sqem_loss(self) -> torch.Tensor:
        """
        Compute SQEM (Squared Quadratic Error Metric) loss.

        Based on: https://dl.acm.org/doi/10.1145/2508363.2508384
        Uses face normals and squared error for better surface fitting.

        Returns:
            SQEM loss tensor
        """
        centers = self.model.centers
        radii = self.model.radii
        samples = self.model.surface_samples

        # Get normalized surface normals
        normals = self.model.surface_normals
        normals = normals / torch.norm(normals, dim=1, keepdim=True)

        # Compute direction vectors from samples to sphere centers
        diff_vec = samples.unsqueeze(1) - centers.unsqueeze(0)

        # Compute signed distance using normal projection
        signed_dist = torch.sum(
            diff_vec * normals.unsqueeze(1), dim=2
        ) - radii.unsqueeze(0)

        # Find closest sphere for each surface sample
        closest_sphere_idx = torch.argmin(
            torch.norm(diff_vec, dim=2) - radii.unsqueeze(0), dim=1
        )

        # Get signed distance to closest sphere
        closest_dist = torch.gather(
            signed_dist, 1, closest_sphere_idx.unsqueeze(1)
        ).squeeze(1)

        # Return mean squared distance
        return torch.mean(closest_dist**2)

    def compute_sqem_loss_sphere_wise(self) -> torch.Tensor:
        """
        Compute sphere-wise SQEM loss.

        Alternative version that computes loss per sphere rather than per surface point.

        Returns:
            Sphere-wise SQEM loss tensor
        """
        centers = self.model.centers
        radii = self.model.radii
        samples = self.model.surface_samples

        # Get normalized surface normals
        normals = self.model.surface_normals
        normals = normals / torch.norm(normals, dim=1, keepdim=True)

        # Compute direction vectors [num_spheres, num_samples, 3]
        diff_vec = samples.unsqueeze(0) - centers.unsqueeze(1)

        # Compute signed distance using normal projection
        signed_dist = torch.sum(
            diff_vec * normals.unsqueeze(0), dim=2
        ) - radii.unsqueeze(1)

        # Find closest sample for each sphere
        closest_sample_idx = torch.argmin(
            torch.norm(diff_vec, dim=2) - radii.unsqueeze(1), dim=1
        )

        # Get signed distance to closest sample
        closest_dist = torch.gather(
            signed_dist, 1, closest_sample_idx.unsqueeze(1)
        ).squeeze(1)

        # Return mean squared distance
        return torch.mean(closest_dist**2)

    def compute_all_losses(self) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Returns:
            Dictionary of loss components
        """
        losses = {
            "coverage_loss": self.compute_coverage_loss(),
            "overlap_penalty": self.compute_overlap_penalty(),
            "boundary_penalty": self.compute_boundary_penalty(),
            "surface_loss": self.compute_surface_loss(),
            "containment_loss": self.compute_containment_loss(),
            "sqem_loss": self.compute_sqem_loss(),
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
        }
