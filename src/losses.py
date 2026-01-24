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

        radii_sum = self.model.radii.unsqueeze(
            1) + self.model.radii.unsqueeze(0)
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
        com = (sphere_masses.unsqueeze(1) *
               self.model.centers).sum(dim=0) / total_mass
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
        center_pairwise = torch.norm(
            centers.unsqueeze(1) - centers.unsqueeze(0), dim=2)

        return inside_to_centers, surface_to_centers, center_pairwise

    def _compute_flatness_loss(self, surface_dists: torch.Tensor) -> torch.Tensor:
        """
        Encourage spheres near flat surfaces to create a flat effective boundary.
        Uses actual mesh face normals, not assumed axes.
        """
        centers = self.model.centers
        radii = self.model.radii
        normals = self.model.surface_normals  # [num_surface, 3]
        samples = self.model.surface_samples  # [num_surface, 3]

        # Cluster surface samples by normal direction to find flat faces
        face_groups = self._cluster_surface_by_normal(
            normals, samples, angle_threshold=0.1
        )

        total_flatness_loss = torch.tensor(0.0, device=self.device)

        for face_normal, face_samples in face_groups:
            # print(f"Processing face with {len(face_samples)} samples")
            if len(face_samples) < 50:
                continue

            # Compute the plane equation for this face
            # Plane: dot(p - face_point, face_normal) = 0
            face_point = face_samples.mean(dim=0)

            # Signed distance from each sphere center to the face plane
            # Positive = same side as normal, negative = opposite side
            center_to_plane = torch.sum(
                (centers - face_point) * face_normal, dim=1)

            # Find spheres that are "near" this face (could be touching it)
            # A sphere touches the face if |center_to_plane| < radius + margin
            margin = radii.mean() * 0.01
            near_face_mask = torch.abs(center_to_plane) < (radii + margin)

            if near_face_mask.sum() < 2:
                continue

            near_centers = centers[near_face_mask]
            near_radii = radii[near_face_mask]
            near_plane_dist = center_to_plane[near_face_mask]

            # The "effective surface" where each sphere meets the face plane
            # If sphere is on positive side: effective_surface = center_dist - radius
            # If sphere is on negative side: effective_surface = center_dist + radius
            # We want these to be equal (all spheres touch the same plane)

            effective_surface = torch.abs(near_plane_dist) - near_radii

            # Penalize variance in effective surface position
            # All spheres near this face should have same effective surface distance
            # if len(effective_surface) > 1:
            #     flatness_loss = torch.var(effective_surface)
            #     total_flatness_loss = total_flatness_loss + flatness_loss

            if len(effective_surface) > 1:
                # Weight by surface area
                area_weight = len(face_samples) / len(samples)
                flatness_loss = torch.var(effective_surface) * area_weight
                total_flatness_loss = total_flatness_loss + flatness_loss

        return total_flatness_loss

    def _cluster_surface_by_normal(self, normals, samples, angle_threshold=0.1):
        """
        Cluster surface samples by their normal direction.
        Returns list of (representative_normal, samples_tensor) for each flat region.

        angle_threshold: normals within this angle (radians) are considered same face
        """
        # Use cosine similarity: cos(angle_threshold) ≈ 1 - angle_threshold^2/2 for small angles
        cos_threshold = np.cos(angle_threshold)

        normals_np = normals.detach().cpu().numpy()
        samples_np = samples.detach().cpu().numpy()

        n = len(normals_np)
        assigned = np.zeros(n, dtype=bool)
        face_groups = []

        for i in range(n):
            if assigned[i]:
                continue

            # Find all samples with similar normal
            ref_normal = normals_np[i]
            similarities = np.abs(
                normals_np @ ref_normal
            )  # abs because n and -n are same plane
            similar_mask = (similarities > cos_threshold) & (~assigned)

            if similar_mask.sum() < 50:  # Skip small clusters
                assigned[i] = True
                continue

            # Mark as assigned
            assigned[similar_mask] = True

            # Compute representative normal (average of cluster)
            cluster_normals = normals_np[similar_mask]
            # Flip normals to same hemisphere before averaging
            signs = np.sign(cluster_normals @ ref_normal)
            aligned_normals = cluster_normals * signs[:, np.newaxis]
            rep_normal = aligned_normals.mean(axis=0)
            rep_normal = rep_normal / np.linalg.norm(rep_normal)

            # Get samples
            cluster_samples = samples_np[similar_mask]

            face_groups.append(
                (
                    torch.tensor(rep_normal, dtype=torch.float32,
                                 device=self.device),
                    torch.tensor(
                        cluster_samples, dtype=torch.float32, device=self.device
                    ),
                )
            )

        return face_groups

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
            "flatness_loss": self._compute_flatness_loss(surface_dists),
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
            "flatness_loss": config.flatness_weight,
        }
