"""
Loss functions for MorphIt sphere packing system.
All losses are GPU-native and vectorized — no CPU round-trips during training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Constants
FOUR_THIRDS_PI = (4.0 / 3.0) * math.pi
TWO_FIFTHS = 2.0 / 5.0


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
        self._flatness_margin = self.model.query_mesh.scale * 0.01

        # 6 axis-aligned probe directions for mesh containment
        self._probe_directions = torch.tensor(
            [[1, 0, 0], [-1, 0, 0],
             [0, 1, 0], [0, -1, 0],
             [0, 0, 1], [0, 0, -1]],
            dtype=torch.float32, device=self.device,
        )

    # ------------------------------------------------------------------
    # Individual loss functions (all GPU-native)
    # ------------------------------------------------------------------

    def _compute_coverage_loss(self, inside_dists: torch.Tensor) -> torch.Tensor:
        """Coverage loss: mean unsigned coverage gap (L1), matching the
        MorphIt paper formulation. Distances in metres are O(1e-2), so the
        L2 form shrinks gradients ~100x near the optimum; L1 keeps the
        constant-gradient pressure that the paper weights were tuned for.
        """
        sphere_coverage = inside_dists - self.model.radii.unsqueeze(0)
        min_coverage = torch.min(sphere_coverage, dim=1)[0]
        return torch.mean(torch.relu(min_coverage))

    def _compute_overlap_penalty(self, pairwise_dists: torch.Tensor) -> torch.Tensor:
        """Overlap penalty: mean linear overlap depth (L1), paper form."""
        n = self.model.centers.shape[0]
        eye_mask = torch.eye(n, device=self.device)
        dists = pairwise_dists + eye_mask * 1e6

        radii_sum = self.model.radii.unsqueeze(
            1) + self.model.radii.unsqueeze(0)
        overlap = torch.relu(radii_sum - dists)
        return torch.mean(overlap)

    def _compute_boundary_penalty(self, surface_dists: torch.Tensor) -> torch.Tensor:
        """Efficient boundary penalty using pre-computed distances."""
        sphere_coverage = surface_dists - self.model.radii.unsqueeze(0)
        return torch.mean(torch.relu(-sphere_coverage))

    def _compute_surface_loss(self, surface_dists: torch.Tensor) -> torch.Tensor:
        """Surface loss: mean |gap| to the closest sphere (L1), paper form."""
        sphere_coverage = surface_dists - self.model.radii.unsqueeze(0)
        closest_dists = torch.min(sphere_coverage, dim=1)[0]
        return torch.mean(torch.abs(closest_dists))

    def _compute_containment_loss(self, pairwise_dists: torch.Tensor) -> torch.Tensor:
        """Efficient containment loss using pre-computed distances."""
        dists = (
            pairwise_dists
            + torch.eye(len(self.model.centers), device=self.device) * 1e6
        )
        containment_depth = self.model.radii.unsqueeze(0) - (
            dists + self.model.radii.unsqueeze(1)
        )
        containment = torch.relu(containment_depth)
        return torch.mean(containment ** 2)

    def _compute_sqem_loss(self, surface_dists: torch.Tensor) -> torch.Tensor:
        """Efficient SQEM loss using pre-computed distances."""
        diff_vec = self.model.surface_samples.unsqueeze(
            1) - self.model.centers.unsqueeze(0)
        signed_dist = (
            torch.sum(diff_vec * self.model.surface_normals.unsqueeze(1), dim=2)
            - self.model.radii.unsqueeze(0)
        )
        closest_sphere_idx = torch.argmin(
            surface_dists - self.model.radii.unsqueeze(0), dim=1
        )
        closest_dist = torch.gather(
            signed_dist, 1, closest_sphere_idx.unsqueeze(1)
        ).squeeze(1)
        return torch.mean(closest_dist ** 2)

    def _compute_hausdorff_surface_loss(self, surface_dists: torch.Tensor) -> torch.Tensor:
        """
        Soft Hausdorff loss targeting the worst-covered surface samples.

        Focuses optimizer attention on the tail of the distance distribution
        so that max surface error keeps decreasing even after the mean is tiny.
        """
        sphere_coverage = surface_dists - self.model.radii.unsqueeze(0)
        closest_dists = torch.min(sphere_coverage, dim=1)[0]  # [S]

        # Target the worst 15% of surface samples
        k = max(1, int(len(closest_dists) * 0.15))
        topk_dists = torch.topk(torch.relu(closest_dists), k, largest=True)[0]
        return torch.mean(topk_dists ** 2)

    def _compute_mesh_containment_loss(self) -> torch.Tensor:
        """
        Penalize sphere surface points that protrude outside the mesh.

        Pure GPU, zero precomputation. For each probe point on each sphere's
        surface, finds the nearest mesh surface sample and checks whether the
        probe is on the outside (positive normal direction = outside).

        Cost: one cdist of [N*6, S] — same order as surface_loss.
        """
        centers = self.model.centers        # [N, 3]
        radii = self.model.radii            # [N]
        surf = self.model.surface_samples   # [S, 3]
        norms = self.model.surface_normals  # [S, 3]

        # Probe points on sphere surfaces: [N, 6, 3]
        probe_pts = (
            centers.unsqueeze(1)
            + radii.unsqueeze(1).unsqueeze(2) *
            self._probe_directions.unsqueeze(0)
        )
        pts_flat = probe_pts.reshape(-1, 3)  # [N*6, 3]

        # Find nearest surface sample for each probe point: [N*6, S]
        dists = torch.cdist(pts_flat, surf)
        nearest_idx = torch.argmin(dists, dim=1)  # [N*6]

        # Vector from nearest surface point to probe point
        nearest_surf_pt = surf[nearest_idx]        # [N*6, 3]
        nearest_normal = norms[nearest_idx]         # [N*6, 3]
        vec_to_probe = pts_flat - nearest_surf_pt   # [N*6, 3]

        # Signed distance: positive = outside the mesh surface
        signed_dist = torch.sum(vec_to_probe * nearest_normal, dim=1)  # [N*6]

        # Penalize only outside protrusion
        return torch.mean(torch.relu(signed_dist) ** 2)

    def _compute_mass_loss(self) -> torch.Tensor:
        """Sphere-pack total mass vs mesh mass."""
        sphere_volumes = FOUR_THIRDS_PI * (self.model.radii ** 3)
        total_mass = self.model.config.model.density * sphere_volumes.sum()
        return (total_mass - self.model.mesh_mass) ** 2

    def _compute_com_loss(self) -> torch.Tensor:
        """Sphere-pack CoM vs mesh CoM."""
        sphere_volumes = FOUR_THIRDS_PI * (self.model.radii ** 3)
        sphere_masses = self.model.config.model.density * sphere_volumes
        total_mass = sphere_masses.sum()
        com = (sphere_masses.unsqueeze(1) *
               self.model.centers).sum(dim=0) / total_mass
        return torch.sum((com - self.model.mesh_com) ** 2)

    def _compute_inertia_loss(self) -> torch.Tensor:
        """Parallel-axis inertia tensor, fully vectorized."""
        radii = self.model.radii
        centers = self.model.centers
        density = self.model.config.model.density

        m = density * FOUR_THIRDS_PI * (radii ** 3)

        I_body_scalar = (TWO_FIFTHS * m * radii ** 2).sum()

        c_sq = (centers * centers).sum(dim=1)
        trace_term = (m * c_sq).sum()

        weighted_centers = m.unsqueeze(1) * centers
        outer_sum = weighted_centers.T @ centers

        eye3 = torch.eye(3, device=self.device)
        I = (I_body_scalar + trace_term) * eye3 - outer_sum

        return torch.sum((I - self.model.mesh_inertia) ** 2)

    def _compute_flatness_loss(self, surface_dists: torch.Tensor) -> torch.Tensor:
        """
        Encourage spheres near flat faces to form a flush boundary.
        Clustering is cached (normals don't change) and runs on GPU.
        """
        centers = self.model.centers
        radii = self.model.radii
        samples = self.model.surface_samples

        face_groups = self._get_face_groups_cached()

        total = torch.tensor(0.0, device=self.device)
        margin = self._flatness_margin

        for face_normal, sample_indices, area_weight in face_groups:
            face_point = samples[sample_indices].mean(dim=0)
            center_to_plane = (centers - face_point) @ face_normal
            near_mask = torch.abs(center_to_plane) < (radii + margin)
            if near_mask.sum() < 2:
                continue
            near_plane_dist = center_to_plane[near_mask]
            near_radii = radii[near_mask]
            effective_surface = torch.abs(near_plane_dist) - near_radii
            total = total + torch.var(effective_surface) * area_weight

        return total

    def _get_face_groups_cached(self):
        """Return cached face groups, computing on first call."""
        if not hasattr(self, '_face_groups_cache') or self._face_groups_cache is None:
            self._face_groups_cache = self._cluster_surface_by_normal_gpu(
                self.model.surface_normals,
                self.model.surface_samples,
                angle_threshold=0.1,
            )
        return self._face_groups_cache

    def reset_flatness_cache(self):
        """Call after density control or any change to surface samples."""
        self._face_groups_cache = None

    @staticmethod
    def _cluster_surface_by_normal_gpu(normals, samples, angle_threshold=0.1):
        """GPU-native greedy clustering by normal direction."""
        device = normals.device
        n = len(normals)
        cos_threshold = math.cos(angle_threshold)
        min_cluster = max(10, int(n * 0.05))

        assigned = torch.zeros(n, dtype=torch.bool, device=device)
        face_groups = []
        total_samples = float(n)

        for i in range(n):
            if assigned[i]:
                continue
            ref = normals[i]
            sims = torch.abs(normals @ ref)
            similar = (sims > cos_threshold) & (~assigned)
            if similar.sum().item() < min_cluster:
                assigned[i] = True
                continue
            assigned[similar] = True
            cluster_normals = normals[similar]
            signs = torch.sign(cluster_normals @ ref)
            aligned = cluster_normals * signs.unsqueeze(1)
            rep = aligned.mean(dim=0)
            rep = rep / rep.norm()
            indices = similar.nonzero(as_tuple=False).squeeze(1)
            area_weight = len(indices) / total_samples
            face_groups.append((rep, indices, area_weight))

        return face_groups

    # ------------------------------------------------------------------
    # Distance matrix precomputation
    # ------------------------------------------------------------------

    def _compute_distance_matrices(self):
        """Pre-compute distance matrices used across multiple loss functions."""
        centers = self.model.centers
        inside_to_centers = torch.cdist(self.model.inside_samples, centers)
        surface_to_centers = torch.cdist(self.model.surface_samples, centers)
        center_pairwise = torch.cdist(centers, centers)
        return inside_to_centers, surface_to_centers, center_pairwise

    # ------------------------------------------------------------------
    # Aggregate interface
    # ------------------------------------------------------------------

    def compute_all_losses(self, weights: Dict[str, float] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components efficiently using pre-computed distances.
        Losses with weight == 0 are skipped entirely (returned as zero tensor)
        so unused losses don't pay their compute cost. Pass weights=None to
        force every loss to be computed.
        """
        inside_dists, surface_dists, pairwise_dists = self._compute_distance_matrices()

        zero = torch.tensor(0.0, device=self.device)

        def _skip(name):
            return weights is not None and weights.get(name, 0.0) == 0.0

        losses = {
            "coverage_loss":         zero if _skip("coverage_loss") else self._compute_coverage_loss(inside_dists),
            "overlap_penalty":       zero if _skip("overlap_penalty") else self._compute_overlap_penalty(pairwise_dists),
            "boundary_penalty":      zero if _skip("boundary_penalty") else self._compute_boundary_penalty(surface_dists),
            "surface_loss":          zero if _skip("surface_loss") else self._compute_surface_loss(surface_dists),
            "containment_loss":      zero if _skip("containment_loss") else self._compute_containment_loss(pairwise_dists),
            "sqem_loss":             zero if _skip("sqem_loss") else self._compute_sqem_loss(surface_dists),
            "hausdorff_loss":        zero if _skip("hausdorff_loss") else self._compute_hausdorff_surface_loss(surface_dists),
            "mesh_containment_loss": zero if _skip("mesh_containment_loss") else self._compute_mesh_containment_loss(),
            "mass_loss":             zero if _skip("mass_loss") else self._compute_mass_loss(),
            "com_loss":              zero if _skip("com_loss") else self._compute_com_loss(),
            "inertia_loss":          zero if _skip("inertia_loss") else self._compute_inertia_loss(),
            "flatness_loss":         zero if _skip("flatness_loss") else self._compute_flatness_loss(surface_dists),
        }

        return losses

    def compute_weighted_total_loss(self, weights: Dict[str, float]) -> torch.Tensor:
        """Compute weighted total loss (skips zero-weight losses)."""
        losses = self.compute_all_losses(weights=weights)
        total_loss = torch.tensor(0.0, device=self.device)
        for loss_name, loss_value in losses.items():
            if loss_name in weights and weights[loss_name] != 0.0:
                total_loss += weights[loss_name] * loss_value
        return total_loss

    def get_loss_weights_from_config(self, config) -> Dict[str, float]:
        """Extract loss weights from configuration."""
        return {
            "coverage_loss": config.coverage_weight,
            "overlap_penalty": config.overlap_weight,
            "boundary_penalty": config.boundary_weight,
            "surface_loss": config.surface_weight,
            "containment_loss": config.containment_weight,
            "sqem_loss": config.sqem_weight,
            "hausdorff_loss": config.hausdorff_weight,
            "mesh_containment_loss": config.mesh_containment_weight,
            "mass_loss": config.mass_weight,
            "com_loss": config.com_weight,
            "inertia_loss": config.inertia_weight,
            "flatness_loss": config.flatness_weight,
        }
