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

        # Normalization constants for the three physics losses.
        # Each loss is divided by a fixed ground-truth magnitude so that
        # weight=1 has the same gradient pressure across objects of any
        # scale: "100 % relative error => loss=1". Without this the
        # raw losses span 20+ orders of magnitude across our test set
        # (bunny ‖I‖² ≈ 1e-4, vase ‖I‖² ≈ 1e25), so a single weight
        # cannot work for both. Constants only depend on the mesh
        # ground truth, so we cache them once at construction.
        _EPS = 1e-20
        self._mass_norm_sq = float(self.model.mesh_mass) ** 2 + _EPS
        self._com_norm_sq = float(self.model.query_mesh.scale) ** 2 + _EPS
        self._inertia_norm_sq = float(
            torch.sum(self.model.mesh_inertia ** 2).item()) + _EPS

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

    def _compute_mesh_containment_loss(
        self, surface_dists: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize sphere centers outside the mesh.

        Center-only signed-distance form: for each sphere center, locate
        its nearest surface sample (argmin over the already-built
        ``surface_dists`` matrix) and project the offset onto that
        sample's normal. Positive ⇒ center is outside ⇒ squared
        ReLU penalty.

        This replaces the previous 6-probe heuristic. The center-only
        form is

            • cheaper — argmin + a few elementwise ops on the existing
              ``surface_dists`` [S, N], no new ``cdist``;
            • geometrically proper — a true signed-distance-from-surface
              quantity rather than 6 axis-aligned probe heuristics;
            • equivalent in gradient magnitude on fully-escaped spheres
              (6 probes × 1/(N·6) = 1/N, same as the center-only term).

        Surface-protrusion (sphere bulging past the boundary with center
        still inside) is already handled by ``boundary_penalty`` (engulfed
        surface samples) and ``sqem_loss`` (squared surface mismatch), so
        nothing of value is lost in the swap.
        """
        centers = self.model.centers        # [N, 3]
        surf = self.model.surface_samples   # [S, 3]
        norms = self.model.surface_normals  # [S, 3]

        # surface_dists is [S, N]; argmin along S gives nearest sample
        # for each center. detach so the argmin doesn't try to backprop.
        nearest_idx = surface_dists.argmin(dim=0).detach()  # [N]
        nearest_surf = surf[nearest_idx]                     # [N, 3]
        nearest_normal = norms[nearest_idx]                  # [N, 3]

        vec = centers - nearest_surf                         # [N, 3]
        signed_dist = (vec * nearest_normal).sum(dim=1)      # [N]

        return torch.mean(torch.relu(signed_dist) ** 2)

    def _compute_mass_loss(self) -> torch.Tensor:
        """Relative mass error squared: ((m_sphere - m_mesh) / m_mesh)².

        Dimensionless. Weight=1 ↔ a 100 % mass error contributes 1.0 to
        the total loss, so the same weight is meaningful across objects
        of very different scales (bunny ≈ 2 kg vs vase ≈ 1e9 kg).

        Reads from model.masses (uniform-density default: density × volume).
        """
        total_mass = self.model.masses.sum()
        return (total_mass - self.model.mesh_mass) ** 2 / self._mass_norm_sq

    def _compute_com_loss(self) -> torch.Tensor:
        """Squared COM offset normalized by mesh-bbox-diagonal squared.

        Returns ‖com_sphere − com_mesh‖² / mesh.scale². Dimensionless.
        Weight=1 ↔ a COM offset equal to one full bounding-box diagonal
        contributes 1.0; a 1 % offset contributes 1e-4.
        """
        sphere_masses = self.model.masses
        total_mass = sphere_masses.sum()
        com = (sphere_masses.unsqueeze(1) *
               self.model.centers).sum(dim=0) / total_mass
        return torch.sum((com - self.model.mesh_com) ** 2) / self._com_norm_sq

    def _compute_inertia_loss(self) -> torch.Tensor:
        """Frobenius-relative inertia error, computed about the sphere
        body's own COM so the comparison is frame-consistent with
        ``trimesh.moment_inertia`` (which is about the mesh centroid).

        Earlier version expressed sphere centers in the WORLD frame and
        applied parallel-axis from the origin; this introduced a
        systematic ~m·d² term unrelated to packing quality whenever
        mesh COM ≠ origin. For the vase (z-COM = 73 m) the bias was
        ~3.8× the true Frobenius gap. Centering on the sphere COM
        removes the bias; the residual is then the true intrinsic
        inertia mismatch between the two bodies.

        Returns ‖I_sphere(COM) − I_mesh(COM)‖²_F / ‖I_mesh‖²_F —
        dimensionless. Weight=1 ↔ a 100 % Frobenius error gives 1.0.
        """
        radii = self.model.radii
        centers = self.model.centers
        m = self.model.masses

        sphere_com = (m.unsqueeze(1) * centers).sum(dim=0) / m.sum()
        centered = centers - sphere_com  # express each center wrt sphere COM

        I_body_scalar = (TWO_FIFTHS * m * radii ** 2).sum()
        c_sq = (centered * centered).sum(dim=1)
        trace_term = (m * c_sq).sum()
        weighted_centers = m.unsqueeze(1) * centered
        outer_sum = weighted_centers.T @ centered

        eye3 = torch.eye(3, device=self.device)
        I = (I_body_scalar + trace_term) * eye3 - outer_sum

        return torch.sum((I - self.model.mesh_inertia) ** 2) / self._inertia_norm_sq

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
            "mesh_containment_loss": zero if _skip("mesh_containment_loss") else self._compute_mesh_containment_loss(surface_dists),
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
