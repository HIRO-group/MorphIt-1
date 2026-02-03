"""
Adaptive density control module for MorphIt sphere packing.
Handles sphere pruning and addition based on quality metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from inside_mesh import check_mesh_contains


@dataclass
class PruningConfig:
    """Configuration for sphere pruning."""
    min_radius_threshold: float = 0.005
    # Prune if badness exceeds this (even if at target count)
    # badness_threshold: float = 5000.0

    # Badness weights
    outside_mesh_penalty: float = 1000.0
    tiny_radius_penalty: float = 500.0
    contained_penalty: float = 800.0
    low_surface_contribution_weight: float = 100.0
    low_unique_coverage_weight: float = 20.0
    far_from_surface_weight: float = 20.0
    # Distance threshold to consider "far"
    far_from_surface_threshold: float = 0.05


@dataclass
class AddingConfig:
    """Configuration for sphere adding."""
    surface_coverage_threshold: float = 0.002  # Surface points farther than this are "uncovered"
    # Volume points farther than this are "uncovered"
    volume_coverage_threshold: float = 0.02
    min_separation_factor: float = 0.5        # min_separation = mean_radius * this
    radius_gap_factor: float = 0.5            # new_radius = local_gap * this

    # Candidate scoring weights
    surface_priority_weight: float = 5.0      # Surface coverage matters more
    volume_priority_weight: float = 1.0
    spread_weight: float = 0.5                # Reward spread from existing spheres

    # How many random exploration candidates to add
    num_random_candidates: int = 50


class SphereScorer:
    """
    Computes quality/badness scores for spheres to guide pruning decisions.
    """

    def __init__(self, model, config: PruningConfig):
        """
        Initialize sphere scorer.

        Args:
            model: MorphIt model instance
            config: Pruning configuration
        """
        self.model = model
        self.config = config
        self.device = model.device

    def compute_all_badness_scores(self) -> torch.Tensor:
        """
        Compute badness score for each sphere.
        Higher score = worse sphere = more likely to be pruned.

        Returns:
            Tensor of shape [num_spheres] with badness scores
        """
        num_spheres = len(self.model.radii)
        scores = torch.zeros(num_spheres, device=self.device)

        with torch.no_grad():
            # Hard criteria
            scores += self._score_outside_mesh()
            scores += self._score_tiny_radius()
            scores += self._score_contained_spheres()

            # Soft criteria
            scores += self._score_surface_contribution()
            scores += self._score_unique_coverage()
            scores += self._score_far_from_surface()

        return scores

    def _score_outside_mesh(self) -> torch.Tensor:
        """Penalize spheres with centers outside the mesh."""
        centers_np = self.model.centers.detach().cpu().numpy()
        inside_mask = check_mesh_contains(self.model.query_mesh, centers_np)
        outside_mask = torch.tensor(
            ~inside_mask, device=self.device, dtype=torch.float32)
        return outside_mask * self.config.outside_mesh_penalty

    def _score_tiny_radius(self) -> torch.Tensor:
        """Penalize spheres with very small radius."""
        tiny_mask = (self.model.radii <
                     self.config.min_radius_threshold).float()
        return tiny_mask * self.config.tiny_radius_penalty

    def _score_contained_spheres(self) -> torch.Tensor:
        """Penalize spheres that are fully contained inside another sphere."""
        centers = self.model.centers
        radii = self.model.radii
        num_spheres = len(radii)

        # Pairwise distances between centers
        pairwise_dists = torch.cdist(centers.unsqueeze(
            0), centers.unsqueeze(0)).squeeze(0)

        # Sphere j is contained in sphere i if: radius_i > dist_ij + radius_j
        # Rearranged: radius_i - dist_ij - radius_j > 0
        # containment_depth[i,j] = how much sphere j is inside sphere i
        containment_depth = radii.unsqueeze(
            0) - pairwise_dists - radii.unsqueeze(1)

        # Mask diagonal (sphere can't contain itself)
        eye_mask = torch.eye(num_spheres, device=self.device) * -1000
        containment_depth = containment_depth + eye_mask

        # For each sphere j, check if ANY sphere i contains it
        max_containment, _ = torch.max(containment_depth, dim=0)
        contained_mask = (max_containment > 0).float()

        return contained_mask * self.config.contained_penalty

    def _score_surface_contribution(self) -> torch.Tensor:
        """
        Penalize spheres that contribute little to surface approximation.
        Surface contribution = number of surface points where this sphere is closest.
        """
        centers = self.model.centers
        radii = self.model.radii
        surface_samples = self.model.surface_samples

        # Distance from each surface sample to each sphere surface
        dists_to_centers = torch.cdist(surface_samples, centers)
        dists_to_surface = dists_to_centers - radii.unsqueeze(0)

        # Find closest sphere for each surface sample
        closest_sphere_idx = torch.argmin(dists_to_surface, dim=1)

        # Count how many surface samples each sphere is closest to
        contribution_counts = torch.bincount(
            closest_sphere_idx,
            minlength=len(radii)
        ).float()

        # Normalize by total surface samples
        contribution_ratio = contribution_counts / len(surface_samples)

        # Low contribution = high penalty
        # Using 1/(x + epsilon) so spheres with 0 contribution get high penalty
        penalty = 1.0 / (contribution_ratio + 0.01)

        return penalty * self.config.low_surface_contribution_weight

    def _score_unique_coverage(self) -> torch.Tensor:
        """
        Penalize spheres that don't uniquely cover any interior points.
        Unique coverage = points covered ONLY by this sphere.
        """
        centers = self.model.centers
        radii = self.model.radii
        inside_samples = self.model.inside_samples

        # Distance from each inside sample to each sphere surface
        dists_to_centers = torch.cdist(inside_samples, centers)
        dists_to_surface = dists_to_centers - radii.unsqueeze(0)

        # Which spheres cover each point? (point is inside sphere if dist_to_surface < 0)
        covered_by = (dists_to_surface < 0)  # [num_inside, num_spheres]

        # Count how many spheres cover each point
        coverage_count = covered_by.sum(dim=1)  # [num_inside]

        # Find points covered by exactly one sphere
        uniquely_covered_mask = (coverage_count == 1)  # [num_inside]

        # For each sphere, count uniquely covered points
        unique_counts = torch.zeros(len(radii), device=self.device)
        for sphere_idx in range(len(radii)):
            # Points that are uniquely covered AND covered by this sphere
            unique_by_this = uniquely_covered_mask & covered_by[:, sphere_idx]
            unique_counts[sphere_idx] = unique_by_this.sum()

        # Normalize
        unique_ratio = unique_counts / max(len(inside_samples), 1)

        # Low unique coverage = high penalty
        penalty = 1.0 / (unique_ratio + 0.01)

        return penalty * self.config.low_unique_coverage_weight

    def _score_far_from_surface(self) -> torch.Tensor:
        """
        Penalize spheres that are far from mesh surface.
        These are deep interior spheres that don't help with surface approximation.
        """
        centers = self.model.centers
        surface_samples = self.model.surface_samples

        # Distance from each sphere center to nearest surface point
        dists_to_surface_samples = torch.cdist(centers, surface_samples)
        min_dist_to_surface, _ = torch.min(dists_to_surface_samples, dim=1)

        # Only penalize if beyond threshold
        excess_distance = torch.relu(
            min_dist_to_surface - self.config.far_from_surface_threshold)

        return excess_distance * self.config.far_from_surface_weight


class CandidateGenerator:
    """
    Generates and scores candidate positions for new spheres.
    """

    def __init__(self, model, config: AddingConfig):
        """
        Initialize candidate generator.

        Args:
            model: MorphIt model instance
            config: Adding configuration
        """
        self.model = model
        self.config = config
        self.device = model.device

    def generate_candidates(self) -> Dict[str, torch.Tensor]:
        """
        Generate candidate positions for new spheres.

        Returns:
            Dictionary with:
                - 'positions': [N, 3] tensor of candidate positions
                - 'priorities': [N] tensor of priority scores (higher = more needed)
                - 'sources': list of source labels for debugging
        """
        all_positions = []
        all_priorities = []
        all_sources = []

        with torch.no_grad():
            # Source 1: Poorly covered surface points (HIGH PRIORITY)
            surface_positions, surface_priorities = self._get_surface_candidates()
            if len(surface_positions) > 0:
                all_positions.append(surface_positions)
                all_priorities.append(
                    surface_priorities * self.config.surface_priority_weight)
                all_sources.extend(['surface'] * len(surface_positions))

            # Source 2: Poorly covered volume points (LOWER PRIORITY)
            volume_positions, volume_priorities = self._get_volume_candidates()
            if len(volume_positions) > 0:
                all_positions.append(volume_positions)
                all_priorities.append(
                    volume_priorities * self.config.volume_priority_weight)
                all_sources.extend(['volume'] * len(volume_positions))

            # Source 3: Random exploration candidates
            random_positions = self._get_random_candidates()
            if len(random_positions) > 0:
                random_priorities = torch.rand(
                    len(random_positions), device=self.device) * 0.1
                all_positions.append(random_positions)
                all_priorities.append(random_priorities)
                all_sources.extend(['random'] * len(random_positions))

        if len(all_positions) == 0:
            return {
                'positions': torch.empty((0, 3), device=self.device),
                'priorities': torch.empty(0, device=self.device),
                'sources': []
            }

        return {
            'positions': torch.cat(all_positions, dim=0),
            'priorities': torch.cat(all_priorities, dim=0),
            'sources': all_sources
        }

    def _get_surface_candidates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get candidates from poorly covered surface points."""
        centers = self.model.centers
        radii = self.model.radii
        surface_samples = self.model.surface_samples

        # Distance from each surface sample to nearest sphere surface
        dists_to_centers = torch.cdist(surface_samples, centers)
        dists_to_sphere_surface = dists_to_centers - radii.unsqueeze(0)
        min_dists, _ = torch.min(dists_to_sphere_surface, dim=1)

        # Find poorly covered surface points
        poorly_covered_mask = min_dists > self.config.surface_coverage_threshold

        if poorly_covered_mask.sum() == 0:
            return torch.empty((0, 3), device=self.device), torch.empty(0, device=self.device)

        positions = surface_samples[poorly_covered_mask]
        # Higher distance = higher priority
        priorities = min_dists[poorly_covered_mask]

        return positions, priorities

    def _get_volume_candidates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get candidates from poorly covered interior points."""
        centers = self.model.centers
        radii = self.model.radii
        inside_samples = self.model.inside_samples

        # Distance from each inside sample to nearest sphere surface
        dists_to_centers = torch.cdist(inside_samples, centers)
        dists_to_sphere_surface = dists_to_centers - radii.unsqueeze(0)
        min_dists, _ = torch.min(dists_to_sphere_surface, dim=1)

        # Find poorly covered interior points
        poorly_covered_mask = min_dists > self.config.volume_coverage_threshold

        if poorly_covered_mask.sum() == 0:
            return torch.empty((0, 3), device=self.device), torch.empty(0, device=self.device)

        positions = inside_samples[poorly_covered_mask]
        priorities = min_dists[poorly_covered_mask]

        return positions, priorities

    def _get_random_candidates(self) -> torch.Tensor:
        """Get random candidates inside mesh for exploration."""
        num_candidates = self.config.num_random_candidates

        # Sample from inside_samples (already guaranteed to be inside mesh)
        if len(self.model.inside_samples) <= num_candidates:
            return self.model.inside_samples.clone()

        indices = torch.randperm(len(self.model.inside_samples))[
            :num_candidates]
        return self.model.inside_samples[indices]

    def score_candidates(
        self,
        candidates: Dict[str, torch.Tensor],
        existing_centers: torch.Tensor
    ) -> torch.Tensor:
        """
        Score candidates considering spread from existing spheres.

        Args:
            candidates: Output from generate_candidates()
            existing_centers: Current sphere centers

        Returns:
            Final scores for each candidate (higher = better)
        """
        positions = candidates['positions']
        priorities = candidates['priorities']

        if len(positions) == 0:
            return torch.empty(0, device=self.device)

        # Start with coverage priority
        scores = priorities.clone()

        # Add spread bonus (distance from existing centers)
        if len(existing_centers) > 0:
            dists_to_existing = torch.cdist(positions, existing_centers)
            min_dist_to_existing, _ = torch.min(dists_to_existing, dim=1)
            scores += min_dist_to_existing * self.config.spread_weight

        return scores

    def select_positions_with_diversity(
        self,
        candidates: Dict[str, torch.Tensor],
        scores: torch.Tensor,
        num_to_select: int,
        min_separation: float,
        existing_centers: torch.Tensor
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Select candidate positions with minimum separation constraint.

        Args:
            candidates: Output from generate_candidates()
            scores: Candidate scores from score_candidates()
            num_to_select: Number of positions to select
            min_separation: Minimum distance between selected positions
            existing_centers: Current sphere centers

        Returns:
            Tuple of (selected_positions [M, 3], selected_sources list)
        """
        positions = candidates['positions']
        sources = candidates['sources']

        if len(positions) == 0:
            return torch.empty((0, 3), device=self.device), []

        # Sort by score (descending)
        sorted_indices = torch.argsort(scores, descending=True)

        selected_positions = []
        selected_sources = []
        selected_tensor = torch.empty((0, 3), device=self.device)

        for idx in sorted_indices:
            if len(selected_positions) >= num_to_select:
                break

            position = positions[idx]

            # Check separation from existing spheres
            if len(existing_centers) > 0:
                dists_to_existing = torch.norm(
                    existing_centers - position.unsqueeze(0), dim=1)
                if torch.min(dists_to_existing) < min_separation:
                    continue

            # Check separation from already selected positions
            if len(selected_positions) > 0:
                dists_to_selected = torch.norm(
                    selected_tensor - position.unsqueeze(0), dim=1)
                if torch.min(dists_to_selected) < min_separation:
                    continue

            # Accept this candidate
            selected_positions.append(position)
            selected_sources.append(sources[idx.item()])
            selected_tensor = torch.cat(
                [selected_tensor, position.unsqueeze(0)], dim=0)

        if len(selected_positions) == 0:
            return torch.empty((0, 3), device=self.device), []

        return torch.stack(selected_positions), selected_sources


class DensityController:
    """
    Adaptive density control for sphere packing optimization.
    Maintains target sphere count while optimizing sphere quality.
    """

    def __init__(self, model, config):
        """
        Initialize density controller.

        Args:
            model: MorphIt model instance
            config: MorphIt configuration (contains target sphere count)
        """
        self.model = model
        self.config = config
        self.device = model.device

        # Initialize sub-components with their configs
        self.pruning_config = PruningConfig(
            # min_radius_threshold=config.model.radius_threshold,
        )
        self.adding_config = AddingConfig(
            # surface_coverage_threshold=config.model.coverage_threshold,
        )

        self.scorer = SphereScorer(model, self.pruning_config)
        self.candidate_generator = CandidateGenerator(
            model, self.adding_config)

        # Target count from config
        self.target_count = config.model.num_spheres

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
        """
        min_interval = self.config.training.density_control_min_interval
        patience = self.config.training.density_control_patience
        grad_threshold = self.config.training.density_control_grad_threshold

        if iteration - self.last_density_control_iter < min_interval:
            return False

        if iteration - self.last_density_control_iter > min_interval * 2:
            return True

        if len(loss_history) < patience or len(position_grad_history) < patience:
            return False

        recent_losses = loss_history[-patience:]
        loss_change = abs(recent_losses[0] - recent_losses[-1]) / max(
            abs(recent_losses[0]), 1e-5
        )
        loss_plateaued = loss_change < 0.01

        recent_pos_grads = position_grad_history[-patience:]
        recent_rad_grads = radius_grad_history[-patience:]

        grads_small = (
            sum(g < grad_threshold for g in recent_pos_grads) > patience // 2
            and sum(g < grad_threshold for g in recent_rad_grads) > patience // 2
        )

        should_densify = loss_plateaued or grads_small

        if should_densify:
            print("\n--- Density Control Trigger ---")
            print(
                f"Loss plateau: {loss_plateaued} (change: {loss_change:.6f})")
            print(f"Small gradients: {grads_small}")

        return should_densify

    def iterative_refinement(self, max_iterations: int = 3) -> Tuple[int, int]:
        """
        Iteratively prune bad spheres and add to gaps until stable.

        This allows the system to:
        1. Remove bad spheres → creates gaps
        2. Fill gaps with new spheres → better positioned
        3. Repeat until sphere count stabilizes near target

        Args:
            max_iterations: Maximum refinement iterations

        Returns:
            Tuple of (total_added, total_removed)
        """
        print("\n" + "=" * 50)
        print("ITERATIVE REFINEMENT")
        print("=" * 50)

        total_added = 0
        total_removed = 0

        for i in range(max_iterations):
            print(f"\n--- Refinement Iteration {i + 1}/{max_iterations} ---")

            current_count = len(self.model.radii)
            print(
                f"Current spheres: {current_count}, Target: {self.target_count}")

            # Prune worst spheres (if over target or outliers exist)
            removed = self._prune_phase()
            total_removed += removed

            # Add spheres to gaps (if under target or gaps exist)
            added = self._add_phase()
            total_added += added

            # Check if we've stabilized
            net_change = abs(added - removed)
            at_target = len(self.model.radii) == self.target_count

            print(
                f"Iteration {i + 1}: removed={removed}, added={added}, net_change={net_change}")

            # Stop early if stable (no changes or at target with minimal changes)
            if net_change == 0:
                print("Converged: no changes made")
                break

            if at_target and added <= 1 and removed <= 1:
                print("Converged: at target with minimal adjustments")
                break

        # Update model sphere count
        self.model.num_spheres = len(self.model._radii)

        print("\n" + "-" * 50)
        print(f"REFINEMENT COMPLETE:")
        print(f"  Total removed: {total_removed}")
        print(f"  Total added: {total_added}")
        print(f"  Final count: {self.model.num_spheres}")
        print("=" * 50)

        return total_added, total_removed

    def adaptive_density_control(self) -> Tuple[int, int]:
        """
        Perform adaptive density control with iterative refinement.

        Returns:
            Tuple of (spheres_added, spheres_removed)
        """
        return self.iterative_refinement(max_iterations=3)

    # def adaptive_density_control(self) -> Tuple[int, int]:
    #     """
    #     Perform adaptive density control.

    #     Logic:
    #     1. Compute badness scores for all spheres
    #     2. If over target OR any sphere exceeds badness threshold: prune worst spheres
    #     3. If under target OR essential coverage missing: add new spheres

    #     Returns:
    #         Tuple of (spheres_added, spheres_removed)
    #     """
    #     print("\n" + "=" * 50)
    #     print("ADAPTIVE DENSITY CONTROL")
    #     print("=" * 50)

    #     current_count = len(self.model.radii)
    #     print(f"Current spheres: {current_count}")
    #     print(f"Target spheres: {self.target_count}")

    #     spheres_removed = 0
    #     spheres_added = 0

    #     # === PHASE 1: Pruning ===
    #     spheres_removed = self._prune_phase()

    #     # === PHASE 2: Adding ===
    #     spheres_added = self._add_phase()

    #     # Update model sphere count
    #     self.model.num_spheres = len(self.model._radii)

    #     # Summary
    #     print("-" * 50)
    #     print(f"SUMMARY:")
    #     print(f"  Removed: {spheres_removed}")
    #     print(f"  Added: {spheres_added}")
    #     print(f"  Final count: {self.model.num_spheres}")
    #     print("=" * 50)

    #     return spheres_added, spheres_removed

    def _prune_phase(self) -> int:
        """
        Pruning phase: remove worst spheres.

        Logic:
        - If over target: prune down to target (remove worst)
        - Optionally: prune worst 1-5% if significantly worse than average

        Returns:
            Number of spheres removed
        """
        print("\n--- Pruning Phase ---")

        current_count = len(self.model.radii)

        if current_count == 0:
            print("No spheres to prune")
            return 0

        # Compute badness scores
        badness_scores = self.scorer.compute_all_badness_scores()

        print(
            f"Badness scores: min={badness_scores.min():.2f}, max={badness_scores.max():.2f}, mean={badness_scores.mean():.2f}")

        # Determine how many to prune
        num_to_prune = 0

        # Case 1: Over target - must prune down to target
        if current_count > self.target_count:
            num_to_prune = current_count - self.target_count
            print(
                f"Over target by {num_to_prune}, will prune worst {num_to_prune}")

        # Case 2: At or below target - optionally prune worst 1-5% if they're outliers
        else:
            # Prune worst 5% only if they're significantly worse than mean (2x)
            worst_percent = 0.1
            num_candidates = max(1, int(current_count * worst_percent))

            mean_badness = badness_scores.mean()
            worst_scores, worst_indices = torch.topk(
                badness_scores, k=num_candidates)

            # Only prune if worst sphere is >2x mean badness
            if worst_scores[0] > mean_badness * 1.5:
                # Count how many are >2x mean
                outliers = (worst_scores > mean_badness * 2).sum().item()
                num_to_prune = min(outliers, num_candidates)
                print(
                    f"Found {num_to_prune} outlier spheres (>{mean_badness * 2:.1f} badness)")
            else:
                print("No outlier spheres to prune")

        if num_to_prune == 0:
            print("No spheres to prune")
            return 0

        # Safety: never prune below 20% of target
        min_keep = max(1, int(self.target_count * 0.2))
        max_can_prune = current_count - min_keep

        if num_to_prune > max_can_prune:
            print(
                f"WARNING: Limiting pruning from {num_to_prune} to {max_can_prune} (keeping {min_keep} spheres)")
            num_to_prune = max_can_prune

        if num_to_prune <= 0:
            print("Cannot prune any spheres (at minimum count)")
            return 0

        # Get indices of worst spheres
        _, worst_indices = torch.topk(badness_scores, k=num_to_prune)

        # Create prune mask
        prune_mask = torch.zeros(
            current_count, dtype=torch.bool, device=self.device)
        prune_mask[worst_indices] = True

        # Report what we're removing
        pruned_scores = badness_scores[prune_mask]
        print(
            f"Pruning {num_to_prune} spheres (badness: {pruned_scores.min():.1f} - {pruned_scores.max():.1f})")

        # Perform pruning
        keep_mask = ~prune_mask
        self.model._centers = nn.Parameter(self.model._centers[keep_mask])
        self.model._radii = nn.Parameter(self.model._radii[keep_mask])

        print(f"After pruning: {len(self.model._radii)} spheres")

        return num_to_prune

    def _add_phase(self) -> int:
        """
        Adding phase: add spheres where needed.

        Add if:
        - Current count < target count
        - Essential surface coverage is missing

        Returns:
            Number of spheres added
        """
        print("\n--- Adding Phase ---")

        current_count = len(self.model.radii)

        # Calculate how many spheres we need
        num_to_add = max(0, self.target_count - current_count)

        # Check if essential surface coverage is missing
        surface_coverage_gap = self._compute_surface_coverage_gap()
        print(f"Surface coverage gap: {surface_coverage_gap:.4f}")

        # If we have significant coverage gap, we might add more spheres
        # even if at target count (up to a limit)
        if surface_coverage_gap > 0.1 and num_to_add == 0:
            # Allow adding a few more spheres for essential coverage
            max_extra = max(1, int(self.target_count * 0.1))  # Up to 10% extra
            num_to_add = min(max_extra, int(surface_coverage_gap * 10))
            print(
                f"Adding {num_to_add} extra spheres for essential surface coverage")

        if num_to_add == 0:
            print("No spheres to add")
            return 0

        print(f"Need to add: {num_to_add} spheres")

        # Generate candidates
        candidates = self.candidate_generator.generate_candidates()
        print(f"Generated {len(candidates['positions'])} candidates")

        if len(candidates['positions']) == 0:
            print("No candidates available")
            return 0

        # Score candidates
        scores = self.candidate_generator.score_candidates(
            candidates,
            self.model.centers
        )

        # Calculate minimum separation
        mean_radius = self.model.radii.mean().item()
        min_separation = mean_radius * self.adding_config.min_separation_factor
        print(f"Min separation: {min_separation:.4f}")

        # Select positions with diversity
        new_positions, new_sources = self.candidate_generator.select_positions_with_diversity(
            candidates,
            scores,
            num_to_add,
            min_separation,
            self.model.centers
        )

        if len(new_positions) == 0:
            print("Could not find valid positions for new spheres")
            return 0

        print(f"Selected {len(new_positions)} positions")
        print(
            f"Sources: {dict((s, new_sources.count(s)) for s in set(new_sources))}")

        # Compute radii for new spheres
        new_radii = self._compute_new_radii(new_positions)

        print(
            f"New radii: min={new_radii.min():.4f}, max={new_radii.max():.4f}")

        # Add new spheres
        self.model._centers = nn.Parameter(
            torch.cat([self.model._centers, new_positions], dim=0)
        )
        self.model._radii = nn.Parameter(
            torch.cat([self.model._radii, new_radii], dim=0)
        )

        print(f"After adding: {len(self.model._radii)} spheres")

        return len(new_positions)

    def _compute_surface_coverage_gap(self) -> float:
        """
        Compute how much of the surface is poorly covered.

        Returns:
            Fraction of surface points that are uncovered (0.0 to 1.0)
        """
        with torch.no_grad():
            centers = self.model.centers
            radii = self.model.radii
            surface_samples = self.model.surface_samples

            dists_to_centers = torch.cdist(surface_samples, centers)
            dists_to_surface = dists_to_centers - radii.unsqueeze(0)
            min_dists, _ = torch.min(dists_to_surface, dim=1)

            uncovered = (
                min_dists > self.adding_config.surface_coverage_threshold).float()

            return uncovered.mean().item()

    def _compute_new_radii(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute initial radii for new spheres based on local coverage gap.

        Args:
            positions: [N, 3] positions for new spheres

        Returns:
            [N] tensor of radii
        """
        centers = self.model.centers
        radii = self.model.radii

        # Distance from each new position to nearest existing sphere surface
        dists_to_centers = torch.cdist(positions, centers)
        dists_to_surface = dists_to_centers - radii.unsqueeze(0)
        local_gaps, _ = torch.min(dists_to_surface, dim=1)

        # Radius = gap * factor (start smaller to avoid immediate overlap)
        new_radii = torch.relu(local_gaps) * \
            self.adding_config.radius_gap_factor

        # Ensure minimum radius
        min_radius = self.pruning_config.min_radius_threshold * 2
        new_radii = torch.clamp(new_radii, min=min_radius)

        # Cap at mean radius to avoid creating giant spheres
        max_radius = radii.mean() * 1.5
        new_radii = torch.clamp(new_radii, max=max_radius)

        return new_radii

    def update_last_density_control_iter(self, iteration: int):
        """Update the last density control iteration."""
        self.last_density_control_iter = iteration

    def get_sphere_quality_report(self) -> Dict:
        """
        Generate a quality report for all spheres.
        Useful for debugging and visualization.

        Returns:
            Dictionary with quality metrics for each sphere
        """
        with torch.no_grad():
            badness_scores = self.scorer.compute_all_badness_scores()

            # Individual components
            outside_scores = self.scorer._score_outside_mesh()
            tiny_scores = self.scorer._score_tiny_radius()
            contained_scores = self.scorer._score_contained_spheres()
            surface_contrib_scores = self.scorer._score_surface_contribution()
            unique_coverage_scores = self.scorer._score_unique_coverage()
            far_from_surface_scores = self.scorer._score_far_from_surface()

            return {
                'total_badness': badness_scores.cpu().numpy(),
                'outside_mesh': outside_scores.cpu().numpy(),
                'tiny_radius': tiny_scores.cpu().numpy(),
                'contained': contained_scores.cpu().numpy(),
                'low_surface_contribution': surface_contrib_scores.cpu().numpy(),
                'low_unique_coverage': unique_coverage_scores.cpu().numpy(),
                'far_from_surface': far_from_surface_scores.cpu().numpy(),
                'radii': self.model.radii.cpu().numpy(),
                'threshold': self.pruning_config.badness_threshold,
            }
