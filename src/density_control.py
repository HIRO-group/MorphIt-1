"""
Annealed Partial Re-Packing density control for MorphIt sphere packing.

Core idea: every density control pass scores spheres by marginal value,
culls the lowest-value ones, and re-seeds via farthest-point placement,
all governed by a temperature schedule that cools over successive passes.
Sphere count is strictly preserved — the user's requested budget is honoured.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict

from inside_mesh import check_mesh_contains


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus: log(exp(x) - 1). Numerically stable."""
    return x + torch.log(-torch.expm1(-x))


class DensityController:
    """
    Density control via annealed partial re-packing.

    Three components:
        1. Marginal value scoring — how much worse does coverage get
           if this sphere is removed?
        2. Farthest-point re-seeding — place new spheres at the worst gaps.
        3. Temperature schedule — start aggressive, cool down each pass.
    """

    def __init__(self, model, config, losses=None):
        """
        Initialize density controller.

        Args:
            model: MorphIt model instance
            config: MorphIt configuration
            losses: MorphItLosses instance (used during the warmup pass to
                run a few mini-optimization steps over only the freshly-
                placed spheres). Required when warmup_steps > 0.
        """
        self.model = model
        self.config = config
        self.device = model.device
        self.losses = losses

        # Fixed sphere budget
        self.target_count = config.model.num_spheres

        # Temperature schedule: initial replacement fraction; cools each pass
        self.temperature = 0.4
        self.cooling_factor = float(config.training.density_control_cooling_factor)
        self.min_temperature = 0.04

        # Trigger tracking
        self.last_density_control_iter = 0

        # Warmup: a few optimizer steps over the new spheres only,
        # with survivors frozen, so they settle before main optimization.
        self.warmup_steps = int(config.training.density_control_warmup_steps)
        self.warmup_lr = config.training.center_lr * 5

    # ------------------------------------------------------------------
    # Public interface (matches what training.py expects)
    # ------------------------------------------------------------------

    def should_perform_density_control(
        self,
        loss_history: List[float],
        position_grad_history: List[float],
        radius_grad_history: List[float],
        iteration: int,
    ) -> bool:
        """Determine if density control should be triggered."""
        min_interval = self.config.training.density_control_min_interval
        patience = self.config.training.density_control_patience
        grad_threshold = self.config.training.density_control_grad_threshold

        # Respect minimum interval
        if iteration - self.last_density_control_iter < min_interval:
            return False

        # Force trigger if we've waited too long
        if iteration - self.last_density_control_iter > min_interval * 2:
            return True

        # Need enough history
        if len(loss_history) < patience or len(position_grad_history) < patience:
            return False

        # Check for loss plateau
        recent_losses = loss_history[-patience:]
        loss_change = abs(recent_losses[0] - recent_losses[-1]) / max(
            abs(recent_losses[0]), 1e-5
        )
        loss_plateaued = loss_change < 0.01

        # Check for small gradients
        recent_pos_grads = position_grad_history[-patience:]
        recent_rad_grads = radius_grad_history[-patience:]
        grads_small = (
            sum(g < grad_threshold for g in recent_pos_grads) > patience // 2
            and sum(g < grad_threshold for g in recent_rad_grads) > patience // 2
        )

        should_trigger = loss_plateaued or grads_small

        if should_trigger:
            print("\n--- Density Control Trigger ---")
            print(f"Loss plateau: {loss_plateaued} (change: {loss_change:.6f})")
            print(f"Small gradients: {grads_small}")
            print(f"Temperature: {self.temperature:.3f}")

        return should_trigger

    def adaptive_density_control(self) -> Tuple[int, int]:
        """
        Perform one pass of count-preserving annealed partial re-packing.

        Cull the k lowest-marginal-value spheres, then re-seed exactly k
        new spheres at the k worst-covered interior points (farthest-point
        placement). Sphere count is strictly preserved — no drift, no
        floor — so the user's requested budget is honoured. The cull/add
        cycle is purely a *reshuffling* mechanism: poor placements get
        replaced by fresh spheres at uncovered interior regions.

        Returns:
            Tuple of (spheres_added, spheres_removed). These are equal.
        """
        n = len(self.model.radii)

        # Identify bad spheres (collapsed-tiny or escaped-outside) up
        # front. They are force-removed regardless of marginal value;
        # if more bad spheres exist than the temperature schedule would
        # otherwise cull, we widen the cull to include all of them.
        bad_mask = self._identify_bad_spheres()
        n_bad = int(bad_mask.sum().item())

        # Count-preserving: cull k, add k. Always keep ≥1 survivor so the
        # marginal-value ranking and the re-seed have a reference pack.
        k_temp = max(1, int(self.target_count * self.temperature))
        k = max(k_temp, n_bad)
        k = min(k, self.target_count - 1, n - 1)

        if k <= 0:
            print(f"Skipping density control: count {n} too low to cull")
            return 0, 0

        print(f"\n{'=' * 50}")
        print(f"COUNT-PRESERVING RE-PACK (temperature={self.temperature:.3f})")
        print(f"{'=' * 50}")
        # One-line cull breakdown: total / bad-forced / score-driven.
        # Grep-friendly tag "[cull]" for log scanning across many runs.
        n_score = k - n_bad
        print(f"[cull] {k}/{n} spheres → reseed {k}  "
              f"(bad: {n_bad} [tiny/outside], by score: {n_score}; "
              f"target={self.target_count}, preserved)")

        # --- Score: marginal value of each sphere ---
        values = self._compute_marginal_values()
        print(
            f"Marginal values: min={values.min():.4f}, "
            f"max={values.max():.4f}, mean={values.mean():.4f}"
        )

        # Bad spheres are pinned to the bottom of the ranking so they
        # always land inside the cull window, regardless of how lucky
        # their current marginal-value score happens to be.
        if n_bad > 0:
            values = values.clone()
            values[bad_mask] = float("-inf")

        # --- Cull: remove the k lowest-value spheres ---
        # Work in real-radius space (post-softplus).
        real_radii = self.model.radii
        keep_indices = torch.argsort(values, descending=True)[: n - k]
        surviving_centers = self.model._centers.data[keep_indices].clone()
        surviving_radii = real_radii[keep_indices].detach().clone()
        # Per-sphere mass — when registered — has to be reordered by the
        # same `keep_indices` as centers/radii. Without this the survivor
        # at position i ends up paired with the *original* sphere i's
        # mass, not its own, so the physics losses compute gradients
        # against scrambled mass↔center pairings on every DC fire.
        psm = self.model._log_masses is not None
        if psm:
            surviving_masses = self.model.masses[keep_indices].detach().clone()

        removed_values = values[torch.argsort(values, descending=False)[:k]]
        print(
            f"Removed {k} spheres (marginal values: "
            f"{removed_values.min():.4f} - {removed_values.max():.4f})"
        )

        # --- Re-seed: farthest-point placement of k new spheres ---
        new_centers, new_radii = self._farthest_point_reseed(
            surviving_centers, surviving_radii, k
        )
        k_added = len(new_centers)
        print(
            f"Placed {k_added}/{k} new spheres  "
            f"(count preserved: {n} -> {n - k + k_added})"
        )

        # --- Combine (in real space) then convert to raw space ---
        all_centers = torch.cat([surviving_centers, new_centers], dim=0)
        all_radii_real = torch.cat([surviving_radii, new_radii], dim=0)

        self.model._centers = nn.Parameter(all_centers)
        self.model._radii = nn.Parameter(_inverse_softplus(all_radii_real))
        # PSM writeback: new spheres are initialised to the median
        # surviving mass — adaptive to whatever the optimizer has
        # learned, and equal to mesh_mass/N before any mass training
        # has happened (matches morphit.py's init convention).
        if psm:
            new_mass_init = float(surviving_masses.median().item())
            new_masses = torch.full(
                (len(new_centers),),
                new_mass_init,
                device=surviving_masses.device,
                dtype=surviving_masses.dtype,
            )
            all_masses = torch.cat([surviving_masses, new_masses], dim=0)
            self.model._log_masses = nn.Parameter(_inverse_softplus(all_masses))
        self.model.num_spheres = len(all_radii_real)

        # --- Warmup: mini-optimization for the new spheres ---
        self._warmup_new_spheres(n_surviving=len(surviving_centers))

        # --- Cool down ---
        old_temp = self.temperature
        self.temperature = max(
            self.min_temperature, self.temperature * self.cooling_factor
        )
        print(f"Temperature: {old_temp:.3f} -> {self.temperature:.3f}")
        print(f"Final sphere count: {self.model.num_spheres}")
        print(f"{'=' * 50}")

        return k_added, k

    def update_last_density_control_iter(self, iteration: int):
        """Update the last density control iteration."""
        self.last_density_control_iter = iteration

    # ------------------------------------------------------------------
    # Core algorithms
    # ------------------------------------------------------------------

    def _identify_bad_spheres(self) -> torch.Tensor:
        """
        Mark spheres for force-removal regardless of marginal value:

          1. **Tiny radius** — radius below
             ``density_control_min_radius_fraction * mesh.scale``. These
             spheres collapsed during optimization (typically driven down
             by overlap or boundary pressure) and contribute essentially
             zero coverage. The reseed step seeds new spheres at the
             survivor median, so the cleanup is constructive.

          2. **Center outside the mesh** — distinct from "sphere bulges
             past the surface" (which V intentionally allows via low
             ``boundary_weight``). A sphere whose *center* has drifted
             out is hemorrhaging coverage on both sides of the mesh
             boundary and should be reseeded back into a useful gap.

        Returns:
            ``[num_spheres]`` bool tensor; True = should be culled.
        """
        with torch.no_grad():
            radii = self.model.radii
            mesh = self.model.query_mesh

            min_radius = (
                float(self.config.training.density_control_min_radius_fraction)
                * float(mesh.scale)
            )
            too_small = radii < min_radius

            # check_mesh_contains operates on numpy points + a trimesh.
            centers_np = self.model.centers.detach().cpu().numpy()
            try:
                inside_np = check_mesh_contains(mesh, centers_np)
                inside = torch.from_numpy(np.asarray(inside_np)).to(
                    device=self.device, dtype=torch.bool
                )
                outside = ~inside
            except Exception as exc:  # pragma: no cover — defensive
                # Don't let a containment query failure block density
                # control entirely; just skip the outside-mesh rule.
                print(f"  outside-mesh check failed ({exc}); "
                      f"skipping outside-mesh cull")
                outside = torch.zeros_like(too_small, dtype=torch.bool)

            return too_small | outside

    def _compute_marginal_values(self) -> torch.Tensor:
        """
        Compute the marginal value of each sphere for *the variant's own
        loss*. A V run (coverage-dominant) judges spheres by how much
        interior coverage they guard; an S run (surface-dominant) judges
        them by how much surface coverage they guard; B blends.

        For each sample point, find the closest sphere and the 2nd-closest
        sphere's surfaces. The "gap" (2nd - 1st) is the coverage hole that
        would open if the closest sphere were removed. Gaps are accumulated
        per sphere separately over interior and surface samples, then
        combined with weights taken from the training config:

            value = w_interior * interior_gap + w_surface * surface_gap

        where w_interior = coverage_weight and
        w_surface       = surface_weight + boundary_weight + sqem_weight.

        This preserves V's interior-guarding spheres (previously culled
        because they scored 0 on pure-surface marginal value) while still
        letting S/B prioritize their surface-side responsibilities.

        Returns:
            [num_spheres] tensor; higher = more valuable (harder to lose).
        """
        with torch.no_grad():
            centers = self.model.centers
            radii = self.model.radii
            n_spheres = len(radii)

            w_interior = float(self.config.training.coverage_weight)
            w_surface = float(
                self.config.training.surface_weight
                + self.config.training.boundary_weight
                + self.config.training.sqem_weight
            )

            def _score(samples: torch.Tensor) -> torch.Tensor:
                dists = torch.cdist(samples, centers) - radii.unsqueeze(0)
                top2_dists, top2_idx = torch.topk(
                    dists, k=min(2, n_spheres), dim=1, largest=False
                )
                closest_idx = top2_idx[:, 0]
                closest_dist = top2_dists[:, 0]
                if n_spheres >= 2:
                    second_dist = top2_dists[:, 1]
                else:
                    second_dist = torch.full_like(closest_dist, 1e6)
                gap = second_dist - closest_dist
                out = torch.zeros(n_spheres, device=self.device)
                out.scatter_add_(0, closest_idx, gap)
                return out

            values = torch.zeros(n_spheres, device=self.device)
            if w_interior > 0:
                values = values + w_interior * _score(self.model.inside_samples)
            if w_surface > 0:
                values = values + w_surface * _score(self.model.surface_samples)
            return values

    def _farthest_point_reseed(
        self,
        surviving_centers: torch.Tensor,
        surviving_radii: torch.Tensor,
        k_max: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Place exactly ``k_max`` new spheres via iterative farthest-point
        sampling on the INTERIOR of the mesh. Candidates are taken from
        ``inside_samples``.

        Count-preserving: the caller has already culled ``k_max`` spheres,
        so we must place exactly ``k_max`` back (no gap-based early exit).
        Even when the worst remaining gap is small, a fresh sphere seeded
        there is what allows the optimizer to reshuffle out of a poor
        equilibrium on subsequent steps.

        Args:
            surviving_centers: [M, 3] centers of kept spheres
            surviving_radii:   [M]    radii of kept spheres
            k_max:             exact number of new spheres to place

        Returns:
            Tuple of (new_centers [k_max, 3], new_radii [k_max]).
        """
        with torch.no_grad():
            candidates = self.model.inside_samples  # [N, 3]
            n_cand = len(candidates)

            # Current coverage: distance from each candidate to nearest
            # surviving sphere's surface.
            if len(surviving_centers) > 0:
                dists = torch.cdist(candidates, surviving_centers)
                dists_to_surface = dists - surviving_radii.unsqueeze(0)
                min_dists, _ = torch.min(dists_to_surface, dim=1)  # [N]
            else:
                min_dists = torch.full((n_cand,), 1e6, device=self.device)

            new_centers_list = []
            new_radii_list = []

            if len(surviving_radii) > 0:
                radius_cap = surviving_radii.median().item() * 1.5
                # Seed new spheres at the survivor median so they are
                # immediately useful. Previously 0.5 * min left many
                # fresh spheres at ~0.001 that never had time to grow
                # before training ended — dead weight that hurt V's
                # coverage at large n.
                default_radius = surviving_radii.median().item()
            else:
                radius_cap = 1e6
                default_radius = 1e-3

            mesh_scale = self.model.query_mesh.scale

            for _ in range(k_max):
                worst_idx = torch.argmax(min_dists)
                new_center = candidates[worst_idx].clone()
                gap_at_point = min_dists[worst_idx].item()

                # Always start at least at median survivor size so fresh
                # spheres contribute immediately; if the local gap is
                # larger, use that (fill the uncovered hole).
                new_radius = max(gap_at_point, default_radius)
                new_radius = min(new_radius, radius_cap, mesh_scale * 0.3)

                new_center_t = new_center.unsqueeze(0)
                new_radius_t = torch.tensor(
                    [new_radius], device=self.device, dtype=torch.float32
                )

                new_centers_list.append(new_center)
                new_radii_list.append(new_radius_t)

                dist_to_new = (
                    torch.norm(candidates - new_center_t, dim=1) - new_radius
                )
                min_dists = torch.min(min_dists, dist_to_new)

            new_centers = torch.stack(new_centers_list, dim=0)
            new_radii = torch.cat(new_radii_list, dim=0)

            return new_centers, new_radii

    def _warmup_new_spheres(self, n_surviving: int):
        """
        Run a few optimizer steps on only the new spheres while keeping
        survivors frozen (gradient zeroed + data restored afterwards).
        Lets new spheres settle into reasonable positions before the main
        optimizer sees them.
        """
        if self.warmup_steps <= 0:
            return

        if self.losses is None:
            print("Warmup skipped: density controller has no losses handle")
            return

        print(f"Warming up new spheres ({self.warmup_steps} steps)...")

        with torch.no_grad():
            survivor_centers = self.model._centers.data[:n_surviving].clone()
            survivor_radii = self.model._radii.data[:n_surviving].clone()

        temp_optimizer = torch.optim.Adam(
            [
                {"params": self.model._centers, "lr": self.warmup_lr},
                {"params": self.model._radii, "lr": self.warmup_lr * 0.5},
            ]
        )

        loss_weights = self.losses.get_loss_weights_from_config(
            self.config.training)

        for _ in range(self.warmup_steps):
            temp_optimizer.zero_grad()

            all_losses = self.losses.compute_all_losses(weights=loss_weights)
            total_loss = torch.tensor(0.0, device=self.device)
            for name, value in all_losses.items():
                if name in loss_weights:
                    total_loss = total_loss + loss_weights[name] * value

            total_loss.backward()

            # Freeze survivors by zeroing their gradients.
            if self.model._centers.grad is not None:
                self.model._centers.grad.data[:n_surviving] = 0.0
            if self.model._radii.grad is not None:
                self.model._radii.grad.data[:n_surviving] = 0.0

            temp_optimizer.step()

        # Restore survivors exactly (protect against floating-point drift).
        with torch.no_grad():
            self.model._centers.data[:n_surviving] = survivor_centers
            self.model._radii.data[:n_surviving] = survivor_radii

        print(f"Warmup complete (final loss: {total_loss.item():.6f})")
