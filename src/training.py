"""
Training module for MorphIt sphere packing optimization.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, Optional

from inside_mesh import check_mesh_contains
from losses import MorphItLosses
from density_control import DensityController
from convergence_tracker import ConvergenceTracker
from logger import SphereEvolutionLogger


class MorphItTrainer:
    """
    Training manager for MorphIt sphere packing optimization.
    """

    def __init__(self, model, config):
        """
        Initialize trainer.

        Args:
            model: MorphIt model instance
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.device = model.device

        # Initialize components. The density controller borrows the
        # losses instance so its warmup pass can compute the same
        # weighted total loss as the main training loop.
        self.losses = MorphItLosses(model)
        self.density_controller = DensityController(
            model, config, losses=self.losses)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize tracking
        self.convergence_tracker = None
        self.evolution_logger = None

        # Training state
        self.current_iteration = 0
        self.density_control_count = 0
        self.training_start_time = None

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with configured learning rates."""
        return torch.optim.Adam(
            [
                {"params": self.model._centers, "lr": self.config.training.center_lr},
                {"params": self.model._radii, "lr": self.config.training.radius_lr},
            ]
        )

    def _reset_optimizer(self):
        """Reset optimizer after parameter changes."""
        self.optimizer = self._create_optimizer()
        print("Optimizer reset after parameter changes")

    def setup_logging(self):
        """Setup logging and tracking systems."""
        # Setup convergence tracker
        model_name = Path(self.model.mesh_path).stem
        self.convergence_tracker = ConvergenceTracker(model_name)

        # Setup evolution logger
        self.evolution_logger = SphereEvolutionLogger("sphere_evolution")
        self.model.evolution_logger = self.evolution_logger

        # Log initial state
        self.evolution_logger.log_spheres(self.model, 0, "initial")

    def setup_rendering(self):
        """Setup rendering if PyVista is available."""
        if hasattr(self.model, "pl") and self.model.pl is not None:
            render_interval = self.config.visualization.render_interval
            self.model.initialize_render_thread(render_interval=render_interval)

    def train(self) -> ConvergenceTracker:
        """
        Main training loop.

        Returns:
            Convergence tracker with training history
        """
        print("\n=== Starting MorphIt Training ===")

        # Setup logging and rendering
        self.setup_logging()
        self.setup_rendering()

        # Get loss weights
        loss_weights = self.losses.get_loss_weights_from_config(self.config.training)

        self.training_start_time = time.time()
        # Training loop
        for iteration in range(self.config.training.iterations):
            self.current_iteration = iteration

            # Perform training step
            loss_info = self._training_step(loss_weights)

            # Update tracking
            self._update_tracking(iteration, loss_info)

            # Handle rendering
            self._handle_rendering(iteration)

            # Print progress
            if iteration % self.config.training.verbose_frequency == 0:
                self._print_progress(iteration, loss_info)

                # Check for convergence
                if self._check_convergence(iteration):
                    break

            # Handle density control
            if self._should_perform_density_control(iteration):
                self._perform_density_control(iteration)

            # Log sphere evolution
            if (
                self.config.training.logging_enabled and iteration % 1 == 0
            ):  # Log every iteration
                self.evolution_logger.log_spheres(self.model, iteration, "training")

        # Cleanup and finalize
        self._finalize_training()

        return self.convergence_tracker

    def _training_step(self, loss_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform a single training step.

        Args:
            loss_weights: Dictionary of loss weights

        Returns:
            Dictionary with loss information and gradients
        """
        iter_start_time = time.time()

        # Zero gradients
        self.optimizer.zero_grad()

        # Compute losses. Pass weights so zero-weight losses are skipped
        # entirely — important because flatness/mesh_containment/etc. are
        # expensive and many configs leave them at 0.
        losses = self.losses.compute_all_losses(weights=loss_weights)

        # Compute weighted total loss
        total_loss = torch.tensor(0.0, device=self.device)
        weighted_losses = {}
        for loss_name, loss_value in losses.items():
            if loss_name in loss_weights:
                weight = loss_weights[loss_name]
                weighted_loss = weight * loss_value
                total_loss += weighted_loss
                weighted_losses[loss_name] = weighted_loss.item()

        # Backward pass
        total_loss.backward()

        # Get gradient information
        grad_info = self._get_gradient_info()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model._centers, self.config.training.grad_clip_norm
        )
        torch.nn.utils.clip_grad_norm_(
            self.model._radii, self.config.training.grad_clip_norm * 0.5
        )  # Smaller for radii

        # Update parameters
        self.optimizer.step()

        # ── BAND-AID ────────────────────────────────────────────────
        # Hard containment projection: snap any sphere whose center
        # drifted outside the mesh back to the nearest surface point
        # (just inside, by 1e-4 × mesh.scale). Sample-based: uses the
        # cached surface_samples + normals, same approximation
        # mesh_containment_loss uses for its soft penalty.
        #
        # Why this is a band-aid: under MorphIt-V's weak boundary
        # gradients, individual training steps can still push a center
        # past the mesh surface faster than mesh_containment_loss
        # pulls it back. The projection clamps each step's overshoot.
        # The "real" fix would be a gradient pipeline that never
        # produces an outward step in the first place — that's an open
        # problem at the loss/optimizer level. Until then, projection
        # + density-control bad-sphere cull together keep escapes
        # bounded; the final-iteration prune in _finalize_training
        # catches whatever survives the last training segment.
        #
        # An SDF-based hard projection (ray-cast sign on a precomputed
        # voxel grid) was tried — it eliminated essentially all escapes
        # at concave regions like bunny ears, but at +14 % per-iter
        # cost and a 0.5 MB precomputed grid per mesh. Reverted in
        # favour of keeping MorphIt "meant to be fast"; the sample
        # projection + final prune gives the same end-state guarantee
        # at much lower cost.
        # ────────────────────────────────────────────────────────────
        self._project_centers_inside_mesh()

        # Calculate timing
        iter_time = time.time() - iter_start_time

        return {
            "total_loss": total_loss.item(),
            "weighted_losses": weighted_losses,
            "raw_losses": {k: v.item() for k, v in losses.items()},
            "grad_info": grad_info,
            "iter_time": iter_time,
        }

    def _project_centers_inside_mesh(self) -> int:
        """Sample-based hard projection — BAND-AID. See the long comment
        at the call site (``_training_step``) for why this is here and
        what would replace it.

        Geometry: for each sphere center, find its nearest precomputed
        surface sample, project the offset onto that sample's normal.
        Positive signed distance ⇒ center is outside the mesh ⇒ move
        ``(signed_dist + epsilon)`` along ``-normal`` to land ε inside
        the surface patch. Uses the same approximation as
        ``mesh_containment_loss`` so the soft penalty and the hard clamp
        agree on the surface.

        Cost: one ``cdist [N, S]`` (~0.2 ms at N=64, S=5000). Cheaper
        than ``trimesh.contains`` by ~15×; less robust through sharp
        concavities (bunny ears) — leaks ~1–6 % of spheres at high N.
        The final-iteration prune in ``_finalize_training`` catches
        whatever the projection misses on the last few steps.
        """
        with torch.no_grad():
            centers = self.model._centers.data  # [N, 3]
            surf = self.model.surface_samples    # [S, 3]
            norms = self.model.surface_normals   # [S, 3]

            dists = torch.cdist(centers, surf)
            nearest_idx = dists.argmin(dim=1)
            nearest_surf = surf[nearest_idx]
            nearest_normal = norms[nearest_idx]

            vec = centers - nearest_surf
            signed_dist = (vec * nearest_normal).sum(dim=1)  # +ve = outside

            outside = signed_dist > 0
            n_out = int(outside.sum().item())
            if n_out > 0:
                epsilon = 1e-4 * float(self.model.query_mesh.scale)
                # Move by (signed_dist + epsilon) along -normal so the
                # projected center lands an ε strictly inside the surface
                # patch we projected to.
                delta = (signed_dist + epsilon).unsqueeze(1) * nearest_normal
                self.model._centers.data = torch.where(
                    outside.unsqueeze(1), centers - delta, centers
                )
            return n_out

    def _get_gradient_info(self) -> Dict[str, float]:
        """Get gradient magnitude information."""
        with torch.no_grad():
            position_grad_mag = (
                self.model._centers.grad.norm(dim=1).mean().item()
                if self.model._centers.grad is not None
                else 0.0
            )
            radius_grad_mag = (
                self.model._radii.grad.norm().mean().item()
                if self.model._radii.grad is not None
                else 0.0
            )

        return {
            "position_grad_mag": position_grad_mag,
            "radius_grad_mag": radius_grad_mag,
        }

    def _update_tracking(self, iteration: int, loss_info: Dict[str, Any]):
        """Update convergence tracking."""
        self.convergence_tracker.update(
            iteration=iteration,
            loss_dict={
                "total": loss_info["total_loss"],
                "components": loss_info["weighted_losses"],
            },
            model=self.model,
            grad_info=loss_info["grad_info"],
            time_taken=loss_info["iter_time"],
        )

    def _handle_rendering(self, iteration: int):
        """Handle rendering if available."""
        if hasattr(self.model, "render_thread"):
            self.model.render_thread.queue_render(iteration)

    def _print_progress(self, iteration: int, loss_info: Dict[str, Any]):
        """Print training progress."""
        total_time = time.time() - self.training_start_time

        print(f"\n[Iter {iteration}] Time: {total_time:.4f}s")
        print(f"Total Loss: {loss_info['total_loss']:.6f}")

        # Print weighted losses
        for name, value in loss_info["weighted_losses"].items():
            print(f"  {name}: {value:.6f}")

        print(f"Spheres: {self.model.num_spheres}")
        print(f"Pos Grad: {loss_info['grad_info']['position_grad_mag']:.6f}")
        print(f"Rad Grad: {loss_info['grad_info']['radius_grad_mag']:.6f}")

    def _check_convergence(self, iteration: int) -> bool:
        """Check if training has converged."""
        if not self.config.training.early_stopping:
            return False

        if iteration <= self.config.training.convergence_patience:
            return False

        analysis = self.convergence_tracker.analyze_convergence(
            window_size=self.config.training.convergence_patience,
            threshold=self.config.training.convergence_threshold,
        )

        if analysis["converged"]:
            print("\n=== Training Converged ===")
            for key, value in analysis.items():
                print(f"  {key}: {value}")
            print(f"Stopping at iteration {iteration}")
            return True

        return False

    def _should_perform_density_control(self, iteration: int) -> bool:
        """Check if density control should be performed."""
        if not self.config.training.density_control_enabled:
            return False
        return self.density_controller.should_perform_density_control(
            self.convergence_tracker.metrics["total_loss"],
            self.convergence_tracker.metrics["gradient_info"]["position_grad_mag"],
            self.convergence_tracker.metrics["gradient_info"]["radius_grad_mag"],
            iteration,
        )

    def _perform_density_control(self, iteration: int):
        """Perform density control operations."""
        print(f"\n[Iter {iteration}] Performing adaptive density control")

        # Perform density control
        spheres_added, spheres_removed = (
            self.density_controller.adaptive_density_control()
        )

        # Record event
        self.convergence_tracker.record_density_control(
            iteration, spheres_added, spheres_removed
        )

        # Update controller state
        self.density_controller.update_last_density_control_iter(iteration)
        self.density_control_count += 1

        # Reset optimizer if parameters changed
        if spheres_added > 0 or spheres_removed > 0:
            self._reset_optimizer()
            # Surface samples don't change here, but cached face groups
            # in the flatness loss must be invalidated whenever the
            # sphere set changes, since the cluster-by-normal cache
            # holds tensor indices computed against the previous mesh
            # state. Cheap no-op if flatness loss is disabled.
            self.losses.reset_flatness_cache()

    def _finalize_training(self):
        """Finalize training and cleanup."""
        # Stop rendering
        if hasattr(self.model, "render_thread"):
            self.model.stop_render_thread()

        # ── BAND-AID ────────────────────────────────────────────────
        # Final-iteration escape prune. Mid-training, density control's
        # ``_identify_bad_spheres`` already flags any sphere whose
        # center has drifted outside the mesh and force-culls + reseeds
        # it (see density_control.py). The per-step projection in
        # ``_training_step`` also clamps escapes every iteration. What
        # neither catches: a sphere that drifts outside in the *last*
        # training segment (after the last density-control pass and
        # past the projection's per-step approximation in a concave
        # region). This prune mops those up so output URDFs never
        # contain a sphere with its center past the mesh boundary.
        #
        # Side effect: can drop the final sphere count below the
        # user's requested ``num_spheres``. Density control is
        # count-preserving on its own; this prune is the only thing
        # that breaks that invariant.
        # ────────────────────────────────────────────────────────────
        n_pruned = self._prune_escaped_spheres()
        if n_pruned > 0:
            print(f"\n[final prune] removed {n_pruned} sphere(s) whose "
                  f"centers drifted outside the mesh during the last "
                  f"training segment")

        # Log final state
        self.evolution_logger.log_spheres(self.model, self.current_iteration, "final")
        self.evolution_logger.save_complete_evolution()

        # Print summary
        self._print_training_summary()

    def _prune_escaped_spheres(self) -> int:
        """Strip spheres whose centers are outside the mesh. Returns
        the count removed. See the band-aid comment in
        ``_finalize_training`` for why this exists.
        """
        with torch.no_grad():
            centers_np = self.model.centers.detach().cpu().numpy()
            try:
                inside = check_mesh_contains(self.model.query_mesh, centers_np)
            except Exception as exc:
                # Containment query failure is rare but should not block
                # finalization. Skip the prune and log; spheres just stay
                # wherever they ended up.
                print(f"[final prune] containment check failed ({exc}); "
                      f"skipping escaped-sphere prune")
                return 0

            keep_mask = torch.from_numpy(np.asarray(inside, dtype=bool)).to(
                device=self.model.device, dtype=torch.bool
            )
            n_remove = int((~keep_mask).sum().item())
            if n_remove == 0:
                return 0

            # Need ≥1 sphere left for downstream code (and a no-sphere
            # URDF is useless anyway). If pruning would empty the pack,
            # leave one sphere behind and warn.
            if int(keep_mask.sum().item()) == 0:
                print("[final prune] WARNING: every sphere has its center "
                      "outside the mesh. Keeping the first one to avoid "
                      "an empty packing; the result will not be useful.")
                keep_mask[0] = True
                n_remove = int((~keep_mask).sum().item())

            self.model._centers = nn.Parameter(
                self.model._centers.data[keep_mask].clone()
            )
            self.model._radii = nn.Parameter(
                self.model._radii.data[keep_mask].clone()
            )
            self.model.num_spheres = int(keep_mask.sum().item())
            return n_remove

    def _print_training_summary(self):
        """Print training summary."""
        print(f"\n=== Training Complete ===")
        print(f"Density control operations: {self.density_control_count}")
        print(f"Final sphere count: {self.model.num_spheres}")
        print("=" * 26)


def train_morphit(model, config: Optional[Dict[str, Any]] = None) -> ConvergenceTracker:
    """
    Convenience function to train MorphIt model.

    Args:
        model: MorphIt model instance
        config: Optional configuration updates

    Returns:
        Convergence tracker with training history
    """
    # Update config if provided
    if config is not None:
        from config import update_config_from_dict

        model.config = update_config_from_dict(model.config, config)

    # Create trainer and run training
    trainer = MorphItTrainer(model, model.config)
    return trainer.train()
