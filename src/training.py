"""
Training module for MorphIt sphere packing optimization.
"""

import torch
import torch.nn as nn
import time
from pathlib import Path
from typing import Dict, Any, Optional

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

        # Initialize components
        self.losses = MorphItLosses(model)
        self.density_controller = DensityController(model, config)

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
                {"params": self.model._centers,
                    "lr": self.config.training.center_lr},
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
            self.model.initialize_render_thread(
                render_interval=render_interval)

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
        loss_weights = self.losses.get_loss_weights_from_config(
            self.config.training)

        self.training_start_time = time.time()
        # self.density_controller.prune_spheres()
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
                self.evolution_logger.log_spheres(
                    self.model, iteration, "training")

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

        # Compute losses
        losses = self.losses.compute_all_losses()

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

        # Calculate timing
        iter_time = time.time() - iter_start_time

        return {
            "total_loss": total_loss.item(),
            "weighted_losses": weighted_losses,
            "raw_losses": {k: v.item() for k, v in losses.items()},
            "grad_info": grad_info,
            "iter_time": iter_time,
        }

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

    def _finalize_training(self):
        """Finalize training and cleanup."""
        # Stop rendering
        if hasattr(self.model, "render_thread"):
            self.model.stop_render_thread()

        # Final pruning
        self.density_controller.prune_spheres()

        # Log final state
        self.evolution_logger.log_spheres(
            self.model, self.current_iteration, "final")
        self.evolution_logger.save_complete_evolution()

        # Print summary
        self._print_training_summary()

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
