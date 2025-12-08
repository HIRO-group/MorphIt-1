"""
Convergence tracking module for MorphIt training.
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for Numpy types"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


class ConvergenceTracker:
    def __init__(self, model_name, save_dir="results/training_logs"):
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # Initialize metrics storage
        self.metrics = {
            "total_loss": [],
            "component_losses": {
                "coverage_loss": [],
                "overlap_penalty": [],
                "boundary_penalty": [],
                "surface_loss": [],
                "containment_loss": [],
                "sqem_loss": [],
                "mass_loss": [],
                "com_loss": [],
                "inertia_loss": [],
            },
            "gradient_info": {"position_grad_mag": [], "radius_grad_mag": []},
            "sphere_stats": {
                "num_spheres": [],
                "min_radius": [],
                "max_radius": [],
                "mean_radius": [],
            },
            "iterations": [],
            "time_per_iteration": [],
            "density_control_events": [],
        }

        self.last_save_time = time.time()
        self.save_interval = 60  # Save every minute by default

    def update(self, iteration, loss_dict, model, grad_info, time_taken):
        """Update all metrics for the current iteration"""
        # Record iteration number
        self.metrics["iterations"].append(iteration)

        # Record losses
        self.metrics["total_loss"].append(loss_dict["total"])
        for key, value in loss_dict["components"].items():
            self.metrics["component_losses"][key].append(value)

        # Record gradient info
        for key, value in grad_info.items():
            self.metrics["gradient_info"][key].append(value)

        # Record sphere statistics
        with torch.no_grad():
            radii = model.radii.detach().cpu().numpy()
            self.metrics["sphere_stats"]["num_spheres"].append(len(radii))
            self.metrics["sphere_stats"]["min_radius"].append(float(np.min(radii)))
            self.metrics["sphere_stats"]["max_radius"].append(float(np.max(radii)))
            self.metrics["sphere_stats"]["mean_radius"].append(float(np.mean(radii)))

        # Record timing info
        self.metrics["time_per_iteration"].append(time_taken)

        # Save periodically
        current_time = time.time()
        if current_time - self.last_save_time > self.save_interval:
            self.save()
            self.last_save_time = current_time

    def record_density_control(self, iteration, spheres_added, spheres_removed):
        """Record a density control event"""
        self.metrics["density_control_events"].append(
            {
                "iteration": iteration,
                "spheres_added": spheres_added,
                "spheres_removed": spheres_removed,
            }
        )

    def save(self):
        """Save metrics to JSON file"""
        filename = self.save_dir / f"{self.model_name}_training_log.json"
        with open(filename, "w") as f:
            json.dump(self.metrics, f, indent=4, cls=NumpyEncoder)
        print(f"Saved training metrics to {filename}")

    def analyze_convergence(self, window_size=20, threshold=0.01):
        """
        Analyze if training has converged

        Args:
            window_size: Number of iterations to look back
            threshold: Relative change threshold for considering convergence

        Returns:
            dict: Convergence analysis for different metrics
        """
        if len(self.metrics["total_loss"]) < window_size:
            return {"converged": False, "reason": "Not enough data points"}

        recent_loss = self.metrics["total_loss"][-window_size:]
        loss_change = abs(recent_loss[0] - recent_loss[-1]) / max(
            abs(recent_loss[0]), 1e-5
        )

        recent_grads_pos = self.metrics["gradient_info"]["position_grad_mag"][
            -window_size:
        ]
        recent_grads_rad = self.metrics["gradient_info"]["radius_grad_mag"][
            -window_size:
        ]

        grad_small = (
            sum(g < 1e-4 for g in recent_grads_pos) > window_size // 2
            and sum(g < 1e-4 for g in recent_grads_rad) > window_size // 2
        )

        analysis = {
            "converged": (loss_change < threshold or grad_small),
            "loss_change": loss_change,
            "grad_small": grad_small,
        }

        return analysis

    def plot_training_metrics(self, save_fig=True):
        """Plot training metrics and optionally save figure"""
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # Plot total loss
        ax = axes[0, 0]
        ax.plot(self.metrics["iterations"], self.metrics["total_loss"])
        ax.set_title("Total Loss")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (log scale)")
        ax.grid(True)

        # Plot component losses
        ax = axes[0, 1]
        for name, values in self.metrics["component_losses"].items():
            ax.plot(self.metrics["iterations"], values, label=name)
        ax.set_title("Component Losses")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (log scale)")
        ax.legend()
        ax.grid(True)

        # Plot gradient magnitudes
        ax = axes[1, 0]
        for name, values in self.metrics["gradient_info"].items():
            ax.plot(self.metrics["iterations"], values, label=name)
        ax.set_title("Gradient Magnitudes")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Magnitude (log scale)")
        ax.legend()
        ax.grid(True)

        # Plot sphere stats
        ax = axes[1, 1]
        ax.plot(
            self.metrics["iterations"],
            self.metrics["sphere_stats"]["num_spheres"],
            label="Count",
        )
        ax2 = ax.twinx()
        ax2.plot(
            self.metrics["iterations"],
            self.metrics["sphere_stats"]["mean_radius"],
            "r-",
            label="Mean Radius",
        )
        ax.set_title("Sphere Statistics")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Count")
        ax2.set_ylabel("Radius")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax.grid(True)

        # Plot min/max radii
        ax = axes[2, 0]
        ax.plot(
            self.metrics["iterations"],
            self.metrics["sphere_stats"]["min_radius"],
            label="Min Radius",
        )
        ax.plot(
            self.metrics["iterations"],
            self.metrics["sphere_stats"]["max_radius"],
            label="Max Radius",
        )
        ax.set_title("Radius Range")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Radius")
        ax.legend()
        ax.grid(True)

        # Plot time per iteration
        ax = axes[2, 1]
        ax.plot(self.metrics["iterations"], self.metrics["time_per_iteration"])
        ax.set_title("Time per Iteration")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Time (seconds)")
        ax.grid(True)

        # Mark density control events
        for event in self.metrics["density_control_events"]:
            iteration = event["iteration"]
            for row in range(3):
                for col in range(2):
                    axes[row, col].axvline(
                        x=iteration, color="k", linestyle="--", alpha=0.3
                    )

        plt.tight_layout()

        if save_fig:
            fig_path = self.save_dir / f"{self.model_name}_training_metrics.png"
            plt.savefig(fig_path)
            print(f"Saved training metrics plot to {fig_path}")

        return fig

    def analyze_convergence_windows(self, window_sizes=[10, 20, 50, 100]):
        """Analyze convergence across multiple window sizes"""
        results = {}
        for window in window_sizes:
            if len(self.metrics["total_loss"]) < window:
                continue

            windows = []
            for i in range(len(self.metrics["total_loss"]) - window + 1):
                window_metrics = {
                    "iteration_start": self.metrics["iterations"][i],
                    "iteration_end": self.metrics["iterations"][i + window - 1],
                    "loss_change": abs(
                        self.metrics["total_loss"][i]
                        - self.metrics["total_loss"][i + window - 1]
                    )
                    / max(abs(self.metrics["total_loss"][i]), 1e-5),
                    "position_grad_small": sum(
                        g < 1e-3
                        for g in self.metrics["gradient_info"]["position_grad_mag"][
                            i : i + window
                        ]
                    )
                    > window // 2,
                    "radius_grad_small": sum(
                        g < 1e-3
                        for g in self.metrics["gradient_info"]["radius_grad_mag"][
                            i : i + window
                        ]
                    )
                    > window // 2,
                }
                windows.append(window_metrics)

            results[f"window_{window}"] = windows

        return results
