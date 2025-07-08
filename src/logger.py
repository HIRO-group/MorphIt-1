"""
Simple evolution logger for MorphIt sphere packing.
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List


class SphereEvolutionLogger:
    """
    Simple logger for tracking sphere evolution during training.
    """

    def __init__(
        self, log_name: str = "sphere_evolution", save_dir: str = "results/evolution"
    ):
        """
        Initialize evolution logger.

        Args:
            log_name: Name for the log files
            save_dir: Directory to save logs
        """
        self.log_name = log_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Storage for evolution data
        self.evolution_data = []
        self.start_time = time.time()

    def log_spheres(
        self, model, iteration: int, stage: str, metadata: Optional[Dict] = None
    ):
        """
        Log sphere state at a given iteration.

        Args:
            model: MorphIt model instance
            iteration: Current iteration number
            stage: Stage of training (e.g., "initial", "training", "final")
            metadata: Additional metadata to store
        """
        # Get sphere data
        with torch.no_grad():
            centers = model.centers.detach().cpu().numpy()
            radii = model.radii.detach().cpu().numpy()

        # Create log entry
        log_entry = {
            "iteration": iteration,
            "stage": stage,
            "timestamp": time.time() - self.start_time,
            "num_spheres": len(centers),
            "sphere_data": {
                "centers": centers.tolist(),
                "radii": radii.tolist(),
            },
            "statistics": {
                "min_radius": float(np.min(radii)),
                "max_radius": float(np.max(radii)),
                "mean_radius": float(np.mean(radii)),
                "std_radius": float(np.std(radii)),
                "total_volume": float(np.sum(4 / 3 * np.pi * radii**3)),
            },
        }

        # Add metadata if provided
        if metadata:
            log_entry["metadata"] = metadata

        # Store the entry
        self.evolution_data.append(log_entry)

        # Optionally save individual snapshots for important stages
        if stage in ["initial", "final"]:
            self._save_snapshot(log_entry, stage)

    def _save_snapshot(self, log_entry: Dict, stage: str):
        """Save individual snapshot for important stages."""
        filename = self.save_dir / f"{self.log_name}_{stage}.json"
        with open(filename, "w") as f:
            json.dump(log_entry, f, indent=4, cls=NumpyEncoder)

    def save_complete_evolution(self):
        """Save complete evolution history."""
        filename = self.save_dir / f"{self.log_name}_complete.json"

        evolution_summary = {
            "log_name": self.log_name,
            "total_entries": len(self.evolution_data),
            "total_time": time.time() - self.start_time,
            "evolution_data": self.evolution_data,
        }

        with open(filename, "w") as f:
            json.dump(evolution_summary, f, indent=4, cls=NumpyEncoder)

        print(f"Saved complete evolution log to {filename}")
        return filename

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution."""
        if not self.evolution_data:
            return {"message": "No evolution data recorded"}

        initial_entry = self.evolution_data[0]
        final_entry = self.evolution_data[-1]

        return {
            "initial_spheres": initial_entry["num_spheres"],
            "final_spheres": final_entry["num_spheres"],
            "initial_total_volume": initial_entry["statistics"]["total_volume"],
            "final_total_volume": final_entry["statistics"]["total_volume"],
            "total_iterations": len(self.evolution_data),
            "total_time": time.time() - self.start_time,
            "stages_recorded": list(
                set(entry["stage"] for entry in self.evolution_data)
            ),
        }

    def clear_evolution_data(self):
        """Clear stored evolution data."""
        self.evolution_data = []
        self.start_time = time.time()


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class SimpleRenderThread:
    """
    Simple placeholder for render thread functionality.
    This is a simplified version that doesn't actually create threads.
    """

    def __init__(self, model, render_interval: int = 5):
        """
        Initialize render thread.

        Args:
            model: MorphIt model instance
            render_interval: Interval between renders
        """
        self.model = model
        self.render_interval = render_interval
        self.last_render_iteration = 0
        self.active = False

    def start(self):
        """Start the render thread (placeholder)."""
        self.active = True
        print("Render thread started (simplified mode)")

    def stop(self):
        """Stop the render thread."""
        self.active = False
        print("Render thread stopped")

    def queue_render(self, iteration: int):
        """Queue a render operation."""
        if not self.active:
            return

        # Only render at specified intervals
        if iteration - self.last_render_iteration >= self.render_interval:
            self.model.pv_render()
            self.last_render_iteration = iteration


# For backward compatibility, create an alias
RenderThread = SimpleRenderThread
