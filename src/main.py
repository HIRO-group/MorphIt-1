"""
Main entry point for MorphIt sphere packing system.
"""

import time
from pathlib import Path

from config import get_config, update_config_from_dict
from morphit import MorphIt
from training import train_morphit
from visualization import visualize_packing


def main():
    """Main function to run MorphIt sphere packing."""
    print("=== MorphIt Sphere Packing System ===")

    start_time = time.time()

    # Load configuration
    config = get_config("MorphIt-B")  # Use MorphIt-B loss configuration

    # Update configuration for this run
    config_updates = {
        "model.num_spheres": 15,
        "model.mesh_path": "../mesh_models/fr3/collision/link0.obj",
        "training.iterations": 500,
        "training.verbose_frequency": 10,
        "training.density_control_min_interval": 50,
        "visualization.off_screen": False,
        "visualization.save_video": False,
        "visualization.video_filename": "morphit_evolution.mp4",
    }

    config = update_config_from_dict(config, config_updates)

    # Create model
    print("Initializing MorphIt model...")
    model = MorphIt(config)

    # Initialize visualization
    print("Setting up visualization...")
    model.pv_init(
        off_screen=config.visualization.off_screen,
        save_video=config.visualization.save_video,
        filename=config.visualization.video_filename,
    )

    # Show initial state
    print("Initial state:")
    model.print_statistics()

    # Visualize initial packing
    print("Visualizing initial packing...")
    visualize_packing(
        model,
        show_sample_points=config.visualization.show_sample_points,
        show_surface_points=config.visualization.show_surface_points,
        sphere_color=config.visualization.sphere_color,
        sphere_opacity=config.visualization.sphere_opacity,
    )

    # Train the model
    print("\nStarting training...")
    tracker = train_morphit(model)

    # Final pruning
    print("\nFinal pruning...")
    from density_control import DensityController

    density_controller = DensityController(model, config)
    density_controller.prune_spheres()

    # Show final state
    print("\nFinal state:")
    model.print_statistics()

    # Save results
    print("\nSaving results...")
    model.save_results()
    tracker.save()

    # Visualize final packing
    print("Visualizing final packing...")
    visualize_packing(model)

    # Plot training metrics
    print("Plotting training metrics...")
    tracker.plot_training_metrics()

    # Print convergence analysis
    convergence_analysis = tracker.analyze_convergence_windows([20, 50, 100])
    print("\n=== Convergence Analysis ===")
    for window_size, windows in convergence_analysis.items():
        if windows:
            last_window = windows[-1]
            print(f"Window size {window_size}:")
            print(f"  Loss change: {last_window['loss_change']:.6f}")
            print(f"  Position gradients small: {last_window['position_grad_small']}")
            print(f"  Radius gradients small: {last_window['radius_grad_small']}")

    # Print total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.4f} seconds")

    print("\n=== MorphIt Complete ===")


if __name__ == "__main__":
    # Run the main experiment
    main()
