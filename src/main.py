"""
Main entry point for MorphIt sphere packing system.
"""

from config import get_config, update_config_from_dict
from morphit import MorphIt
from training import train_morphit
from visualization import visualize_packing


def main():
    """Main function to run MorphIt sphere packing."""
    print("=== MorphIt Sphere Packing System ===")

    # Load configuration
    config = get_config("MorphIt-B")  # Use MorphIt-B loss configuration

    # Update configuration for this run
    config_updates = {
        "model.num_spheres": 15,
        "model.mesh_path": "../mesh_models/fr3/collision/link0.obj",
        "training.iterations": 50,
        "training.verbose_frequency": 10,
        "training.logging_enabled": False,
        "training.density_control_min_interval": 25,
        "visualization.enabled": False,
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
        enabled=config.visualization.enabled,
        off_screen=config.visualization.off_screen,
        save_video=config.visualization.save_video,
        filename=config.visualization.video_filename,
    )

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

    print("\n=== MorphIt Complete ===")


if __name__ == "__main__":
    # Run the main experiment
    main()
