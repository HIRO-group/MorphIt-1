"""
Example script to run MorphIt
"""

from config import get_config, update_config_from_dict
from morphit import MorphIt
from training import train_morphit
from visualization import visualize_packing

import numpy as np
import trimesh


def main():
    print("=== Hello, I'm MorphIt ===")

    # Load a pre-set configuration of weights
    # Choose between MorphIt-B, MorphIt-S, and MorphIt-V
    config = get_config("MorphIt-B")

    # BOX
    config_updates = {
        "model.num_spheres": 27,
        "model.mesh_path": "../mesh_models/box.obj",
        # For bigger or smaller shapes than the default panda link, these parameters are useful to adjust
        # "model.mesh_path": "../mesh_models/objects/t-shape/t-shape.obj",
        # "model.mesh_path": "../mesh_models/objects/pusher-stick/pusher-stick.obj",
        "model.initialization_method": "grid",  # grid, volume
        "model.radius_threshold": 0.0001,
        "model.coverage_threshold": 0.0001,
        "training.early_stopping": False,
        "training.iterations": 500,
        "training.verbose_frequency": 10,
        "training.logging_enabled": False,
        "training.density_control_min_interval": 260,
        "visualization.enabled": False,
        "visualization.off_screen": False,
        "visualization.save_video": False,
        "visualization.video_filename": "morphit_evolution.mp4",
    }

    # BUNNY
    # config_updates = {
    #     "model.num_spheres": 30,
    #     "model.mesh_path": "../mesh_models/bunny.obj",
    #     # For bigger or smaller shapes than the default panda link, these parameters are useful to adjust
    #     # "model.mesh_path": "../mesh_models/objects/t-shape/t-shape.obj",
    #     # "model.mesh_path": "../mesh_models/objects/pusher-stick/pusher-stick.obj",
    #     "model.initialization_method": "medial",
    #     "model.radius_threshold": 0.0001,
    #     "model.coverage_threshold": 0.0001,
    #     "training.early_stopping": False,
    #     "training.iterations": 800,
    #     "training.verbose_frequency": 10,
    #     "training.logging_enabled": False,
    #     "training.density_control_min_interval": 120,
    #     "training.density_control_patience": 2,
    #     "visualization.enabled": False,
    #     "visualization.off_screen": False,
    #     "visualization.save_video": False,
    #     "visualization.video_filename": "morphit_evolution.mp4",
    # }

    config = update_config_from_dict(config, config_updates)

    print("Initializing MorphIt model...")
    model = MorphIt(config)

    print("Setting up visualization...")
    model.pv_init(
        enabled=config.visualization.enabled,
        off_screen=config.visualization.off_screen,
        save_video=config.visualization.save_video,
        filename=config.visualization.video_filename,
    )

    print("Visualizing initial packing...")
    visualize_packing(
        model,
        show_sample_points=config.visualization.show_sample_points,
        show_surface_points=config.visualization.show_surface_points,
        sphere_color=config.visualization.sphere_color,
        sphere_opacity=config.visualization.sphere_opacity,
    )

    print("\nStarting training...")
    tracker = train_morphit(model)

    print("\nSaving results...")
    model.save_results()
    tracker.save()

    print("Visualizing final packing...")
    visualize_packing(model)

    print("Plotting training metrics...")
    tracker.plot_training_metrics()

    print("\n=== MorphIt Complete ===")


if __name__ == "__main__":
    main()
