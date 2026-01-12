"""
Example script to run MorphIt
"""

from config import get_config, update_config_from_dict
from morphit import MorphIt
from training import train_morphit
from visualization import visualize_packing

import numpy as np
import trimesh

mesh = trimesh.load("../mesh_models/bunny2.obj", force="mesh")

print("=== Mesh Diagnostics ===")
print(f"Is watertight: {mesh.is_watertight}")
print(f"Is volume valid: {mesh.is_volume}")
print(f"Is winding consistent: {mesh.is_winding_consistent}")
print(f"Euler number: {mesh.euler_number}")  # Should be 2 for closed mesh
print(f"Number of bodies: {len(mesh.split())}")  # Should be 1
print(
    f"Has degenerate faces: {mesh.is_degenerate.any() if hasattr(mesh, 'is_degenerate') else 'N/A'}"
)

# Check for holes
edges = mesh.edges_sorted
edges_unique, counts = np.unique(edges, axis=0, return_counts=True)
boundary_edges = edges_unique[counts == 1]
print(f"Boundary edges (holes): {len(boundary_edges)}")


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
        "model.initialization_method": "volume",  # grid, volume
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

    # config_updates = {
    #     "model.num_spheres": 100,
    #     "model.mesh_path": "../mesh_models/bunny2.obj",
    #     # For bigger or smaller shapes than the default panda link, these parameters are useful to adjust
    #     # "model.mesh_path": "../mesh_models/objects/t-shape/t-shape.obj",
    #     # "model.mesh_path": "../mesh_models/objects/pusher-stick/pusher-stick.obj",
    #     "model.initialization_method": "volume",
    #     "model.radius_threshold": 0.001,
    #     "model.coverage_threshold": 0.001,
    #     "training.early_stopping": False,
    #     "training.iterations": 1000,
    #     "training.verbose_frequency": 10,
    #     "training.logging_enabled": False,
    #     "training.density_control_min_interval": 350,
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
