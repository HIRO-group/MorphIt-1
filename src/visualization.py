"""
Visualization functions for MorphIt sphere packing system.
"""

import numpy as np
import pyvista as pv
import time
from typing import Optional, Tuple
import torch


def visualize_packing(
    model,
    scaled_mesh: Optional[object] = None,
    show_sample_points: bool = False,
    show_surface_points: bool = False,
    sample_points_subsample: int = 100000,
    surface_points_subsample: int = 100000,
    sample_point_color: str = "red",
    surface_point_color: str = "green",
    sphere_color: str = "blue",
    sphere_opacity: float = 0.9,
    point_size: int = 5,
    mesh_color: str = "white",
    mesh_line_width: float = 1.5,
    mesh_opacity: float = 0.8,
    camera_config: Optional[dict] = None,
):
    """
    Visualize sphere packing with PyVista.

    Args:
        model: MorphIt model instance
        scaled_mesh: Optional additional mesh to display
        show_sample_points: Whether to show interior sample points
        show_surface_points: Whether to show surface sample points
        sample_points_subsample: Number of interior sample points to visualize
        surface_points_subsample: Number of surface sample points to visualize
        sample_point_color: Color for interior sample points
        surface_point_color: Color for surface sample points
        sphere_color: Color for spheres
        sphere_opacity: Opacity for spheres (0-1)
        point_size: Size of points visualization
        mesh_color: Color for mesh wireframes
        mesh_line_width: Line width for mesh wireframes
        mesh_opacity: Opacity for mesh wireframes
        camera_config: Optional camera configuration dictionary
    """
    print("Starting visualization...")
    start_time = time.time()

    # Default camera configuration
    if camera_config is None:
        camera_config = {
            "position": (1.0, 1.0, 1.0),
            "focal_point": (0.0, 0.0, 0.0),
            "view_up": (0.0, 0.0, 1.0),
            "azimuth": 80,
            "elevation": 120,
            "roll": 120,
            "zoom": 1.5,
        }

    # Get sphere data
    centers = model.centers.detach().cpu().numpy()
    radii = model.radii.detach().cpu().numpy()

    # Create plotter
    plotter = pv.Plotter()

    # Add main mesh
    _add_mesh_to_plotter(
        plotter, model.query_mesh, mesh_color, mesh_line_width, mesh_opacity
    )

    # Add scaled mesh if provided
    if scaled_mesh is not None:
        _add_mesh_to_plotter(
            plotter, scaled_mesh, mesh_color, mesh_line_width, mesh_opacity
        )

    # Add spheres
    _add_spheres_to_plotter(plotter, centers, radii, sphere_color, sphere_opacity)

    # Add sample points if requested
    if show_sample_points and hasattr(model, "inside_samples"):
        _add_sample_points_to_plotter(
            plotter,
            model.inside_samples,
            sample_points_subsample,
            sample_point_color,
            point_size,
            "interior",
        )

    # Add surface points if requested
    if show_surface_points and hasattr(model, "surface_samples"):
        _add_sample_points_to_plotter(
            plotter,
            model.surface_samples,
            surface_points_subsample,
            surface_point_color,
            point_size,
            "surface",
        )

    # Configure camera
    _configure_camera(plotter, camera_config)

    # Show visualization
    plotter.show()

    end_time = time.time()
    print(f"Visualization completed in {end_time - start_time:.2f} seconds")


def _add_mesh_to_plotter(plotter, mesh, color, line_width, opacity):
    """Add mesh to plotter as wireframe."""
    pv_mesh = pv.PolyData(
        mesh.vertices,
        np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).flatten(),
    )
    plotter.add_mesh(
        pv_mesh,
        style="wireframe",
        color=color,
        line_width=line_width,
        opacity=opacity,
    )


def _add_spheres_to_plotter(plotter, centers, radii, color, opacity):
    """Add spheres to plotter."""
    # Sort spheres by radius (larger first for better visualization)
    sort_idx = np.argsort(-radii)
    centers = centers[sort_idx]
    radii = radii[sort_idx]

    # Create spheres as multiblock for efficiency
    spheres = pv.MultiBlock()
    for center, radius in zip(centers, radii):
        sphere = pv.Sphere(radius=radius, center=center)
        spheres.append(sphere)

    # Add all spheres at once
    plotter.add_mesh(spheres, color=color, opacity=opacity)


def _add_sample_points_to_plotter(
    plotter, samples, subsample, color, point_size, point_type
):
    """Add sample points to plotter."""
    # Convert to numpy if needed
    if hasattr(samples, "detach"):
        all_points = samples.detach().cpu().numpy()
    else:
        all_points = samples

    print(f"Adding {len(all_points)} {point_type} sample points")

    # Subsample if needed
    if len(all_points) > subsample:
        idx = np.random.choice(len(all_points), subsample, replace=False)
        points = all_points[idx]
        print(f"Subsampled to {len(points)} points")
    else:
        points = all_points

    # Create and add point cloud
    point_cloud = pv.PolyData(points)
    plotter.add_mesh(
        point_cloud,
        color=color,
        point_size=point_size,
        render_points_as_spheres=True,
    )


def _configure_camera(plotter, camera_config):
    """Configure camera settings."""
    plotter.camera_position = [
        camera_config["position"],
        camera_config["focal_point"],
        camera_config["view_up"],
    ]
    plotter.camera.azimuth = camera_config["azimuth"]
    plotter.camera.elevation = camera_config["elevation"]
    plotter.camera.roll = camera_config["roll"]
    plotter.camera.zoom(camera_config["zoom"])


class MorphItVisualizer:
    """
    Advanced visualization class for MorphIt with PyVista integration.
    """

    def __init__(self, model, config):
        """
        Initialize visualizer.

        Args:
            model: MorphIt model instance
            config: Visualization configuration
        """
        self.model = model
        self.config = config
        self.plotter = None
        self.sphere_list = []

    def __del__(self):
        if hasattr(self, "plotter") and self.plotter is not None:
            try:
                self.plotter.close()
            except:
                pass  # Ignore cleanup errors

    def pv_init(
        self,
        enabled: bool = False,
        off_screen: bool = False,
        save_video: bool = False,
        filename: str = "morphit.mp4",
    ):
        """
        Initialize PyVista plotter.

        Args:
            off_screen: Whether to run in off-screen mode
            save_video: Whether to save video
            filename: Video filename
        """
        if enabled == False:
            print(f"Disabled pyvista visualization.")
            return

        self.plotter = pv.Plotter(off_screen=off_screen)
        self.off_screen = off_screen
        self.save_video = save_video

        if save_video:
            self.plotter.open_movie(filename)

        # Add mesh
        self.plotter.add_mesh(
            self.model.query_mesh,
            color=self.config.mesh_color,
            style="wireframe",
            line_width=self.config.mesh_line_width,
            opacity=self.config.mesh_opacity,
        )

        # Add spheres
        self._add_initial_spheres()

        # Configure camera
        self._configure_initial_camera()

        if not off_screen:
            self.plotter.show(interactive_update=True, auto_close=False)

    def _add_initial_spheres(self):
        """Add initial spheres to plotter."""
        self.sphere_list = []
        centers = self.model.centers.detach().cpu().numpy()
        radii = self.model.radii.detach().cpu().numpy()

        for center, radius in zip(centers, radii):
            sphere = pv.Sphere(center=center, radius=radius)
            self.sphere_list.append(sphere)
            self.plotter.add_mesh(
                sphere,
                color=self.config.sphere_color,
                opacity=self.config.sphere_opacity,
            )

    def _configure_initial_camera(self):
        """Configure initial camera settings."""
        self.plotter.camera_position = [
            self.config.camera_position,
            self.config.camera_focal_point,
            self.config.camera_view_up,
        ]
        self.plotter.camera.azimuth = self.config.camera_azimuth
        self.plotter.camera.elevation = self.config.camera_elevation
        self.plotter.camera.roll = self.config.camera_roll
        self.plotter.camera.zoom(self.config.camera_zoom)

    def pv_render(self):
        """Render current sphere state."""
        if self.plotter is None:
            return

        # Get current sphere data
        centers = self.model.centers.detach().cpu().numpy()
        radii = self.model.radii.detach().cpu().numpy()

        # Update existing spheres
        for i, (sphere, center, radius) in enumerate(
            zip(self.sphere_list, centers, radii)
        ):
            if i < len(self.sphere_list):
                sphere.points = pv.Sphere(center=center, radius=radius).points

        # Handle sphere count changes
        if len(centers) != len(self.sphere_list):
            self._handle_sphere_count_change(centers, radii)

        # Render or update
        if self.off_screen:
            self.plotter.render()
        else:
            self.plotter.update()

        # Write frame for video
        if self.save_video:
            self.plotter.write_frame()

    def _handle_sphere_count_change(self, centers, radii):
        """Handle changes in sphere count during training."""
        current_count = len(self.sphere_list)
        new_count = len(centers)

        if new_count > current_count:
            # Add new spheres
            for i in range(current_count, new_count):
                sphere = pv.Sphere(center=centers[i], radius=radii[i])
                self.sphere_list.append(sphere)
                self.plotter.add_mesh(
                    sphere,
                    color=self.config.sphere_color,
                    opacity=self.config.sphere_opacity,
                )
        elif new_count < current_count:
            # Remove excess spheres (this is more complex in PyVista)
            # For now, we'll recreate all spheres
            self._recreate_all_spheres(centers, radii)

    def _recreate_all_spheres(self, centers, radii):
        """Recreate all spheres (fallback for complex updates)."""
        # Clear existing spheres
        # Note: This is a simplified approach
        self.sphere_list = []

        # Add new spheres
        for center, radius in zip(centers, radii):
            sphere = pv.Sphere(center=center, radius=radius)
            self.sphere_list.append(sphere)
            self.plotter.add_mesh(
                sphere,
                color=self.config.sphere_color,
                opacity=self.config.sphere_opacity,
            )

    def pv_close(self):
        """Close PyVista plotter."""
        if self.plotter is not None:
            self.plotter.close()

    def pv_screenshot(self, filename: str):
        """Take screenshot."""
        if self.plotter is not None:
            self.plotter.screenshot(filename)

    def save_visualization_config(self, filename: str):
        """Save current visualization configuration."""
        import json
        from pathlib import Path

        config_dict = {
            "camera_position": self.config.camera_position,
            "camera_focal_point": self.config.camera_focal_point,
            "camera_view_up": self.config.camera_view_up,
            "camera_azimuth": self.config.camera_azimuth,
            "camera_elevation": self.config.camera_elevation,
            "camera_roll": self.config.camera_roll,
            "camera_zoom": self.config.camera_zoom,
            "sphere_color": self.config.sphere_color,
            "sphere_opacity": self.config.sphere_opacity,
            "mesh_color": self.config.mesh_color,
            "mesh_line_width": self.config.mesh_line_width,
            "mesh_opacity": self.config.mesh_opacity,
        }

        with open(filename, "w") as f:
            json.dump(config_dict, f, indent=4)

        print(f"Visualization config saved to {filename}")
