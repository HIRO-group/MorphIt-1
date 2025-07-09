#!/usr/bin/env python3
"""
Standalone script to visualize sphere packing results from JSON files.
Usage: python visualize_packing.py [results_file.json]
"""

import sys
import json
import numpy as np
import pyvista as pv
import trimesh
from pathlib import Path
import argparse


def load_packing_results(json_file):
    """Load sphere packing results from JSON file."""
    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        centers = np.array(data["centers"], dtype=np.float32)
        radii = np.array(data["radii"], dtype=np.float32)
        mesh_path = data["mesh_path"]

        print(f"Loaded {len(centers)} spheres from {json_file}")
        print(f"Mesh path: {mesh_path}")

        return centers, radii, mesh_path

    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)


def load_mesh(mesh_path):
    """Load mesh from file path."""
    try:
        # Try to load the mesh
        if not Path(mesh_path).exists():
            print(f"Warning: Mesh file not found at {mesh_path}")
            print("Proceeding without mesh visualization...")
            return None

        mesh = trimesh.load(mesh_path, force="mesh")
        print(
            f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces"
        )
        return mesh

    except Exception as e:
        print(f"Warning: Could not load mesh from {mesh_path}: {e}")
        print("Proceeding without mesh visualization...")
        return None


def create_pyvista_mesh(trimesh_mesh):
    """Convert trimesh to PyVista mesh."""
    if trimesh_mesh is None:
        return None

    try:
        # Create PyVista mesh from trimesh
        faces_with_counts = np.hstack(
            [np.full((len(trimesh_mesh.faces), 1), 3), trimesh_mesh.faces]
        ).flatten()

        pv_mesh = pv.PolyData(trimesh_mesh.vertices, faces_with_counts)
        return pv_mesh

    except Exception as e:
        print(f"Error converting mesh to PyVista: {e}")
        return None


def create_sphere_multiblock(centers, radii, max_spheres=None):
    """Create PyVista MultiBlock of spheres."""
    if max_spheres is not None and len(centers) > max_spheres:
        print(f"Warning: Too many spheres ({len(centers)}), showing only {max_spheres}")
        # Sort by radius (largest first) for better visualization
        sort_idx = np.argsort(-radii)[:max_spheres]
        centers = centers[sort_idx]
        radii = radii[sort_idx]

    spheres = pv.MultiBlock()

    print(f"Creating {len(centers)} spheres...")
    for i, (center, radius) in enumerate(zip(centers, radii)):
        if i % 20 == 0:  # Progress indicator
            print(f"  Created {i}/{len(centers)} spheres")

        sphere = pv.Sphere(
            radius=radius, center=center, theta_resolution=16, phi_resolution=16
        )
        spheres.append(sphere)

    print(f"Created {len(spheres)} spheres")
    return spheres


def visualize_packing(centers, radii, mesh=None, config=None):
    """Create PyVista visualization of sphere packing."""

    # Default configuration (matching MorphIt visualization exactly)
    default_config = {
        "sphere_color": "blue",
        "sphere_opacity": 0.3,
        "mesh_color": "white",
        "mesh_line_width": 1.5,
        "mesh_opacity": 0.8,
        "camera_position": (1.0, 1.0, 1.0),
        "camera_focal_point": (0.0, 0.0, 0.0),
        "camera_view_up": (0.0, 0.0, 1.0),
        "camera_azimuth": 80,
        "camera_elevation": 120,
        "camera_roll": 120,
        "camera_zoom": 1.5,
        "max_spheres": 100,  # Limit for performance
        "window_size": (1024, 768),
        "show_axes": False,  # MorphIt doesn't show axes by default
        "show_edges": True,
    }

    # Update with user config
    if config:
        default_config.update(config)

    # Create plotter
    plotter = pv.Plotter(window_size=default_config["window_size"])

    # Set background color only if specified
    if "background_color" in default_config:
        plotter.set_background(default_config["background_color"])

    # Add mesh if available
    if mesh is not None:
        pv_mesh = create_pyvista_mesh(mesh)
        if pv_mesh is not None:
            plotter.add_mesh(
                pv_mesh,
                style="wireframe" if default_config["show_edges"] else "surface",
                color=default_config["mesh_color"],
                line_width=default_config["mesh_line_width"],
                opacity=default_config["mesh_opacity"],
                label="Mesh",
            )

    # Add spheres
    spheres = create_sphere_multiblock(centers, radii, default_config["max_spheres"])
    plotter.add_mesh(
        spheres,
        color=default_config["sphere_color"],
        opacity=default_config["sphere_opacity"],
        label="Spheres",
    )

    # Configure camera (matching MorphIt exactly)
    plotter.camera_position = [
        default_config["camera_position"],
        default_config["camera_focal_point"],
        default_config["camera_view_up"],
    ]
    plotter.camera.azimuth = default_config["camera_azimuth"]
    plotter.camera.elevation = default_config["camera_elevation"]
    plotter.camera.roll = default_config["camera_roll"]
    plotter.camera.zoom(default_config["camera_zoom"])

    # Add axes only if requested (MorphIt doesn't show axes by default)
    if default_config["show_axes"]:
        plotter.add_axes()

    # Add legend
    plotter.add_legend()

    # Add text info
    info_text = f"Spheres: {len(centers)}\n"
    info_text += f"Radii: {radii.min():.3f} - {radii.max():.3f}\n"
    info_text += f"Mean radius: {radii.mean():.3f}"

    plotter.add_text(info_text, position="upper_left", font_size=12)

    # Show visualization
    print("Launching visualization...")
    plotter.show()


def print_statistics(centers, radii):
    """Print statistics about the sphere packing."""
    print("\n" + "=" * 50)
    print("SPHERE PACKING STATISTICS")
    print("=" * 50)

    print(f"Number of spheres: {len(centers)}")
    print(f"Radius statistics:")
    print(f"  Min: {radii.min():.4f}")
    print(f"  Max: {radii.max():.4f}")
    print(f"  Mean: {radii.mean():.4f}")
    print(f"  Std: {radii.std():.4f}")

    # Volume statistics
    sphere_volumes = (4 / 3) * np.pi * (radii**3)
    total_volume = sphere_volumes.sum()

    print(f"Volume statistics:")
    print(f"  Total sphere volume: {total_volume:.4f}")
    print(f"  Largest sphere volume: {sphere_volumes.max():.4f}")
    print(f"  Smallest sphere volume: {sphere_volumes.min():.4f}")

    # Center bounds
    print(f"Center bounds:")
    print(f"  X: [{centers[:, 0].min():.3f}, {centers[:, 0].max():.3f}]")
    print(f"  Y: [{centers[:, 1].min():.3f}, {centers[:, 1].max():.3f}]")
    print(f"  Z: [{centers[:, 2].min():.3f}, {centers[:, 2].max():.3f}]")

    print("=" * 50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualize sphere packing results")
    parser.add_argument("json_file", nargs="?", help="Path to JSON results file")
    parser.add_argument(
        "--no-mesh", action="store_true", help="Skip mesh visualization"
    )
    parser.add_argument(
        "--max-spheres", type=int, default=100, help="Maximum spheres to show"
    )
    parser.add_argument("--sphere-color", default="blue", help="Sphere color")
    parser.add_argument(
        "--sphere-opacity", type=float, default=0.3, help="Sphere opacity"
    )
    parser.add_argument("--mesh-color", default="white", help="Mesh color")
    parser.add_argument(
        "--background", default=None, help="Background color (default: PyVista default)"
    )
    parser.add_argument(
        "--stats-only", action="store_true", help="Only print statistics"
    )
    parser.add_argument("--show-axes", action="store_true", help="Show coordinate axes")

    args = parser.parse_args()

    # Determine JSON file path
    if args.json_file:
        json_file = Path(args.json_file)
    else:
        # Default to most recent file in results directory
        results_dir = Path("results/output")
        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            sys.exit(1)

        json_files = list(results_dir.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {results_dir}")
            sys.exit(1)

        # Use most recent file
        json_file = max(json_files, key=lambda f: f.stat().st_mtime)
        print(f"Using most recent results file: {json_file}")

    if not json_file.exists():
        print(f"JSON file not found: {json_file}")
        sys.exit(1)

    # Load results
    centers, radii, mesh_path = load_packing_results(json_file)

    # Print statistics
    print_statistics(centers, radii)

    if args.stats_only:
        return

    # Load mesh
    mesh = None if args.no_mesh else load_mesh(mesh_path)

    # Create visualization config
    vis_config = {
        "sphere_color": args.sphere_color,
        "sphere_opacity": args.sphere_opacity,
        "mesh_color": args.mesh_color,
        "max_spheres": args.max_spheres,
        "show_axes": args.show_axes,
    }

    # Only set background if specified
    if args.background:
        vis_config["background_color"] = args.background

    # Visualize
    visualize_packing(centers, radii, mesh, vis_config)


if __name__ == "__main__":
    main()
