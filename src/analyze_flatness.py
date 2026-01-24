#!/usr/bin/env python3
"""
Standalone script to analyze flatness loss and check bottom surface alignment.
Visualizes which faces and spheres are triggered by flatness loss.
"""

import numpy as np
import trimesh
import json
import pyvista as pv
from pathlib import Path
from typing import List, Tuple, Dict


class FlatnessAnalyzer:
    """Analyzes flatness loss for sphere packing results."""

    def __init__(self, mesh_path: str, results_path: str):
        """
        Initialize analyzer.

        Args:
            mesh_path: Path to mesh OBJ file
            results_path: Path to morphit_results.json
        """
        print(f"Loading mesh: {mesh_path}")
        self.mesh = trimesh.load(mesh_path, force="mesh")

        print(f"Loading results: {results_path}")
        with open(results_path, "r") as f:
            data = json.load(f)

        self.centers = np.array(data["centers"], dtype=np.float32)
        self.radii = np.array(data["radii"], dtype=np.float32)

        print(f"Loaded {len(self.centers)} spheres")
        print(
            f"Mesh: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")

        # Sample surface points and normals (matching MorphIt's approach)
        self.surface_samples, self.surface_normals = self._sample_surface(1000)

    def _sample_surface(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample points on mesh surface with normals."""
        print(f"Sampling {num_samples} surface points...")
        points, face_indices = trimesh.sample.sample_surface(
            self.mesh, num_samples)
        normals = self.mesh.face_normals[face_indices]
        return points.astype(np.float32), normals.astype(np.float32)

    def cluster_surface_by_normal(
        self, angle_threshold: float = 0.1
    ) -> List[Tuple[np.ndarray, np.ndarray, List[int]]]:
        """
        Cluster surface samples by their normal direction to find flat regions.

        Args:
            angle_threshold: Normals within this angle (radians) are same face

        Returns:
            List of (representative_normal, samples, sample_indices) for each flat region
        """
        print(
            f"\nClustering surface by normals (threshold: {angle_threshold} rad)...")

        cos_threshold = np.cos(angle_threshold)

        n = len(self.surface_normals)
        assigned = np.zeros(n, dtype=bool)
        face_groups = []

        for i in range(n):
            if assigned[i]:
                continue

            # Find all samples with similar normal
            ref_normal = self.surface_normals[i]
            similarities = np.abs(self.surface_normals @ ref_normal)
            similar_mask = (similarities > cos_threshold) & (~assigned)

            if similar_mask.sum() < 20:  # Skip small clusters
                assigned[i] = True
                continue

            # Mark as assigned
            assigned[similar_mask] = True

            # Compute representative normal
            cluster_normals = self.surface_normals[similar_mask]
            signs = np.sign(cluster_normals @ ref_normal)
            aligned_normals = cluster_normals * signs[:, np.newaxis]
            rep_normal = aligned_normals.mean(axis=0)
            rep_normal = rep_normal / np.linalg.norm(rep_normal)

            # Get samples
            cluster_samples = self.surface_samples[similar_mask]
            sample_indices = np.where(similar_mask)[0]

            face_groups.append((rep_normal, cluster_samples, sample_indices))

        print(f"Found {len(face_groups)} distinct flat regions")
        return face_groups

    def analyze_flatness_loss(
        self, angle_threshold: float = 0.1, min_samples: int = 20
    ) -> Dict:
        """
        Analyze which faces and spheres are triggered by flatness loss.

        Args:
            angle_threshold: Normal angle threshold for clustering
            min_samples: Minimum samples per face group

        Returns:
            Dictionary with analysis results
        """
        print("\n" + "=" * 70)
        print("FLATNESS LOSS ANALYSIS")
        print("=" * 70)

        face_groups = self.cluster_surface_by_normal(angle_threshold)

        total_flatness_loss = 0.0
        face_results = []

        for face_idx, (face_normal, face_samples, sample_indices) in enumerate(face_groups):
            if len(face_samples) < min_samples:
                continue

            # Compute plane equation for this face
            face_point = face_samples.mean(axis=0)

            # Signed distance from each sphere center to the face plane
            center_to_plane = np.sum(
                (self.centers - face_point) * face_normal, axis=1)

            # Find spheres near this face
            margin = self.radii.mean() * 0.05
            near_face_mask = np.abs(center_to_plane) < (self.radii + margin)

            if near_face_mask.sum() < 2:
                continue

            near_centers = self.centers[near_face_mask]
            near_radii = self.radii[near_face_mask]
            near_plane_dist = center_to_plane[near_face_mask]
            near_sphere_indices = np.where(near_face_mask)[0]

            # Calculate effective surface
            effective_surface = np.abs(near_plane_dist) - near_radii

            # Calculate flatness metrics
            flatness_variance = np.var(effective_surface)
            area_weight = len(face_samples) / len(self.surface_samples)
            weighted_loss = flatness_variance * area_weight

            total_flatness_loss += weighted_loss

            # Store results
            face_result = {
                'face_idx': face_idx,
                'normal': face_normal,
                'face_point': face_point,
                'num_samples': len(face_samples),
                'sample_indices': sample_indices,
                'num_spheres': len(near_sphere_indices),
                'sphere_indices': near_sphere_indices,
                'flatness_variance': flatness_variance,
                'area_weight': area_weight,
                'weighted_loss': weighted_loss,
                'effective_surface': effective_surface,
                'effective_surface_range': effective_surface.max() - effective_surface.min(),
                'effective_surface_mean': effective_surface.mean(),
            }
            face_results.append(face_result)

            # Print face info
            print(f"\nFace {face_idx}:")
            print(
                f"  Normal: [{face_normal[0]:6.3f}, {face_normal[1]:6.3f}, {face_normal[2]:6.3f}]")
            print(f"  Samples: {len(face_samples)}")
            print(f"  Spheres: {len(near_sphere_indices)}")
            print(f"  Flatness variance: {flatness_variance:.6f}")
            print(
                f"  Effective surface range: {effective_surface.max() - effective_surface.min():.6f}")
            print(f"  Weighted loss: {weighted_loss:.6f}")

        # Sort by weighted loss
        face_results.sort(key=lambda x: x['weighted_loss'], reverse=True)

        print(f"\n{'=' * 70}")
        print(f"TOTAL FLATNESS LOSS: {total_flatness_loss:.6f}")
        print(f"Number of faces analyzed: {len(face_results)}")
        print(f"{'=' * 70}")

        return {
            'total_loss': total_flatness_loss,
            'face_results': face_results,
            'num_faces': len(face_results),
        }

    def check_bottom_surface(
        self,
        bottom_direction: str = '-z',
        alignment_tolerance: float = 0.001,
        height_threshold_pct: float = 0.05,
        absolute_height_tol: float = None
    ) -> Dict:
        """
        Check if spheres create a flat bottom surface matching the mesh.

        Args:
            bottom_direction: Direction pointing down ('x', 'y', 'z', '-x', '-y', '-z')
            alignment_tolerance: Distance tolerance for flatness (in mesh units)
            height_threshold_pct: Percentage of mesh height range to consider as "bottom" (default 0.05 = 5%)
            absolute_height_tol: If provided, use this absolute distance from min height instead of percentage

        Returns:
            Dictionary with bottom surface analysis
        """
        print("\n" + "=" * 70)
        print("BOTTOM SURFACE FLATNESS ANALYSIS")
        print("=" * 70)

        # Define direction
        direction_map = {
            'x': np.array([1, 0, 0]), '-x': np.array([-1, 0, 0]),
            'y': np.array([0, 1, 0]), '-y': np.array([0, -1, 0]),
            'z': np.array([0, 0, 1]), '-z': np.array([0, 0, -1])
        }
        down_direction = direction_map[bottom_direction]
        outward_normal = -down_direction  # Face normal points outward

        print(f"Bottom direction: {bottom_direction}")
        print(f"Down vector: {down_direction}")

        # Strategy: Find faces that are at the LOWEST position AND have upward normals
        # This correctly identifies the flat bottom surface

        # 1. Find the minimum height in the mesh
        vertex_heights = np.dot(self.mesh.vertices, down_direction)
        min_height = vertex_heights.min()
        max_height = vertex_heights.max()
        height_range = max_height - min_height

        print(
            f"\nMesh height range: {min_height:.6f} to {max_height:.6f} ({height_range*1000:.3f} mm)")

        # 2. Find faces whose vertices are near the minimum height
        # A face is on the bottom if ALL its vertices are near the bottom
        if absolute_height_tol is not None:
            height_threshold = min_height + absolute_height_tol
            print(
                f"Using ABSOLUTE height threshold: {absolute_height_tol*1000:.3f} mm from minimum")
        else:
            height_threshold = min_height + height_range * height_threshold_pct
            print(
                f"Using PERCENTAGE height threshold: {height_threshold_pct*100:.1f}% of range")

        print(f"  Threshold Z value: {height_threshold:.6f}")
        print(
            f"  Distance from minimum: {(height_threshold-min_height)*1000:.3f} mm")

        bottom_faces_mask = np.zeros(len(self.mesh.faces), dtype=bool)
        bottom_face_max_heights = []

        for i, face in enumerate(self.mesh.faces):
            face_vertex_heights = vertex_heights[face]
            # All vertices must be in bottom region
            if np.all(face_vertex_heights < height_threshold):
                bottom_faces_mask[i] = True
                bottom_face_max_heights.append(face_vertex_heights.max())

        num_faces_by_height = bottom_faces_mask.sum()
        print(
            f"Faces in bottom region (all vertices z < {height_threshold:.6f}): {num_faces_by_height}")

        if num_faces_by_height > 0:
            print(
                f"  Highest vertex in bottom faces: {max(bottom_face_max_heights):.6f}")
            print(
                f"  Range in bottom faces: {(max(bottom_face_max_heights) - min_height)*1000:.3f} mm")

        # 3. Further filter by normal direction (should point upward)
        face_normals = self.mesh.face_normals
        alignment_with_outward = np.dot(face_normals, outward_normal)
        upward_normal_mask = alignment_with_outward > 0.7  # Stricter threshold

        print(
            f"Faces with upward normals (dot product > 0.7): {upward_normal_mask.sum()}")

        # Combine both criteria
        bottom_faces_mask = bottom_faces_mask & upward_normal_mask

        num_bottom_faces = bottom_faces_mask.sum()
        print(f"Bottom faces (by height AND normal): {num_bottom_faces}")

        if num_bottom_faces == 0:
            print("WARNING: No bottom faces detected!")
            print("Trying with only height-based detection...")
            # Fallback to just height
            bottom_faces_mask = np.zeros(len(self.mesh.faces), dtype=bool)
            for i, face in enumerate(self.mesh.faces):
                face_vertex_heights = vertex_heights[face]
                if np.all(face_vertex_heights < height_threshold):
                    bottom_faces_mask[i] = True
            num_bottom_faces = bottom_faces_mask.sum()
            if num_bottom_faces == 0:
                return {'error': 'No bottom faces found'}

        # Sample points on bottom surface
        bottom_face_indices = np.where(bottom_faces_mask)[0]
        bottom_samples = []

        for face_idx in bottom_face_indices:
            face = self.mesh.faces[face_idx]
            vertices = self.mesh.vertices[face]
            # Sample multiple points per triangle
            for _ in range(10):
                r1, r2 = np.random.random(), np.random.random()
                if r1 + r2 > 1:
                    r1, r2 = 1 - r1, 1 - r2
                point = vertices[0] + r1 * (vertices[1] - vertices[0]) + \
                    r2 * (vertices[2] - vertices[0])
                bottom_samples.append(point)

        bottom_samples = np.array(bottom_samples)
        print(f"Sampled {len(bottom_samples)} points on bottom mesh surface")

        # Check mesh bottom flatness
        sample_heights = np.dot(bottom_samples, down_direction)
        mesh_bottom_range = sample_heights.max() - sample_heights.min()
        mesh_bottom_mean = sample_heights.mean()
        mesh_bottom_std = sample_heights.std()

        print(f"\n--- MESH BOTTOM SURFACE ---")
        print(f"Height range: {mesh_bottom_range*1000:.3f} mm")
        print(f"Mean height: {mesh_bottom_mean:.6f}")
        print(f"Std dev: {mesh_bottom_std*1000:.3f} mm")

        if mesh_bottom_range < alignment_tolerance:
            print(
                f"✓ Mesh bottom IS FLAT (range < {alignment_tolerance*1000:.1f} mm)")
            mesh_is_flat = True
        else:
            print(
                f"⚠ Mesh bottom has {mesh_bottom_range*1000:.3f} mm variation")
            if mesh_bottom_range < alignment_tolerance * 5:
                print(f"  (This is acceptable for a nominally flat surface)")
                mesh_is_flat = True
            else:
                print(f"  (This is NOT flat)")
                mesh_is_flat = False

        # Find spheres covering bottom
        closest_sphere_indices = set()
        for sample in bottom_samples:
            dists = np.linalg.norm(self.centers - sample, axis=1)
            surface_dists = dists - self.radii
            closest_idx = np.argmin(surface_dists)
            closest_sphere_indices.add(closest_idx)

        bottom_sphere_indices = np.array(list(closest_sphere_indices))
        print(f"\n--- SPHERE BOTTOM COVERAGE ---")
        print(
            f"Identified {len(bottom_sphere_indices)} spheres covering bottom")

        # Calculate contact heights for spheres
        bottom_centers = self.centers[bottom_sphere_indices]
        bottom_radii = self.radii[bottom_sphere_indices]

        # Contact point = center + radius * down_direction
        contact_points = bottom_centers + \
            bottom_radii[:, np.newaxis] * down_direction
        contact_heights = np.dot(contact_points, down_direction)

        sphere_bottom_range = contact_heights.max() - contact_heights.min()
        sphere_bottom_mean = contact_heights.mean()
        sphere_bottom_std = contact_heights.std()

        print(f"Height range: {sphere_bottom_range*1000:.3f} mm")
        print(f"Mean height: {sphere_bottom_mean:.6f}")
        print(f"Std dev: {sphere_bottom_std*1000:.3f} mm")

        if sphere_bottom_range < alignment_tolerance * 2:
            print(
                f"✓ Sphere bottom IS FLAT (range < {alignment_tolerance*2*1000:.1f} mm)")
            spheres_are_flat = True
        else:
            print(
                f"✗ Sphere bottom is NOT flat (range = {sphere_bottom_range*1000:.3f} mm)")
            spheres_are_flat = False

        # Detailed breakdown
        print(f"\n--- INDIVIDUAL SPHERE CONTACT HEIGHTS ---")
        sorted_indices = np.argsort(contact_heights)
        for idx in sorted_indices:
            i = bottom_sphere_indices[idx]
            center_height = np.dot(bottom_centers[idx], down_direction)
            deviation_from_mean = contact_heights[idx] - sphere_bottom_mean
            print(f"  Sphere {i:3d}: center_h={center_height:7.4f}, "
                  f"radius={bottom_radii[idx]:6.4f}, "
                  f"contact_h={contact_heights[idx]:7.4f}, "
                  f"dev={deviation_from_mean:+7.4f} ({deviation_from_mean*1000:+6.2f}mm)")

        # Alignment offset between mesh and spheres
        height_offset = abs(mesh_bottom_mean - sphere_bottom_mean)
        print(f"\n--- ALIGNMENT OFFSET ---")
        print(f"Mesh mean height:   {mesh_bottom_mean:.6f}")
        print(f"Sphere mean height: {sphere_bottom_mean:.6f}")
        print(f"Offset: {height_offset*1000:.3f} mm")

        # Final assessment
        print(f"\n{'=' * 70}")
        print("ASSESSMENT:")
        print(f"{'=' * 70}")

        if not mesh_is_flat:
            print(f"⚠ MESH BOTTOM HAS VARIATION")
            print(
                f"  The mesh itself has {mesh_bottom_range*1000:.3f} mm variation")
            print(f"  This may be due to mesh artifacts or modeling")
            status = "MESH_NOT_FLAT"
        elif not spheres_are_flat:
            print(f"✗ SPHERES CREATE UNEVEN BOTTOM")
            print(
                f"  Mesh is reasonably flat ({mesh_bottom_range*1000:.3f} mm)")
            print(
                f"  But spheres have {sphere_bottom_range*1000:.3f} mm variation")
            print(f"  This will cause simulation instability!")
            status = "UNEVEN"

            # Identify the problematic spheres
            deviation_threshold = alignment_tolerance * 1.5
            problem_spheres = []
            for idx, dev in enumerate(contact_heights - sphere_bottom_mean):
                if abs(dev) > deviation_threshold:
                    problem_spheres.append((bottom_sphere_indices[idx], dev))

            if problem_spheres:
                print(
                    f"\n  Problem spheres (deviation > {deviation_threshold*1000:.1f} mm):")
                for sphere_idx, dev in problem_spheres:
                    print(f"    Sphere {sphere_idx}: {dev*1000:+6.2f} mm")
        else:
            print(f"✓ BOTTOM SURFACE IS WELL-ALIGNED")
            print(f"  Mesh range: {mesh_bottom_range*1000:.3f} mm")
            print(f"  Sphere range: {sphere_bottom_range*1000:.3f} mm")
            status = "FLAT_AND_ALIGNED"

        print(f"{'=' * 70}")

        return {
            'num_spheres': len(bottom_sphere_indices),
            'sphere_indices': bottom_sphere_indices,
            'mesh_is_flat': mesh_is_flat,
            'spheres_are_flat': spheres_are_flat,
            'mesh_bottom_range': mesh_bottom_range,
            'mesh_bottom_mean': mesh_bottom_mean,
            'sphere_bottom_range': sphere_bottom_range,
            'sphere_bottom_mean': sphere_bottom_mean,
            'height_offset': height_offset,
            'status': status,
            'contact_heights': contact_heights,
            'bottom_samples': bottom_samples,
            # Add face indices for visualization
            'bottom_face_indices': bottom_face_indices,
        }

    def visualize_flatness_analysis(
        self, flatness_results: Dict, bottom_results: Dict = None
    ):
        """
        Visualize flatness analysis with PyVista.

        Color Legend:
        - SPHERES: Rainbow colors - each color represents spheres near a different flat face
        - SURFACE SAMPLES: Rainbow colors - surface points grouped by face normal direction
        - BOTTOM SAMPLES: RED - points on the actual bottom surface of the mesh

        Args:
            flatness_results: Results from analyze_flatness_loss()
            bottom_results: Optional results from check_bottom_surface()
        """
        print("\n" + "=" * 70)
        print("CREATING VISUALIZATION")
        print("=" * 70)
        print("\nColor Legend:")
        print("  - Spheres: Rainbow colors = different flat face groups")
        print("  - Surface Samples: Rainbow colors = face groups")
        print("  - Bottom Faces: YELLOW = detected bottom faces (surface)")
        print("  - Bottom Samples: RED = sample points on bottom")
        print("=" * 70)

        plotter = pv.Plotter(window_size=(1400, 900))

        # Add mesh
        faces_with_counts = np.hstack(
            [np.full((len(self.mesh.faces), 1), 3), self.mesh.faces]
        ).flatten()
        pv_mesh = pv.PolyData(self.mesh.vertices, faces_with_counts)
        plotter.add_mesh(
            pv_mesh,
            style='wireframe',
            color='white',
            line_width=1.5,
            opacity=0.6,
            label='Mesh'
        )

        # Highlight bottom faces if available
        if bottom_results and 'bottom_face_indices' in bottom_results:
            bottom_face_indices = bottom_results['bottom_face_indices']
            if len(bottom_face_indices) > 0:
                # Create a mesh with just the bottom faces
                bottom_faces = self.mesh.faces[bottom_face_indices]
                bottom_faces_with_counts = np.hstack(
                    [np.full((len(bottom_faces), 1), 3), bottom_faces]
                ).flatten()
                bottom_mesh = pv.PolyData(
                    self.mesh.vertices, bottom_faces_with_counts)
                plotter.add_mesh(
                    bottom_mesh,
                    color='yellow',
                    opacity=0.7,
                    style='surface',
                    label='Detected Bottom Faces'
                )
                print(
                    f"Highlighted {len(bottom_face_indices)} detected bottom faces in YELLOW")

        # Color code spheres by which face group they belong to
        sphere_colors = np.zeros(len(self.centers))

        # Assign colors to spheres based on face groups
        face_results = flatness_results['face_results']
        # Top 10 faces
        for face_idx, face_result in enumerate(face_results[:10]):
            sphere_indices = face_result['sphere_indices']
            sphere_colors[sphere_indices] = face_idx + 1

        # Highlight bottom spheres if available
        if bottom_results and 'sphere_indices' in bottom_results:
            bottom_sphere_indices = bottom_results['sphere_indices']
            sphere_colors[bottom_sphere_indices] = - \
                1  # Special color for bottom

        # Add all spheres
        spheres = pv.MultiBlock()
        for i, (center, radius) in enumerate(zip(self.centers, self.radii)):
            sphere = pv.Sphere(radius=radius, center=center,
                               theta_resolution=16, phi_resolution=16)
            sphere['color'] = np.full(sphere.n_points, sphere_colors[i])
            spheres.append(sphere)

        plotter.add_mesh(
            spheres,
            scalars='color',
            cmap='rainbow',
            opacity=0.4,
            label='Spheres'
        )

        # Add surface samples colored by face group
        if len(face_results) > 0:
            sample_colors = np.zeros(len(self.surface_samples))
            for face_idx, face_result in enumerate(face_results[:10]):
                sample_indices = face_result['sample_indices']
                sample_colors[sample_indices] = face_idx + 1

            surface_points = pv.PolyData(self.surface_samples)
            surface_points['face_group'] = sample_colors
            plotter.add_mesh(
                surface_points,
                scalars='face_group',
                cmap='rainbow',
                point_size=8,
                render_points_as_spheres=True,
                label='Surface Samples'
            )

        # Add bottom samples if available
        if bottom_results and 'bottom_samples' in bottom_results:
            bottom_points = pv.PolyData(bottom_results['bottom_samples'])
            plotter.add_mesh(
                bottom_points,
                color='red',
                point_size=12,
                render_points_as_spheres=True,
                label='Bottom Surface'
            )

        # Add text info
        info_text = f"Total Flatness Loss: {flatness_results['total_loss']:.6f}\n"
        info_text += f"Flat Faces: {flatness_results['num_faces']}\n"
        info_text += f"Total Spheres: {len(self.centers)}\n"

        if bottom_results and 'status' in bottom_results:
            info_text += f"\nBottom Status: {bottom_results['status']}\n"
            if 'bottom_face_indices' in bottom_results:
                info_text += f"Bottom Faces: {len(bottom_results['bottom_face_indices'])}\n"
            info_text += f"Bottom Spheres: {bottom_results['num_spheres']}\n"
            if 'sphere_bottom_range' in bottom_results:
                info_text += f"Sphere Range: {bottom_results['sphere_bottom_range']*1000:.2f} mm\n"
                info_text += f"Mesh Range: {bottom_results['mesh_bottom_range']*1000:.2f} mm"

        plotter.add_text(info_text, position='upper_left', font_size=11)

        # Add top faces info
        top_faces_text = "Top 5 Faces by Loss:\n"
        for i, face_result in enumerate(face_results[:5]):
            top_faces_text += f"{i+1}. Spheres:{face_result['num_spheres']:2d} "
            top_faces_text += f"Loss:{face_result['weighted_loss']:.4f}\n"

        plotter.add_text(top_faces_text, position='upper_right', font_size=10)

        # Set camera
        plotter.camera_position = [
            (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
        plotter.camera.azimuth = 80
        plotter.camera.elevation = 120
        plotter.camera.zoom(1.5)

        plotter.add_legend()

        print("Launching visualization...")
        plotter.show()

    def print_summary(self, flatness_results: Dict, bottom_results: Dict = None):
        """Print comprehensive summary."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        # Flatness summary
        print("\nFlatness Loss Summary:")
        print(f"  Total flatness loss: {flatness_results['total_loss']:.6f}")
        print(f"  Number of flat faces: {flatness_results['num_faces']}")

        print("\nTop 5 faces by weighted loss:")
        for i, face_result in enumerate(flatness_results['face_results'][:5]):
            print(f"  {i+1}. Face {face_result['face_idx']}:")
            print(f"     Spheres: {face_result['num_spheres']}")
            print(f"     Weighted loss: {face_result['weighted_loss']:.6f}")
            print(f"     Variance: {face_result['flatness_variance']:.6f}")
            print(
                f"     Surface range: {face_result['effective_surface_range']:.6f}")

        # Bottom surface summary
        if bottom_results and 'status' in bottom_results:
            print("\nBottom Surface Summary:")
            print(f"  Status: {bottom_results['status']}")
            if 'bottom_face_indices' in bottom_results:
                print(
                    f"  Bottom faces detected: {len(bottom_results['bottom_face_indices'])}")
            print(f"  Bottom spheres: {bottom_results['num_spheres']}")

            if 'mesh_is_flat' in bottom_results:
                print(f"  Mesh is flat: {bottom_results['mesh_is_flat']}")
                print(
                    f"  Spheres are flat: {bottom_results['spheres_are_flat']}")
                print(
                    f"  Mesh range: {bottom_results['mesh_bottom_range']*1000:.3f} mm")
                print(
                    f"  Sphere range: {bottom_results['sphere_bottom_range']*1000:.3f} mm")
                print(
                    f"  Height offset: {bottom_results['height_offset']*1000:.3f} mm")

        print("=" * 70)


def main():
    """Main function."""
    # File paths
    MESH_FILE = "../mesh_models/bunny.obj"
    RESULTS_FILE = "results/output/morphit_results.json"

    # ============ ADJUSTABLE PARAMETERS ============
    # Use ABSOLUTE height tolerance (in meters) instead of percentage
    # This looks for faces within this distance from the minimum Z
    # For a bunny with flat bottom, try 0.001 to 0.005 (1-5mm)
    ABSOLUTE_HEIGHT_TOL = 0.003  # 3mm from the lowest point

    # Set to None to use percentage-based detection instead
    # HEIGHT_THRESHOLD_PCT = 0.02  # 2% of height range

    # Tolerance for flatness checks (in mesh units, typically meters)
    ALIGNMENT_TOLERANCE = 0.001  # 1mm
    # ===============================================

    print("=" * 70)
    print("FLATNESS ANALYZER FOR MORPHIT")
    print("=" * 70)
    print("\nThis script analyzes:")
    print("1. Which faces trigger the flatness loss")
    print("2. Which spheres are near each flat face")
    print("3. Whether the bottom surface is flat and aligned")
    print("\nAdjust ABSOLUTE_HEIGHT_TOL if red dots appear on wrong surfaces")
    print("=" * 70)

    # Check files exist
    if not Path(MESH_FILE).exists():
        print(f"ERROR: Mesh file not found: {MESH_FILE}")
        return

    if not Path(RESULTS_FILE).exists():
        print(f"ERROR: Results file not found: {RESULTS_FILE}")
        return

    # Create analyzer
    analyzer = FlatnessAnalyzer(MESH_FILE, RESULTS_FILE)

    # Analyze flatness loss
    flatness_results = analyzer.analyze_flatness_loss(
        angle_threshold=0.1,  # radians (~5.7 degrees)
        min_samples=20
    )

    # Check bottom surface
    bottom_results = analyzer.check_bottom_surface(
        bottom_direction='-z',  # negative Z is down
        alignment_tolerance=ALIGNMENT_TOLERANCE,
        absolute_height_tol=ABSOLUTE_HEIGHT_TOL  # Use absolute tolerance
    )

    # Print summary
    analyzer.print_summary(flatness_results, bottom_results)

    # Visualize
    print("\nGenerating visualization...")
    analyzer.visualize_flatness_analysis(flatness_results, bottom_results)


if __name__ == "__main__":
    main()
