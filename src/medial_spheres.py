"""
Medial Spheres for Shape Approximation

Implementation based on the paper:
"Medial Spheres for Shape Approximation" by Stolpner, Kry, and Siddiqi (2012)
IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 34, No. 6

This module computes a set of medial spheres that approximate a 3D solid represented
by a closed triangle mesh. The spheres are internal and tangent to the solid's boundary.

Algorithm Overview (from Section 3 of the paper):
1. Partition space into voxels with side length σ (voxel_size)
2. For each voxel interior to or intersected by the solid Ω:
   a. Circumscribe the voxel with a sphere S
   b. Sample points p on S
   c. For each p, compute q = p + γ(p - B(p)) where B(p) is nearest boundary point
   d. If B(p) ≠ B(q), binary search on segment (p,q) to find medial point m
   e. Estimate object angle ∠AmB at m
   f. Keep m if object angle > threshold and m is inside Ω
3. Output at most one medial sphere per voxel (the one with largest radius)

Key Properties:
- Spheres are internal and tangent to the boundary (enables exact error analysis)
- At most one sphere center per voxel (controls sphere density)
- Object angle filtering removes spheres in flat/unimportant regions
"""

import numpy as np
import trimesh
import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ----------------------------
# Data Classes
# ----------------------------


@dataclass
class MedialSphereResult:
    """Result container for medial sphere computation."""

    centers: np.ndarray  # (N, 3) sphere centers
    radii: np.ndarray  # (N,) sphere radii
    computation_time: float
    num_voxels_processed: int
    num_candidates_found: int
    num_spheres_output: int


# ----------------------------
# Utility: Sampling on a sphere
# ----------------------------


def fibonacci_sphere(samples: int) -> np.ndarray:
    """
    Generate approximately uniformly distributed points on a unit sphere
    using the Fibonacci spiral method.

    This method provides good coverage with O(n) points without requiring
    expensive optimization or rejection sampling.

    Args:
        samples: Number of points to generate

    Returns:
        Array of shape (samples, 3) with unit vectors

    References:
        - Gonzalez, "Measurement of Areas on a Sphere Using Fibonacci and
          Latitude-Longitude Lattices"
    """
    if samples <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    if samples == 1:
        return np.array([[0.0, 0.0, 1.0]])

    indices = np.arange(samples, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle ≈ 2.39996 radians

    # y goes from 1 to -1
    y = 1.0 - (indices / (samples - 1.0)) * 2.0

    # radius at each y level
    radius = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))

    # angle increments by golden angle
    theta = phi * indices

    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    return np.stack([x, y, z], axis=1)


# ----------------------------
# Mesh proximity helper
# ----------------------------


class MeshProximityQuery:
    """
    Wrapper for efficient mesh proximity queries.

    Provides methods for:
    - Finding nearest points on mesh surface
    - Testing if points are inside the mesh

    Uses trimesh's ProximityQuery which builds a BVH for efficient queries.
    """

    def __init__(self, mesh: trimesh.Trimesh):
        """
        Initialize proximity query structure.

        Args:
            mesh: A watertight trimesh.Trimesh object
        """
        self.mesh = mesh
        self._proximity = trimesh.proximity.ProximityQuery(mesh)
        logger.debug(
            f"Initialized MeshProximityQuery with {len(mesh.faces)} faces")

    def nearest_point(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the nearest point on the mesh surface for each query point.

        This implements B(p) from the paper - the nearest point on boundary B to point p.

        Args:
            points: Query points, shape (N, 3) or (3,)

        Returns:
            Tuple of:
                - surface_points: Nearest points on mesh, shape (N, 3)
                - distances: Distances to nearest points, shape (N,)
                - triangle_ids: IDs of triangles containing nearest points, shape (N,)
        """
        points = np.atleast_2d(points)
        surface_points, distances, triangle_ids = self._proximity.on_surface(
            points)
        return surface_points, distances, triangle_ids

    def is_inside(self, points: np.ndarray) -> np.ndarray:
        """
        Test if points are inside the mesh.

        Args:
            points: Query points, shape (N, 3) or (3,)

        Returns:
            Boolean array of shape (N,) indicating inside/outside
        """
        points = np.atleast_2d(points)
        return self.mesh.contains(points)


# ----------------------------
# Object angle computation (Section 3, Eq. for ∠AmB)
# ----------------------------


def compute_object_angle(m: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the object angle ∠AmB at medial point m.

    The object angle is defined in Section 3 of the paper as the angle
    at m between vectors (A - m) and (B - m), where A and B are two
    distinct nearest boundary points to m.

    From the paper: "The object angle is a valuable simplification criterion
    for the medial surface. [...] removal of medial balls having a small
    object angle has a small impact on the volume of the reconstructed object."

    Args:
        m: Medial point, shape (3,)
        A: First nearest boundary point, shape (3,)
        B: Second nearest boundary point, shape (3,)

    Returns:
        Object angle in radians (0 to π)
    """
    v1 = A - m
    v2 = B - m

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-12 or norm2 < 1e-12:
        logger.debug(
            "Degenerate object angle computation (zero-length vector)")
        return 0.0

    # Compute cosine of angle, clipped for numerical stability
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return float(np.arccos(cos_theta))


# ----------------------------
# Line construction for finding q (Section 3)
# ----------------------------


def compute_opposite_point_on_sphere(
    p: np.ndarray, B_p: np.ndarray, center: np.ndarray, radius: float
) -> np.ndarray:
    """
    Compute the second point q on sphere S along the line from B(p) through p.

    From Section 3 of the paper:
    "We then consider pairs of points (p, q) such that q = p + γ(p − B(p)),
    and p and q both lie on S."

    This finds the point q on sphere S(center, radius) such that:
    - q lies on the ray from B(p) through p
    - q ≠ p (we want the other intersection)

    Mathematically, we solve:
        |p + λ·d - center|² = radius²
    where d = p - B(p) is the direction from B(p) to p.

    Since p is already on the sphere, one solution is λ=0.
    We want the other solution: λ = -2(d·v)/(d·d) where v = p - center.

    Args:
        p: Point on sphere S, shape (3,)
        B_p: Nearest boundary point to p, B(p), shape (3,)
        center: Center of sphere S, shape (3,)
        radius: Radius of sphere S

    Returns:
        Point q on sphere S along the ray, shape (3,)
    """
    d = p - B_p  # direction vector
    v = p - center  # vector from center to p

    d_dot_d = np.dot(d, d)

    if d_dot_d < 1e-15:
        # p coincides with B(p), degenerate case
        # Return antipodal point on sphere
        logger.debug("Degenerate case in opposite point computation: p ≈ B(p)")
        return center - v  # antipodal point

    # Solve quadratic: |v + λd|² = R²
    # Expanding: |v|² + 2λ(d·v) + λ²|d|² = R²
    # Since |v| = R (p on sphere): 2λ(d·v) + λ²|d|² = 0
    # Solutions: λ = 0 or λ = -2(d·v)/|d|²
    lam = -2.0 * np.dot(d, v) / d_dot_d

    q = p + lam * d

    # Project back to sphere surface for numerical stability
    q_centered = q - center
    q_norm = np.linalg.norm(q_centered)

    if q_norm < 1e-12:
        # q is at center, fallback to antipodal
        logger.debug("Opposite point at center, using antipodal")
        return center - v

    return center + (radius / q_norm) * q_centered


# ----------------------------
# Binary search for medial point (Section 3)
# ----------------------------


def binary_search_medial_point(
    p: np.ndarray,
    B_p: np.ndarray,
    q: np.ndarray,
    B_q: np.ndarray,
    proximity: MeshProximityQuery,
    same_boundary_tol: float,
    position_tol: float,
    max_iterations: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Binary search along segment [p, q] to find the medial point.

    From Section 3 of the paper:
    "For those pairs of points (p, q) that have different nearest boundary
    points, we perform binary search on the segment (p, q) to determine a
    location within a user-chosen tolerance ε of the medial surface on (p, q)."

    The medial surface intersects segment (p, q) because B(p) ≠ B(q).
    We find the point m where the nearest boundary "switches" from one
    region to another.

    Args:
        p: Start point of segment, shape (3,)
        B_p: Nearest boundary point to p, shape (3,)
        q: End point of segment, shape (3,)
        B_q: Nearest boundary point to q, shape (3,)
        proximity: MeshProximityQuery object
        same_boundary_tol: Tolerance for considering two boundary points the same
        position_tol: Tolerance for binary search convergence (ε in paper)
        max_iterations: Maximum binary search iterations

    Returns:
        Tuple of (m, A, B) where:
            - m: Approximate medial point, or None if search failed
            - A: Nearest boundary point on "left" side of m
            - B: Nearest boundary point on "right" side of m
    """
    left = p.copy()
    A_left = B_p.copy()
    right = q.copy()
    A_right = B_q.copy()

    for iteration in range(max_iterations):
        segment_length = np.linalg.norm(right - left)

        if segment_length < position_tol:
            logger.debug(
                f"Binary search converged after {iteration} iterations")
            break

        # Midpoint
        mid = 0.5 * (left + right)

        # Find nearest boundary to midpoint
        B_mid, _, _ = proximity.nearest_point(mid)
        B_mid = B_mid[0]

        # Determine which side of the medial surface mid is on
        # by checking which boundary point it's closer to
        dist_to_left = np.linalg.norm(B_mid - A_left)

        if dist_to_left < same_boundary_tol:
            # mid is on the same side as left (same nearest boundary region)
            left = mid
            A_left = B_mid
        else:
            # mid is on the same side as right
            right = mid
            A_right = B_mid

    # Medial point is approximately at the midpoint of final interval
    m = 0.5 * (left + right)

    return m, A_left, A_right


# ----------------------------
# Progress reporter
# ----------------------------


class ProgressReporter:
    """Simple progress reporter for long-running operations."""

    def __init__(
        self, total: int, report_interval: float = 2.0, description: str = "Processing"
    ):
        self.total = total
        self.report_interval = report_interval
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.last_report_time = self.start_time

    def update(self, increment: int = 1):
        """Update progress and print if interval elapsed."""
        self.current += increment
        current_time = time.time()

        if current_time - self.last_report_time >= self.report_interval:
            elapsed = current_time - self.start_time
            percent = 100.0 * self.current / self.total if self.total > 0 else 0

            if self.current > 0:
                eta = elapsed * (self.total - self.current) / self.current
                eta_str = f", ETA: {eta:.1f}s"
            else:
                eta_str = ""

            logger.info(
                f"{self.description}: {self.current}/{self.total} ({percent:.1f}%), "
                f"elapsed: {elapsed:.1f}s{eta_str}"
            )
            self.last_report_time = current_time

    def finish(self):
        """Report final statistics."""
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        logger.info(
            f"{self.description}: Completed {self.current} items in {elapsed:.2f}s "
            f"({rate:.1f} items/sec)"
        )


# ----------------------------
# Main algorithm (Section 3)
# ----------------------------


def compute_medial_spheres(
    mesh: trimesh.Trimesh,
    voxel_size: float,
    samples_per_voxel: int = 32,
    angle_threshold: float = 0.6,
    same_boundary_tol: float = 1e-4,
    medial_position_tol: float = 1e-4,
    max_binary_iterations: int = 25,
    verbose: bool = True,
) -> MedialSphereResult:
    """
    Compute medial spheres approximating a 3D solid.

    This implements the algorithm from Section 3 of Stolpner et al. (2012).
    The output spheres are:
    - Internal to the solid (centers inside Ω)
    - Tangent to the boundary (touch but don't cross ∂Ω)
    - Well-distributed (at most one sphere per voxel)
    - Significant (object angle above threshold)

    Args:
        mesh: Closed triangle mesh representing the solid Ω
        voxel_size: Side length σ of voxels partitioning space.
                    Smaller values give more spheres but take longer.
                    Paper suggests values giving 400-500 spheres for good results.
        samples_per_voxel: Number of points to sample on each voxel's
                          circumscribed sphere. More samples find more medial
                          points but increase computation time. Default 32.
        angle_threshold: Minimum object angle θ in radians. Paper uses 0.6 rad
                        (≈34°). Larger values keep only more "significant"
                        medial points.
        same_boundary_tol: Relative tolerance for considering two boundary
                          points as "the same" (scaled by mesh size)
        medial_position_tol: Relative tolerance ε for binary search convergence
                            (scaled by mesh size)
        max_binary_iterations: Maximum iterations for binary search
        verbose: Whether to print progress information

    Returns:
        MedialSphereResult containing centers, radii, and statistics

    Example:
        >>> mesh = trimesh.load("model.obj")
        >>> result = compute_medial_spheres(mesh, voxel_size=mesh.scale/50)
        >>> print(f"Found {len(result.radii)} medial spheres")
    """
    start_time = time.time()

    # Ensure we have a proper Trimesh
    if not isinstance(mesh, trimesh.Trimesh):
        if hasattr(mesh, "dump"):
            mesh = mesh.dump().sum()
        else:
            raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")

    # Compute mesh scale for relative tolerances
    bbox_min, bbox_max = mesh.bounds
    mesh_scale = np.linalg.norm(bbox_max - bbox_min)

    # Convert relative tolerances to absolute
    abs_same_boundary_tol = same_boundary_tol * mesh_scale
    abs_position_tol = medial_position_tol * mesh_scale

    logger.info(f"Mesh bounds: {bbox_min} to {bbox_max}")
    logger.info(f"Mesh scale (diagonal): {mesh_scale:.4f}")
    logger.info(f"Voxel size: {voxel_size:.4f}")
    logger.info(
        f"Absolute tolerances - boundary: {abs_same_boundary_tol:.6f}, position: {abs_position_tol:.6f}"
    )

    # Build proximity query structure
    logger.info("Building proximity query structure...")
    proximity = MeshProximityQuery(mesh)

    # Voxelize the mesh
    # This creates a voxel grid where voxels intersecting/inside the mesh are filled
    logger.info("Voxelizing mesh...")
    voxel_grid = mesh.voxelized(pitch=voxel_size)

    # Fill interior voxels (paper processes both boundary and interior voxels)
    voxel_filled = voxel_grid.fill()
    voxel_centers = voxel_filled.points  # Centers of filled voxels

    num_voxels = len(voxel_centers)
    logger.info(f"Number of voxels to process: {num_voxels}")

    # Precompute sampling directions on unit sphere
    sample_directions = fibonacci_sphere(samples_per_voxel)
    logger.info(f"Sampling {samples_per_voxel} directions per voxel")

    # Circumscribed sphere radius for a cube with side length σ
    # The circumscribed sphere passes through all 8 corners
    # Radius = (σ√3)/2 = half the space diagonal
    circum_radius = np.sqrt(3.0) * voxel_size / 2.0
    logger.debug(f"Circumscribed sphere radius: {circum_radius:.4f}")

    # Storage for results
    centers_list: List[np.ndarray] = []
    radii_list: List[float] = []
    candidates_found = 0

    # Progress reporting
    progress = ProgressReporter(num_voxels, description="Processing voxels")

    # Process each voxel
    for voxel_idx, voxel_center in enumerate(voxel_centers):
        # Track best sphere found in this voxel
        best_medial_point = None
        best_radius = -np.inf

        # Sample points on circumscribed sphere S(voxel_center, circum_radius)
        sample_points = voxel_center + \
            circum_radius * sample_directions  # (K, 3)

        # Batch query: find nearest boundary points for all samples at once
        B_samples, distances_samples, _ = proximity.nearest_point(
            sample_points)

        # Process each sample point
        for sample_idx, p in enumerate(sample_points):
            B_p = B_samples[sample_idx]  # Nearest boundary point to p

            # Compute opposite point q on sphere S along direction (p - B(p))
            q = compute_opposite_point_on_sphere(
                p, B_p, voxel_center, circum_radius)

            # Find nearest boundary point to q
            B_q_arr, _, _ = proximity.nearest_point(q.reshape(1, 3))
            B_q = B_q_arr[0]

            # Check if nearest boundary points differ (indicating medial crossing)
            if np.linalg.norm(B_p - B_q) < abs_same_boundary_tol:
                # Same nearest boundary region, no medial crossing on this line
                continue

            # Binary search to find approximate medial point
            m, A, B = binary_search_medial_point(
                p,
                B_p,
                q,
                B_q,
                proximity,
                abs_same_boundary_tol,
                abs_position_tol,
                max_binary_iterations,
            )

            if m is None:
                continue

            # Check if medial point is inside the solid
            if not proximity.is_inside(m.reshape(1, 3))[0]:
                logger.debug(f"Medial point outside solid, skipping")
                continue

            # Compute object angle using the two boundary points
            # A and B are the nearest boundary points from each side of the medial crossing
            theta = compute_object_angle(m, A, B)

            if theta < angle_threshold:
                logger.debug(
                    f"Object angle {theta:.3f} rad below threshold {angle_threshold:.3f}"
                )
                continue

            candidates_found += 1

            # Compute radius (distance from m to nearest boundary)
            B_m, dist_m, _ = proximity.nearest_point(m.reshape(1, 3))
            radius = float(dist_m[0])

            # Keep the sphere with largest radius in this voxel
            if radius > best_radius:
                best_radius = radius
                best_medial_point = m.copy()

        # Add best sphere from this voxel (if any found)
        if best_medial_point is not None and best_radius > 0:
            centers_list.append(best_medial_point)
            radii_list.append(best_radius)

        if verbose:
            progress.update()

    if verbose:
        progress.finish()

    # Convert to arrays
    if len(centers_list) == 0:
        centers = np.zeros((0, 3), dtype=np.float32)
        radii = np.zeros((0,), dtype=np.float32)
    else:
        centers = np.array(centers_list, dtype=np.float32)
        radii = np.array(radii_list, dtype=np.float32)

    computation_time = time.time() - start_time

    logger.info(f"Computation completed in {computation_time:.2f}s")
    logger.info(f"Candidates found: {candidates_found}")
    logger.info(f"Spheres output: {len(radii)}")
    logger.info(
        f"Radius range: [{radii.min():.4f}, {radii.max():.4f}]"
        if len(radii) > 0
        else "No spheres"
    )

    return MedialSphereResult(
        centers=centers,
        radii=radii,
        computation_time=computation_time,
        num_voxels_processed=num_voxels,
        num_candidates_found=candidates_found,
        num_spheres_output=len(radii),
    )


# ----------------------------
# Visualization
# ----------------------------


def create_sphere_mesh(
    center: np.ndarray, radius: float, subdivisions: int = 2
) -> trimesh.Trimesh:
    """
    Create a sphere mesh at the given center and radius.

    Args:
        center: Sphere center, shape (3,)
        radius: Sphere radius
        subdivisions: Icosphere subdivision level (higher = smoother)

    Returns:
        trimesh.Trimesh representing the sphere
    """
    sphere = trimesh.creation.icosphere(
        subdivisions=subdivisions, radius=radius)
    sphere.apply_translation(center)
    return sphere


def visualize_medial_spheres(
    mesh: trimesh.Trimesh,
    centers: np.ndarray,
    radii: np.ndarray,
    max_spheres_to_show: int = 200,
    mesh_opacity: float = 0.3,
    sphere_color: tuple = (0.2, 0.5, 1.0, 0.6),
    mesh_color: tuple = (0.8, 0.8, 0.8, 0.3),
    show_largest: bool = True,
) -> trimesh.Scene:
    """
    Create a visualization of the mesh with medial spheres.

    Args:
        mesh: Original mesh
        centers: Sphere centers, shape (N, 3)
        radii: Sphere radii, shape (N,)
        max_spheres_to_show: Limit number of spheres for performance
        mesh_opacity: Opacity of the original mesh (0-1)
        sphere_color: RGBA color for spheres
        mesh_color: RGBA color for mesh
        show_largest: If True, show the largest spheres when limiting

    Returns:
        trimesh.Scene that can be shown with scene.show()
    """
    scene = trimesh.Scene()

    # Add original mesh (semi-transparent)
    mesh_visual = mesh.copy()
    mesh_visual.visual.face_colors = [int(c * 255) for c in mesh_color]
    # scene.add_geometry(mesh_visual, node_name="original_mesh")

    # Select spheres to show
    num_spheres = len(radii)
    if num_spheres > max_spheres_to_show:
        if show_largest:
            # Show largest spheres
            indices = np.argsort(radii)[-max_spheres_to_show:]
        else:
            # Random sample
            indices = np.random.choice(
                num_spheres, max_spheres_to_show, replace=False)
        logger.info(f"Showing {max_spheres_to_show} of {num_spheres} spheres")
    else:
        indices = np.arange(num_spheres)

    # Add spheres
    sphere_color_rgba = [int(c * 255) for c in sphere_color]

    for i, idx in enumerate(indices):
        sphere = create_sphere_mesh(centers[idx], radii[idx], subdivisions=1)
        sphere.visual.face_colors = sphere_color_rgba
        scene.add_geometry(sphere, node_name=f"sphere_{i}")

    logger.info(f"Created visualization with {len(indices)} spheres")

    return scene


# ----------------------------
# Demo / Test
# ----------------------------


# def create_test_mesh(mesh_type: str = "box", **kwargs) -> trimesh.Trimesh:
#     """
#     Create a simple test mesh.

#     Args:
#         mesh_type: One of "box", "sphere", "cylinder", "torus"
#         **kwargs: Additional arguments passed to creation function

#     Returns:
#         trimesh.Trimesh
#     """
#     if mesh_type == "box":
#         extents = kwargs.get("extents", [1.0, 1.0, 1.0])
#         mesh = trimesh.creation.box(extents=extents)
#     elif mesh_type == "sphere":
#         radius = kwargs.get("radius", 0.5)
#         mesh = trimesh.creation.icosphere(radius=radius, subdivisions=3)
#     elif mesh_type == "cylinder":
#         radius = kwargs.get("radius", 0.3)
#         height = kwargs.get("height", 1.0)
#         mesh = trimesh.creation.cylinder(radius=radius, height=height)
#     elif mesh_type == "torus":
#         major_radius = kwargs.get("major_radius", 0.5)
#         minor_radius = kwargs.get("minor_radius", 0.2)
#         mesh = trimesh.creation.torus(
#             major_radius=major_radius, minor_radius=minor_radius
#         )
#     else:
#         raise ValueError(f"Unknown mesh type: {mesh_type}")

#     return mesh


# def run_demo():
#     """Run a demonstration of the medial spheres algorithm."""
#     logger.info("=" * 60)
#     logger.info("Medial Spheres Demo")
#     logger.info("=" * 60)

#     # Create test mesh - using a box for faster testing
#     logger.info("\nCreating test mesh (box)...")
#     mesh = create_test_mesh("box", extents=[0.5, 0.5, 0.5])
#     logger.info(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
#     logger.info(f"Mesh bounds: {mesh.bounds}")
#     logger.info(f"Mesh volume: {mesh.volume:.4f}")

#     # Compute medial spheres with coarser voxel size for speed
#     logger.info("\nComputing medial spheres...")
#     voxel_size = mesh.scale / 8  # Coarser for speed

#     result = compute_medial_spheres(
#         mesh,
#         voxel_size=voxel_size,
#         samples_per_voxel=32,
#         angle_threshold=0.6,  # ~34 degrees, as in paper
#         verbose=True,
#     )

#     # Print results
#     logger.info("\n" + "=" * 60)
#     logger.info("Results Summary")
#     logger.info("=" * 60)
#     logger.info(f"Number of spheres: {result.num_spheres_output}")
#     logger.info(f"Computation time: {result.computation_time:.2f}s")
#     logger.info(f"Voxels processed: {result.num_voxels_processed}")
#     logger.info(f"Candidates found: {result.num_candidates_found}")

#     if result.num_spheres_output > 0:
#         logger.info(f"\nSphere statistics:")
#         logger.info(f"  Radius min: {result.radii.min():.4f}")
#         logger.info(f"  Radius max: {result.radii.max():.4f}")
#         logger.info(f"  Radius mean: {result.radii.mean():.4f}")
#         logger.info(f"  Radius std: {result.radii.std():.4f}")

#         # Compute approximate volume coverage
#         sphere_volume = np.sum(4 / 3 * np.pi * result.radii**3)
#         logger.info(f"\nVolume analysis:")
#         logger.info(f"  Total sphere volume (with overlap): {sphere_volume:.4f}")
#         logger.info(f"  Mesh volume: {mesh.volume:.4f}")
#         logger.info(f"  Volume ratio: {sphere_volume/mesh.volume:.2f}")

#     # Create 3D visualization (GLB for external viewing)
#     logger.info("\nCreating 3D visualization...")
#     scene = visualize_medial_spheres(
#         mesh,
#         result.centers,
#         result.radii,
#         max_spheres_to_show=200,
#         mesh_color=(0.7, 0.7, 0.7, 0.3),
#         sphere_color=(0.2, 0.6, 1.0, 0.7),
#     )

#     scene.show()

#     return result, scene, mesh


# if __name__ == "__main__":
#     result, scene, mesh = run_demo()

#     # Print first few spheres
#     if result.num_spheres_output > 0:
#         logger.info("\nFirst 5 spheres:")
#         for i in range(min(5, result.num_spheres_output)):
#             logger.info(f"  Center: {result.centers[i]}, Radius: {result.radii[i]:.4f}")
