import numpy as np
import trimesh

# ----------------------------
# Utility: sampling on a sphere
# ----------------------------


def fibonacci_sphere(samples: int) -> np.ndarray:
    """
    Approximately uniform points on unit sphere.
    Returns array of shape (samples, 3).
    """
    if samples <= 1:
        return np.array([[0., 0., 1.]])

    indices = np.arange(samples, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle

    y = 1.0 - (indices / (samples - 1.0)) * 2.0
    radius = np.sqrt(1.0 - y * y)

    theta = phi * indices

    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    return np.stack([x, y, z], axis=1)


# ----------------------------
# Mesh proximity helper
# ----------------------------

class MeshNearest:
    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.prox = trimesh.proximity.ProximityQuery(mesh)

    def nearest_point(self, p: np.ndarray):
        """
        p: (..., 3)
        returns:
            surf_points: (..., 3)
            tri_ids:     (...,)
        """
        surf_points, distances, tri_ids = self.prox.on_surface(p)
        return surf_points, tri_ids

    def is_inside(self, p: np.ndarray) -> np.ndarray:
        """
        p: (..., 3)
        returns bool array (...,) whether points are inside the mesh.
        """
        # contains() can take many points at once
        return self.mesh.contains(p)


# ----------------------------
# Object angle (Eq. ffAmB)
# ----------------------------

def object_angle(m: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """
    Angle ∠AmB at m (in radians).
    """
    v1 = A - m
    v2 = B - m
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_theta))


# ----------------------------
# q on same sphere via paper’s line construction
# ----------------------------

def second_point_on_sphere(p: np.ndarray,
                           A: np.ndarray,
                           center: np.ndarray,
                           R: float) -> np.ndarray:
    """
    Given point p on sphere S(center, R) and nearest boundary point A = B(p),
    find q on same sphere along direction (p - A) as in the paper:
        q = p + λ (p - A)
    with |q - center| = R.

    We solve:
        |(p - center) + λ d|^2 = R^2
    with d = (p - A), |p - center| = R.
    """
    d = p - A
    v = p - center
    denom = np.dot(d, d)
    if denom == 0.0:
        # Degenerate; just return antipodal point.
        q = center - v
    else:
        lam = -2.0 * np.dot(d, v) / (denom + 1e-15)
        q = p + lam * d

    # Numerical projection back to sphere
    q_c = q - center
    norm_qc = np.linalg.norm(q_c)
    if norm_qc == 0.0:
        # fallback
        return center - v
    return center + (R / norm_qc) * q_c


# ----------------------------
# Binary search for medial point on segment
# ----------------------------

def find_medial_on_segment(p: np.ndarray,
                           Ap: np.ndarray,
                           q: np.ndarray,
                           Aq: np.ndarray,
                           nearest: MeshNearest,
                           same_tol: float,
                           max_iter: int,
                           pos_tol: float):
    """
    Given endpoints p, q and their nearest points Ap, Aq with Ap != Aq,
    binary search along segment [p, q] for a point where nearest boundary
    'switches'. Returns (m, A_left, A_right) or (None, None, None) if fails.
    """
    left, A_left = p.copy(), Ap.copy()
    right, A_right = q.copy(), Aq.copy()

    for _ in range(max_iter):
        if np.linalg.norm(right - left) < pos_tol:
            break

        mid = 0.5 * (left + right)
        A_mid, _ = nearest.nearest_point(mid[None, :])
        A_mid = A_mid[0]

        # Decide which side mid is 'closer' to in terms of nearest boundary
        if np.linalg.norm(A_mid - A_left) < same_tol:
            left, A_left = mid, A_mid
        else:
            right, A_right = mid, A_mid

    m = 0.5 * (left + right)
    return m, A_left, A_right


# ----------------------------
# Main algorithm: medial spheres
# ----------------------------

def medial_spheres_trimesh(
    mesh: trimesh.Trimesh,
    voxel_size: float,
    samples_per_voxel: int = 32,
    angle_threshold: float = 0.6,  # radians, per paper
    same_boundary_tol: float = 1e-4,
    medial_pos_tol: float = 1e-4,
    max_binary_iters: int = 25,
):
    """
    Compute medial spheres (centers, radii) approximating the solid represented
    by a closed triangle mesh, following Stolpner et al. (2012) in spirit.

    Returns:
        centers: (N, 3) float32 array
        radii:   (N,)   float32 array
    """
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()

    # Normalize same_boundary_tol and medial_pos_tol to mesh scale
    bbox_min = mesh.bounds[0]
    bbox_max = mesh.bounds[1]
    scale = np.linalg.norm(bbox_max - bbox_min)
    same_tol = same_boundary_tol * scale
    pos_tol = medial_pos_tol * scale

    nearest = MeshNearest(mesh)

    # Voxelize mesh – fills interior.
    # This gives us voxel centers inside/intersecting the solid.
    vox_grid = mesh.voxelized(pitch=voxel_size)
    vox_filled = vox_grid.fill()  # ensure interior voxels as well
    voxel_centers = vox_filled.points  # (M, 3)

    # Precompute sampling directions on unit sphere
    dirs = fibonacci_sphere(samples_per_voxel)

    centers = []
    radii = []

    R = np.sqrt(3.0) * voxel_size / 2.0

    for c in voxel_centers:
        # circumscribed sphere S(c, R)
        best_m = None
        best_r = -np.inf

        # Sample points on S
        points_on_sphere = c[None, :] + R * dirs  # (K, 3)
        # Nearest boundary for all p at once
        A_all, tri_ids_all = nearest.nearest_point(points_on_sphere)

        for i, p in enumerate(points_on_sphere):
            A = A_all[i]

            # Build q via line construction
            q = second_point_on_sphere(p, A, c, R)
            B_arr, tri_ids_q = nearest.nearest_point(q[None, :])
            B = B_arr[0]

            # Check if nearest boundary changed
            if np.linalg.norm(A - B) < same_tol:
                continue  # no medial crossing along [p, q]

            # Binary search to find medial point approx.
            m, A_left, A_right = find_medial_on_segment(
                p, A, q, B, nearest,
                same_tol=same_tol,
                max_iter=max_binary_iters,
                pos_tol=pos_tol,
            )

            # Check inside mesh
            if not nearest.is_inside(m[None, :])[0]:
                continue

            # Compute object angle using A_left and A_right
            theta = object_angle(m, A_left, A_right)
            if theta < angle_threshold:
                continue

            # Radius is distance to nearest boundary at m
            Am, _ = nearest.nearest_point(m[None, :])
            Am = Am[0]
            r = float(np.linalg.norm(m - Am))

            if r > best_r:
                best_r = r
                best_m = m

        if best_m is not None and best_r > 0:
            centers.append(best_m)
            radii.append(best_r)

    if len(centers) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    centers = np.array(centers, dtype=np.float32)
    radii = np.array(radii, dtype=np.float32)
    return centers, radii


# USAGE

mesh = trimesh.load("your_model.obj", process=True)

centers, radii = medial_spheres_trimesh(
    mesh,
    voxel_size=mesh.scale / 50.0,  # tune this
    samples_per_voxel=32,
    angle_threshold=0.6,
)

print("Number of spheres:", len(radii))
print("First few centers:", centers[:5])
print("First few radii:", radii[:5])
