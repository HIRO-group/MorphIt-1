import numpy as np
import trimesh
import json
import os


def check_sphere_alignment(mesh_path, results_path, tolerance=0.003):
    """
    Checks if spheres along the mesh boundaries are aligned (flat).

    Args:
        mesh_path: Path to the .obj box mesh
        results_path: Path to the morphit_results.json file
        tolerance: Distance threshold to consider a sphere part of a "face" group
    """
    print(f"--- Checking Alignment for {os.path.basename(mesh_path)} ---")

    # 1. Load Data
    mesh = trimesh.load(mesh_path, force="mesh")
    with open(results_path, "r") as f:
        data = json.load(f)

    centers = np.array(data["centers"])
    radii = np.array(data["radii"])

    # 2. Identify Box Faces (Cluster by Normal)
    # We group faces that point in the same direction to identify "Sides"
    face_normals = mesh.face_normals
    # Round normals to avoid float errors when finding unique directions
    rounded_normals = np.round(face_normals, decimals=3)
    unique_normals, indices = np.unique(
        rounded_normals, axis=0, return_inverse=True)

    print(f"Detected {len(unique_normals)} distinct planar regions (sides).")

    issues_found = False

    for i, normal in enumerate(unique_normals):
        # Find a point on this plane (take the first vertex of the first face in this group)
        face_idx = np.where(indices == i)[0][0]
        plane_origin = mesh.vertices[mesh.faces[face_idx][0]]

        # 3. Find Spheres Belonging to this Face
        # Distance from sphere center to plane: dot(center - origin, normal)
        # We want spheres where (Distance_to_Plane - Radius) is close to 0

        # Vector from plane origin to sphere centers
        vecs = centers - plane_origin
        # Projected distance to the plane
        dists_to_plane = np.dot(vecs, normal)

        # Distance from the sphere's *surface* to the plane
        # Assuming spheres are inside, this should be close to 0 for surface spheres
        surface_gaps = np.abs(dists_to_plane) - radii

        # Filter for spheres that are close to this wall (e.g., within 1cm or tolerance)
        # We look for spheres that are meant to be touching this wall
        close_mask = surface_gaps < tolerance

        if not np.any(close_mask):
            continue

        relevant_gaps = surface_gaps[close_mask]

        # 4. Calculate Alignment Statistics
        # Ideally, all 'relevant_gaps' should be identical (or close to 0)
        gap_variance = np.var(relevant_gaps)
        gap_range = np.max(relevant_gaps) - np.min(relevant_gaps)
        mean_offset = np.mean(relevant_gaps)

        print(f"\nSide with Normal {normal}:")
        print(f"  Spheres on this side: {len(relevant_gaps)}")
        print(f"  Mean Offset from Wall: {mean_offset:.6f} m")
        print(f"  Max Misalignment (Range): {gap_range:.6f} m")

        # Heuristic for "Bad" alignment
        if gap_range > 0.001:  # more than 5mm variance
            print(
                f"  [WARNING] Uneven surface detected! Variance: {gap_variance:.8f}")
            issues_found = True
        else:
            print(f"  [OK] Surface is aligned.")

    if issues_found:
        print(
            "\nSUMMARY: The simulator might be unstable. Some sides have uneven sphere depths."
        )
    else:
        print("\nSUMMARY: Spheres are well-aligned. Simulation should be stable.")


if __name__ == "__main__":
    # Update these paths to match your setup
    MESH_FILE = "../mesh_models/box.obj"
    RESULTS_FILE = "results/output/morphit_results.json"

    if os.path.exists(MESH_FILE) and os.path.exists(RESULTS_FILE):
        # check_sphere_alignment(MESH_FILE, RESULTS_FILE)
        check_sphere_alignment(MESH_FILE, RESULTS_FILE, bottom_direction='-z')
    else:
        print("Please check your file paths in the script.")
