"""
Simple script to compare physical properties between mesh and sphere approximation.
Uses Drake's built-in functions for accurate calculation of sphere system properties.
"""

import numpy as np
import trimesh
from pydrake.all import (
    MultibodyPlant,
    RigidTransform,
    SpatialInertia,
    UnitInertia,
)
import json


# =============================================================================
# STEP 1: Load and analyze mesh with Trimesh
# =============================================================================
def load_mesh_with_trimesh(mesh_path, density=1.0):
    """
    Load mesh and compute physical properties using Trimesh.

    Args:
        mesh_path: Path to .obj file
        density: Material density (kg/m^3)

    Returns:
        dict with mass, com, inertia
    """
    print("\n" + "=" * 70)
    print("STEP 1: Loading mesh with Trimesh")
    print("=" * 70)

    # Load mesh
    mesh = trimesh.load(mesh_path)
    print(f"✓ Loaded mesh from: {mesh_path}")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Volume: {mesh.volume:.6f} m³")

    # Compute properties
    mass = mesh.volume * density
    com = mesh.center_mass
    inertia = mesh.moment_inertia * density

    print(f"\n✓ Trimesh Physical Properties (density={density}):")
    print(f"  Mass: {mass:.6f} kg")
    print(f"  COM: [{com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f}] m")
    print(f"  Inertia tensor:")
    print(f"    [{inertia[0,0]:.6f}, {inertia[0,1]:.6f}, {inertia[0,2]:.6f}]")
    print(f"    [{inertia[1,0]:.6f}, {inertia[1,1]:.6f}, {inertia[1,2]:.6f}]")
    print(f"    [{inertia[2,0]:.6f}, {inertia[2,1]:.6f}, {inertia[2,2]:.6f}]")

    return {"mass": mass, "com": com, "inertia": inertia, "volume": mesh.volume}


# =============================================================================
# STEP 2: Calculate properties using Drake's built-in functions
# =============================================================================
def calculate_sphere_properties_with_drake(centers, radii, density=1.0):
    """
    Use Drake's MultibodyPlant to calculate physical properties of spheres.
    This leverages Drake's built-in mass matrix and spatial inertia calculations.

    Args:
        centers: (N, 3) array of sphere centers
        radii: (N,) array of sphere radii
        density: Material density (kg/m^3)

    Returns:
        dict with mass, com, inertia calculated by Drake
    """
    print("\n" + "=" * 70)
    print("STEP 2: Using Drake to calculate sphere system properties")
    print("=" * 70)

    # Create a MultibodyPlant
    plant = MultibodyPlant(time_step=0.0)

    # Add each sphere as a rigid body
    print(f"✓ Adding {len(radii)} spheres to Drake plant...")
    for i, (center, radius) in enumerate(zip(centers, radii)):
        mass = density * (4.0 / 3.0) * np.pi * (radius**3)

        # Create spatial inertia for a solid sphere about its center
        # For a solid sphere: I = (2/5) * m * r²
        I_sphere = (2.0 / 5.0) * mass * radius**2

        # Create unit inertia (normalized by mass)
        unit_inertia = UnitInertia(I_sphere / mass, I_sphere / mass, I_sphere / mass)

        # Create spatial inertia
        M_BBo_B = SpatialInertia(mass, [0, 0, 0], unit_inertia)

        # Add body to plant
        body = plant.AddRigidBody(f"sphere_{i}", M_BBo_B)

        if i < 3:  # Debug first few spheres
            print(f"  Debug sphere {i}:")
            print(f"    Mass: {mass:.6f} kg, Radius: {radius:.6f} m")
            print(f"    Center: [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")
            print(f"    I_diag: {I_sphere:.6f}")

    # Finalize the plant
    plant.Finalize()
    print(
        f"✓ Drake plant finalized with {plant.num_bodies()-1} spheres"
    )  # -1 for world

    # Create a context
    context = plant.CreateDefaultContext()

    # Set positions for each sphere (using quaternion floating joints)
    # Note: Drake automatically creates floating joints for free bodies
    for i, center in enumerate(centers):
        body = plant.GetBodyByName(f"sphere_{i}")
        if body.is_floating():
            plant.SetFreeBodyPose(
                context, body, RigidTransform([center[0], center[1], center[2]])
            )

    # Use Drake's built-in functions to calculate properties
    print(f"\n✓ Using Drake's CalcTotalMass()...")
    total_mass = plant.CalcTotalMass(context)

    print(f"✓ Using Drake's CalcCenterOfMassPositionInWorld()...")
    com = plant.CalcCenterOfMassPositionInWorld(context)

    print(f"✓ Using Drake's CalcMassMatrix()...")
    # Get mass matrix M(q)
    M = plant.CalcMassMatrix(context)

    # For a system of floating bodies, the mass matrix has a specific structure
    # We need to extract the spatial inertia about the origin
    # For now, we'll compute it manually as Drake's mass matrix is in generalized coordinates

    # Calculate inertia about world origin using Drake's CalcSpatialInertia
    # This gives us the composite spatial inertia
    all_bodies = [plant.GetBodyByName(f"sphere_{i}") for i in range(len(radii))]
    body_indices = [body.index() for body in all_bodies]

    print(f"✓ Using Drake's CalcSpatialInertia()...")
    spatial_inertia_W = plant.CalcSpatialInertia(
        context, plant.world_frame(), body_indices
    )

    # Extract inertia tensor about world origin
    inertia_about_world_origin = (
        spatial_inertia_W.CalcRotationalInertia().CopyToFullMatrix3()
    )

    # IMPORTANT: Trimesh gives inertia about COM, so we need to shift Drake's inertia
    # from world origin to COM using the parallel axis theorem
    # I_com = I_origin - m * (com^T * com * I - com * com^T)
    com_array = np.array([com[0], com[1], com[2]])
    com_sq = np.dot(com_array, com_array)
    parallel_axis_term = total_mass * (
        com_sq * np.eye(3) - np.outer(com_array, com_array)
    )
    inertia = inertia_about_world_origin - parallel_axis_term

    print(f"\n✓ Drake Physical Properties (density={density}):")
    print(f"  Total Mass: {total_mass:.6f} kg")
    print(f"  COM: [{com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f}] m")
    print(f"  Inertia tensor about COM:")
    print(f"    [{inertia[0,0]:.6f}, {inertia[0,1]:.6f}, {inertia[0,2]:.6f}]")
    print(f"    [{inertia[1,0]:.6f}, {inertia[1,1]:.6f}, {inertia[1,2]:.6f}]")
    print(f"    [{inertia[2,0]:.6f}, {inertia[2,1]:.6f}, {inertia[2,2]:.6f}]")

    return {
        "mass": total_mass,
        "com": np.array([com[0], com[1], com[2]]),
        "inertia": inertia,
        "volume": sum((4.0 / 3.0) * np.pi * (r**3) for r in radii),
    }


# =============================================================================
# STEP 3: Load sphere configuration and compute properties
# =============================================================================
def load_sphere_config(results_path):
    """
    Load sphere centers and radii from MorphIt results.

    Args:
        results_path: Path to JSON results file

    Returns:
        centers (N, 3), radii (N,)
    """
    print("\n" + "=" * 70)
    print("STEP 3: Loading sphere configuration")
    print("=" * 70)

    with open(results_path, "r") as f:
        data = json.load(f)

    centers = np.array(data["centers"])
    radii = np.array(data["radii"])

    print(f"✓ Loaded sphere configuration")
    print(f"  Number of spheres: {len(radii)}")
    print(f"  Radii range: [{radii.min():.6f}, {radii.max():.6f}] m")
    print(f"  Centers range:")
    print(f"    X: [{centers[:,0].min():.6f}, {centers[:,0].max():.6f}]")
    print(f"    Y: [{centers[:,1].min():.6f}, {centers[:,1].max():.6f}]")
    print(f"    Z: [{centers[:,2].min():.6f}, {centers[:,2].max():.6f}]")

    return centers, radii


# =============================================================================
# STEP 4: Compare properties
# =============================================================================
def compare_properties(mesh_props, sphere_props):
    """
    Compare physical properties between mesh and sphere approximation.

    Args:
        mesh_props: dict from load_mesh_with_trimesh
        sphere_props: dict from calculate_sphere_properties_with_drake
    """
    print("\n" + "=" * 70)
    print("STEP 4: COMPARISON RESULTS")
    print("=" * 70)

    # Mass comparison
    mass_error = abs(sphere_props["mass"] - mesh_props["mass"])
    mass_error_pct = (mass_error / mesh_props["mass"]) * 100

    print(f"\n1. MASS COMPARISON:")
    print(f"   Mesh mass:   {mesh_props['mass']:.6f} kg")
    print(f"   Sphere mass: {sphere_props['mass']:.6f} kg")
    print(f"   Absolute error: {mass_error:.6f} kg")
    print(f"   Percentage error: {mass_error_pct:.2f}%")

    if mass_error_pct < 1.0:
        print(f"   ✓ EXCELLENT match (<1%)")
    elif mass_error_pct < 5.0:
        print(f"   ✓ GOOD match (<5%)")
    elif mass_error_pct < 10.0:
        print(f"   ⚠ ACCEPTABLE match (<10%)")
    else:
        print(f"   ✗ POOR match (>10%)")

    # COM comparison
    com_error = np.linalg.norm(sphere_props["com"] - mesh_props["com"])

    print(f"\n2. CENTER OF MASS COMPARISON:")
    print(
        f"   Mesh COM:   [{mesh_props['com'][0]:.6f}, {mesh_props['com'][1]:.6f}, {mesh_props['com'][2]:.6f}] m"
    )
    print(
        f"   Sphere COM: [{sphere_props['com'][0]:.6f}, {sphere_props['com'][1]:.6f}, {sphere_props['com'][2]:.6f}] m"
    )
    print(f"   Euclidean distance: {com_error:.6f} m ({com_error*1000:.3f} mm)")

    if com_error < 0.001:
        print(f"   ✓ EXCELLENT match (<1mm)")
    elif com_error < 0.005:
        print(f"   ✓ GOOD match (<5mm)")
    elif com_error < 0.01:
        print(f"   ⚠ ACCEPTABLE match (<10mm)")
    else:
        print(f"   ✗ POOR match (>10mm)")

    # Inertia comparison
    inertia_error = np.linalg.norm(
        sphere_props["inertia"] - mesh_props["inertia"], "fro"
    )
    inertia_norm = np.linalg.norm(mesh_props["inertia"], "fro")
    inertia_error_pct = (inertia_error / inertia_norm) * 100

    print(f"\n3. INERTIA TENSOR COMPARISON:")
    print(f"   Mesh inertia:")
    print(
        f"     [{mesh_props['inertia'][0,0]:.6f}, {mesh_props['inertia'][0,1]:.6f}, {mesh_props['inertia'][0,2]:.6f}]"
    )
    print(
        f"     [{mesh_props['inertia'][1,0]:.6f}, {mesh_props['inertia'][1,1]:.6f}, {mesh_props['inertia'][1,2]:.6f}]"
    )
    print(
        f"     [{mesh_props['inertia'][2,0]:.6f}, {mesh_props['inertia'][2,1]:.6f}, {mesh_props['inertia'][2,2]:.6f}]"
    )
    print(f"   Sphere inertia:")
    print(
        f"     [{sphere_props['inertia'][0,0]:.6f}, {sphere_props['inertia'][0,1]:.6f}, {sphere_props['inertia'][0,2]:.6f}]"
    )
    print(
        f"     [{sphere_props['inertia'][1,0]:.6f}, {sphere_props['inertia'][1,1]:.6f}, {sphere_props['inertia'][1,2]:.6f}]"
    )
    print(
        f"     [{sphere_props['inertia'][2,0]:.6f}, {sphere_props['inertia'][2,1]:.6f}, {sphere_props['inertia'][2,2]:.6f}]"
    )
    print(f"   Frobenius norm error: {inertia_error:.6f}")
    print(f"   Percentage error: {inertia_error_pct:.2f}%")

    if inertia_error_pct < 5.0:
        print(f"   ✓ EXCELLENT match (<5%)")
    elif inertia_error_pct < 10.0:
        print(f"   ✓ GOOD match (<10%)")
    elif inertia_error_pct < 20.0:
        print(f"   ⚠ ACCEPTABLE match (<20%)")
    else:
        print(f"   ✗ POOR match (>20%)")

    print("\n" + "=" * 70)


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main comparison function."""

    # Configuration
    mesh_path = "../mesh_models/box.obj"  # Change to your mesh
    results_path = "results/output/morphit_results.json"  # Change to your results
    density = 1.0  # kg/m^3

    print("\n" + "=" * 70)
    print("PHYSICAL PROPERTIES COMPARISON: MESH vs SPHERES")
    print("=" * 70)
    print(f"Mesh: {mesh_path}")
    print(f"Results: {results_path}")
    print(f"Density: {density} kg/m³")

    # Step 1: Load mesh with Trimesh
    mesh_props = load_mesh_with_trimesh(mesh_path, density)

    # Step 2: Load sphere configuration
    centers, radii = load_sphere_config(results_path)

    # Step 3: Use Drake to compute sphere properties
    sphere_props = calculate_sphere_properties_with_drake(centers, radii, density)

    # Step 4: Compare
    compare_properties(mesh_props, sphere_props)

    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()
