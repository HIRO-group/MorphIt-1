import trimesh

# Box dimensions [width, depth, height]
box_size = [0.05, 0.05, 0.05]

# Create the box
mesh = trimesh.creation.box(extents=box_size)

rho = 4.0  # or whatever density you want
mesh.apply_translation(-mesh.center_mass)  # move COM to (0,0,0)


V_mesh = mesh.volume
M_target = rho * V_mesh
COM_target = mesh.center_mass  # shape (3,)

# moment of inertia about some origin; be consistent later
I_target = mesh.moment_inertia  # 3x3
print("Target mass:", M_target)
print("Target COM:", COM_target)
print("Target inertia:\n", I_target)


# Save the box
output_path = "../../mesh_models/box.obj"  # Change this to your desired path
mesh.export(output_path)

print(f"Box saved to: {output_path}")
