import trimesh

box_size = [0.2, 0.2, 0.2]
mesh = trimesh.creation.box(extents=box_size)
rho = 4.0
mesh.apply_translation(-mesh.center_mass)  # move COM to (0,0,0)
V_mesh = mesh.volume
M_target = rho * V_mesh
COM_target = mesh.center_mass
I_target = mesh.moment_inertia
print("Target mass:", M_target)
print("Target COM:", COM_target)
print("Target inertia:\n", I_target)
output_path = "../../mesh_models/box.obj"
mesh.export(output_path)
print(f"Box saved to: {output_path}")
