import trimesh
import torch

mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
rho = 1.0  # or whatever density you want
mesh.apply_translation(-mesh.center_mass)  # move COM to (0,0,0)


V_mesh = mesh.volume
M_target = rho * V_mesh

COM_target = mesh.center_mass  # shape (3,)

# moment of inertia about some origin; be consistent later
I_target = mesh.moment_inertia  # 3x3
print("Target mass:", M_target)
print("Target COM:", COM_target)
print("Target inertia:\n", I_target)


N = 10  # number of spheres, for example

# parameters to optimize:
# centers: (N, 3)
centers = torch.nn.Parameter(torch.randn(N, 3))

# unconstrained radii params: (N,)
radii_unconstrained = torch.nn.Parameter(torch.zeros(N))  # start near 0


def get_radii():
    # smooth, strictly positive
    return torch.nn.functional.softplus(radii_unconstrained) + 1e-6


def mass_spheres(centers, radii, rho):
    m = rho * (4.0 / 3.0) * torch.pi * radii**3   # (N,)
    M = m.sum()
    return M, m


def com_spheres(centers, m):
    M = m.sum()
    COM = (m[:, None] * centers).sum(dim=0) / M
    return COM


def inertia_spheres_about_origin(centers, radii, m):
    # centers: (N, 3), radii: (N,), m: (N,)
    N = centers.shape[0]
    I = torch.zeros(3, 3, dtype=centers.dtype, device=centers.device)

    # 2/5 m_i R_i^2 part
    I_center_diag = (2.0 / 5.0) * m * radii**2  # (N,)

    eye3 = torch.eye(3, dtype=centers.dtype, device=centers.device)

    for i in range(N):
        ci = centers[i]             # (3,)
        mi = m[i]
        Ri_term = I_center_diag[i]  # scalar tensor

        # I_center about sphere center: diag(Ri_term, Ri_term, Ri_term)
        I_center = eye3 * Ri_term   # (3,3), stays differentiable

        # parallel axis term: mi (|c|^2 I - c c^T)
        ci_sq = (ci * ci).sum()
        I_parallel = mi * (ci_sq * eye3 - torch.outer(ci, ci))

        I += I_center + I_parallel

    return I


def loss(centers, radii_unconstrained, rho,
         M_target, COM_target, I_target,
         wM=1.0, wCOM=1.0, wI=1.0):

    radii = torch.nn.functional.softplus(radii_unconstrained) + 1e-6

    M, m = mass_spheres(centers, radii, rho)
    COM = com_spheres(centers, m)
    I = inertia_spheres_about_origin(centers, radii, m)

    # scalar losses
    mass_loss = (M - M_target)**2
    com_loss = torch.sum((COM - COM_target)**2)
    inertia_loss = torch.sum((I - I_target)**2)  # Frobenius norm squared

    # optional: keep centers roughly inside the cube, radii small, etc.
    # Example: soft penalty for going outside bounds [-L/2, L/2]
    # This keeps it differentiable (no hard clipping).
    # L is the cube side length
    L = 1.0
    penalty = torch.relu(torch.abs(centers) - (L / 2)).pow(2).sum()

    return wM * mass_loss + wCOM * com_loss + wI * inertia_loss + penalty


optimizer = torch.optim.Adam([centers, radii_unconstrained], lr=1e-2)

# fixed target tensors on the right device/dtype
M_target_t = torch.tensor(M_target, dtype=centers.dtype, device=centers.device)
COM_target_t = torch.tensor(
    COM_target, dtype=centers.dtype, device=centers.device)
I_target_t = torch.tensor(I_target, dtype=centers.dtype, device=centers.device)

num_steps = 10000

for step in range(1, num_steps + 1):
    optimizer.zero_grad()
    L = loss(
        centers,
        radii_unconstrained,
        rho,
        M_target_t,
        COM_target_t,
        I_target_t
    )
    L.backward()
    optimizer.step()

    # progress logging
    if step == 1 or step % 100 == 0:
        with torch.no_grad():
            radii = get_radii()
            M, m = mass_spheres(centers, radii, rho)
            COM = com_spheres(centers, m)
            I = inertia_spheres_about_origin(centers, radii, m)

            mass_err = (M - M_target_t).abs().item()
            com_err = torch.norm(COM - COM_target_t).item()
            inertia_err = torch.norm(I - I_target_t).item()

        print(
            f"step {step:5d} | "
            f"loss={L.item():.6e} | "
            f"dM={mass_err:.3e} | "
            f"dCOM={com_err:.3e} | "
            f"dI={inertia_err:.3e}"
        )

print("Optimization finished.")


with torch.no_grad():
    centers_np = centers.detach().cpu().numpy()
    radii_np = get_radii().detach().cpu().numpy()

# make a copy of the original mesh so we don't modify it further
box_mesh = mesh.copy()


box_edges = box_mesh.copy().edges_unique
box_lines = trimesh.load_path(box_mesh.vertices[box_edges])

sphere_meshes = []
for c, r in zip(centers_np, radii_np):
    # skip essentially-zero-radius spheres
    if r <= 1e-5:
        continue

    s = trimesh.creation.icosphere(subdivisions=3, radius=float(r))
    s.apply_translation(c)
    # color spheres (RGBA)
    s.visual.face_colors = [200, 50, 50, 200]
    sphere_meshes.append(s)

scene = trimesh.Scene([box_lines] + sphere_meshes)
# In a notebook this will open an interactive viewer (if supported),
# in a script it will open a window using the default viewer.
scene.show()
