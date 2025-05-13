import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre
from scipy.optimize import newton
from mpl_toolkits.mplot3d import Axes3D

def parse_mesh_with_nodes(mesh_path):
    with open(mesh_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    e_index = lines.index("elements") + 1
    num_elements = int(lines[e_index])
    elem_lines = lines[e_index + 1:e_index + 1 + num_elements]
    elements = [(int(parts.split()[2]), int(parts.split()[3])) for parts in elem_lines]
    n_index = lines.index("nodes") + 1
    assert lines[n_index] == "FiniteElementSpace"
    coord_start = n_index + 4
    coords = [float(line) for line in lines[coord_start:]]
    return np.array(coords), elements

def read_gf(gf_path):
    with open(gf_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    start = next(i for i, l in enumerate(lines) if l.startswith("Ordering")) + 1
    return np.array([float(line) for line in lines[start:]])

def gauss_lobatto_points(order):
    if order == 1:
        return np.array([-1.0, 1.0])
    Pn = Legendre.basis(order)
    dPn = Pn.deriv()
    interior = [newton(dPn, -np.cos(np.pi * i / order)) for i in range(1, order)]
    return np.array([-1.0] + interior + [1.0])

def lagrange_basis_matrix(x_nodes, x_eval):
    N = len(x_nodes)
    L = np.ones((len(x_eval), N))
    for i in range(N):
        for j in range(N):
            if i != j:
                L[:, i] *= (x_eval - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
    return L

def plot_dg_solution_3d_disconnected(mesh_file, gf_pattern, vmin, vmax, num_vel, time_index):
    vertices, elements = parse_mesh_with_nodes(mesh_file)
    num_elements = len(elements)

    velocity_indices = list(range(num_vel))
    physical_velocities = np.linspace(vmin, vmax, num_vel)

    sample_coeffs = read_gf(gf_pattern.format(0, time_index))
    dofs_per_elem = len(sample_coeffs) // num_elements
    order = dofs_per_elem - 1

    ref_nodes = gauss_lobatto_points(order)
    x_eval = np.linspace(-1, 1, 100)
    basis = lagrange_basis_matrix(ref_nodes, x_eval)

    x_phys_list = []
    for i in range(num_elements):
        x0, x1 = vertices[2*i], vertices[2*i+1]
        x_phys = 0.5 * (x1 - x0) * x_eval + 0.5 * (x0 + x1)
        x_phys_list.append(x_phys)
    x_all = np.concatenate(x_phys_list)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    for idx, v_idx in enumerate(velocity_indices):
        path = gf_pattern.format(v_idx, time_index)
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue

        coeffs = read_gf(path)
        u_vals_list = []
        for i in range(num_elements):
            coeff_local = coeffs[i*dofs_per_elem:(i+1)*dofs_per_elem]
            u_vals = basis @ coeff_local
            u_vals_list.append(u_vals)
        u_full = np.concatenate(u_vals_list)

        v_array = np.full_like(x_all, physical_velocities[idx])
        ax.plot(x_all, v_array, u_full, color='black')

    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_zlabel("u(x, v)")
    ax.set_title(f"DG Solution at Time Step {time_index}")
    plt.tight_layout()
    plt.show()

def animate_dg_solution_3d_disconnected(mesh_file, gf_pattern, vmin, vmax, num_vel,
                                        start=0, stop=1000, step=100, delay=0.1):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    vertices, elements = parse_mesh_with_nodes(mesh_file)
    num_elements = len(elements)

    velocity_indices = list(range(num_vel))
    physical_velocities = np.linspace(vmin, vmax, num_vel)

    sample_coeffs = read_gf(gf_pattern.format(0, start))
    dofs_per_elem = len(sample_coeffs) // num_elements
    order = dofs_per_elem - 1

    ref_nodes = gauss_lobatto_points(order)
    x_eval = np.linspace(-1, 1, 100)
    basis = lagrange_basis_matrix(ref_nodes, x_eval)

    # Prepare physical x grid
    x_phys_list = []
    for i in range(num_elements):
        x0, x1 = vertices[2*i], vertices[2*i+1]
        x_phys = 0.5 * (x1 - x0) * x_eval + 0.5 * (x0 + x1)
        x_phys_list.append(x_phys)
    x_all = np.concatenate(x_phys_list)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Initialize empty plot lines
    line_objs = []
    for idx in range(num_vel):
        v_array = np.full_like(x_all, physical_velocities[idx])
        line, = ax.plot(x_all, v_array, np.zeros_like(x_all), color='black')
        line_objs.append(line)

    ax.set_xlim(x_all.min(), x_all.max())
    ax.set_ylim(physical_velocities.min(), physical_velocities.max())
    ax.set_zlim(0, 1.1)  # adjust as needed
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_zlabel("u(x, v)")

    for t in range(start, stop + 1, step):
        for idx, v_idx in enumerate(velocity_indices):
            path = gf_pattern.format(v_idx, t)
            if not os.path.exists(path):
                print(f"Missing: {path}")
                continue

            coeffs = read_gf(path)
            u_vals_list = []
            for i in range(num_elements):
                coeff_local = coeffs[i*dofs_per_elem:(i+1)*dofs_per_elem]
                u_vals = basis @ coeff_local
                u_vals_list.append(u_vals)
            u_full = np.concatenate(u_vals_list)

            line_objs[idx].set_data(x_all, np.full_like(x_all, physical_velocities[idx]))
            line_objs[idx].set_3d_properties(u_full)

        ax.set_title(f"Time Step: {t}")
        plt.pause(delay)

    plt.show()

def animate_dg_solution_3d_wireframe(mesh_file, gf_pattern, vmin, vmax, num_vel,
                                     start=0, stop=1000, step=100, delay=0.1):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    vertices, elements = parse_mesh_with_nodes(mesh_file)
    num_elements = len(elements)

    velocity_indices = list(range(num_vel))
    physical_velocities = np.linspace(vmin, vmax, num_vel)

    # Sample to get DOFs and order
    sample_coeffs = read_gf(gf_pattern.format(0, start))
    dofs_per_elem = len(sample_coeffs) // num_elements
    order = dofs_per_elem - 1

    ref_nodes = gauss_lobatto_points(order)
    x_eval = np.linspace(-1, 1, 10)
    basis = lagrange_basis_matrix(ref_nodes, x_eval)

    # Build full physical x grid
    x_phys_list = []
    for i in range(num_elements):
        x0, x1 = vertices[2*i], vertices[2*i+1]
        x_phys = 0.5 * (x1 - x0) * x_eval + 0.5 * (x0 + x1)
        x_phys_list.append(x_phys)
    x_all = np.concatenate(x_phys_list)

    X, V = np.meshgrid(x_all, physical_velocities)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    wire = None
    ax.set_xlim(x_all.min(), x_all.max())
    ax.set_ylim(physical_velocities.min(), physical_velocities.max())
    ax.set_zlim(0, 1.1)
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_zlabel("u(x, v)")

    for t in range(start, stop + 1, step):
        u_data = []
        for v_idx in velocity_indices:
            path = gf_pattern.format(v_idx, t)
            if not os.path.exists(path):
                print(f"Missing: {path}")
                u_data.append(np.zeros_like(x_all))
                continue

            coeffs = read_gf(path)
            u_vals_list = []
            for i in range(num_elements):
                coeff_local = coeffs[i*dofs_per_elem:(i+1)*dofs_per_elem]
                u_vals = basis @ coeff_local
                u_vals_list.append(u_vals)
            u_data.append(np.concatenate(u_vals_list))

        U = np.array(u_data)

        if wire:
            wire.remove()
        wire = ax.plot_wireframe(X, V, U, color='black', rstride=1, cstride=1)

        ax.set_title(f"Time Step: {t}")
        plt.pause(delay)

    plt.show()


# Example call:
# plot_dg_solution_3d_disconnected("ex9.mesh", "ex9-v{}-{}.gf", vmin=-1.0, vmax=1.0, num_vel=3, time_index=10)
# animate_dg_solution_3d_disconnected("ex9.mesh", "ex9-v{}-{}.gf",
                                    #  vmin=-1.0, vmax=1.0, num_vel=9,
                                    #  start=10, stop=1000, step=10, delay=0.1)
animate_dg_solution_3d_wireframe("ex9.mesh", "gf_out/ex9-v{}-{}.gf",
                                 vmin=-1.0, vmax=1.0, num_vel=9,
                                 start=0, stop=48000, step=400, delay=0.1)


