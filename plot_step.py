import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
import os
import time

# --- Parse MFEM 1D mesh with high-order geometry in 'nodes' ---
def parse_mesh_with_nodes(mesh_path):
    with open(mesh_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Parse elements
    e_index = lines.index("elements") + 1
    num_elements = int(lines[e_index])
    elem_lines = lines[e_index + 1:e_index + 1 + num_elements]
    elements = [(int(parts.split()[2]), int(parts.split()[3])) for parts in elem_lines]

    # Parse nodes (geometry)
    n_index = lines.index("nodes") + 1
    assert lines[n_index] == "FiniteElementSpace"
    coord_start = n_index + 4
    coords = [float(line) for line in lines[coord_start:]]
    
    return np.array(coords), elements

# --- Parse MFEM GridFunction (.gf) file ---
def read_gf(gf_path):
    with open(gf_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    start = next(i for i, l in enumerate(lines) if l.startswith("Ordering")) + 1
    return np.array([float(line) for line in lines[start:]])

# --- Gauss-Lobatto points on [-1, 1] ---
def gauss_lobatto_points(n):
    if n == 2:
        return np.array([-1.0, 1.0])
    x_int, _ = leggauss(n - 2)
    return np.concatenate(([-1.0], x_int, [1.0]))

# --- Build Lagrange basis matrix for evaluation points ---
def lagrange_basis_matrix(x_nodes, x_eval):
    N = len(x_nodes)
    L = np.ones((len(x_eval), N))
    for i in range(N):
        for j in range(N):
            if i != j:
                L[:, i] *= (x_eval - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
    return L

# --- Main visualization function ---
def visualize_dg_solution(mesh_file, gf_file):
    vertices, elements = parse_mesh_with_nodes(mesh_file)
    coeffs = read_gf(gf_file)

    num_elements = len(elements)
    dofs_per_elem = len(coeffs) // num_elements
    order = dofs_per_elem - 1

    ref_nodes = gauss_lobatto_points(order + 1)
    x_eval = np.linspace(-1, 1, 100)
    basis = lagrange_basis_matrix(ref_nodes, x_eval)

    plt.figure(figsize=(10, 4))
    idx = 0
    for i in range(num_elements):
        v0, v1 = 2*i, 2*i+1
        x0, x1 = vertices[v0], vertices[v1]
        x_phys = 0.5 * (x1 - x0) * x_eval + 0.5 * (x0 + x1)
        coeff_local = coeffs[idx:idx + dofs_per_elem]
        u_vals = basis @ coeff_local
        plt.plot(x_phys, u_vals, label=f'Elem [{x0:.2f}, {x1:.2f}]')
        idx += dofs_per_elem

    plt.title(f"DG solution using Gauss-Lobatto basis (order {order})")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def animate_dg_solution(mesh_file, base_gf_pattern, start=0, stop=10000, step=100, delay=0.1):
    vertices, elements = parse_mesh_with_nodes(mesh_file)
    num_elements = len(elements)
    geometry = [(vertices[2*i], vertices[2*i+1]) for i in range(num_elements)]

    # Load all valid solutions
    file_indices = list(range(start, stop + 1, step))
    all_solutions = []
    for i in file_indices:
        fname = base_gf_pattern.format(i)
        if os.path.exists(fname):
            sol = read_gf(fname)
            all_solutions.append((i, sol))
        else:
            print(f"Missing: {fname}")

    if not all_solutions:
        print("No valid files loaded.")
        return

    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 4))
    lines = []
    ref_nodes = gauss_lobatto_points(len(all_solutions[0][1]) // num_elements)
    x_eval = np.linspace(-1, 1, 100)
    basis = lagrange_basis_matrix(ref_nodes, x_eval)

    for e, (x0, x1) in enumerate(geometry):
        x_phys = 0.5 * (x1 - x0) * x_eval + 0.5 * (x1 + x0)
        line, = ax.plot(x_phys, np.zeros_like(x_phys))
        lines.append(line)

    ax.set_title("DG Solution over Time")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True)

    # Animate
    for t_index, coeffs in all_solutions:
        for e, (x0, x1) in enumerate(geometry):
            dofs_per_elem = len(coeffs) // num_elements
            coeff_local = coeffs[e * dofs_per_elem : (e + 1) * dofs_per_elem]
            u_vals = basis @ coeff_local
            x_phys = 0.5 * (x1 - x0) * x_eval + 0.5 * (x0 + x1)
            lines[e].set_data(x_phys, u_vals)

        ax.set_title(f"Time step: {t_index}")
        ax.relim()
        ax.autoscale_view()
        plt.pause(delay)

    plt.show()

# === Example usage ===
# visualize_dg_solution("ex9.mesh", "ex9-0.gf")

# === Example animation usage ===
# ex9-{i}.gf from i = 0 to 10000 with step size 100
animate_dg_solution("ex9.mesh", "ex9-{}.gf", start=0, stop=10000, step=100, delay=0.1)
