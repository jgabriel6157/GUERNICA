import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre
from scipy.optimize import newton
from scipy.integrate import quad

A = 0.291e-7
P = 0
U = 13.6 / 20
X = 0.232
K = 0.39
n0 = 5*10**18  
Tw = 2  
csL = np.sqrt(20)  
csR = -np.sqrt(20)  
Crec = 4.98*10**18  
Lz = 40

def sigma(A, P, U, X, K):
    return A * (1 + P * np.sqrt(U)) * U**K * np.exp(-U) / (X + U) * 1e-6

def integrand_1(v, z):
    return (1 / np.sqrt(2 * np.pi * Tw)) * Crec * np.exp(-((v - csL) ** 2) / (2 * Tw)) * \
           np.exp(-n0 * sigma(A, P, U, X, K) * np.sqrt(1.672e-27 / 1.602e-19) * (z) / v)

def integrand_2(v, z):
    return (1 / np.sqrt(2 * np.pi * Tw)) * Crec * np.exp(-((v - csR) ** 2) / (2 * Tw)) * \
           np.exp(-n0 * sigma(A, P, U, X, K) * np.sqrt(1.672e-27 / 1.602e-19) * (z - Lz) / v)

def f(z):
    integral_1, _ = quad(integrand_1, 0, np.inf, args=(z,))
    integral_2, _ = quad(integrand_2, -np.inf, 0, args=(z,))
    return integral_1 + integral_2

# --- Parse MFEM 1D mesh with linear geometry ---
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

# --- Parse MFEM GridFunction (.gf) file ---
def read_gf(gf_path):
    with open(gf_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    start = next(i for i, l in enumerate(lines) if l.startswith("Ordering")) + 1
    return np.array([float(line) for line in lines[start:]])

# --- Compute Gauss-Lobatto nodes on [-1, 1] ---
def gauss_lobatto_points(order):
    if order == 1:
        return np.array([-1.0, 1.0])
    Pn = Legendre.basis(order)
    dPn = Pn.deriv()
    interior = [newton(dPn, -np.cos(np.pi * i / order)) for i in range(1, order)]
    return np.array([-1.0] + interior + [1.0])

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

    ref_nodes = gauss_lobatto_points(order)
    x_eval = np.linspace(-1, 1, 100)
    basis = lagrange_basis_matrix(ref_nodes, x_eval)

    plt.figure(figsize=(10, 4))
    idx = 0
    for i in range(num_elements):
        # Use linear geometry from mesh
        v0, v1 = 2*i, 2*i+1
        x0, x1 = vertices[v0], vertices[v1]
        x_phys = 0.5 * (x1 - x0) * x_eval + 0.5 * (x0 + x1)

        coeff_local = coeffs[idx:idx + dofs_per_elem]
        u_vals = basis @ coeff_local           

        plt.plot(x_phys, u_vals*1e18, color='k')
        idx += dofs_per_elem

    x_vals = np.linspace(0, Lz, 100)
    f_vals = [f(z) for z in x_vals]
    plt.plot(x_vals,f_vals,color='red',linestyle = '--')

    plt.title(f"DG Solution (Order {order}) on Linear Mesh")
    plt.xlim(vertices.min(), vertices.max())
    plt.ylim(1e9, 1e19)
    plt.yscale('log')
    plt.xlabel("x")
    plt.ylabel("rho(x)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Animate DG solution over time ---
def animate_dg_solution(mesh_file, base_gf_pattern, start=0, stop=1000, step=100, delay=0.1):
    vertices, elements = parse_mesh_with_nodes(mesh_file)
    num_elements = len(elements)
    
    # Load one GF file to get DOFs per element
    sample_gf_path = base_gf_pattern.format(start)
    sample_coeffs = read_gf(sample_gf_path)
    dofs_per_elem = len(sample_coeffs) // num_elements
    order = dofs_per_elem - 1

    ref_nodes = gauss_lobatto_points(order)
    x_eval = np.linspace(-1, 1, 100)
    basis = lagrange_basis_matrix(ref_nodes, x_eval)

    # Precompute geometry per element
    x_phys_list = []
    for i in range(num_elements):
        x0, x1 = vertices[2*i], vertices[2*i+1]
        x_phys = 0.5 * (x1 - x0) * x_eval + 0.5 * (x0 + x1)
        x_phys_list.append(x_phys)

    # Load all solutions
    solutions = []
    time_indices = []
    for i in range(start, stop + 1, step):
        path = base_gf_pattern.format(i)
        if os.path.exists(path):
            coeffs = read_gf(path)
            solutions.append(coeffs)
            time_indices.append(i)
        else:
            print(f"Missing: {path}")

    if not solutions:
        print("No valid .gf files found.")
        return

    # Animate
    fig, ax = plt.subplots(figsize=(10, 4))
    lines = [ax.plot(x, np.zeros_like(x), color='k')[0] for x in x_phys_list]

    ax.set_xlim(vertices.min(), vertices.max())
    ax.set_ylim(1e8, 1e19)
    ax.set_yscale('log')
    ax.set_xlabel("x")
    ax.set_ylabel("rho(x)")
    ax.grid(True)

    for t_idx, coeffs in zip(time_indices, solutions):
        for i in range(num_elements):
            coeff_local = coeffs[i*dofs_per_elem:(i+1)*dofs_per_elem]
            u_vals = basis @ coeff_local
            lines[i].set_data(x_phys_list[i], u_vals*1e18)

        ax.set_title(f"Time Step: {t_idx}")
        plt.pause(delay)

    plt.show()

# === Example usage ===
visualize_dg_solution("ex9.mesh", "gf_out/rho-9000.gf")

# animate_dg_solution("ex9.mesh", "gf_out/rho-{}.gf", start=0, stop=9000, step=100, delay=0.1)
