import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre
from scipy.optimize import newton
from scipy.integrate import quad

Lz = 40
zvals = np.linspace(0,Lz,73)
#tw = 2
sol = [9.530037918593606e+18,3.443585377795066e+18,1.1798778383970465e+18,4.230035393563359e+17,1.582902613583898e+17,6.149948822451518e+16,2.472530326865515e+16,
       1.0255050148172988e+16,4380327202107013.0,1923718648188707.2,867818305504052.6,401746227850319.7,190725403969430.44,92780866066238.86,46213707070264.734,
       23548111437137.75,12262806802297.824,6519188821336.82,3533903271155.998,1950887233760.4268,1095407474005.471,624794875291.5898,361564797707.91644,
       212038887327.73297,125878241642.13467,75570061369.60495,45835537586.36156,28062240135.22628,17327437665.918846,10780669536.719915,6751335947.825615,
       4249109178.072075,2680495390.223425,1685151768.5609396,1043217405.1694884,602606475.811542,302962114.23153406,602606475.8111047,1043217405.1681938,
       1685151768.5584273,2680495390.218846,4249109178.0642195,6751335947.81239,10780669536.698355,17327437665.88374,28062240135.16903,45835537586.2684,
       75570061369.4519,125878241641.89076,212038887327.32016,361564797707.2179,624794875290.4581,1095407474003.4694,1950887233756.92,3533903271149.823,
       6519188821325.937,12262806802278.195,23548111437110.336,46213707070202.67,92780866066119.61,190725403969180.28,401746227849867.94,867818305503060.1,
       1923718648186979.5,4380327202101790.5,1.0255050148166076e+16,2.4725303268610548e+16,6.149948822447443e+16,1.582902613581344e+17,4.230035393560502e+17,
       1.1798778383881987e+18,3.4435853777890196e+18,9.530037918590149e+18]
#tw = 10
# sol = [9.271348489836261e+18,3.8754428088949514e+18,1.6932338616168102e+18,7.754379043645197e+17,3.703097076228575e+17,1.822157351491099e+17,
#        9.21811376675369e+16,4.767955433672934e+16,2.5171769786945176e+16,1.3524586186453064e+16,7384923039433558.0,4090955994446276.0,2296485632385380.5,
#        1304838202117868.2,749739790080419.5,435267778721258.56,255142715477517.84,150906047778989.8,90007234113567.34,54108963673047.09,32770176667558.055,
#        19985768033112.9,12269481498689.963,7579465572786.979,4709903064214.832,2943114536287.073,1848788332094.2627,1167104784356.1567,740140862116.5347,
#        471297430651.7465,301123988646.1213,192810267953.17917,123423571847.6819,78543255380.94543,49099902858.73616,28561938896.348938,14404363732.721174,
#        28561938896.348713,49099902858.74805,78543255380.95578,123423571847.68909,192810267953.1872,301123988646.12506,471297430651.776,740140862116.5443,
#        1167104784356.1626,1848788332094.2788,2943114536287.0835,4709903064215.169,7579465572787.246,12269481498690.16,19985768033114.184,32770176667558.723,
#        54108963673046.71,90007234113566.84,150906047778987.78,255142715477515.3,435267778721187.4,749739790080367.8,1304838202117811.5,2296485632385492.0,
#        4090955994446296.5,7384923039433890.0,1.3524586186453012e+16,2.5171769786939908e+16,4.76795543367135e+16,9.21811376674128e+16,1.822157351490363e+17,
#        3.703097076227858e+17,7.754379043650042e+17,1.6932338616167905e+18,3.875442808894982e+18,9.271348489836345e+18]

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

    plt.plot(zvals,sol,color='red',linestyle = '--')

    plt.title(f"DG Solution (Order {order}) on Linear Mesh")
    plt.xlim(vertices.min(), vertices.max())
    plt.ylim(1e8, 1e19)
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
