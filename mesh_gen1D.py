def generate_mfem_1d_mesh(num_elements=4, length=1.0, periodic=True):
    dx = length / num_elements
    if periodic:
        filename = f"mesh/mesh_1D_{num_elements}_{length}_periodic.mesh"
    else:
        filename = f"mesh/mesh_1D_{num_elements}_{length}_dirchlet.mesh"

    with open(filename, "w") as f:
        f.write("MFEM mesh v1.0\n\n")
        f.write("#\n# MFEM Geometry Types (see fem/geom.hpp):\n#\n")
        f.write("# POINT       = 0\n")
        f.write("# SEGMENT     = 1\n")
        f.write("# TRIANGLE    = 2\n")
        f.write("# SQUARE      = 3\n")
        f.write("# TETRAHEDRON = 4\n")
        f.write("# CUBE        = 5\n")
        f.write("# PRISM       = 6\n")
        f.write("# PYRAMID     = 7\n#\n\n")

        f.write("dimension\n1\n\n")
        f.write("elements\n")
        f.write(f"{num_elements}\n")
        for i in range(num_elements - 1):
            f.write(f"1 1 {i} {i+1}\n")
        if periodic:
            f.write(f"1 1 {num_elements - 1} 0\n")  # periodic wraparound
        else:
            f.write(f"1 1 {num_elements - 1} {num_elements}\n")

        if periodic:
            f.write("\nboundary\n0\n\n")
        else:
            f.write("\nboundary\n2\n")
            f.write("1 0 0\n")
            f.write(f"2 0 {num_elements}\n\n")

        f.write("vertices\n")
        if periodic:
            f.write(f"{num_elements}\n\n")
        else:
            f.write(f"{num_elements+1}\n\n")

        f.write("nodes\n")
        f.write("FiniteElementSpace\n")
        f.write("FiniteElementCollection: L2_T1_1D_P1\n")
        f.write("VDim: 1\n")
        f.write("Ordering: 1\n\n")

        for i in range(num_elements):
            x0 = i * dx
            x1 = (i + 1) * dx
            f.write(f"{x0}\n{x1}\n")

    print(f"Mesh written to: {filename}")


# Example usage
generate_mfem_1d_mesh(num_elements=224, length=40.0, periodic=False)
