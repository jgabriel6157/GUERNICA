#include "mfem.hpp"
#include "InputConfig.hxx"
#include "DG_Advection.hxx"

#include <fstream>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <memory>
#include <sstream>
#include <cstring>

using namespace mfem;
using namespace std;
namespace fs = std::filesystem;

double u0_function(const Vector &x)
{
    return exp(-50.0 * pow(x(0) - 0.5, 2));
}

double inflow_function(const Vector &)
{
    return 0.0;
}

int main(int argc, char *argv[])
{
    // Load config
    InputConfig config("input.cfg");
    std::string mesh_file_str = config.Get<std::string>("mesh_file", "mesh/periodic-segment.mesh");
    const char *mesh_file = mesh_file_str.c_str();
    int order = config.Get<int>("order", 4);
    int ode_solver_type = config.Get<int>("ode_solver_type", 3);
    double t_final = config.Get<double>("t_final", 10.0);
    double dt = config.Get<double>("dt", 0.001);
    int vis_steps = config.Get<int>("vis_steps", 100);
    double vmin = config.Get<double>("vmin", 0.0);
    double vmax = config.Get<double>("vmax", 1.0);
    int num_vel = config.Get<int>("num_vel", 2);
    const char *device_config = "cpu";

    int precision = 8;
    cout.precision(precision);

    // Parse CLI args
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
    args.AddOption(&order, "-o", "--order", "Finite element order.");
    args.AddOption(&ode_solver_type, "-s", "--ode-solver", "ODE solver type.");
    args.AddOption(&t_final, "-tf", "--t-final", "Final time.");
    args.AddOption(&dt, "-dt", "--time-step", "Time step.");
    args.AddOption(&vis_steps, "-vs", "--visualization-steps", "Steps between visualization outputs.");
    args.AddOption(&device_config, "-d", "--device", "Device configuration string.");
    args.Parse();
    if (!args.Good()) { args.PrintUsage(cout); return 1; }
    args.PrintOptions(cout);

    // Output dir
    std::string out_dir = "gf_out";
    if (fs::exists(out_dir)) { fs::remove_all(out_dir); }
    fs::create_directory(out_dir);

    Device device(device_config);
    device.Print();

    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();

    // Velocity grid (assumed num_vel >= 2 by config)
    std::vector<double> vNodes(num_vel);
    for (int i = 0; i < num_vel; i++)
    {
        vNodes[i] = vmin + i * (vmax - vmin) / (num_vel - 1);
    }
    const int Nv = (int)vNodes.size();

    // FE space
    DG_FECollection fec(order, dim, BasisType::GaussLobatto);
    FiniteElementSpace fes(&mesh, &fec);
    const int Ndof = fes.GetVSize();
    cout << "Number of unknowns per velocity: " << Ndof << endl;

    // Build the element-major operator (advection, 1D)
    DG_Advection op(fes, vNodes, t_final);

    // Introspection for element-major packing
    const int NE = op.GetNE();
    const int Ntot = op.GlobalSize();
    const std::vector<int> &elem_base = op.ElemBase();

    // Allocate element-major global vectors
    Vector U(Ntot);
    U = 0.0;
    Vector dUdt(Ntot);
    dUdt = 0.0;

    // Initialize U with projected scalar initial condition, replicated across velocities
    {
        GridFunction u0_gf(&fes);
        FunctionCoefficient u0c(u0_function);
        u0_gf.ProjectCoefficient(u0c);

        Array<int> vdofs;
        Vector Ue; // element-local from L-layout

        for (int e = 0; e < NE; ++e)
        {
            const int base = elem_base[e];
            const int ld   = op.Ldof(e);

            fes.GetElementVDofs(e, vdofs);
            Ue.SetSize(ld);
            u0_gf.GetSubVector(vdofs, Ue);

            for (int iv = 0; iv < Nv; ++iv)
            {
                double *dst = U.Write() + base + iv*ld;
                std::memcpy(dst, Ue.Read(), ld * sizeof(double));
            }
        }
    }

    // Save mesh and initial solutions (unpack to L-layout per velocity)
    ofstream omesh("ex9.mesh");
    omesh.precision(precision);
    mesh.Print(omesh);

    auto dump_fields = [&](int step, double tcur)
    {
        Vector Ub(Ndof); // L-layout buffer
        Array<int> vdofs;

        for (int iv = 0; iv < Nv; ++iv)
        {
            Ub = 0.0;
            for (int e = 0; e < NE; ++e)
            {
                const int base = elem_base[e];
                const int ld   = op.Ldof(e);
                fes.GetElementVDofs(e, vdofs);

                Vector Uslab(const_cast<double*>(U.Read()) + base + iv*ld, ld);
                Ub.SetSubVector(vdofs, Uslab);
            }

            GridFunction ui(&fes);
            ui = Ub;

            ostringstream name;
            name << out_dir << "/ex9-v" << iv << "-" << step << ".gf";
            ofstream sol_out(name.str());
            sol_out.precision(precision);
            ui.Save(sol_out);
        }

        cout << "Step " << step << ", time = " << tcur << endl;
    };

    dump_fields(0, 0.0);

    // ODE solver
    std::unique_ptr<ODESolver> solver;
    switch (ode_solver_type)
    {
        case 1:  solver = std::make_unique<ForwardEulerSolver>(); break;
        case 2:  solver = std::make_unique<RK2Solver>(1.0); break;
        case 3:  solver = std::make_unique<RK3SSPSolver>(); break;
        case 4:  solver = std::make_unique<RK4Solver>(); break;
        case 6:  solver = std::make_unique<RK6Solver>(); break;
        default: solver = std::make_unique<RK3SSPSolver>(); break;
    }
    solver->Init(op);

    // Time loop
    double t = 0.0;
    int ti = 0;
    while (t < t_final - 1e-8*dt)
    {
        double dt_real = std::min(dt, t_final - t);
        solver->Step(U, t, dt_real);
        ti++;

        if ((ti % vis_steps) == 0 || t + 1e-8*dt >= t_final)
        {
            dump_fields(ti, t);
        }
    }

    return 0;
}
