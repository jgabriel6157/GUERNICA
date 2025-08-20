#include "mfem.hpp"
#include "InputConfig.hxx"
#include "DG_Solver.hxx"
#include "FE_Evolution.hxx"
#include "DG_Advection.hxx"

#include <fstream>
#include <iostream>
#include <cmath>
#include <filesystem>

using namespace mfem;
using namespace std;
namespace fs = std::filesystem;

VectorFunctionCoefficient MakeVelocityCoefficient(int dim, double v_value)
{
    return VectorFunctionCoefficient(dim, [=](const Vector &x, Vector &v) 
    {
        v.SetSize(dim);
        v = 0.0;
        v(0) = v_value;
    });
}

double u0_function(const Vector &x)
{
    return exp(-50.0 * pow(x(0) - 0.5, 2));
    // return 1.0;
}

double inflow_function(const Vector &x)
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
    bool pa = false, ea = false, fa = false;
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
    args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa", "--no-partial-assembly", "Enable Partial Assembly.");
    args.AddOption(&ea, "-ea", "--element-assembly", "-no-ea", "--no-element-assembly", "Enable Element Assembly.");
    args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa", "--no-full-assembly", "Enable Full Assembly.");
    args.AddOption(&device_config, "-d", "--device", "Device configuration string.");

    args.Parse();
    if (!args.Good()) { args.PrintUsage(cout); return 1; }
    args.PrintOptions(cout);

    std::string out_dir = "gf_out";
    if (fs::exists(out_dir)) 
    {
        fs::remove_all(out_dir);
    }
    fs::create_directory(out_dir);

    Device device(device_config);
    device.Print();

    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();

    // Velocity grid
    std::vector<double> vNodes(num_vel);
    for (int i = 0; i < num_vel; i++) 
    {
        vNodes[i] = vmin + i * (vmax - vmin) / (num_vel - 1);
    }
    const int Nv = vNodes.size();

    // FE space
    DG_FECollection fec(order, dim, BasisType::GaussLobatto);
    FiniteElementSpace fes(&mesh, &fec);
    const int Ndof = fes.GetVSize();
    cout << "Number of unknowns per velocity: " << Ndof << endl;

    Array<int> block_offsets(Nv+1);
    for (int i=0; i<=Nv; i++)
    {
        block_offsets[i] = i*Ndof;
    }

    BlockVector U(block_offsets);
    BlockVector dU(block_offsets);

    // Initial condition into each block
    FunctionCoefficient u0(u0_function);
    for (int iv = 0; iv < Nv; ++iv)
    {
        GridFunction ui(&fes, U.GetBlock(iv).GetData());
        ui.ProjectCoefficient(u0);
    }

    // Save mesh and initial solutions
    ofstream omesh("ex9.mesh");
    omesh.precision(precision);
    mesh.Print(omesh);

    for (int i = 0; i < Nv; i++) 
    {
        ostringstream name;
        name << "gf_out/ex9-v" << i << "-0.gf";
        ofstream sol_out(name.str());
        GridFunction ui(&fes, U.GetBlock(i).GetData());
        sol_out.precision(precision);
        ui.Save(sol_out);
    }

    // Build the element-outer operator (advection, 1D)
    DG_Advection op(fes, vNodes, t_final);

    // One ODE solver for the whole stacked system
    std::unique_ptr<ODESolver> solver;
    switch (ode_solver_type) {
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
            cout << "Step " << ti << ", time = " << t << endl;
            for (int iv = 0; iv < Nv; ++iv)
            {
                ostringstream name; name << "gf_out/ex9-v" << iv << "-" << ti << ".gf";
                ofstream sol_out(name.str());
                GridFunction ui(&fes, U.GetBlock(iv).GetData());
                ui.Save(sol_out);
            }
        }
    }

    return 0;
}
