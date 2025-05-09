#include "mfem.hpp"
#include "InputConfig.hxx"
#include "DG_Solver.hxx"
#include "FE_Evolution.hxx"

#include <fstream>
#include <iostream>
#include <cmath>

using namespace mfem;
using namespace std;

// Velocity coefficient factory for a given velocity value
VectorFunctionCoefficient MakeVelocityCoefficient(int dim, double v_value)
{
    return VectorFunctionCoefficient(dim, [=](const Vector &x, Vector &v) 
    {
        v.SetSize(dim);
        v = 0.0;
        v(0) = v_value;
    });
}

// Initial condition
double u0_function(const Vector &x)
{
    return exp(-50.0 * pow(x(0) - 0.5, 2));
}

// Inflow boundary condition
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

    Device device(device_config);
    device.Print();

    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();

    // Velocity nodes (DVM)
    std::vector<double> vNodes(num_vel);
    for (int i = 0; i < num_vel; i++)
    {
        vNodes[i] = vmin + i * (vmax - vmin) / (num_vel - 1);
    }
    int Nv = vNodes.size();    

    // FE space
    DG_FECollection fec(order, dim, BasisType::GaussLobatto);
    FiniteElementSpace fes(&mesh, &fec);
    cout << "Number of unknowns per velocity: " << fes.GetVSize() << endl;

    // Solution per velocity node
    std::vector<GridFunction> u;
    for (int i = 0; i < Nv; i++) 
    {
        u.emplace_back(&fes);
        FunctionCoefficient u0(u0_function);
        u[i].ProjectCoefficient(u0);
    }

    // Mass, convection, RHS vector, evolution, solver
    std::vector<BilinearForm*> m(Nv), k(Nv);
    std::vector<Vector> b(Nv);
    std::vector<FE_Evolution*> evolution(Nv);
    std::vector<ODESolver*> solver(Nv);

    for (int i = 0; i < Nv; i++)
    {
        // Velocity coefficient for v_i
        auto vel_coeff = MakeVelocityCoefficient(dim, vNodes[i]);
        FunctionCoefficient inflow(inflow_function);

        // Mass
        m[i] = new BilinearForm(&fes);
        m[i]->AddDomainIntegrator(new MassIntegrator);
        if (pa) m[i]->SetAssemblyLevel(AssemblyLevel::PARTIAL);
        else if (ea) m[i]->SetAssemblyLevel(AssemblyLevel::ELEMENT);
        else if (fa) m[i]->SetAssemblyLevel(AssemblyLevel::FULL);
        m[i]->Assemble();
        m[i]->Finalize();

        // Convection
        k[i] = new BilinearForm(&fes);
        k[i]->AddDomainIntegrator(new ConvectionIntegrator(vel_coeff, -1.0));
        k[i]->AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(vel_coeff, -1.0));
        k[i]->AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(vel_coeff, -1.0));
        if (pa) k[i]->SetAssemblyLevel(AssemblyLevel::PARTIAL);
        else if (ea) k[i]->SetAssemblyLevel(AssemblyLevel::ELEMENT);
        else if (fa) k[i]->SetAssemblyLevel(AssemblyLevel::FULL);
        k[i]->Assemble();
        k[i]->Finalize();

        // Boundary flow
        LinearForm bform(&fes);
        bform.AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow, vel_coeff, -1.0));
        bform.Assemble();
        b[i] = bform;

        // Evolution operator and solver
        evolution[i] = new FE_Evolution(*m[i], *k[i], b[i]);
        evolution[i]->SetTime(0.0);

        switch (ode_solver_type)
        {
            case 1:  solver[i] = new ForwardEulerSolver; break;
            case 2:  solver[i] = new RK2Solver(1.0); break;
            case 3:  solver[i] = new RK3SSPSolver; break;
            case 4:  solver[i] = new RK4Solver; break;
            case 6:  solver[i] = new RK6Solver; break;
            case 11: solver[i] = new BackwardEulerSolver; break;
            case 12: solver[i] = new SDIRK23Solver(2); break;
            case 13: solver[i] = new SDIRK33Solver; break;
            case 22: solver[i] = new ImplicitMidpointSolver; break;
            case 23: solver[i] = new SDIRK23Solver; break;
            case 24: solver[i] = new SDIRK34Solver; break;
            default: cerr << "Unknown ODE solver type: " << ode_solver_type << endl; return 3;
        }

        solver[i]->Init(*evolution[i]);
    }

    // Save mesh
    ofstream omesh("ex9.mesh");
    omesh.precision(precision);
    mesh.Print(omesh);

    // Time loop
    double t = 0.0;
    for (int ti = 0; t < t_final - 1e-8 * dt; ti++)
    {
        double dt_real = min(dt, t_final - t);
        for (int i = 0; i < Nv; i++)
        {
            double current_time = t;
            solver[i]->Step(u[i], current_time, dt_real);
        }
        t+=dt_real;

        if ((ti+1) % vis_steps == 0 || t + 1e-8*dt >= t_final)
        {
            cout << "Step " << ti+1 << ", time = " << t << endl;
            for (int i = 0; i < Nv; i++)
            {
                ostringstream name;
                name << "ex9-v" << i << "-" << (ti+1) << ".gf";
                ofstream sol_out(name.str());
                sol_out.precision(precision);
                u[i].Save(sol_out);
            }
        }
    }

    // Cleanup
    for (int i = 0; i < Nv; i++) 
    {
        delete m[i];
        delete k[i];
        delete evolution[i];
        delete solver[i];
    }

    return 0;
}
