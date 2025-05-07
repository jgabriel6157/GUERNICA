#include "mfem.hpp"
#include "InputConfig.hxx"
#include "DG_Solver.hxx"
#include "FE_Evolution.hxx"

#include <fstream>
#include <iostream>
#include <cmath>

using namespace mfem;
using namespace std;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
    v(0) = 1.0;
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
    std::string mesh_file_str = config.Get<std::string>("mesh_file", "external/mfem/data/periodic-segment.mesh");
    const char *mesh_file = mesh_file_str.c_str();
    int order = config.Get<int>("order", 4);
    int ode_solver_type = config.Get<int>("ode_solver_type", 3);
    double t_final = config.Get<double>("t_final", 10.0);
    double dt = config.Get<double>("dt", 0.001);
    bool visualization = config.Get<bool>("visualization", true);
    bool paraview = config.Get<bool>("paraview", true);
    int vis_steps = config.Get<int>("vis_steps", 100);
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
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis", "--no-visualization", "Enable/disable GLVis.");
    args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview", "--no-paraview-datafiles", "Enable/disable ParaView.");
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

    // Setup ODE solver
    ODESolver *ode_solver = NULL;
    switch (ode_solver_type)
    {
        case 1:  ode_solver = new ForwardEulerSolver; break;
        case 2:  ode_solver = new RK2Solver(1.0); break;
        case 3:  ode_solver = new RK3SSPSolver; break;
        case 4:  ode_solver = new RK4Solver; break;
        case 6:  ode_solver = new RK6Solver; break;
        case 11: ode_solver = new BackwardEulerSolver; break;
        case 12: ode_solver = new SDIRK23Solver(2); break;
        case 13: ode_solver = new SDIRK33Solver; break;
        case 22: ode_solver = new ImplicitMidpointSolver; break;
        case 23: ode_solver = new SDIRK23Solver; break;
        case 24: ode_solver = new SDIRK34Solver; break;
        default: cerr << "Unknown ODE solver type: " << ode_solver_type << endl; return 3;
    }

    // Finite Element space
    DG_FECollection fec(order, dim, BasisType::GaussLobatto);
    FiniteElementSpace fes(&mesh, &fec);
    cout << "Number of unknowns: " << fes.GetVSize() << endl;

    // Define forms
    VectorFunctionCoefficient velocity(dim, velocity_function);
    FunctionCoefficient inflow(inflow_function);
    FunctionCoefficient u0(u0_function);

    BilinearForm m(&fes);
    BilinearForm k(&fes);
    if (pa) m.SetAssemblyLevel(AssemblyLevel::PARTIAL);
    else if (ea) m.SetAssemblyLevel(AssemblyLevel::ELEMENT);
    else if (fa) m.SetAssemblyLevel(AssemblyLevel::FULL);

    m.AddDomainIntegrator(new MassIntegrator);
    constexpr double alpha = -1.0;
    k.AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
    k.AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity, alpha));
    k.AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity, alpha));

    LinearForm b(&fes);
    b.AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow, velocity, alpha));

    m.Assemble();
    k.Assemble();
    b.Assemble();
    m.Finalize();
    k.Finalize();

    GridFunction u(&fes);
    u.ProjectCoefficient(u0);

    {
        ofstream omesh("ex9.mesh");
        omesh.precision(precision);
        mesh.Print(omesh);
        ofstream osol("ex9-0.gf");
        osol.precision(precision);
        u.Save(osol);
    }

    ParaViewDataCollection *pd = nullptr;
    if (paraview)
    {
        pd = new ParaViewDataCollection("Example9", &mesh);
        pd->SetPrefixPath("ParaView");
        pd->RegisterField("solution", &u);
        pd->SetLevelsOfDetail(order);
        pd->SetDataFormat(VTKFormat::BINARY);
        pd->SetHighOrderOutput(true);
        pd->SetCycle(0);
        pd->SetTime(0.0);
        pd->Save();
    }

    socketstream sout;
    if (visualization)
    {
        sout.open("localhost", 19916);
        if (!sout)
        {
            cout << "GLVis server not found. Visualization disabled." << endl;
            visualization = false;
        }
        else
        {
            sout.precision(precision);
            sout << "solution\n" << mesh << u << "pause\n" << flush;
        }
    }

    FE_Evolution adv(m, k, b);
    double t = 0.0;
    adv.SetTime(t);
    ode_solver->Init(adv);

    for (int ti = 0; t < t_final - 1e-8*dt; ti++)
    {
        double dt_real = min(dt, t_final - t);
        ode_solver->Step(u, t, dt_real);

        if ((ti+1) % vis_steps == 0 || t + 1e-8*dt >= t_final)
        {
            cout << "Step " << ti+1 << ", t = " << t << endl;
            ofstream sol_out("ex9-" + to_string(ti+1) + ".gf");
            sol_out.precision(precision);
            u.Save(sol_out);

            if (visualization)
            {
                sout << "solution\n" << mesh << u << "pause 0.1\n" << flush;
            }

            if (paraview)
            {
                pd->SetCycle(ti+1);
                pd->SetTime(t);
                pd->Save();
            }
        }
    }

    delete ode_solver;
    delete pd;
    return 0;
}
