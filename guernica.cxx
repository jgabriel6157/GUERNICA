#include "mfem.hpp"
#include "InputConfig.hxx"
#include "DG_Solver.hxx"
#include "FE_Evolution.hxx"
#include "IonizationOperator.hxx"
#include "OperatorToTimeDependent.hxx"

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
    double ionization_rate = config.Get<double>("ionization_rate",0.0);
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
    int Nv = vNodes.size();

    // FE space
    DG_FECollection fec(order, dim, BasisType::GaussLobatto);
    FiniteElementSpace fes(&mesh, &fec);
    cout << "Number of unknowns per velocity: " << fes.GetVSize() << endl;

    // Shared mass matrix
    BilinearForm *m = new BilinearForm(&fes);
    m->AddDomainIntegrator(new MassIntegrator);
    if (pa) m->SetAssemblyLevel(AssemblyLevel::PARTIAL);
    else if (ea) m->SetAssemblyLevel(AssemblyLevel::ELEMENT);
    else if (fa) m->SetAssemblyLevel(AssemblyLevel::FULL);
    m->Assemble();
    m->Finalize();

    // Solution vectors and structures per velocity
    std::vector<GridFunction> u;
    std::vector<BilinearForm*> k(Nv);
    std::vector<Vector> b(Nv);
    std::vector<FE_Evolution*> evolution(Nv);
    std::vector<ODESolver*> solver(Nv);
    std::vector<IonizationOperator*> ionization(Nv);
    std::vector<Operator*> rhs_operator(Nv);
    std::vector<OperatorToTimeDependent*> td_rhs_operator(Nv);

    // Solver factory
    auto make_solver = [&]() -> ODESolver* 
    {
        switch (ode_solver_type)
        {
            case 1:  return new ForwardEulerSolver;
            case 2:  return new RK2Solver(1.0);
            case 3:  return new RK3SSPSolver;
            case 4:  return new RK4Solver;
            case 6:  return new RK6Solver;
            case 11: return new BackwardEulerSolver;
            case 12: return new SDIRK23Solver(2);
            case 13: return new SDIRK33Solver;
            case 22: return new ImplicitMidpointSolver;
            case 23: return new SDIRK23Solver;
            case 24: return new SDIRK34Solver;
            default:
                cerr << "Unknown ODE solver type: " << ode_solver_type << endl;
                exit(3);
        }
    };

    for (int i = 0; i < Nv; i++)
    {
        u.emplace_back(&fes);
        FunctionCoefficient u0(u0_function);
        u[i].ProjectCoefficient(u0);

        auto vel_coeff = MakeVelocityCoefficient(dim, vNodes[i]);
        FunctionCoefficient inflow(inflow_function);

        // Convection matrix
        k[i] = new BilinearForm(&fes);
        k[i]->AddDomainIntegrator(new ConvectionIntegrator(vel_coeff, -1.0));
        k[i]->AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(vel_coeff, -1.0));
        k[i]->AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(vel_coeff, -1.0));
        if (pa) k[i]->SetAssemblyLevel(AssemblyLevel::PARTIAL);
        else if (ea) k[i]->SetAssemblyLevel(AssemblyLevel::ELEMENT);
        else if (fa) k[i]->SetAssemblyLevel(AssemblyLevel::FULL);
        k[i]->Assemble();
        k[i]->Finalize();

        // Boundary flow vector
        LinearForm bform(&fes);
        bform.AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow, vel_coeff, -1.0));
        bform.Assemble();
        b[i] = bform;

        // Evolution and solver
        evolution[i] = new FE_Evolution(*m, *k[i], b[i]);
        evolution[i]->SetTime(0.0);
        
        // Add collision operator: -S*u
        ionization[i] = new IonizationOperator(fes.GetVSize(), ionization_rate);

        // Combine: du/dt = transport + ionization
        rhs_operator[i] = new SumOperator(evolution[i],1.0,ionization[i],1.0,false,false);
        td_rhs_operator[i] = new OperatorToTimeDependent(*rhs_operator[i]);

        solver[i] = make_solver();
        solver[i]->Init(*td_rhs_operator[i]);
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
        sol_out.precision(precision);
        u[i].Save(sol_out);
    }

    // Time loop
    double t = 0.0;
    for (int ti = 0; t < t_final - 1e-8 * dt; ti++)
    {
        double dt_real = min(dt, t_final - t);
        for (int i = 0; i < Nv; i++) 
        {
            double local_t = t;
            solver[i]->Step(u[i], local_t, dt_real);
        }
        t += dt_real;

        if ((ti+1) % vis_steps == 0 || t + 1e-8*dt >= t_final)
        {
            cout << "Step " << ti+1 << ", time = " << t << endl;
            for (int i = 0; i < Nv; i++) 
            {
                ostringstream name;
                name << "gf_out/ex9-v" << i << "-" << (ti+1) << ".gf";
                ofstream sol_out(name.str());
                sol_out.precision(precision);
                u[i].Save(sol_out);
            }
        }
    }

    // Cleanup
    delete m;
    for (int i = 0; i < Nv; i++) 
    {
        delete k[i];
        delete evolution[i];
        delete ionization[i];
        delete rhs_operator[i];
        delete td_rhs_operator[i];
        delete solver[i];
    }

    return 0;
}
