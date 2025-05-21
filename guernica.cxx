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

Vector integrate1D(const std::vector<GridFunction> &u,
                   const std::vector<double> &vNodes,
                   int power)
{
    int nv = vNodes.size();
    int ndof = u[0].Size();
    double dv = vNodes[1] - vNodes[0];

    Vector result(ndof);
    result = 0.0;

    // Simpson's rule
    for (int i = 0; i < nv; i++)
    {
        const double *f_i = u[i].GetData();
        double v = vNodes[i];
        double w;

        if (i == 0 || i == nv - 1)
            w = 1.0;
        else if (i % 2 == 1)
            w = 4.0;
        else
            w = 2.0;

        double factor = w * pow(v, power);
        for (int j = 0; j < ndof; j++)
        {
            result[j] += factor * f_i[j];
        }
    }

    result *= dv / 3.0;
    return result;
}

void ComputeMoments(const std::vector<GridFunction> &u,
                    const std::vector<double> &vNodes,
                    const FiniteElementSpace &fes,
                    GridFunction &rho,
                    GridFunction &u_bulk,
                    GridFunction &T)
{
    Vector rho_v = integrate1D(u, vNodes, 0);
    Vector mom_v = integrate1D(u, vNodes, 1);
    Vector E_v   = integrate1D(u, vNodes, 2);

    int ndof = fes.GetVSize();

    rho.SetSize(ndof);
    u_bulk.SetSize(ndof);
    T.SetSize(ndof);

    for (int i = 0; i < ndof; i++)
    {
        double r = rho_v[i];
        double u = mom_v[i] / r;
        double E = E_v[i];

        rho[i] = r;
        u_bulk[i] = u;
        T[i] = (E - r * u * u) / r;
    }
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

    GridFunction rho(&fes), u_bulk(&fes), T(&fes);

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
        double v_i = vNodes[i];

        FunctionCoefficient u0([=](const Vector &x)
        {
            double xVal = x(0);
            double rho;
            if (x(0)<20)
            {
                rho = 5*(pow(cosh((20.0+(xVal-20.0))/2),-2)+1e-6);
            }
            else
            {
                rho = 5*(pow(cosh((20.0-(xVal-20.0))/2),-2)+1e-6);
            }
            double T = 2;
            double u_bulk = 0.0;
    
            double coeff = rho / sqrt(2.0 * M_PI * T);
            double exponent = -pow(v_i - u_bulk, 2) / (2.0 * T);
    
            return coeff * exp(exponent);
        });
    
        u.emplace_back(&fes);
        u[i].ProjectCoefficient(u0);

        auto vel_coeff = MakeVelocityCoefficient(dim, vNodes[i]);
        FunctionCoefficient inflow([=](const Vector &x)
        {
            double T_rec = 2;
            double rho_rec = 4.98;
            double u_rec = sqrt(20.0);

            double sign = (v_i > 0) ? 1.0 : -1.0;
            double u_shift = (x(0) < 1e-6) ? u_rec : -u_rec;  // left = +u_rec, right = -u_rec

            double coeff = rho_rec / sqrt(2.0 * M_PI * T_rec);
            double exponent = -pow(v_i - u_shift, 2) / (2.0 * T_rec);
            return coeff * exp(exponent);
        });

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

    ComputeMoments(u, vNodes, fes, rho, u_bulk, T);

    std::ostringstream rname;
    rname << "gf_out/rho-" << 0 << ".gf";
    std::ofstream rout(rname.str());
    rout.precision(precision);
    rho.Save(rout);

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

        ComputeMoments(u, vNodes, fes, rho, u_bulk, T);

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

                std::ostringstream rname;
                rname << "gf_out/rho-" << (ti+1) << ".gf";
                std::ofstream rout(rname.str());
                rout.precision(precision);
                rho.Save(rout);
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
