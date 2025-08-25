#include "mfem.hpp"
#include "InputConfig.hxx"
#include "DG_Advection.hxx"
#include "IonizationOperator.hxx"
#include "SumTDep.hxx"

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

void UnpackElementMajor(const mfem::Vector &U,
                        mfem::FiniteElementSpace &fes,
                        const std::vector<int> &elem_base,
                        const std::vector<double> &vNodes,
                        std::vector<mfem::GridFunction> &u_out)
{
    const int Nv = (int)vNodes.size();
    const int NE = fes.GetMesh()->GetNE();

    u_out.clear();
    u_out.reserve(Nv);
    for (int iv = 0; iv < Nv; ++iv) { u_out.emplace_back(&fes); u_out.back() = 0.0; }

    mfem::Array<int> vdofs;
    for (int e = 0; e < NE; ++e) {
        const int ld = fes.GetFE(e)->GetDof();
        const int base = elem_base[e];
        fes.GetElementVDofs(e, vdofs);
        for (int iv = 0; iv < Nv; ++iv) {
            mfem::Vector Ue(const_cast<double*>(U.Read()) + base + iv*ld, ld);
            u_out[iv].SetSubVector(vdofs, Ue);
        }
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
    bool ionization = config.Get<bool>("ionization", false);
    double nu0 = config.Get<double>("nu0",0.0);
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

    mfem::GridFunction rho(&fes), u_bulk(&fes), T(&fes);

    // Build the element-major operator (advection, 1D)
    auto adv = std::make_shared<DG_Advection>(fes, vNodes, t_final);

    adv->BuildInflow([&](int iv, const mfem::Vector &x) -> double 
    {
        const double v_i     = vNodes[iv];
        const double T_rec   = 2.0;
        const double rho_rec = 4.98;
        const double u_rec   = std::sqrt(20.0);

        // same side logic as your old code: left ~ x≈0 uses +u_rec, right uses -u_rec
        const double u_shift = (x(0) < 1e-6) ? +u_rec : -u_rec;

        const double coeff   = rho_rec / std::sqrt(2.0 * M_PI * T_rec);
        const double dv      = v_i - u_shift;
        const double exponent= - (dv*dv) / (2.0 * T_rec);

        return coeff * std::exp(exponent);
    });

    // Introspection for element-major packing
    const int NE = adv->GetNE();
    const int Ntot = adv->GlobalSize();
    const std::vector<int> &elem_base = adv->ElemBase();

    // Allocate element-major global vectors
    Vector U(Ntot);
    U = 0.0;
    Vector dUdt(Ntot);
    dUdt = 0.0;

    // Initialize U with projected scalar initial condition, replicated across velocities
    Array<int> vdofs;
    Vector Ue; // element-local buffer

    for (int iv = 0; iv < Nv; ++iv)
    {
        const double v_i = vNodes[iv];

        // f0(x, v_i) per your formula
        FunctionCoefficient f0([=](const Vector &x)
        {
            const double xVal = x(0);

            double rho;
            if (xVal < 20.0)
            {
                // rho = 5*(cosh((20 + (x-20))/2)^(-2) + 1e-6)
                rho = 5.0 * ( std::pow(std::cosh((20.0 + (xVal - 20.0))/2.0), -2.0) + 1e-6 );
            }
            else
            {
                // rho = 5*(cosh((20 - (x-20))/2)^(-2) + 1e-6)
                rho = 5.0 * ( std::pow(std::cosh((20.0 - (xVal - 20.0))/2.0), -2.0) + 1e-6 );
            }

            const double T      = 2.0;
            const double u_bulk = 0.0;

            const double coeff    = rho / std::sqrt(2.0 * M_PI * T);
            const double dv       = v_i - u_bulk;
            const double exponent = -(dv*dv) / (2.0 * T);
            return coeff * std::exp(exponent);
        });

        // Project f0 onto the FE space for this velocity, then pack into U (element-major)
        GridFunction u0_gf(&fes);
        u0_gf.ProjectCoefficient(f0);

        for (int e = 0; e < NE; ++e)
        {
            const int base = elem_base[e];
            const int ld   = adv->Ldof(e);          // use your operator’s ldof accessor

            fes.GetElementVDofs(e, vdofs);
            Ue.SetSize(ld);
            u0_gf.GetSubVector(vdofs, Ue);

            double *dst = U.Write() + base + iv*ld;
            std::memcpy(dst, Ue.Read(), ld * sizeof(double));
        }
    }

    // Save mesh and initial solutions
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
                const int ld   = adv->Ldof(e);
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

        std::ostringstream rname;
        rname << "gf_out/rho-" << (step) << ".gf";
        std::ofstream rout(rname.str());
        rout.precision(precision);
        rho.Save(rout);

        cout << "Step " << step << ", time = " << tcur << endl;
    };

    std::vector<mfem::GridFunction> u_vs;
    UnpackElementMajor(U, fes, adv->ElemBase(), vNodes, u_vs);
    ComputeMoments(u_vs, vNodes, fes, rho, u_bulk, T);
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
    SumTDep rhs;
    rhs.Add(adv);
    if (ionization)
    {
        mfem::ConstantCoefficient nu_coeff(nu0);

        // Build ionization operator (diagonal sink: dU/dt += -nu U)
        auto ion = std::make_shared<IonizationOperator>(fes, vNodes, nu_coeff, t_final);

        // Composite RHS = advection + ionization
        rhs.Add(ion);
    }
    solver->Init(rhs);

    // Time loop
    double t = 0.0;
    int ti = 0;
    while (t < t_final - 1e-8*dt)
    {
        double dt_real = std::min(dt, t_final - t);
        solver->Step(U, t, dt_real);
        ti++;

        UnpackElementMajor(U, fes, adv->ElemBase(), vNodes, u_vs);
        ComputeMoments(u_vs, vNodes, fes, rho, u_bulk, T);

        if ((ti % vis_steps) == 0 || t + 1e-8*dt >= t_final)
        {
            dump_fields(ti, t);
        }
    }

    return 0;
}
