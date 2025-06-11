#include "BGKOperator.hxx"
#include "Integrators.hxx"
#include <cmath>
#include <iostream>

using namespace mfem;

BGKEquilibrium::BGKEquilibrium(const FiniteElementSpace &fes_,
                               const std::vector<double> &vNodes_)
    : fes(fes_), vNodes(vNodes_)
{
    int Nv = vNodes.size();
    int ndof = fes.GetVSize();
    fM.resize(Nv, Vector(ndof));
    alpha0.resize(ndof);
    alpha1.resize(ndof);
    alpha2.resize(ndof);
}

void BGKEquilibrium::Update(const std::vector<GridFunction> &u, bool first_time)
{
    int ndof = fes.GetVSize();
    int Nv = vNodes.size();

    Vector rho = integrate1D(u, vNodes, 0);
    Vector mom = integrate1D(u, vNodes, 1);
    Vector E   = integrate1D(u, vNodes, 2);

    for (int i = 0; i < ndof; i++)
    {
        bool test = false;
        double r = rho[i], m = mom[i], e = E[i];
        double u_bulk = m / r;
        double T = (e - r * u_bulk * u_bulk) / r;

        Vector alpha(3);
        if (first_time)
        {
            alpha[0] = log(r / sqrt(2.0 * M_PI * T))-u_bulk*u_bulk/(2.0*T);
            alpha[1] = -u_bulk / T;
            alpha[2] = 1.0 / (2.0 * T);
        }
        else
        {
            alpha[0] = alpha0[i];
            alpha[1] = alpha1[i];
            alpha[2] = alpha2[i];
        }
        // std::cout<<i<<"\n";
        // if (i==138)
        // {
        //     test = true;
        //     // alpha.Print();
        // }
        SolveForAlpha(alpha, vNodes, r, m, e, test);

        if (!std::isfinite(alpha[0]) || !std::isfinite(alpha[1]) || !std::isfinite(alpha[2]))
        {
            std::cerr << "Non-finite alpha at dof " << i << std::endl;
            std::cerr << "Trying with first time guess" << std::endl;

            alpha[0] = log(r / sqrt(2.0 * M_PI * T))-u_bulk*u_bulk/(2.0*T);
            alpha[1] = -u_bulk / T;
            alpha[2] = 1.0 / (2.0 * T);

            SolveForAlpha(alpha, vNodes, r, m, e, test);
            if (!std::isfinite(alpha[0]) || !std::isfinite(alpha[1]) || !std::isfinite(alpha[2]))
            {
                std::cout << "RIP" << "\n";
            }
        }

        alpha0[i] = alpha[0];
        alpha1[i] = alpha[1];
        alpha2[i] = alpha[2];

        for (int j = 0; j < Nv; j++)
        {
            double v = vNodes[j];
            fM[j][i] = exp(alpha[0] - alpha[1]*v - alpha[2]*v*v);
            // double val = fM[j][i];
            // if (!std::isfinite(val))
            // {
            //     std::cerr << "NaN in fM at v=" << j << ", dof=" << i << std::endl;
            // }
        }
    }
}

const Vector &BGKEquilibrium::GetFM(int vIndex) const
{
    return fM[vIndex];
}

void BGKEquilibrium::SolveForAlpha(Vector &alpha,
                                   const std::vector<double> &v,
                                   double rho_target,
                                   double mom_target,
                                   double E_target,
                                   bool test,
                                   int max_iter,
                                   double tol)
{
    int Nv = v.size();
    double dv = v[1] - v[0];
    Vector fM(Nv), res(3), delta(3);
    DenseMatrix J(3);
    std::vector<double> w(Nv);

    for (int i = 0; i < Nv; i++)
    {
        w[i] = (i == 0 || i == Nv - 1) ? 1.0 : ((i % 2 == 0) ? 2.0 : 4.0);
    }
    if (test)
    {
        // alpha.Print();
        std::cout<<rho_target<<", "<<mom_target<<", "<<E_target<<"\n";
    }
    for (int iter = 0; iter < max_iter; iter++)
    {
        // if (test) std::cout << iter << "\n";
        double a0 = alpha[0], a1 = alpha[1], a2 = alpha[2];
        double m0 = 0.0, m1 = 0.0, m2 = 0.0, m3 = 0.0, m4 = 0.0;

        for (int j = 0; j < Nv; j++)
        {
            double vj = v[j];
            double v2 = vj*vj, v3 = v2*vj, v4 = v2*v2;
            double val = exp(a0 - a1*vj - a2*v2);
            fM[j] = val;
            m0 += w[j]*val;
            m1 += w[j]*vj*val;
            m2 += w[j]*v2*val;
            m3 += w[j]*v3*val;
            m4 += w[j]*v4*val;
        }

        m0 *= dv / 3.0;
        m1 *= dv / 3.0;
        m2 *= dv / 3.0;
        m3 *= dv / 3.0;
        m4 *= dv / 3.0;

        res[0] = m0 - rho_target;
        res[1] = m1 - mom_target;
        res[2] = m2 - E_target;

        if (test) res.Print();

        if (res.Norml2() < tol) return;

        J(0,0) = m0;  J(0,1) = -m1;  J(0,2) = -m2;
        J(1,0) = m1;  J(1,1) = -m2;  J(1,2) = -m3;
        J(2,0) = m2;  J(2,1) = -m3;  J(2,2) = -m4;
        
        J.Invert();
        J.Mult(res, delta);
        delta *= -1.0;
        alpha += delta;
        if (test)
        {
            // res.Print();
            // alpha.Print();
            // J.Invert();
        }
    }

    std::cerr << "Warning: Newton failed to converge\n";
}

BGKOperator::BGKOperator(GridFunction &f_,
                         double nu_val_,
                         const FiniteElementSpace &fes_,
                         const Vector &fM_)
    : Operator(f_.Size()), f(f_), nu_val(nu_val_), fes(fes_), fM(fM_)
{
}

void BGKOperator::Mult(const Vector &x, Vector &y) const
{
    int ndof = fes.GetVSize();
    y.SetSize(ndof);
    for (int i = 0; i < ndof; i++)
    {
        y[i] = nu_val * (fM[i] - x[i]);
    }
    // y.Print();
}
