#include "ChargeExchangeOperator.hxx"
#include <cmath>

ChargeExchangeOperator::ChargeExchangeOperator(const GridFunction &fn_i,
                                               const GridFunction &ni,
                                               const GridFunction &ui,
                                               const GridFunction &Ti,
                                               double v_i_,
                                               double S_)
    : Operator(fn_i.Size()), fn(fn_i), ni_x(ni), ui_x(ui), Ti_x(Ti), v_i(v_i_), S(S_)
{
    rho_n.SetSize(fn.Size());
    rho_n = 0.0;
}

void ChargeExchangeOperator::SetNeutralDensity(const Vector &rho_in)
{
    rho_n = rho_in;
}

void ChargeExchangeOperator::Mult(const Vector &x, Vector &y) const
{
    const double *f = x.GetData();
    const double *ni = ni_x.GetData();
    const double *ui = ui_x.GetData();
    const double *Ti = Ti_x.GetData();
    const double *rho = rho_n.GetData();

    double *out = y.GetData();

    for (int i = 0; i < Height(); i++)
    {
        double coeff = ni[i] / sqrt(2.0 * M_PI * Ti[i]);
        double exponent = -pow(v_i - ui[i], 2) / (2.0 * Ti[i]);
        double fi_v = coeff * exp(exponent);

        out[i] = -S * (ni[i] * f[i] - rho[i] * fi_v);
    }
}
