#pragma once

#include "mfem.hpp"
using namespace mfem;

class ChargeExchangeOperator : public Operator
{
private:
    double S, v_i;
    const GridFunction &fn;    // f_n(x, v_i)
    const GridFunction &ni_x;  // n_i(x)
    const GridFunction &ui_x;  // u_i(x)
    const GridFunction &Ti_x;  // T_i(x)

    Vector rho_n;              // neutral density at x

public:
    ChargeExchangeOperator(const GridFunction &fn_i,
                           const GridFunction &ni,
                           const GridFunction &ui,
                           const GridFunction &Ti,
                           double v_i, double S_);

    void SetNeutralDensity(const Vector &rho_in);

    virtual void Mult(const Vector &x, Vector &y) const override;
};
