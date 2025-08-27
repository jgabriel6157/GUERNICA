#ifndef CHARGE_EXCHANGE_OPERATOR_HXX
#define CHARGE_EXCHANGE_OPERATOR_HXX

#include "mfem.hpp"
#include <vector>
#include <cmath>

/// Element-major charge-exchange operator:
/// dU/dt += -S * ( n_i(x) * f(x,v) - rho_n(x) * F_i(x; u_i, T_i) )
/// F_i(x;u,T) = n_i(x)/sqrt(2*pi*T) * exp(-(v - u)^2 / (2*T))
class ChargeExchangeOperator : public mfem::TimeDependentOperator
{
public:
  ChargeExchangeOperator(mfem::FiniteElementSpace &fes,
                         const std::vector<double> &vNodes,
                         const mfem::GridFunction &ni,   // ion density
                         const mfem::GridFunction &ui,   // ion bulk velocity
                         const mfem::GridFunction &Ti,   // ion temperature
                         double S);                      // CX rate

  // Update rho_n(x) each step (neutrals’ density)
  void SetNeutralDensity(const mfem::GridFunction &rho);
  void SetNeutralDensity(const mfem::Vector &rho);

  void Mult(const mfem::Vector &U, mfem::Vector &dUdt) const override;

  // Optional helpers
  int    GetNE()        const { return NE_; }
  int    GetNv()        const { return Nv_; }
  int    GlobalSize()   const { return ndof_total_; }
  int    Ldof(int e)    const { return ldof_e_[e]; }
  const std::vector<int>& ElemBase() const { return elem_base_; }

private:
  // layout / sizes
  mfem::FiniteElementSpace &fes_;
  mfem::Mesh &mesh_;
  int NE_, Nv_, ndof_total_;
  std::vector<int> ldof_e_, elem_base_;
  std::vector<double> vNodes_;

  // ion background fields (live on fes dofs)
  const mfem::GridFunction &ni_;
  const mfem::GridFunction &ui_;
  const mfem::GridFunction &Ti_;

  // neutral density at dofs (updated by SetNeutralDensity)
  mfem::Vector rho_dof_;

  // rate
  double S_;
};

#endif // CHARGE_EXCHANGE_OPERATOR_HXX
