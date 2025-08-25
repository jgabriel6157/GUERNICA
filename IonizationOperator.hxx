#ifndef IONIZATIONOperator_HXX
#define IONIZATIONOperator_HXX

#include "mfem.hpp"
#include <vector>
#include <iostream>
#include <functional>

/// dU/dt = -nu(x) * U  (no face terms; diagonal in velocity & dofs)
class IonizationOperator : public mfem::TimeDependentOperator
{
public:
  IonizationOperator(mfem::FiniteElementSpace &fes,
                     const std::vector<double> &vNodes,
                     mfem::Coefficient &nu_coeff,
                     double t_final = 0.0);

  void Mult(const mfem::Vector &U, mfem::Vector &dUdt) const override;
  void SetTime(double t) override { mfem::TimeDependentOperator::SetTime(t); }
  void ProjectNu(); // call if nu_coeff changes (e.g., new ne/Te)

private:
  mfem::FiniteElementSpace &fes_;
  mfem::Mesh &mesh_;
  int NE_, Nv_, ldof_max_, ndof_total_;
  std::vector<int> ldof_e_, elem_base_;

  mfem::Coefficient &nu_coeff_;
  mutable mfem::GridFunction nu_gf_;        // nu on fes dofs (collocated)
  mutable mfem::Vector nue_, tmp_;          // element scratch
};

#endif
