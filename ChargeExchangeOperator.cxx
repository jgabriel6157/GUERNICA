#include "ChargeExchangeOperator.hxx"
#include "mfem.hpp"

using namespace mfem;

ChargeExchangeOperator::ChargeExchangeOperator(FiniteElementSpace &fes,
                                               const std::vector<double> &vNodes,
                                               const GridFunction &ni,
                                               const GridFunction &ui,
                                               const GridFunction &Ti,
                                               double S)
  : TimeDependentOperator(0),
    fes_(fes),
    mesh_(*fes.GetMesh()),
    NE_(mesh_.GetNE()),
    Nv_((int)vNodes.size()),
    ndof_total_(0),
    vNodes_(vNodes),
    ni_(ni), ui_(ui), Ti_(Ti),
    rho_dof_(fes.GetVSize()),
    S_(S)
{
  // element-major sizes and offsets
  ldof_e_.resize(NE_);
  elem_base_.resize(NE_ + 1);
  elem_base_[0] = 0;
  for (int e = 0; e < NE_; ++e)
  {
    ldof_e_[e] = fes_.GetFE(e)->GetDof();
    elem_base_[e+1] = elem_base_[e] + Nv_ * ldof_e_[e];
  }
  ndof_total_ = elem_base_[NE_];
  height = width = ndof_total_;

  rho_dof_ = 0.0;
}

void ChargeExchangeOperator::SetNeutralDensity(const GridFunction &rho)
{
  MFEM_VERIFY(rho.Size() == rho_dof_.Size(), "rho size mismatch");
  rho_dof_ = rho;
}

void ChargeExchangeOperator::SetNeutralDensity(const Vector &rho)
{
  MFEM_VERIFY(rho.Size() == rho_dof_.Size(), "rho size mismatch");
  rho_dof_ = rho;
}

void ChargeExchangeOperator::Mult(const Vector &U, Vector &dUdt) const
{
  MFEM_VERIFY(U.Size() == ndof_total_ && dUdt.Size() == ndof_total_,
              "ChargeExchange: element-major vector size mismatch");

  const double *ni_ptr  = ni_.Read();
  const double *ui_ptr  = ui_.Read();
  const double *Ti_ptr  = Ti_.Read();
  const double *rho_ptr = rho_dof_.Read();

  Array<int> vdofs;
  for (int e = 0; e < NE_; ++e)
  {
    const int ld   = ldof_e_[e];
    const int base = elem_base_[e];

    fes_.GetElementVDofs(e, vdofs);

    for (int iv = 0; iv < Nv_; ++iv)
    {
      const double v = vNodes_[iv];

      const double *Ue = U.GetData()    + base + iv*ld;
      double       *Re = dUdt.GetData() + base + iv*ld;

      for (int i = 0; i < ld; ++i)
      {
        const int gd = vdofs[i];

        const double ni = ni_ptr[gd];
        const double ui = ui_ptr[gd];
        const double Ti = Ti_ptr[gd];
        const double rn = rho_ptr[gd];

        double Fi = 0.0;
        if (Ti > 0.0)
        {
          const double coeff = ni / std::sqrt(2.0 * M_PI * Ti);
          const double dv    = v - ui;
          Fi = coeff * std::exp(-(dv*dv) / (2.0 * Ti));
        }

        // accumulate (caller should have zeroed dUdt)
        Re[i] += -S_ * ( ni * Ue[i] - rn * Fi );
      }
    }
  }
}
