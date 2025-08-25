#include "IonizationOperator.hxx"
using namespace mfem;

IonizationOperator::IonizationOperator(FiniteElementSpace &fes,
                                       const std::vector<double> &vNodes,
                                       Coefficient &nu_coeff,
                                       double t_final)
  : TimeDependentOperator(0),
    fes_(fes), mesh_(*fes.GetMesh()),
    NE_(mesh_.GetNE()), Nv_((int)vNodes.size()),
    ldof_max_(0), ndof_total_(0),
    nu_coeff_(nu_coeff), nu_gf_(&fes_)
{
  ldof_e_.resize(NE_);
  elem_base_.resize(NE_);

  int base = 0;
  for (int e = 0; e < NE_; ++e) {
    const int ld = fes_.GetFE(e)->GetDof();
    ldof_e_[e] = ld;
    ldof_max_ = std::max(ldof_max_, ld);
    elem_base_[e] = base;
    base += ld * Nv_;
  }
  ndof_total_ = base;
  height = width = ndof_total_;

  nue_.SetSize(ldof_max_);
  tmp_.SetSize(ldof_max_);

  ProjectNu();
}

void IonizationOperator::ProjectNu()
{
  nu_gf_.ProjectCoefficient(nu_coeff_);
}

void IonizationOperator::Mult(const Vector &U, Vector &dUdt) const
{
  // assumes caller zeroed dUdt (common pattern when summing operators)
  Array<int> vdofs;
  for (int e = 0; e < NE_; ++e) 
  {
    const int ld = ldof_e_[e];
    const int base = elem_base_[e];

    fes_.GetElementVDofs(e, vdofs);
    for (int i = 0; i < ld; ++i) { nue_[i] = nu_gf_(vdofs[i]); }

    for (int iv = 0; iv < Nv_; ++iv) 
    {
      const double *Ue = U.GetData()     + base + iv*ld;
      double       *dE = dUdt.GetData()  + base + iv*ld;
      for (int i = 0; i < ld; ++i) { dE[i] += -nue_[i] * Ue[i]; }
    }
  }
}
