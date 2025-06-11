#pragma once

#include "mfem.hpp"
#include <vector>

class BGKEquilibrium
{
private:
    const mfem::FiniteElementSpace &fes;
    const std::vector<double> &vNodes;
    std::vector<mfem::Vector> fM;
    std::vector<double> alpha0, alpha1, alpha2;

public:
    BGKEquilibrium(const mfem::FiniteElementSpace &fes_,
                   const std::vector<double> &vNodes_);

    void Update(const std::vector<mfem::GridFunction> &u, bool first_time = false);

    const mfem::Vector &GetFM(int vIndex) const;

private:
    void SolveForAlpha(mfem::Vector &alpha,
                       const std::vector<double> &v,
                       double rho_target,
                       double mom_target,
                       double E_target,
                       bool test,
                       int max_iter = 20,
                       double tol = 1e-12);
};


class BGKOperator : public mfem::Operator
{
private:
    mfem::GridFunction &f;
    double nu_val;
    const mfem::FiniteElementSpace &fes;
    const mfem::Vector &fM;

public:
    BGKOperator(mfem::GridFunction &f_,
                double nu_val_,
                const mfem::FiniteElementSpace &fes_,
                const mfem::Vector &fM_);

    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
};