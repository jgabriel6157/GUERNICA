#ifndef FE_EVOLUTION_HXX
#define FE_EVOLUTION_HXX

#include "mfem.hpp"
#include "DG_Solver.hxx"

class FE_Evolution : public mfem::TimeDependentOperator
{
private:
    mfem::BilinearForm &M, &K;
    const mfem::Vector &b;
    mfem::Solver *M_prec;
    mfem::CGSolver M_solver;
    DG_Solver *dg_solver;
    mutable mfem::Vector z;

public:
    FE_Evolution(mfem::BilinearForm &M_, mfem::BilinearForm &K_, const mfem::Vector &b_);
    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
    virtual void ImplicitSolve(const double dt, const mfem::Vector &x, mfem::Vector &k) override;
    virtual ~FE_Evolution();
};

#endif
