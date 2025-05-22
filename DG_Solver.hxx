#pragma once

#include "mfem.hpp"

class DG_Solver : public mfem::Solver
{
private:
    mfem::SparseMatrix &M, &K, A;
    mfem::GMRESSolver linear_solver;
    mfem::BlockILU prec;
    double dt;

public:
    DG_Solver(mfem::SparseMatrix &M_, mfem::SparseMatrix &K_, const mfem::FiniteElementSpace &fes);
    void SetTimeStep(double dt_);
    virtual void SetOperator(const mfem::Operator &op) override;
    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
};