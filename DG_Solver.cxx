#include "DG_Solver.hxx"

using namespace mfem;

DG_Solver::DG_Solver(SparseMatrix &M_, SparseMatrix &K_, const FiniteElementSpace &fes)
    : M(M_), K(K_), prec(fes.GetFE(0)->GetDof(), BlockILU::Reordering::MINIMUM_DISCARDED_FILL), dt(-1.0)
{
    linear_solver.iterative_mode = false;
    linear_solver.SetRelTol(1e-9);
    linear_solver.SetAbsTol(0.0);
    linear_solver.SetMaxIter(100);
    linear_solver.SetPrintLevel(0);
    linear_solver.SetPreconditioner(prec);
}

void DG_Solver::SetTimeStep(double dt_)
{
    if (dt_ != dt)
    {
        dt = dt_;
        A = K;
        A *= -dt;
        A += M;
        linear_solver.SetOperator(A);
    }
}

void DG_Solver::SetOperator(const Operator &op)
{
    linear_solver.SetOperator(op);
}

void DG_Solver::Mult(const Vector &x, Vector &y) const
{
    linear_solver.Mult(x, y);
}
