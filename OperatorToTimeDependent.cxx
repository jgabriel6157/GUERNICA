#include "OperatorToTimeDependent.hxx"

using namespace mfem;

OperatorToTimeDependent::OperatorToTimeDependent(Operator &wrapped)
    : TimeDependentOperator(wrapped.Height()), op(wrapped) {}

void OperatorToTimeDependent::Mult(const Vector &x, Vector &y) const
{
    op.Mult(x, y);
}

void OperatorToTimeDependent::ImplicitSolve(double, const Vector &, Vector &)
{
    MFEM_ABORT("ImplicitSolve not implemented for this wrapper.");
}
