#ifndef OPERATOR_TO_TIME_DEPENDENT_HXX
#define OPERATOR_TO_TIME_DEPENDENT_HXX

#include "mfem.hpp"

class OperatorToTimeDependent : public mfem::TimeDependentOperator
{
private:
    mfem::Operator &op;

public:
    OperatorToTimeDependent(mfem::Operator &wrapped);

    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

    virtual void ImplicitSolve(double dt, const mfem::Vector &x, mfem::Vector &k) override;
};

#endif
