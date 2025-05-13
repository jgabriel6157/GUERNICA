#ifndef IONIZATION_OPERATOR_HXX
#define IONIZATION_OPERATOR_HXX

#include "mfem.hpp"

class IonizationOperator : public mfem::Operator
{
private:
    double rate;

public:
    IonizationOperator(int size, double rate_);

    void SetRate(double r);

    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
};

#endif
