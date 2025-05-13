#include "IonizationOperator.hxx"

using namespace mfem;

IonizationOperator::IonizationOperator(int size, double rate_)
    : Operator(size), rate(rate_) {}

void IonizationOperator::SetRate(double r)
{
    rate = r;
}

void IonizationOperator::Mult(const Vector &x, Vector &y) const
{
    y = x;
    y *= -rate;
}
