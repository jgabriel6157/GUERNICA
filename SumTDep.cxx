#include "SumTDep.hxx"
#include "mfem.hpp"
#include <utility>   // for std::move

using namespace mfem;

SumTDep::SumTDep()
  : TimeDependentOperator(0)
{}

void SumTDep::Add(std::shared_ptr<TimeDependentOperator> op)
{
  if (!op) { MFEM_ABORT("SumTDep::Add: null operator"); }
  if (ops_.empty())
  {
    height = op->Height();
    width  = op->Width();
  }
  else
  {
    MFEM_VERIFY(op->Height() == height && op->Width() == width,
                "All summed operators must have matching sizes");
  }
  ops_.push_back(std::move(op));
}

void SumTDep::Mult(const Vector &x, Vector &y) const
{
  y = 0.0;
  for (const auto &op : ops_)
  {
    Vector tmp(y.Size());
    tmp = 0.0;
    op->Mult(x, tmp);
    y += tmp;
  }
}

void SumTDep::SetTime(double t)
{
  TimeDependentOperator::SetTime(t);
  for (const auto &op : ops_) { op->SetTime(t); }
}
