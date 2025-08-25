#ifndef SUM_TDEP_HXX
#define SUM_TDEP_HXX
#include "mfem.hpp"
#include <memory>
#include <vector>
#include <cassert>
#include <iostream>

/// N-ary sum of TimeDependentOperator's with SetTime fan-out.
/// Owns children via shared_ptr to guarantee lifetime.
class SumTDep : public mfem::TimeDependentOperator 
{
public:
  SumTDep() : mfem::TimeDependentOperator(0) {}

  void Add(std::shared_ptr<mfem::TimeDependentOperator> op) 
  {
    if (!op) { MFEM_ABORT("SumTDep::Add: null operator"); }
    if (ops_.empty()) 
    {
        height = op->Height(); width = op->Width();
    }
    else 
    {
      MFEM_VERIFY(op->Height() == height && op->Width() == width,
                  "All summed operators must have matching sizes");
    }
    ops_.push_back(std::move(op));
  }

  void Mult(const mfem::Vector &x, mfem::Vector &y) const override 
  {
    y = 0.0;
    for (const auto &op : ops_) 
    {
      mfem::Vector tmp(y.Size()); tmp = 0.0;
      op->Mult(x, tmp);
      y += tmp;
    }
  }

  void SetTime(double t) override 
  {
    mfem::TimeDependentOperator::SetTime(t);
    for (const auto &op : ops_) { op->SetTime(t); }
  }

private:
  std::vector<std::shared_ptr<mfem::TimeDependentOperator>> ops_;
};
#endif // SUM_TDEP_HXX
