#ifndef SUM_TDEP_HXX
#define SUM_TDEP_HXX

#include "mfem.hpp"
#include <memory>
#include <vector>

/// N-ary sum of TimeDependentOperator's with SetTime fan-out.
/// Owns children via shared_ptr to guarantee lifetime.
class SumTDep : public mfem::TimeDependentOperator
{
public:
  SumTDep();

  // Add a child operator. All children must have identical (height, width).
  void Add(std::shared_ptr<mfem::TimeDependentOperator> op);

  // y = sum_i (op_i x)
  void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

  // Forwards time to all children
  void SetTime(double t) override;

  // Optional: number of summed terms
  std::size_t NumOps() const { return ops_.size(); }

private:
  std::vector<std::shared_ptr<mfem::TimeDependentOperator>> ops_;
};

#endif // SUM_TDEP_HXX
