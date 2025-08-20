#ifndef DG_ADVECTION_HXX
#define DG_ADVECTION_HXX

#include "mfem.hpp"
#include <vector>
#include <memory>

class DG_Advection : public mfem::TimeDependentOperator
{
public:
  DG_Advection(mfem::FiniteElementSpace &fes,
               const std::vector<double> &vNodes,
               double t_final);

  // RHS: dUdt = A(U)
  void Mult(const mfem::Vector &U, mfem::Vector &dUdt) const override;

  // Optional helpers
  double MaxCharSpeed() const { return vmax_; }

private:
  // ---- Types for precomputed face blocks ----
  struct FaceBlock
  {
    // Interior face coupling: [eL <- {eL,eR}], [eR <- {eL,eR}]
    mfem::DenseMatrix A_LL; // (ldof_L x ldof_L)
    mfem::DenseMatrix A_LR; // (ldof_L x ldof_R)
    mfem::DenseMatrix A_RL; // (ldof_R x ldof_L)
    mfem::DenseMatrix A_RR; // (ldof_R x ldof_R)
    int eL = -1;
    int eR = -1;
  };

  struct BdrFaceBlock
  {
    mfem::DenseMatrix A_bdr; // (ldof_e x ldof_e)
    int e   = -1;            // adjacent element
    int face = -1;           // optional metadata
  };

  // ---- Precompute helpers ----
  void PrecomputeElementMatrices(const std::vector<double> &vNodes);
  void PrecomputeFaceBlocks(const std::vector<double> &vNodes);

private:
  // ---- Core references / sizes ----
  mfem::FiniteElementSpace &fes_;
  int Nv_;                // number of velocities
  int ndof_;              // global dofs per velocity block (fes_.GetVSize())
  int ndof_face_;         // placeholder; not strictly required in current impl
  double vmax_;           // max |v| for CFL checks
  double t_final_;

  // ---- Velocity grid (stored so faces can use it) ----
  std::vector<double> vNodes_;

  // ---- Restrictions ----
  std::unique_ptr<mfem::L2ElementRestriction> elemR_;
  // Note: L2FaceRestriction is not required by the current implementation,
  // but you can add it later if you switch to face E-vectors.
  // std::unique_ptr<mfem::L2FaceRestriction> faceR_;

  // ---- Precomputed element data ----
  std::vector<mfem::DenseMatrix> M_e_;                     // per-element mass
  std::vector<std::vector<mfem::DenseMatrix>> K_e_;        // [iv][e] volume convection
  std::vector<mfem::Vector> MinvLumped_;                   // per-element lumped M^{-1}

  // ---- Precomputed face data (per velocity) ----
  std::vector<std::vector<FaceBlock>> IFace_;              // [iv][interior face]
  std::vector<std::vector<BdrFaceBlock>> BFace_;           // [iv][boundary face]

  // ---- Scratch buffers (reused in Mult) ----
  mutable mfem::Vector Ue_;   // element-local dofs (size = ldof)
  mutable mfem::Vector dUe_;  // element-local residual (size = ldof)
};

#endif // DG_ADVECTION_HXX
