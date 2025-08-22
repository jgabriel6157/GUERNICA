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

  // RHS: dUdt = A(U) with element-major layout
  void Mult(const mfem::Vector &U, mfem::Vector &dUdt) const override;

  // Optional helpers
  double MaxCharSpeed() const { return vmax_; }

  // --- Minimal introspection needed by main for element-major vectors ---
  int    GetNE()        const { return NE_; }
  int    GetNv()        const { return Nv_; }
  int    Ldof(int e)    const { return ldof_e_[e]; }
  const std::vector<int>& ElemBase() const { return elem_base_; }
  int    GlobalSize()   const { return ndof_total_; }

private:
  // ---- Types for precomputed face blocks (unit velocity) ----
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

  // ---- Precompute helpers (now unit-velocity only) ----
  void PrecomputeElementMatrices();
  void PrecomputeFaceBlocks();

private:
  // ---- Core references / sizes ----
  mfem::FiniteElementSpace &fes_;
  mfem::Mesh               &mesh_;
  int dim_;
  int NE_;                         // number of elements
  int Nv_;                         // number of velocities
  int ndof_total_;                 // total unknowns in element-major layout
  double vmax_;                    // max |v| for CFL checks
  double t_final_;

  // ---- Element-major indexing ----
  // Global vector is concatenation of element slabs; slab e has size Nv_ * ldof_e_[e]
  std::vector<int> ldof_e_;        // ldof per element
  std::vector<int> elem_base_;     // base offset of element slab in global vector (size NE_+1)

  // ---- Velocity grid ----
  std::vector<double> vNodes_;

  // ---- Precomputed element data (unit velocity) ----
  std::vector<mfem::DenseMatrix> M_e_;        // per-element mass
  std::vector<mfem::DenseMatrix> Minv_e_;     // per-element mass inverse (consistent)
  std::vector<mfem::DenseMatrix> K_e_unit_;   // per-element volume convection for v=+1 in x

  // Face blocks for v=+1 and v=-1 (needed for correct upwinding with sign)
  std::vector<FaceBlock>    IFace_pos_, IFace_neg_;
  std::vector<BdrFaceBlock> BFace_pos_, BFace_neg_;

  // ---- Scratch buffers (reused in Mult) ----
  mutable mfem::Vector Ue_;   // element-local dofs (size = ldof)
  mutable mfem::Vector dUe_;  // element-local residual (size = ldof)
};

#endif // DG_ADVECTION_HXX
