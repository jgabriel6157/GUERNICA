#include "DG_Advection.hxx"
#include "mfem.hpp"

using namespace mfem;

// ------------------------ ctor / precompute ------------------------

DG_Advection::DG_Advection(FiniteElementSpace &fes,
                           const std::vector<double> &vNodes,
                           double t_final)
: TimeDependentOperator(0),  // set after we compute element-major size
  fes_(fes),
  mesh_(*fes.GetMesh()),
  dim_(mesh_.Dimension()),
  NE_(mesh_.GetNE()),
  Nv_((int)vNodes.size()),
  ndof_total_(0),
  vmax_(0.0),
  t_final_(t_final),
  vNodes_(vNodes)
{
    for (double v : vNodes_) 
    {
        vmax_ = std::max(vmax_, std::abs(v)); 
    }

    // Build element ldof and element-major base offsets
    ldof_e_.resize(NE_);
    elem_base_.resize(NE_ + 1);
    elem_base_[0] = 0;
    for (int e = 0; e < NE_; ++e)
    {
        ldof_e_[e] = fes_.GetFE(e)->GetDof();
        elem_base_[e+1] = elem_base_[e] + Nv_ * ldof_e_[e];
    }
    ndof_total_ = elem_base_[NE_];

    // Set operator dimensions
    height = ndof_total_;
    width  = ndof_total_;

    // Buffers sized off first element (only used as temp if needed)
    Ue_.SetSize(fes_.GetFE(0)->GetDof());
    dUe_.SetSize(fes_.GetFE(0)->GetDof());

    PrecomputeElementMatrices();
    PrecomputeFaceBlocks();
}

void DG_Advection::PrecomputeElementMatrices()
{
    const int NE = NE_;

    M_e_.resize(NE);
    Minv_e_.resize(NE);
    K_e_unit_.resize(NE);

    // Unit velocity coefficient in +x direction
    VectorFunctionCoefficient vcoef(dim_, [this](const Vector&, Vector &vv)
    {
        vv.SetSize(dim_); vv = 0.0; vv(0) = 1.0;
    });

    MassIntegrator       mi;
    ConvectionIntegrator ki(vcoef, -1.0); // minus from integration by parts baked in

    for (int e = 0; e < NE; ++e)
    {
        const FiniteElement &fe = *fes_.GetFE(e);
        ElementTransformation &Tr = *fes_.GetElementTransformation(e);

        // Element mass
        mi.AssembleElementMatrix(fe, Tr, M_e_[e]);

        // Consistent inverse (explicit)
        DenseMatrixInverse inv(M_e_[e]);  // LU factorization internally
        Minv_e_[e].SetSize(M_e_[e].Height());
        inv.GetInverseMatrix(Minv_e_[e]); // store M_e^{-1}

        // Unit-velocity volume matrix
        ki.AssembleElementMatrix(fe, Tr, K_e_unit_[e]);
    }
}

void DG_Advection::PrecomputeFaceBlocks()
{
    IFace_pos_.clear(); IFace_neg_.clear();
    BFace_pos_.clear(); BFace_neg_.clear();

    // Unit velocities for trace operator
    VectorFunctionCoefficient v_pos(dim_, [this](const Vector&, Vector &vv)
    { vv.SetSize(dim_); vv = 0.0; vv(0) =  1.0; });
    VectorFunctionCoefficient v_neg(dim_, [this](const Vector&, Vector &vv)
    { vv.SetSize(dim_); vv = 0.0; vv(0) = -1.0; });
    NonconservativeDGTraceIntegrator tr_pos(v_pos, -1.0);
    NonconservativeDGTraceIntegrator tr_neg(v_neg, -1.0);

    const int Nfaces = mesh_.GetNumFaces();
    for (int f = 0; f < Nfaces; ++f)
    {
        FaceElementTransformations *FT = mesh_.GetFaceElementTransformations(f);
        if (!FT) { continue; }

        const bool is_bdr = (FT->Elem2No < 0);

        if (!is_bdr)
        {
            const int eL = FT->Elem1No;
            const int eR = FT->Elem2No;

            const FiniteElement &fel = *fes_.GetFE(eL);
            const FiniteElement &fer = *fes_.GetFE(eR);

            const int ldL = fel.GetDof();
            const int ldR = fer.GetDof();

            DenseMatrix Apos(ldL + ldR), Aneg(ldL + ldR);
            tr_pos.AssembleFaceMatrix(fel, fer, *FT, Apos);
            tr_neg.AssembleFaceMatrix(fel, fer, *FT, Aneg);

            auto split = [&](const DenseMatrix &A, std::vector<FaceBlock> &out)
            {
                FaceBlock fb;
                fb.eL = eL; fb.eR = eR;
                fb.A_LL.SetSize(ldL, ldL);
                fb.A_LR.SetSize(ldL, ldR);
                fb.A_RL.SetSize(ldR, ldL);
                fb.A_RR.SetSize(ldR, ldR);
                for (int i = 0; i < ldL; ++i)
                {
                    for (int j = 0; j < ldL; ++j)
                    {
                        fb.A_LL(i,j) = A(i,j);
                    }
                }
                for (int i = 0; i < ldL; ++i)
                {
                    for (int j = 0; j < ldR; ++j)
                    {
                        fb.A_LR(i,j) = A(i, ldL + j);
                    }
                }
                for (int i = 0; i < ldR; ++i)
                {
                    for (int j = 0; j < ldL; ++j)
                    {
                        fb.A_RL(i,j) = A(ldL + i, j);
                    }
                }
                for (int i = 0; i < ldR; ++i)
                {
                    for (int j = 0; j < ldR; ++j)
                    {
                        fb.A_RR(i,j) = A(ldL + i, ldL + j);
                    }
                }
                out.push_back(std::move(fb));
            };
            split(Apos, IFace_pos_);
            split(Aneg, IFace_neg_);
        }
        else
        {
            const int e  = FT->Elem1No;
            const FiniteElement &fe = *fes_.GetFE(e);
            const int lde = fe.GetDof();

            DenseMatrix Ab_pos(lde), Ab_neg(lde); // square
            tr_pos.AssembleFaceMatrix(fe, fe, *FT, Ab_pos);
            tr_neg.AssembleFaceMatrix(fe, fe, *FT, Ab_neg);

            auto pack = [&](const DenseMatrix &Ab, std::vector<BdrFaceBlock> &out)
            {
                BdrFaceBlock bf;
                bf.e = e;
                bf.face = FT->FaceGeom;
                bf.A_bdr.SetSize(lde, lde);
                bf.A_bdr = Ab;
                out.push_back(std::move(bf));
            };
            pack(Ab_pos, BFace_pos_);
            pack(Ab_neg, BFace_neg_);
        }
    }
}

// ----------------------------- Mult --------------------------------

void DG_Advection::Mult(const Vector &U, Vector &dUdt) const
{
    MFEM_VERIFY(U.Size() == ndof_total_ && dUdt.Size() == ndof_total_,
                "DG_Advection: element-major vector size mismatch");
    dUdt = 0.0;

    // -------- Volume (element-outer; scale unit operator by v) --------
    for (int e = 0; e < NE_; ++e)
    {
        const int base = elem_base_[e];
        const int ld   = ldof_e_[e];

        for (int iv = 0; iv < Nv_; ++iv)
        {
            const double *u_ptr = U.GetData()     + base + iv*ld;
            double       *r_ptr = dUdt.GetData()  + base + iv*ld;

            Vector Ue(const_cast<double*>(u_ptr), ld);
            Vector Re(r_ptr, ld);

            Vector tmp(ld); tmp = 0.0;
            K_e_unit_[e].Mult(Ue, tmp);        // tmp = K_unit * Ue
            Re.Add(vNodes_[iv], tmp);          // Re += v * tmp
        }
    }

    // -------- Faces (interior & boundary), choose by sign(v) and scale by |v| --------
    for (int signcase = 0; signcase < 2; ++signcase)
    {
        const bool use_pos = (signcase == 0);
        const auto &IF = use_pos ? IFace_pos_ : IFace_neg_;
        const auto &BF = use_pos ? BFace_pos_ : BFace_neg_;

    // Interior
    for (const auto &fb : IF)
    {
        const int eL = fb.eL, eR = fb.eR;
        const int baseL = elem_base_[eL], baseR = elem_base_[eR];
        const int ldL = ldof_e_[eL], ldR = ldof_e_[eR];

        for (int iv = 0; iv < Nv_; ++iv)
        {
            const double v = vNodes_[iv];
            if ((use_pos && v < 0.0) || (!use_pos && v > 0.0)) continue;
            const double a = std::abs(v);

            Vector UeL(const_cast<double*>(U.GetData()) + baseL + iv*ldL, ldL);
            Vector UeR(const_cast<double*>(U.GetData()) + baseR + iv*ldR, ldR);
            Vector ReL(               dUdt.GetData()    + baseL + iv*ldL, ldL);
            Vector ReR(               dUdt.GetData()    + baseR + iv*ldR, ldR);

            Vector RL(ldL); RL = 0.0;
            Vector RR(ldR); RR = 0.0;

            fb.A_LL.Mult(UeL, RL);
            fb.A_LR.AddMult(UeR, RL);
            fb.A_RL.Mult(UeL, RR);
            fb.A_RR.AddMult(UeR, RR);

            ReL.Add(a, RL);
            ReR.Add(a, RR);
        }
    }

    // Boundary
    for (const auto &bf : BF)
    {
        const int e    = bf.e;
        const int base = elem_base_[e];
        const int ld   = ldof_e_[e];

        for (int iv = 0; iv < Nv_; ++iv)
        {
            const double v = vNodes_[iv];
            if ((use_pos && v < 0.0) || (!use_pos && v > 0.0)) continue;
            const double a = std::abs(v);

            Vector Ue(const_cast<double*>(U.GetData()) + base + iv*ld, ld);
            Vector Re(               dUdt.GetData()    + base + iv*ld, ld);

            Vector Rb(ld); Rb = 0.0;
            bf.A_bdr.Mult(Ue, Rb);
            Re.Add(a, Rb);
        }
    }
    }

    // -------- Apply consistent M^{-1} per (e,iv) in-place --------
    for (int e = 0; e < NE_; ++e)
    {
        const int base = elem_base_[e];
        const int ld   = ldof_e_[e];

        for (int iv = 0; iv < Nv_; ++iv)
        {
            Vector Re(dUdt.GetData() + base + iv*ld, ld);
            Vector tmp(ld); tmp = 0.0;
            Minv_e_[e].Mult(Re, tmp);
            Re = tmp; // overwrite
        }
    }
}
