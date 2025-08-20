#include "DG_Advection.hxx"
#include "mfem.hpp"

using namespace mfem;

// ------------------------ ctor / precompute ------------------------

DG_Advection::DG_Advection(FiniteElementSpace &fes,
                           const std::vector<double> &vNodes,
                           double t_final)
: TimeDependentOperator(fes.GetVSize() * (int)vNodes.size()),
  fes_(fes),
  Nv_((int)vNodes.size()),
  ndof_(fes.GetVSize()),
  ndof_face_(fes.GetFE(0)->GetDof()), // not strictly used; kept for reference
  vmax_(0.0),
  t_final_(t_final),
  vNodes_(vNodes)
{
    for (double v : vNodes_) { vmax_ = std::max(vmax_, std::abs(v)); }

    // L2 DG needs the L2ElementRestriction
    elemR_ = std::make_unique<L2ElementRestriction>(fes_);

    // Buffers sized off first element (uniform p assumed)
    Ue_.SetSize(fes_.GetFE(0)->GetDof());
    dUe_.SetSize(fes_.GetFE(0)->GetDof());

    PrecomputeElementMatrices(vNodes_);
    PrecomputeFaceBlocks(vNodes_);
}

void DG_Advection::PrecomputeElementMatrices(const std::vector<double> &vNodes)
{
    Mesh &mesh = *fes_.GetMesh();
    const int NE = mesh.GetNE();

    M_e_.resize(NE);
    K_e_.assign(Nv_, std::vector<DenseMatrix>(NE));
    Minv_e_.resize(NE);            // << explicit inverse per element
    // (MinvLumped_ no longer used in this option.)

    MassIntegrator mi;

    for (int e = 0; e < NE; ++e)
    {
        const FiniteElement &fe = *fes_.GetFE(e);
        ElementTransformation &Tr = *fes_.GetElementTransformation(e);

        // Element mass
        mi.AssembleElementMatrix(fe, Tr, M_e_[e]);

        // Precompute explicit inverse of the consistent mass matrix
        {
            DenseMatrixInverse inv(M_e_[e]);  // LU factorization internally
            Minv_e_[e].SetSize(M_e_[e].Height());
            inv.GetInverseMatrix(Minv_e_[e]); // store M_e^{-1}
        }

        // Per-velocity convection (volume) matrix; sign -1 baked in
        for (int iv = 0; iv < Nv_; ++iv)
        {
            VectorFunctionCoefficient vcoef(1,
                [v=vNodes[iv]](const Vector &, Vector &vv)
                { vv.SetSize(1); vv = 0.0; vv(0) = v; });

            ConvectionIntegrator ki(vcoef, -1.0); // du/dt = K*u with -1 applied in K
            ki.AssembleElementMatrix(fe, Tr, K_e_[iv][e]);
        }
    }
}

void DG_Advection::PrecomputeFaceBlocks(const std::vector<double> &vNodes)
{
    Mesh &mesh = *fes_.GetMesh();
    const int Nfaces = mesh.GetNumFaces();

    IFace_.assign(Nv_, std::vector<FaceBlock>());
    BFace_.assign(Nv_, std::vector<BdrFaceBlock>());

    for (int iv = 0; iv < Nv_; ++iv)
    {
        VectorFunctionCoefficient vcoef(1,
            [v=vNodes[iv]](const Vector &, Vector &vv)
            { vv.SetSize(1); vv = 0.0; vv(0) = v; });

        NonconservativeDGTraceIntegrator tr(vcoef, -1.0);

        for (int f = 0; f < Nfaces; ++f)
        {
            FaceElementTransformations *FT = mesh.GetFaceElementTransformations(f, 31);
            if (!FT) { continue; }

            const bool bdr = (FT->Elem2No < 0);

            if (!bdr)
            {
                const int eL = FT->Elem1No;
                const int eR = FT->Elem2No;

                const FiniteElement &fel = *fes_.GetFE(eL);
                const FiniteElement &fer = *fes_.GetFE(eR);

                const int ldL = fel.GetDof();
                const int ldR = fer.GetDof();

                DenseMatrix A(ldL + ldR);
                tr.AssembleFaceMatrix(fel, fer, *FT, A);

                FaceBlock fb;
                fb.eL = eL; fb.eR = eR;
                fb.A_LL.SetSize(ldL, ldL);
                fb.A_LR.SetSize(ldL, ldR);
                fb.A_RL.SetSize(ldR, ldL);
                fb.A_RR.SetSize(ldR, ldR);

                for (int i = 0; i < ldL; ++i)
                for (int j = 0; j < ldL; ++j)
                    fb.A_LL(i,j) = A(i,j);

                for (int i = 0; i < ldL; ++i)
                for (int j = 0; j < ldR; ++j)
                    fb.A_LR(i,j) = A(i, ldL + j);

                for (int i = 0; i < ldR; ++i)
                for (int j = 0; j < ldL; ++j)
                    fb.A_RL(i,j) = A(ldL + i, j);

                for (int i = 0; i < ldR; ++i)
                for (int j = 0; j < ldR; ++j)
                    fb.A_RR(i,j) = A(ldL + i, ldL + j);

                IFace_[iv].push_back(std::move(fb));
            }
            else
            {
                const int e  = FT->Elem1No;
                const FiniteElement &fe = *fes_.GetFE(e);
                const int lde = fe.GetDof();

                DenseMatrix Ab(lde); // square
                tr.AssembleFaceMatrix(fe, fe, *FT, Ab);

                BdrFaceBlock bf;
                bf.e = e;
                bf.face = FT->FaceGeom; // metadata only
                bf.A_bdr.SetSize(lde, lde);
                bf.A_bdr = Ab;

                BFace_[iv].push_back(std::move(bf));
            }
        }
    }
}

// ----------------------------- Mult --------------------------------

void DG_Advection::Mult(const Vector &U, Vector &dUdt) const
{
    dUdt = 0.0;

    Mesh &mesh = *fes_.GetMesh();
    const int NE = mesh.GetNE();

    // Per-element offsets into the element-stacked (E-layout) vectors
    std::vector<int> elem_off(NE + 1, 0);
    for (int e = 0; e < NE; ++e) {
        elem_off[e+1] = elem_off[e] + fes_.GetFE(e)->GetDof();
    }
    const int Esize = elem_off[NE];

    // --- Gather ALL velocities once into E-layout; allocate residual buffers ---
    std::vector<mfem::Vector> Ue_all_list(Nv_), Re_all_list(Nv_);
    for (int iv = 0; iv < Nv_; ++iv)
    {
        Ue_all_list[iv].SetSize(Esize);
        Re_all_list[iv].SetSize(Esize);
        Re_all_list[iv] = 0.0;

        const double *Ub_ptr = U.GetData() + iv * ndof_;
        Vector Ub(const_cast<double*>(Ub_ptr), ndof_);
        elemR_->Mult(Ub, Ue_all_list[iv]);
    }

    // -------- ELEMENT-OUTER volume contribution (loop elements, then velocities) --------
    for (int e = 0; e < NE; ++e)
    {
        const int off = elem_off[e];
        const int ld  = elem_off[e+1] - off;

        for (int iv = 0; iv < Nv_; ++iv)
        {
            Vector Ue(Ue_all_list[iv].GetData() + off, ld);
            Vector Re(Re_all_list[iv].GetData() + off, ld);

            // Volume residual already has the correct sign from assembly:
            // Re += K_e_[iv][e] * Ue  (K_e holds the -1 factor for advection)
            K_e_[iv][e].Mult(Ue, Re);
        }
    }

    // -------- Faces (per-velocity, add directly into Re_all_list[iv]) --------
    // Interior faces
    for (int iv = 0; iv < Nv_; ++iv)
    {
        for (const auto &fb : IFace_[iv])
        {
            const int eL   = fb.eL;
            const int eR   = fb.eR;
            const int offL = elem_off[eL];
            const int offR = elem_off[eR];
            const int ldL  = elem_off[eL+1] - offL;
            const int ldR  = elem_off[eR+1] - offR;

            Vector UeL(Ue_all_list[iv].GetData() + offL, ldL);
            Vector UeR(Ue_all_list[iv].GetData() + offR, ldR);
            Vector ReL(Re_all_list[iv].GetData() + offL, ldL);
            Vector ReR(Re_all_list[iv].GetData() + offR, ldR);

            Vector RL(ldL); RL = 0.0;
            Vector RR(ldR); RR = 0.0;

            fb.A_LL.Mult(UeL, RL);
            fb.A_LR.AddMult(UeR, RL);

            fb.A_RL.Mult(UeL, RR);
            fb.A_RR.AddMult(UeR, RR);

            ReL += RL;
            ReR += RR;
        }

        // Boundary faces
        for (const auto &bf : BFace_[iv])
        {
            const int e    = bf.e;
            const int off  = elem_off[e];
            const int lde  = elem_off[e+1] - off;

            Vector Ue(Ue_all_list[iv].GetData() + off, lde);
            Vector Re(Re_all_list[iv].GetData() + off, lde);

            Vector Rb(lde); Rb = 0.0;
            bf.A_bdr.Mult(Ue, Rb);
            Re += Rb;
        }
    }

    // -------- Apply CONSISTENT mass inverse per element --------
    // Re_e := M_e^{-1} * Re_e  for each element e and velocity iv
    for (int e = 0; e < NE; ++e)
    {
        const int off = elem_off[e];
        const int ld  = elem_off[e+1] - off;

        for (int iv = 0; iv < Nv_; ++iv)
        {
            Vector Re(Re_all_list[iv].GetData() + off, ld);
            Vector tmp(ld); tmp = 0.0;
            Minv_e_[e].Mult(Re, tmp);  // tmp = M_e^{-1} * Re
            Re = tmp;                   // overwrite Re with result
        }
    }

    // -------- Scatter back to L-layout per velocity --------
    for (int iv = 0; iv < Nv_; ++iv)
    {
        double *dUb_ptr = dUdt.GetData() + iv * ndof_;
        Vector dUb(dUb_ptr, ndof_);
        elemR_->MultTranspose(Re_all_list[iv], dUb);
    }
}
