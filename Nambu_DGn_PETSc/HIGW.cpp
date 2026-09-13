#include "HIGW.hpp"
/*--------------------------------------------------------------------------*/
PetscErrorCode checkCurlSymmetry(Mat C, const char name[])
{
    PetscFunctionBegin;

    Mat CT = nullptr;
    Mat D  = nullptr;

    PetscReal normC = 0.0;
    PetscReal normD = 0.0;

    // CT = C^T
    PetscCall(MatTranspose(C, MAT_INITIAL_MATRIX, &CT));

    // D = C
    PetscCall(MatDuplicate(C, MAT_COPY_VALUES, &D));

    // D = D - CT = C - C^T
    PetscCall(MatAXPY(D, -1.0, CT, DIFFERENT_NONZERO_PATTERN));

    PetscCall(MatNorm(C, NORM_FROBENIUS, &normC));
    PetscCall(MatNorm(D, NORM_FROBENIUS, &normD));

    PetscPrintf(PETSC_COMM_SELF,
                "%s symmetry check: ||C-C^T||_F = %.16e, ||C||_F = %.16e, rel = %.16e\n",
                name,
                (double)normD,
                (double)normC,
                (double)(normD / (normC + PETSC_SMALL)));

    PetscCall(MatDestroy(&CT));
    PetscCall(MatDestroy(&D));

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode computeHelicity(
    Mat M,
    Mat CURL,
    Vec V,
    const RotationParameters &rot,
    DiagnosticsWorkspace &work,
    double &helicity)
{
  PetscFunctionBeginUser;
  // omega = CURL V
  PetscCall(MatMult(CURL, V, work.omega));

  if (rot.include_rotation) {
    PetscCheck(rot.RossbyNumber != 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                "Rossby number must be nonzero when rotation is enabled.");

    PetscCall(VecAXPY(work.omega, 2.0/rot.RossbyNumber, work.Frot)); // 2 x 0.5 = 1
  }
  // Momega = M omega
  PetscCall(MatMult(M, work.omega, work.Momega));

  // h = 1/2 V^T M omega
  PetscScalar dot = 0.0;
  PetscCall(VecDot(V, work.Momega, &dot));

  // // h = 1/2 V^T M omega
  // PetscScalar dot = 0.0;
  // PetscCall(VecDot(V, work.omega, &dot));


  helicity = 0.5 * PetscRealPart(dot);

  PetscFunctionReturn(PETSC_SUCCESS);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode destroyPicardWorkspace(PicardWorkspace &work)
{
    PetscFunctionBegin;

    PetscCall(ISDestroy(&work.isU));
    PetscCall(ISDestroy(&work.isV));
    PetscCall(ISDestroy(&work.isW));
    PetscCall(ISDestroy(&work.isP));
    PetscCall(VecDestroy(&work.tmpP));

    work.node_to_elem_idx.clear();
    work.elem_pos_list.clear();

    work.Pe_x.clear();
    work.Pe_y.clear();
    work.Pe_z.clear();

    work.u_loc.clear();
    work.v_loc.clear();
    work.w_loc.clear();

    work.bu_loc.clear();
    work.bv_loc.clear();
    work.bw_loc.clear();

    work.NElements = 0;

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode initializePicardWorkspace(
    PicardWorkspace &work,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    PetscInt N_Nodes,
    PetscInt Np)
{
    PetscFunctionBegin;

    const PetscInt NElements = static_cast<PetscInt>(List_Of_Elements.size());

    work.NElements = NElements;

    work.node_to_elem_idx.assign(N_Nodes, -1);
    work.elem_pos_list.assign(NElements, -1);

    for (PetscInt e = 0; e < NElements; ++e)
    {
        const PetscInt pos = static_cast<PetscInt>(List_Of_Elements[e]->get_pos());
        work.elem_pos_list[e] = pos;

        for (PetscInt a = 0; a < Np; ++a) {
            work.node_to_elem_idx[pos + a] = e;
        }
    }

    for (PetscInt i = 0; i < N_Nodes; ++i) {
        PetscCheck(work.node_to_elem_idx[i] >= 0,
                   PETSC_COMM_SELF,
                   PETSC_ERR_ARG_WRONG,
                   "initializePicardWorkspace: node_to_elem_idx incomplete");
    }

    work.Pe_x.assign(Np * Np, 0.0);
    work.Pe_y.assign(Np * Np, 0.0);
    work.Pe_z.assign(Np * Np, 0.0);

    work.u_loc.assign(Np, 0.0);
    work.v_loc.assign(Np, 0.0);
    work.w_loc.assign(Np, 0.0);

    work.bu_loc.assign(Np, 0.0);
    work.bv_loc.assign(Np, 0.0);
    work.bw_loc.assign(Np, 0.0);

    PetscCall(ISCreateStride(PETSC_COMM_SELF, N_Nodes, 0,           1, &work.isU));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, N_Nodes, N_Nodes,     1, &work.isV));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, N_Nodes, 2*N_Nodes,   1, &work.isW));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, N_Nodes, 3*N_Nodes,   1, &work.isP));

    PetscCall(VecCreateSeq(PETSC_COMM_SELF, N_Nodes, &work.tmpP));

    // Reusable pressure assembly workspace
    work.pressure_touched_elems.reserve(32);
    work.pressure_touched_mark.assign(NElements, -1);
    work.pressure_accum.assign(32 * Np, 0.0);
    work.pressure_cols_out.assign(Np, 0);

    work.dense_rows.assign(Np, 0);
    work.dense_cols.assign(Np, 0);
    work.dense_vals.assign(Np * Np, 0.0);

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode computeRelativeNonlinearFieldFromCache(
    const PicardNonlinearCache &cache,
    PicardWorkspace &work,
    const Vec Vmid,
    Vec Nrelative,
    const PetscInt N_Nodes,
    const PetscInt Np)
{
    PetscFunctionBeginUser;

    const PetscScalar *v = nullptr;
    PetscScalar *n = nullptr;

    PetscCall(VecZeroEntries(Nrelative));

    PetscCall(VecGetArrayRead(Vmid, &v));
    PetscCall(VecGetArray(Nrelative, &n));

    for (PetscInt e = 0; e < work.NElements; ++e)
    {
        const PetscInt pos = work.elem_pos_list[e];

        const PetscScalar *MPx =
            cache.MPx_by_elem[e].data();
        const PetscScalar *MPy =
            cache.MPy_by_elem[e].data();
        const PetscScalar *MPz =
            cache.MPz_by_elem[e].data();

        for (PetscInt a = 0; a < Np; ++a)
        {
            work.u_loc[a] =
                v[pos + a];

            work.v_loc[a] =
                v[N_Nodes + pos + a];

            work.w_loc[a] =
                v[2 * N_Nodes + pos + a];

            work.bu_loc[a] = 0.0;
            work.bv_loc[a] = 0.0;
            work.bw_loc[a] = 0.0;
        }

        // Nrelative = -(omega^r x Vmid)
        //
        // x: +omega_z v - omega_y w
        PetscCall(denseMatVecAdd(
            Np,
            MPz,
            work.v_loc.data(),
            +1.0,
            work.bu_loc.data()));

        PetscCall(denseMatVecAdd(
            Np,
            MPy,
            work.w_loc.data(),
            -1.0,
            work.bu_loc.data()));

        // y: -omega_z u + omega_x w
        PetscCall(denseMatVecAdd(
            Np,
            MPz,
            work.u_loc.data(),
            -1.0,
            work.bv_loc.data()));

        PetscCall(denseMatVecAdd(
            Np,
            MPx,
            work.w_loc.data(),
            +1.0,
            work.bv_loc.data()));

        // z: +omega_y u - omega_x v
        PetscCall(denseMatVecAdd(
            Np,
            MPy,
            work.u_loc.data(),
            +1.0,
            work.bw_loc.data()));

        PetscCall(denseMatVecAdd(
            Np,
            MPx,
            work.v_loc.data(),
            -1.0,
            work.bw_loc.data()));

        for (PetscInt a = 0; a < Np; ++a)
        {
            n[pos + a] +=
                work.bu_loc[a];

            n[N_Nodes + pos + a] +=
                work.bv_loc[a];

            n[2 * N_Nodes + pos + a] +=
                work.bw_loc[a];
        }
    }

    PetscCall(VecRestoreArrayRead(Vmid, &v));
    PetscCall(VecRestoreArray(Nrelative, &n));

    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode computeAbsoluteHelicityBoundaryDiagnostics(
    AppCtx &user,
    const Vec ForceHalf,
    const std::vector<std::unique_ptr<Element>>
        &List_Of_Elements,
    AbsoluteHelicityBoundaryDiagnostics &out)
{
    PetscFunctionBeginUser;

    DiagnosticsWorkspace &work = user.diagWork;

    out = AbsoluteHelicityBoundaryDiagnostics{};

    // ------------------------------------------------------------
    // 1. Accepted midpoint velocity
    // ------------------------------------------------------------
    PetscCall(VecCopy(
        user.Velocity_n,
        work.Vmid));

    PetscCall(VecAXPY(
        work.Vmid,
        1.0,
        user.Velocity));

    PetscCall(VecScale(
        work.Vmid,
        0.5));

    // ------------------------------------------------------------
    // 2. Relative-vorticity nonlinear contribution
    //
    // Nrelative = -(CURL Vmid) x Vmid
    // ------------------------------------------------------------
    PetscCall(VecZeroEntries(work.Nrelative));

    if (user.opt.nonlinear == PETSC_TRUE)
    {
        PetscCall(MatMult(
            user.ops.CURLdx,
            work.Vmid,
            user.OmegaStar_x));

        PetscCall(MatMult(
            user.ops.CURLdy,
            work.Vmid,
            user.OmegaStar_y));

        PetscCall(MatMult(
            user.ops.CURLdz,
            work.Vmid,
            user.OmegaStar_z));

        PetscCall(buildPicardNonlinearCache(
            user.picardCache,
            user.picardWork,
            List_Of_Elements,
            user.ops,
            user.Np,
            user.OmegaStar_x,
            user.OmegaStar_y,
            user.OmegaStar_z));

        PetscCall(computeRelativeNonlinearFieldFromCache(
            user.picardCache,
            user.picardWork,
            work.Vmid,
            work.Nrelative,
            user.N_Nodes,
            user.Np));
    }

    // ------------------------------------------------------------
    // 3. Linear rotational contribution
    //
    // ConstantRotationalMat V = (1/Ro) f x V.
    //
    // The momentum contribution is:
    // Nrotation = -(1/Ro) f x Vmid.
    // ------------------------------------------------------------
    PetscCall(VecZeroEntries(work.Nrotation));

    if (user.rot.include_rotation == PETSC_TRUE)
    {
        PetscCheck(
            user.ops.ConstantRotationalMat != nullptr,
            PETSC_COMM_SELF,
            PETSC_ERR_ARG_WRONGSTATE,
            "Rotation is enabled but ConstantRotationalMat is null.");

        PetscCall(MatMult(
            user.ops.ConstantRotationalMat,
            work.Vmid,
            work.Nrotation));

        PetscCall(VecScale(
            work.Nrotation,
            -1.0));
    }

    // Ntotal = Nrelative + Nrotation
    PetscCall(VecCopy(
        work.Nrelative,
        work.Ntotal));

    PetscCall(VecAXPY(
        work.Ntotal,
        1.0,
        work.Nrotation));

    // ------------------------------------------------------------
    // 4. Accepted pressure and pressure gradient
    // ------------------------------------------------------------
    Vec Pressure = nullptr;

    PetscCall(VecGetSubVector(
        user.X,
        user.isp,
        &Pressure));

    PetscCall(MatMult(
        user.ops.GRAD,
        Pressure,
        work.gradP));

    // ------------------------------------------------------------
    // 5. Viscous field A^nu Vmid
    // ------------------------------------------------------------
    PetscCall(ApplyVelocityLaplacianBlockDiag(
        user.ops.Laplacian,
        work.Vmid,
        work.AV,
        user.isu,
        user.isv,
        user.isw));

    double pairing = 0.0;

    // ------------------------------------------------------------
    // 6. Relative + rotational boundary pairing
    // ------------------------------------------------------------
    PetscCall(computeCurlBoundaryPairing(
        work.Ntotal,
        work.Vmid,
        user.ops.CurlBoundary,
        work.boundaryWork,
        pairing));

    out.nonlinear =
        -0.5 * pairing;

    // ------------------------------------------------------------
    // 7. Explicit background-vorticity pressure flux
    //
    // -(1/Ro) F^T M GRAD P
    // ------------------------------------------------------------
    if (user.rot.include_rotation == PETSC_TRUE)
    {
        PetscCheck(
            user.rot.RossbyNumber != 0.0,
            PETSC_COMM_SELF,
            PETSC_ERR_ARG_OUTOFRANGE,
            "Rossby number must be nonzero.");

        PetscCall(MatMult(
            user.ops.M,
            work.gradP,
            work.MAV));

        PetscScalar value = 0.0;

        PetscCall(VecDot(
            work.Frot,
            work.MAV,
            &value));

        out.pressure_background =
            PetscRealPart(value)
            / user.rot.RossbyNumber;
    }

    // ------------------------------------------------------------
    // 8. Pressure boundary pairing
    // ------------------------------------------------------------
    PetscCall(computeCurlBoundaryPairing(
        work.gradP,
        work.Vmid,
        user.ops.CurlBoundary,
        work.boundaryWork,
        pairing));

    out.pressure_pairing =
        0.5 * pairing;

    // ------------------------------------------------------------
    // 9. Viscous boundary contribution
    // ------------------------------------------------------------
    if (user.opt.viscous)
    {
        PetscCheck(
            user.Re != 0.0,
            PETSC_COMM_SELF,
            PETSC_ERR_ARG_OUTOFRANGE,
            "Reynolds number must be nonzero.");

        PetscCall(computeCurlBoundaryPairing(
            work.AV,
            work.Vmid,
            user.ops.CurlBoundary,
            work.boundaryWork,
            pairing));

        out.viscous =
            -0.5 * pairing / user.Re;
    }

    // ------------------------------------------------------------
    // 10. Forcing boundary contribution
    // ------------------------------------------------------------
    if (ForceHalf != nullptr)
    {
        PetscCheck(
            user.opt.Fo != 0.0,
            PETSC_COMM_SELF,
            PETSC_ERR_ARG_OUTOFRANGE,
            "Forcing number must be nonzero.");

        PetscCall(computeCurlBoundaryPairing(
            ForceHalf,
            work.Vmid,
            user.ops.CurlBoundary,
            work.boundaryWork,
            pairing));

        out.forcing =
            -0.5 * pairing
            / (user.opt.Fo * user.opt.Fo);
    }

    // ------------------------------------------------------------
    // 11. Total boundary flux
    // ------------------------------------------------------------
    out.total =
          out.nonlinear
        + out.pressure_background
        + out.pressure_pairing
        + out.viscous
        + out.forcing;

    // ------------------------------------------------------------
    // 12. CURL GRAD compatibility defect
    //
    // D_cg = (CURL GRAD P)^T M Vmid
    // ------------------------------------------------------------
    PetscCall(MatMult(
        user.ops.CURL,
        work.gradP,
        work.curlGradP));
        // ------------------------------------------------------------
        // Solution-specific CURL*GRAD P diagnostics
        // ------------------------------------------------------------
        PetscReal curlGradPInfinity = 0.0;
        PetscReal curlGradPEuclidean = 0.0;

        PetscCall(VecNorm(
            work.curlGradP,
            NORM_INFINITY,
            &curlGradPInfinity));

        PetscCall(VecNorm(
            work.curlGradP,
            NORM_2,
            &curlGradPEuclidean));

        // boundaryWork = M * CURL * GRAD P
        PetscCall(MatMult(
            user.ops.M,
            work.curlGradP,
            work.boundaryWork));

        PetscScalar curlGradPMassSquared = 0.0;

        PetscCall(VecDot(
            work.curlGradP,
            work.boundaryWork,
            &curlGradPMassSquared));

        const double curlGradPMass =
            PetscSqrtReal(
                PetscMax(
                    0.0,
                    PetscRealPart(curlGradPMassSquared)));

    PetscCall(MatMult(
        user.ops.M,
        work.Vmid,
        work.MAV));

    PetscScalar defect = 0.0;

    PetscCall(VecDot(
        work.curlGradP,
        work.MAV,
        &defect));

    out.curl_grad_defect =
        PetscRealPart(defect);
        out.curl_grad_p_inf =
            static_cast<double>(curlGradPInfinity);

        out.curl_grad_p_l2 =
            static_cast<double>(curlGradPEuclidean);

        out.curl_grad_p_mass =
            curlGradPMass;
    PetscCall(VecRestoreSubVector(
        user.X,
        user.isp,
        &Pressure));

    PetscFunctionReturn(PETSC_SUCCESS);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode formRHSPicardDirectFromCache(
    Vec bnonlinear,
    const PicardNonlinearCache &cache,
    PicardWorkspace &work,
    const Vec Velocity_n,
    const DiscreteOperators &ops,
    PetscInt N_Nodes,
    PetscInt Np,
    PetscReal Delta_t)
{
    Vec bU = nullptr, bV = nullptr, bW = nullptr, bP = nullptr;

    const PetscScalar *veln = nullptr;
    PetscScalar *bu = nullptr, *bv = nullptr, *bw = nullptr;

    PetscFunctionBegin;

    const PetscInt NElements = work.NElements;

    PetscCall(VecZeroEntries(bnonlinear));

    PetscCall(VecGetSubVector(bnonlinear, work.isU, &bU));
    PetscCall(VecGetSubVector(bnonlinear, work.isV, &bV));
    PetscCall(VecGetSubVector(bnonlinear, work.isW, &bW));
    PetscCall(VecGetSubVector(bnonlinear, work.isP, &bP));

    PetscCall(VecGetArray(bU, &bu));
    PetscCall(VecGetArray(bV, &bv));
    PetscCall(VecGetArray(bW, &bw));
    PetscCall(VecGetArrayRead(Velocity_n, &veln));

    for (PetscInt e = 0; e < NElements; ++e)
    {
        const PetscInt pos = work.elem_pos_list[e];

        const PetscScalar *MPx = cache.MPx_by_elem[e].data();
        const PetscScalar *MPy = cache.MPy_by_elem[e].data();
        const PetscScalar *MPz = cache.MPz_by_elem[e].data();

        for (PetscInt a = 0; a < Np; ++a) {
            work.u_loc[a] = veln[pos + a];
            work.v_loc[a] = veln[N_Nodes + pos + a];
            work.w_loc[a] = veln[2 * N_Nodes + pos + a];

            work.bu_loc[a] = 0.0;
            work.bv_loc[a] = 0.0;
            work.bw_loc[a] = 0.0;
        }

        // std::fill(work.bu_loc.begin(), work.bu_loc.end(), 0.0);
        // std::fill(work.bv_loc.begin(), work.bv_loc.end(), 0.0);
        // std::fill(work.bw_loc.begin(), work.bw_loc.end(), 0.0);

        double sign = 1.0;
        PetscCall(denseMatVecAdd(Np, MPz, work.v_loc.data(), +Delta_t*sign, work.bu_loc.data()));
        PetscCall(denseMatVecAdd(Np, MPy, work.w_loc.data(), -Delta_t*sign, work.bu_loc.data()));

        PetscCall(denseMatVecAdd(Np, MPz, work.u_loc.data(), -Delta_t*sign, work.bv_loc.data()));
        PetscCall(denseMatVecAdd(Np, MPx, work.w_loc.data(), +Delta_t*sign, work.bv_loc.data()));

        PetscCall(denseMatVecAdd(Np, MPy, work.u_loc.data(), +Delta_t*sign, work.bw_loc.data()));
        PetscCall(denseMatVecAdd(Np, MPx, work.v_loc.data(), -Delta_t*sign, work.bw_loc.data()));

        for (PetscInt a = 0; a < Np; ++a) {
            bu[pos + a] += work.bu_loc[a];
            bv[pos + a] += work.bv_loc[a];
            bw[pos + a] += work.bw_loc[a];
        }
    }

    PetscCall(VecRestoreArrayRead(Velocity_n, &veln));
    PetscCall(VecRestoreArray(bU, &bu));
    PetscCall(VecRestoreArray(bV, &bv));
    PetscCall(VecRestoreArray(bW, &bw));

    //PetscCall(VecZeroEntries(bP));
    //PetscCall(VecZeroEntries(work.tmpP));

    PetscCall(MatMult(ops.DIVdx, bU, bP));
    PetscCall(MatMult(ops.DIVdy, bV, work.tmpP));
    PetscCall(VecAXPY(bP, 1.0, work.tmpP));
    PetscCall(MatMult(ops.DIVdz, bW, work.tmpP));
    PetscCall(VecAXPY(bP, 1.0, work.tmpP));

    PetscCall(VecRestoreSubVector(bnonlinear, work.isU, &bU));
    PetscCall(VecRestoreSubVector(bnonlinear, work.isV, &bV));
    PetscCall(VecRestoreSubVector(bnonlinear, work.isW, &bW));
    PetscCall(VecRestoreSubVector(bnonlinear, work.isP, &bP));

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode formMatrixPicardTotalFromCache(
    Mat A,
    const Mat Alinear,
    const PicardNonlinearCache &cache,
    PicardWorkspace &work,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    const DiscreteOperators &ops,
    PetscInt N_Nodes,
    PetscInt Np,
    PetscReal Delta_t)
{
    PetscFunctionBegin;

    PetscCall(MatCopy(Alinear, A, SAME_NONZERO_PATTERN));

    PetscCall(addNonlinearVelocityAndPressureFromCache(
        A,
        cache,
        work,
        ops,
        N_Nodes,
        Np,
        Delta_t));

    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode addNonlinearVelocityAndPressureFromCache(
    Mat A,
    const PicardNonlinearCache &cache,
    PicardWorkspace &work,
    const DiscreteOperators &ops,
    PetscInt N_Nodes,
    PetscInt Np,
    PetscReal Delta_t)
{
    PetscFunctionBegin;

    const PetscInt NElements = work.NElements;

    // ------------------------------------------------------------
    // Velocity blocks
    // ------------------------------------------------------------

        double sign = -1.0;
        for (PetscInt e = 0; e < NElements; ++e)
        {
            const PetscInt pos = work.elem_pos_list[e];
            const PetscScalar *MPx = cache.MPx_by_elem[e].data();
            const PetscScalar *MPy = cache.MPy_by_elem[e].data();
            const PetscScalar *MPz = cache.MPz_by_elem[e].data();

            PetscCall(addDenseElementBlockValues(A, 0,         N_Nodes,   pos, Np, +Delta_t*sign, MPz, work));
            PetscCall(addDenseElementBlockValues(A, 0,         2*N_Nodes, pos, Np, -Delta_t*sign, MPy, work));
            PetscCall(addDenseElementBlockValues(A, N_Nodes,   0,         pos, Np, -Delta_t*sign, MPz, work));
            PetscCall(addDenseElementBlockValues(A, N_Nodes,   2*N_Nodes, pos, Np, +Delta_t*sign, MPx, work));
            PetscCall(addDenseElementBlockValues(A, 2*N_Nodes, 0,         pos, Np, +Delta_t*sign, MPy, work));
            PetscCall(addDenseElementBlockValues(A, 2*N_Nodes, N_Nodes,   pos, Np, -Delta_t*sign, MPx, work));
        }

        // ------------------------------------------------------------
        // Pressure row
        // ------------------------------------------------------------
        for (PetscInt i = 0; i < N_Nodes; ++i)
        {
            const PetscInt prow = 3 * N_Nodes + i;

            PetscCall(addPressureContributionFromDiv(
                A, ops.DIVdy, i, prow, 0,
                Np, work.node_to_elem_idx, work.elem_pos_list, cache.MPz_by_elem, -Delta_t*sign, work));
            PetscCall(addPressureContributionFromDiv(
                A, ops.DIVdz, i, prow, 0,
                Np, work.node_to_elem_idx, work.elem_pos_list, cache.MPy_by_elem, +Delta_t*sign, work));

            PetscCall(addPressureContributionFromDiv(
                A, ops.DIVdx, i, prow, N_Nodes,
                Np, work.node_to_elem_idx, work.elem_pos_list, cache.MPz_by_elem, +Delta_t*sign, work));
            PetscCall(addPressureContributionFromDiv(
                A, ops.DIVdz, i, prow, N_Nodes,
                Np, work.node_to_elem_idx, work.elem_pos_list, cache.MPx_by_elem, -Delta_t*sign, work));

            PetscCall(addPressureContributionFromDiv(
                A, ops.DIVdx, i, prow, 2 * N_Nodes,
                Np, work.node_to_elem_idx, work.elem_pos_list, cache.MPy_by_elem, -Delta_t*sign, work));
            PetscCall(addPressureContributionFromDiv(
                A, ops.DIVdy, i, prow, 2 * N_Nodes,
                Np, work.node_to_elem_idx, work.elem_pos_list, cache.MPx_by_elem, +Delta_t*sign, work));
        }

        PetscFunctionReturn(0);
    }
/*--------------------------------------------------------------------------*/
PetscErrorCode buildPicardNonlinearCache(
    PicardNonlinearCache &cache,
    PicardWorkspace &work,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    const DiscreteOperators &ops,
    PetscInt Np,
    const Vec OmegaStar_x,
    const Vec OmegaStar_y,
    const Vec OmegaStar_z)
{
    PetscFunctionBegin;

    const PetscInt NElements = static_cast<PetscInt>(List_Of_Elements.size());

    if (cache.NElements != NElements || cache.Np != Np) {
        cache.resize(NElements, Np);
    }

    const PetscScalar *ox = nullptr, *oy = nullptr, *oz = nullptr;
    PetscCall(VecGetArrayRead(OmegaStar_x, &ox));
    PetscCall(VecGetArrayRead(OmegaStar_y, &oy));
    PetscCall(VecGetArrayRead(OmegaStar_z, &oz));

    for (PetscInt e = 0; e < NElements; ++e)
    {
        const auto &elem = List_Of_Elements[e];
        const PetscInt pos = work.elem_pos_list[e];
        const PetscScalar J = static_cast<PetscScalar>(elem->get_detJ());
        const std::vector<PetscScalar> &invM = elem->get_invM_local();

        if (static_cast<PetscInt>(invM.size()) != Np * Np) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
                    "elem->invM_local has wrong size");
        }

        PetscCall(buildLocalProductMatrixFromTensor(
            ops.G1D, ox + pos, Np, J, work.Pe_x.data()));
        PetscCall(buildLocalProductMatrixFromTensor(
            ops.G1D, oy + pos, Np, J, work.Pe_y.data()));
        PetscCall(buildLocalProductMatrixFromTensor(
            ops.G1D, oz + pos, Np, J, work.Pe_z.data()));

        PetscCall(denseMatMat(Np, invM.data(), work.Pe_x.data(), cache.MPx_by_elem[e].data()));
        PetscCall(denseMatMat(Np, invM.data(), work.Pe_y.data(), cache.MPy_by_elem[e].data()));
        PetscCall(denseMatMat(Np, invM.data(), work.Pe_z.data(), cache.MPz_by_elem[e].data()));
    }

    PetscCall(VecRestoreArrayRead(OmegaStar_x, &ox));
    PetscCall(VecRestoreArrayRead(OmegaStar_y, &oy));
    PetscCall(VecRestoreArrayRead(OmegaStar_z, &oz));

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode formRHSLinear(
    Vec blinear,
    const Vec Velocity_n,              // full 3N vector
    const Vec ForceHalf,                   // full 3N vector, may be nullptr
    const DiscreteOperators &ops,
    const RotationParameters &rot,
    const PetscReal Delta_t,
    const PetscReal Re,
    const PetscReal force_scale,
    const PetscBool viscous,
    const PetscBool include_force,
    const PetscBool include_divergence_correction,
    IS isVel,
    IS isP,
    IS isu,
    IS isv,
    IS isw,
    Vec tmp_vel,                       // reusable full 3N vector
    Vec bvel_for_div)                  // reusable full 3N vector
{
    Vec bvel = nullptr;
    Vec bp   = nullptr;

    PetscFunctionBeginUser;

    PetscCall(VecZeroEntries(blinear));

    PetscCall(VecGetSubVector(blinear, isVel, &bvel));
    PetscCall(VecGetSubVector(blinear, isP,   &bp));

    // ------------------------------------------------------------
    // Build the linear velocity RHS excluding the pure bdiag = V^n term
    // ------------------------------------------------------------
    PetscCall(VecZeroEntries(bvel));

    // rotational known term: - Delta_t/2 * CR * V^n
    if (rot.include_rotation == PETSC_TRUE && ops.ConstantRotationalMat) {
        PetscCall(MatMult(ops.ConstantRotationalMat, Velocity_n, tmp_vel));
        PetscCall(VecAXPY(bvel, -Delta_t / 2.0, tmp_vel));
    }

    // viscous known term:
    // + Delta_t/(2Re) * blockdiag(L,L,L) * V^n
    if (viscous == PETSC_TRUE && ops.Laplacian) {
        PetscCall(ApplyVelocityLaplacianBlockDiag(
            ops.Laplacian,
            Velocity_n,
            tmp_vel,
            isu,
            isv,
            isw));

        PetscCall(VecAXPY(bvel, Delta_t / (2.0 * Re), tmp_vel));
    }

    // known external forcing term:
    // bvel <- bvel + Delta_t * force_scale * ForceHalf
    if (include_force == PETSC_TRUE && ForceHalf) {
        PetscCall(VecAXPY(bvel, Delta_t * force_scale, ForceHalf));
    }

    // ------------------------------------------------------------
    // Build the vector used in the pressure RHS
    // ------------------------------------------------------------
    PetscCall(VecCopy(bvel, bvel_for_div));

    if (include_divergence_correction == PETSC_TRUE) {
        const PetscReal a = 1.0;
        PetscCall(VecAXPY(bvel_for_div, a, Velocity_n));
    }

    // pressure RHS = DIV * bvel_for_div
    PetscCall(MatMult(ops.DIV, bvel_for_div, bp));

    // ------------------------------------------------------------
    // Finalize the velocity RHS itself by adding V^n
    // ------------------------------------------------------------
    PetscCall(VecAXPY(bvel, 1.0, Velocity_n));

    PetscCall(VecRestoreSubVector(blinear, isVel, &bvel));
    PetscCall(VecRestoreSubVector(blinear, isP,   &bp));

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode checkMatrixCommutator(
    Mat A,
    Mat B,
    const char *label)
{
    PetscFunctionBeginUser;

    Mat AB = nullptr;
    Mat BA = nullptr;
    Mat commutator = nullptr;

    PetscCall(MatMatMult(
        A,
        B,
        MAT_INITIAL_MATRIX,
        PETSC_DETERMINE,
        &AB));

    PetscCall(MatMatMult(
        B,
        A,
        MAT_INITIAL_MATRIX,
        PETSC_DETERMINE,
        &BA));

    // commutator = AB - BA
    PetscCall(MatDuplicate(
        AB,
        MAT_COPY_VALUES,
        &commutator));

    PetscCall(MatAXPY(
        commutator,
        -1.0,
        BA,
        DIFFERENT_NONZERO_PATTERN));

    PetscCall(MatAssemblyBegin(
        commutator,
        MAT_FINAL_ASSEMBLY));

    PetscCall(MatAssemblyEnd(
        commutator,
        MAT_FINAL_ASSEMBLY));

    PetscReal normA = 0.0;
    PetscReal normB = 0.0;
    PetscReal normCommF = 0.0;
    PetscReal normCommInf = 0.0;

    PetscCall(MatNorm(
        A,
        NORM_FROBENIUS,
        &normA));

    PetscCall(MatNorm(
        B,
        NORM_FROBENIUS,
        &normB));

    PetscCall(MatNorm(
        commutator,
        NORM_FROBENIUS,
        &normCommF));

    PetscCall(MatNorm(
        commutator,
        NORM_INFINITY,
        &normCommInf));

    const PetscReal denominator = normA * normB;

    const PetscReal relativeComm =
        denominator > PETSC_SMALL
        ? normCommF / denominator
        : 0.0;

    PetscCall(PetscPrintf(
        PETSC_COMM_WORLD,
        "%s:\n"
        "  ||Gi*Gj-Gj*Gi||_F   = %.16e\n"
        "  ||Gi*Gj-Gj*Gi||_inf = %.16e\n"
        "  relative commutator = %.16e\n",
        label,
        static_cast<double>(normCommF),
        static_cast<double>(normCommInf),
        static_cast<double>(relativeComm)));

    PetscCall(MatDestroy(&AB));
    PetscCall(MatDestroy(&BA));
    PetscCall(MatDestroy(&commutator));

    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode build_DiscreteOperators_Inertial(
    const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
    const std::vector<std::unique_ptr<Boundary>> &List_Of_Boundaries,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    unsigned int N_Nodes,
    unsigned int N_Order,
    const RotationParameters &rotation,
    DiscreteOperators &ops)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

    InertialRawMatrices raw;

    create_Matrices_Cuboids_Inertial(List_Of_Vertices, List_Of_Boundaries, List_Of_Elements,N_Nodes, N_Order, raw, rotation);
    std::cout << "\033[1;32m Raw Matrices Computed \033[0m" << std::endl;

    TripleProductTensor1D G1D;
    PetscCall(createTripleProductTensor1D(N_Order, ops.G1D));

    checkCurlSymmetry(raw.Curl, "raw.Curl");

    // checkCurlSymmetry(raw.Curl, "raw.Curl");


    // // actually weak curl matrices
    // ops.CURL = raw.Curl;
    // ops.CURLdx = raw.Curlx;
    // ops.CURLdy = raw.Curly;
    // ops.CURLdz = raw.Curlz;

    // ------------------------------------------------------------
    // Preserve the curl boundary matrix
    //
    // C_boundary = C - C^T, where C = M*CURL.
    //
    // Hence:
    //     b_boundary(U,V) = U^T C_boundary V.
    // ------------------------------------------------------------
    Mat CurlTranspose = nullptr;

    PetscCall(MatTranspose(
        raw.Curl,
        MAT_INITIAL_MATRIX,
        &CurlTranspose));

    PetscCall(MatDuplicate(
        raw.Curl,
        MAT_COPY_VALUES,
        &ops.CurlBoundary));

    PetscCall(MatAXPY(
        ops.CurlBoundary,
        -1.0,
        CurlTranspose,
        DIFFERENT_NONZERO_PATTERN));

    PetscCall(MatAssemblyBegin(
        ops.CurlBoundary,
        MAT_FINAL_ASSEMBLY));

    PetscCall(MatAssemblyEnd(
        ops.CurlBoundary,
        MAT_FINAL_ASSEMBLY));

    PetscCall(MatDestroy(&CurlTranspose));

    // ------------------------------------------------------------
    // Build curl operators
    //
    // CURL = M^{-1} C.
    // ------------------------------------------------------------

    // // build curl operators
    MatMatMult(raw.invM,       raw.Curl,   MAT_INITIAL_MATRIX, 1.0, &ops.CURL);
    MatMatMult(raw.invM_small, raw.Curlx, MAT_INITIAL_MATRIX, 1.0, &ops.CURLdx);
    MatMatMult(raw.invM_small, raw.Curly, MAT_INITIAL_MATRIX, 1.0, &ops.CURLdy);
    MatMatMult(raw.invM_small, raw.Curlz, MAT_INITIAL_MATRIX, 1.0, &ops.CURLdz);
    // keep mass matrices
    ops.M = raw.M;
    ops.M_small = raw.M_small;
    //ops.invM = raw.invM;
    //ops.invM_small = raw.invM_small;

    // build gradient operators
    MatMatMult(raw.invM,       raw.E,   MAT_INITIAL_MATRIX, 1.0, &ops.GRAD);
    MatMatMult(raw.invM_small, raw.Edx, MAT_INITIAL_MATRIX, 1.0, &raw.GRADdx);
    MatMatMult(raw.invM_small, raw.Edy, MAT_INITIAL_MATRIX, 1.0, &raw.GRADdy);
    MatMatMult(raw.invM_small, raw.Edz, MAT_INITIAL_MATRIX, 1.0, &raw.GRADdz);
    // build divergence operators
    MatMatMult(raw.invM_small, raw.ET,   MAT_INITIAL_MATRIX, 1.0, &ops.DIV);
    MatMatMult(raw.invM_small, raw.ETdx, MAT_INITIAL_MATRIX, 1.0, &ops.DIVdx);
    MatMatMult(raw.invM_small, raw.ETdy, MAT_INITIAL_MATRIX, 1.0, &ops.DIVdy);
    MatMatMult(raw.invM_small, raw.ETdz, MAT_INITIAL_MATRIX, 1.0, &ops.DIVdz);

    // build rotation operators
    if (rotation.include_rotation == PETSC_TRUE)
    {
        ierr = MatMatMult(raw.invM,       raw.ConstantRotational,  MAT_INITIAL_MATRIX, 1.0, &ops.ConstantRotationalMat); CHKERRQ(ierr);
        ierr = MatMatMult(raw.invM_small, raw.ConstantRotationalx, MAT_INITIAL_MATRIX, 1.0, &ops.ConstantRotationalMatx); CHKERRQ(ierr);
        ierr = MatMatMult(raw.invM_small, raw.ConstantRotationaly, MAT_INITIAL_MATRIX, 1.0, &ops.ConstantRotationalMaty); CHKERRQ(ierr);
        ierr = MatMatMult(raw.invM_small, raw.ConstantRotationalz, MAT_INITIAL_MATRIX, 1.0, &ops.ConstantRotationalMatz); CHKERRQ(ierr);
    }
    else
    {
        ops.ConstantRotationalMat  = nullptr;
        ops.ConstantRotationalMatx = nullptr;
        ops.ConstantRotationalMaty = nullptr;
        ops.ConstantRotationalMatz = nullptr;
    }

    // curl

    // std::cout << "New CURLx = " << std::endl;
    // MatView(raw.Curlx, PETSC_VIEWER_STDOUT_SELF);
    // std::cout << "New invM CURLx = " << std::endl;
    // MatMatMult(raw.invM_small, raw.Curlx, MAT_INITIAL_MATRIX, 1.0, &ops.Laplacian);
    // MatView(ops.Laplacian, PETSC_VIEWER_STDOUT_SELF);
    // std::cout << "Old CURLx = " << std::endl;
    // MatView(ops.CURLdx, PETSC_VIEWER_STDOUT_SELF);
    // pause();

    // ------------------------------------------------------------
    // Diagnostic: global CURL*GRAD compatibility
    // ------------------------------------------------------------
    {
        Mat CurlGrad = nullptr;

        PetscCall(MatMatMult(
            ops.CURL,
            ops.GRAD,
            MAT_INITIAL_MATRIX,
            PETSC_DETERMINE,
            &CurlGrad));

        PetscReal curlGradFrobenius = 0.0;
        PetscReal curlGradInfinity  = 0.0;

        PetscReal curlFrobenius = 0.0;
        PetscReal gradFrobenius = 0.0;

        PetscCall(MatNorm(
            CurlGrad,
            NORM_FROBENIUS,
            &curlGradFrobenius));

        PetscCall(MatNorm(
            CurlGrad,
            NORM_INFINITY,
            &curlGradInfinity));

        PetscCall(MatNorm(
            ops.CURL,
            NORM_FROBENIUS,
            &curlFrobenius));

        PetscCall(MatNorm(
            ops.GRAD,
            NORM_FROBENIUS,
            &gradFrobenius));

        const PetscReal denominator =
            curlFrobenius * gradFrobenius;

        const PetscReal normalizedCurlGrad =
            denominator > PETSC_SMALL
            ? curlGradFrobenius / denominator
            : 0.0;

        PetscPrintf(
            PETSC_COMM_WORLD,
            "CURL-GRAD operator diagnostic:\n"
            "  ||CURL*GRAD||_F             = %.16e\n"
            "  ||CURL*GRAD||_inf           = %.16e\n"
            "  ||CURL*GRAD||_F/"
            "(||CURL||_F ||GRAD||_F)       = %.16e\n",
            (double)curlGradFrobenius,
            (double)curlGradInfinity,
            (double)normalizedCurlGrad);

        PetscCall(MatDestroy(&CurlGrad));
    }

    PetscCall(checkMatrixCommutator(
        raw.GRADdx,
        raw.GRADdy,
        "[GRADdx,GRADdy]"));

    PetscCall(checkMatrixCommutator(
        raw.GRADdx,
        raw.GRADdz,
        "[GRADdx,GRADdz]"));

        PetscCall(checkMatrixCommutator(
            raw.GRADdy,
            raw.GRADdz,
            "[GRADdy,GRADdz]"));
    // PetscCall(checkMatrixCommutator(
    //     ops.CURLdx,
    //     ops.CURLdy,
    //     "[Gx,Gy]"));
    //
    // PetscCall(checkMatrixCommutator(
    //     ops.CURLdx,
    //     ops.CURLdz,
    //     "[Gx,Gz]"));
    //
    // PetscCall(checkMatrixCommutator(
    //     ops.CURLdy,
    //     ops.CURLdz,
    //     "[Gy,Gz]"));

    // laplacian
    MatMatMult(ops.DIV, ops.GRAD, MAT_INITIAL_MATRIX, 1.0, &ops.Laplacian);

    std::cout << "\033[1;32m Matrices computed \033[0m" << std::endl;
    // destroy temporary raw matrices that are no longer needed
    MatDestroy(&raw.E);
    MatDestroy(&raw.ET);
    MatDestroy(&raw.Edx);
    MatDestroy(&raw.Edy);
    MatDestroy(&raw.Edz);
    MatDestroy(&raw.ETdx);
    MatDestroy(&raw.ETdy);
    MatDestroy(&raw.ETdz);
    PetscCall(MatDestroy(&raw.GRADdx));
    PetscCall(MatDestroy(&raw.GRADdy));
    PetscCall(MatDestroy(&raw.GRADdz));
    MatDestroy(&raw.ConstantRotational);
    MatDestroy(&raw.ConstantRotationalx);
    MatDestroy(&raw.ConstantRotationaly);
    MatDestroy(&raw.ConstantRotationalz);

    MatDestroy(&raw.invM);
    MatDestroy(&raw.invM_small);

    PetscCall(MatDestroy(&raw.Curl));
    PetscCall(MatDestroy(&raw.Curlx));
    PetscCall(MatDestroy(&raw.Curly));
    PetscCall(MatDestroy(&raw.Curlz));
    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode computeFrozenOmega(
    const DiscreteOperators &ops,
    const Vec Vk,
    const Vec Vn,
    const unsigned int N_Nodes,
    Vec Vsum,
    Vec OmegaStar,
    Vec OmegaStar_x,
    Vec OmegaStar_y,
    Vec OmegaStar_z)
{

    const PetscScalar *omegaArray = nullptr;
    PetscScalar *ox = nullptr, *oy = nullptr, *oz = nullptr;

    PetscFunctionBegin;

    PetscCall(VecWAXPY(Vsum, 1.0, Vk, Vn));      // Vsum = Vk + Vn

    PetscCall(MatMult(ops.CURL, Vsum, OmegaStar)); // Omega* = 1/4*CURL*(Vk+Vn)
    PetscCall(VecScale(OmegaStar, 0.25));

    PetscCall(VecGetArrayRead(OmegaStar, &omegaArray));
    PetscCall(VecGetArray(OmegaStar_x, &ox));
    PetscCall(VecGetArray(OmegaStar_y, &oy));
    PetscCall(VecGetArray(OmegaStar_z, &oz));

    // for (unsigned int i = 0; i < N_Nodes; ++i)
    // {
    //     ox[i] = omegaArray[i];
    //     oy[i] = omegaArray[N_Nodes + i];
    //     oz[i] = omegaArray[2 * N_Nodes + i];
    // }
    PetscCall(PetscArraycpy(ox, omegaArray,             N_Nodes));
    PetscCall(PetscArraycpy(oy, omegaArray + N_Nodes,   N_Nodes));
    PetscCall(PetscArraycpy(oz, omegaArray + 2*N_Nodes, N_Nodes));

    PetscCall(VecRestoreArrayRead(OmegaStar, &omegaArray));
    PetscCall(VecRestoreArray(OmegaStar_x, &ox));
    PetscCall(VecRestoreArray(OmegaStar_y, &oy));
    PetscCall(VecRestoreArray(OmegaStar_z, &oz));

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode CreateFullPatternMatrix(
    Mat *Apattern,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    const AppCtx &user)
{

    const auto &ops = user.ops;
    const PetscInt N_Nodes = user.N_Nodes;
    const PetscInt Np      = user.Np;

    PetscFunctionBegin;

    std::cout << "user.N_Nodes = " << user.N_Nodes << ", user.Np = " << user.Np << std::endl;
    // Problem Type 2 (with viscous forces, without viscous correction without rotation)
    PetscInt nnz_vel  = 13 * Np+10;
    PetscInt nnz_pres = 25 * Np+10 + 3*Np*Np; // added last term for viscous correction term
    // Adding rotation: might need more

    std::vector<PetscInt> nnz(4 * N_Nodes, 0);
    for (PetscInt i = 0; i < 3 * (PetscInt)N_Nodes; ++i) nnz[i] = nnz_vel;
    for (PetscInt i = 3 * (PetscInt)N_Nodes; i < 4 * (PetscInt)N_Nodes; ++i) nnz[i] = nnz_pres;

    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,
                              4 * N_Nodes, 4 * N_Nodes,
                              0, nnz.data(), Apattern));

    // PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,
    //                           4 * N_Nodes, 4 * N_Nodes,
    //                           60 * Np, nullptr, Apattern));
    std::cout << "Apattern pre-allocated" << std::endl;
    PetscCall(FormMatrixLinearPart(*Apattern,
                     ops.GRAD,
                     ops.DIVdx, ops.DIVdy, ops.DIVdz,
                     ops.ConstantRotationalMat,
                     ops.ConstantRotationalMatx, ops.ConstantRotationalMaty, ops.ConstantRotationalMatz,
                     ops.Laplacian,
                     user.Delta_t, N_Nodes,
                     user.opt.viscous, user.opt.Re,
                     user.opt.include_viscous_divergence_correction));
    //std::cout << "Apattern: Linear part done" << std::endl;
    PetscCall(insertNonlinearVelocityPattern(
        *Apattern, List_Of_Elements, N_Nodes, Np));
    //std::cout << "Apattern: Nonlinear velocity part done" << std::endl;

    PetscCall(insertNonlinearPressurePattern(
        *Apattern, List_Of_Elements, ops, N_Nodes, Np));
    //std::cout << "Apattern: Nonlinear pressure part done" << std::endl;

    PetscCall(MatAssemblyBegin(*Apattern, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*Apattern, MAT_FINAL_ASSEMBLY));
    //std::cout << "Np = " << Np << ", nnz_vel = " << nnz_vel << ", nnz_pres = " << nnz_pres << std::endl;
    printMatrixNNZInfo(*Apattern, "Apattern");
    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode printMatrixNNZInfo(const Mat A, const std::string &name)
{
    PetscFunctionBegin;

    PetscInt m, n;
    PetscCall(MatGetSize(A, &m, &n));

    MatInfo info;
    PetscCall(MatGetInfo(A, MAT_GLOBAL_SUM, &info));

    PetscInt maxRowNnz = 0;
    PetscInt minRowNnz = PETSC_MAX_INT;
    long long sumRowNnz = 0;

    for (PetscInt i = 0; i < m; ++i)
    {
        PetscInt ncols;
        const PetscInt *cols;
        const PetscScalar *vals;

        PetscCall(MatGetRow(A, i, &ncols, &cols, &vals));
        maxRowNnz = std::max(maxRowNnz, ncols);
        minRowNnz = std::min(minRowNnz, ncols);
        sumRowNnz += static_cast<long long>(ncols);
        PetscCall(MatRestoreRow(A, i, &ncols, &cols, &vals));
    }

    const double avgRowNnz = (m > 0) ? static_cast<double>(sumRowNnz) / static_cast<double>(m) : 0.0;

    std::cout << "\nMatrix: " << name << "\n";
    std::cout << "  Size              : " << m << " x " << n << "\n";
    std::cout << "  Actual nnz        : " << sumRowNnz << "\n";
    std::cout << "  Avg nnz/row       : " << avgRowNnz << "\n";
    std::cout << "  Min nnz/row       : " << minRowNnz << "\n";
    std::cout << "  Max nnz/row       : " << maxRowNnz << "\n";
    std::cout << "  PETSc mallocs     : " << info.mallocs << "\n";
    std::cout << "  Memory            : " << info.memory << "\n";
    std::cout << "  NZ allocated      : " << info.nz_allocated << "\n";
    std::cout << "  NZ used           : " << info.nz_used << "\n";
    std::cout << "  NZ unneeded       : " << info.nz_unneeded << "\n";

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode insertNonlinearVelocityPattern(
    Mat Apattern,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    const PetscInt N_Nodes,
    const PetscInt Np)
{
    PetscFunctionBegin;

    for (const auto &elem : List_Of_Elements)
    {
        const PetscInt pos = static_cast<PetscInt>(elem->get_pos());

        // A_UV
        PetscCall(insertDenseElementBlockPattern(Apattern, 0,         N_Nodes,   pos, Np));
        // A_UW
        PetscCall(insertDenseElementBlockPattern(Apattern, 0,         2*N_Nodes, pos, Np));
        // A_VU
        PetscCall(insertDenseElementBlockPattern(Apattern, N_Nodes,   0,         pos, Np));
        // A_VW
        PetscCall(insertDenseElementBlockPattern(Apattern, N_Nodes,   2*N_Nodes, pos, Np));
        // A_WU
        PetscCall(insertDenseElementBlockPattern(Apattern, 2*N_Nodes, 0,         pos, Np));
        // A_WV
        PetscCall(insertDenseElementBlockPattern(Apattern, 2*N_Nodes, N_Nodes,   pos, Np));
    }

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode insertDenseElementBlockPattern(
    Mat A,
    PetscInt row_offset,
    PetscInt col_offset,
    PetscInt elem_pos,
    PetscInt Np)
{
    PetscFunctionBegin;

    std::vector<PetscInt> rows(Np), cols(Np);
    std::vector<PetscScalar> vals(Np * Np, 1.0);

    for (PetscInt i = 0; i < Np; ++i) {
        rows[i] = row_offset + elem_pos + i;
        cols[i] = col_offset + elem_pos + i;
    }

    PetscCall(MatSetValues(A,
                           Np, rows.data(),
                           Np, cols.data(),
                           vals.data(),
                           ADD_VALUES));

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode buildNodeToElemPos(
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    PetscInt N_Nodes,
    PetscInt Np,
    std::vector<PetscInt> &node_to_elem_pos)
{
    PetscFunctionBegin;

    node_to_elem_pos.assign(N_Nodes, -1);

    for (const auto &elem : List_Of_Elements)
    {
        const PetscInt pos = static_cast<PetscInt>(elem->get_pos());
        for (PetscInt a = 0; a < Np; ++a) {
            node_to_elem_pos[pos + a] = pos;
        }
    }

    for (PetscInt i = 0; i < N_Nodes; ++i) {
        if (node_to_elem_pos[i] < 0) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                    "node_to_elem_pos incomplete: some node has no owning element block");
        }
    }

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode insertPressureBlockPatternFromDivRow(
    Mat A,
    const Mat DIV,
    PetscInt div_row,
    PetscInt pressure_row,
    PetscInt target_col_offset,              // 0, N_Nodes, or 2*N_Nodes
    PetscInt Np,
    const std::vector<PetscInt> &node_to_elem_pos)
{
    PetscFunctionBegin;

    PetscInt ncols;
    const PetscInt *cols = nullptr;

    PetscCall(MatGetRow(DIV, div_row, &ncols, &cols, nullptr));

    std::vector<PetscInt> block_cols(Np);
    std::vector<PetscScalar> vals(Np, 1.0);

    // avoid inserting the same element block multiple times for this row
    std::unordered_set<PetscInt> touched_elem_pos;
    touched_elem_pos.reserve(static_cast<std::size_t>(ncols));

    for (PetscInt j = 0; j < ncols; ++j)
    {
        const PetscInt k = cols[j];
        const PetscInt elem_pos = node_to_elem_pos[k];

        if (touched_elem_pos.insert(elem_pos).second)
        {
            for (PetscInt a = 0; a < Np; ++a) {
                block_cols[a] = target_col_offset + elem_pos + a;
            }

            PetscCall(MatSetValues(A,
                                   1, &pressure_row,
                                   Np, block_cols.data(),
                                   vals.data(),
                                   ADD_VALUES));
        }
    }

    PetscCall(MatRestoreRow(DIV, div_row, &ncols, &cols, nullptr));

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode insertNonlinearPressurePattern(
    Mat Apattern,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    const DiscreteOperators &ops,
    const PetscInt N_Nodes,
    const PetscInt Np)
{
    PetscFunctionBegin;

    std::vector<PetscInt> node_to_elem_pos;
    PetscCall(buildNodeToElemPos(
        List_Of_Elements, N_Nodes, Np, node_to_elem_pos));

    for (PetscInt i = 0; i < N_Nodes; ++i)
    {
        const PetscInt prow = 3 * N_Nodes + i;

        // ----------------------------------------------------
        // pU block: A_pU = DIVdy * A_VU + DIVdz * A_WU
        // target columns are in U block [0, N_Nodes)
        // ----------------------------------------------------
        PetscCall(insertPressureBlockPatternFromDivRow(
            Apattern, ops.DIVdy, i, prow, 0, Np, node_to_elem_pos));
        PetscCall(insertPressureBlockPatternFromDivRow(
            Apattern, ops.DIVdz, i, prow, 0, Np, node_to_elem_pos));

        // ----------------------------------------------------
        // pV block: A_pV = DIVdx * A_UV + DIVdz * A_WV
        // target columns are in V block [N_Nodes, 2*N_Nodes)
        // ----------------------------------------------------
        PetscCall(insertPressureBlockPatternFromDivRow(
            Apattern, ops.DIVdx, i, prow, N_Nodes, Np, node_to_elem_pos));
        PetscCall(insertPressureBlockPatternFromDivRow(
            Apattern, ops.DIVdz, i, prow, N_Nodes, Np, node_to_elem_pos));

        // ----------------------------------------------------
        // pW block: A_pW = DIVdx * A_UW + DIVdy * A_VW
        // target columns are in W block [2*N_Nodes, 3*N_Nodes)
        // ----------------------------------------------------
        PetscCall(insertPressureBlockPatternFromDivRow(
            Apattern, ops.DIVdx, i, prow, 2 * N_Nodes, Np, node_to_elem_pos));
        PetscCall(insertPressureBlockPatternFromDivRow(
            Apattern, ops.DIVdy, i, prow, 2 * N_Nodes, Np, node_to_elem_pos));
    }

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode denseMatVecAdd(
    PetscInt Np,
    const PetscScalar *A,   // Np x Np, row-major
    const PetscScalar *x,   // length Np
    PetscScalar alpha,
    PetscScalar *y)         // length Np, accumulated
{
    PetscFunctionBegin;

    for (PetscInt i = 0; i < Np; ++i) {
        PetscScalar sum = 0.0;
        for (PetscInt j = 0; j < Np; ++j) {
            sum += A[i * Np + j] * x[j];
        }
        y[i] += alpha * sum;
    }

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode assembleTotalRHS(Vec b, const Vec blinear, const Vec bnonlinear)
{
    PetscFunctionBegin;
    PetscCall(VecCopy(blinear, b));
    PetscCall(VecAXPY(b, 1.0, bnonlinear));
    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode addPressureContributionFromDiv(
    Mat Anonlinear,
    const Mat DIV,
    PetscInt div_row,
    PetscInt pressure_row,
    PetscInt target_col_offset,
    PetscInt Np,
    const std::vector<PetscInt> &node_to_elem_idx,
    const std::vector<PetscInt> &elem_pos_list,
    const std::vector<std::vector<PetscScalar>> &localBlocks,
    PetscScalar alpha,
    PicardWorkspace &work)
{
    PetscFunctionBegin;

    PetscInt ncols = 0;
    const PetscInt *cols = nullptr;
    const PetscScalar *vals = nullptr;

    PetscCall(MatGetRow(DIV, div_row, &ncols, &cols, &vals));

    work.pressure_touched_elems.clear();

    /*
       We use pressure_touched_mark[eidx] to store the compact local index
       of element eidx in pressure_touched_elems.

       -1 means this element has not yet been touched in this row.
    */

    for (PetscInt j = 0; j < ncols; ++j)
    {
        const PetscInt k_global = cols[j];
        const PetscScalar dval  = vals[j];

        const PetscInt eidx = node_to_elem_idx[k_global];

        // PetscCheck(eidx >= 0,
        //            PETSC_COMM_SELF,
        //            PETSC_ERR_ARG_OUTOFRANGE,
        //            "addPressureContributionFromDiv: invalid node_to_elem_idx");

        PetscInt compact_idx = work.pressure_touched_mark[eidx];

        if (compact_idx < 0)
        {
            compact_idx = static_cast<PetscInt>(work.pressure_touched_elems.size());
            work.pressure_touched_elems.push_back(eidx);
            work.pressure_touched_mark[eidx] = compact_idx;

            const PetscInt required_size =
                static_cast<PetscInt>(work.pressure_touched_elems.size()) * Np;

            if (static_cast<PetscInt>(work.pressure_accum.size()) < required_size) {
                work.pressure_accum.resize(required_size);
            }

            PetscScalar *acc = work.pressure_accum.data() + compact_idx * Np;
            for (PetscInt c = 0; c < Np; ++c) {
                acc[c] = 0.0;
            }
        }

        const PetscInt elem_pos = elem_pos_list[eidx];
        const PetscInt k_local  = k_global - elem_pos;

        // PetscCheck(k_local >= 0 && k_local < Np,
        //            PETSC_COMM_SELF,
        //            PETSC_ERR_ARG_OUTOFRANGE,
        //            "addPressureContributionFromDiv: invalid local node index");

        const auto &Ke = localBlocks[eidx];

        PetscScalar *acc = work.pressure_accum.data() + compact_idx * Np;

        const PetscScalar scale = alpha * dval;
        const PetscScalar *Ke_row = Ke.data() + k_local * Np;

        for (PetscInt c = 0; c < Np; ++c) {
            acc[c] += scale * Ke_row[c];
        }
    }

    PetscCall(MatRestoreRow(DIV, div_row, &ncols, &cols, &vals));

    for (PetscInt q = 0; q < static_cast<PetscInt>(work.pressure_touched_elems.size()); ++q)
    {
        const PetscInt eidx = work.pressure_touched_elems[q];
        const PetscInt elem_pos = elem_pos_list[eidx];

        for (PetscInt c = 0; c < Np; ++c) {
            work.pressure_cols_out[c] = target_col_offset + elem_pos + c;
        }

        PetscScalar *acc = work.pressure_accum.data() + q * Np;

        PetscCall(MatSetValues(
            Anonlinear,
            1, &pressure_row,
            Np, work.pressure_cols_out.data(),
            acc,
            ADD_VALUES));
    }

    /*
       Reset marks only for elements touched in this call.
       This avoids O(NElements) clearing.
    */
    for (PetscInt eidx : work.pressure_touched_elems) {
        work.pressure_touched_mark[eidx] = -1;
    }

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode buildLocalProductMatrixFromTensor(
    const TripleProductTensor1D &G1D,
    const PetscScalar *a_local,   // length Np, x-fastest ordering
    PetscInt Np,
    PetscScalar J,
    PetscScalar *Pe)              // output: Np*Np dense
{
    PetscFunctionBegin;

    const unsigned int n1 = G1D.n1;

    auto idx3 = [n1](unsigned int ix, unsigned int iy, unsigned int iz) -> unsigned int {
        return iz * n1 * n1 + iy * n1 + ix;
    };

    // B(ix,jx,ky,kz)
    auto idx4 = [n1](unsigned int ix, unsigned int jx,
                     unsigned int ky, unsigned int kz) -> unsigned int {
        return (((kz * n1 + ky) * n1 + jx) * n1 + ix);
    };

    // C(ix,jx,iy,jy,kz)
    auto idx5 = [n1](unsigned int ix, unsigned int jx,
                     unsigned int iy, unsigned int jy,
                     unsigned int kz) -> unsigned int {
        return ((((kz * n1 + jy) * n1 + iy) * n1 + jx) * n1 + ix);
    };

    std::vector<PetscScalar> B(n1*n1*n1*n1, 0.0);
    std::vector<PetscScalar> C(n1*n1*n1*n1*n1, 0.0);

    std::fill(Pe, Pe + Np * Np, 0.0);

    // ------------------------------------------------------------
    // First contraction in x:
    //
    // B(ix,jx,ky,kz) =
    //     sum_kx G(ix,kx,jx) a(kx,ky,kz)
    // ------------------------------------------------------------
    for (unsigned int ix = 0; ix < n1; ++ix)
    for (unsigned int jx = 0; jx < n1; ++jx)
    for (unsigned int ky = 0; ky < n1; ++ky)
    for (unsigned int kz = 0; kz < n1; ++kz)
    {
        PetscScalar val = 0.0;

        for (unsigned int kx = 0; kx < n1; ++kx) {
            val += G1D(ix, kx, jx) * a_local[idx3(kx, ky, kz)];
        }

        B[idx4(ix, jx, ky, kz)] = val;
    }

    // ------------------------------------------------------------
    // Second contraction in y:
    //
    // C(ix,jx,iy,jy,kz) =
    //     sum_ky G(iy,ky,jy) B(ix,jx,ky,kz)
    // ------------------------------------------------------------
    for (unsigned int ix = 0; ix < n1; ++ix)
    for (unsigned int jx = 0; jx < n1; ++jx)
    for (unsigned int iy = 0; iy < n1; ++iy)
    for (unsigned int jy = 0; jy < n1; ++jy)
    for (unsigned int kz = 0; kz < n1; ++kz)
    {
        PetscScalar val = 0.0;

        for (unsigned int ky = 0; ky < n1; ++ky) {
            val += G1D(iy, ky, jy) * B[idx4(ix, jx, ky, kz)];
        }

        C[idx5(ix, jx, iy, jy, kz)] = val;
    }

    // ------------------------------------------------------------
    // Third contraction in z:
    //
    // Pe[(ix,iy,iz),(jx,jy,jz)] =
    //     J sum_kz G(iz,kz,jz) C(ix,jx,iy,jy,kz)
    // ------------------------------------------------------------
    for (unsigned int ix = 0; ix < n1; ++ix)
    for (unsigned int iy = 0; iy < n1; ++iy)
    for (unsigned int iz = 0; iz < n1; ++iz)
    {
        const unsigned int i_loc = idx3(ix, iy, iz);

        for (unsigned int jx = 0; jx < n1; ++jx)
        for (unsigned int jy = 0; jy < n1; ++jy)
        for (unsigned int jz = 0; jz < n1; ++jz)
        {
            const unsigned int j_loc = idx3(jx, jy, jz);

            PetscScalar val = 0.0;

            for (unsigned int kz = 0; kz < n1; ++kz) {
                val += G1D(iz, kz, jz) * C[idx5(ix, jx, iy, jy, kz)];
            }

            Pe[i_loc * Np + j_loc] = J * val;
        }
    }

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode denseMatMat(
    PetscInt Np,
    const PetscScalar *A,
    const PetscScalar *B,
    PetscScalar *C)
    {
        PetscFunctionBegin;

        const PetscInt n2 = Np * Np;

        for (PetscInt q = 0; q < n2; ++q) {
            C[q] = 0.0;
        }

        for (PetscInt i = 0; i < Np; ++i)
        {
            const PetscScalar *Ai = A + i * Np;
            PetscScalar       *Ci = C + i * Np;

            for (PetscInt k = 0; k < Np; ++k)
            {
                const PetscScalar aik = Ai[k];
                const PetscScalar *Bk = B + k * Np;

                for (PetscInt j = 0; j < Np; ++j) {
                    Ci[j] += aik * Bk[j];
                }
            }
        }

        PetscFunctionReturn(0);
    }
/*--------------------------------------------------------------------------*/
PetscErrorCode addDenseElementBlockValues(
    Mat A,
    PetscInt row_offset,
    PetscInt col_offset,
    PetscInt elem_pos,
    PetscInt Np,
    PetscScalar alpha,
    const PetscScalar *Ke,
    PicardWorkspace &work)
{
    PetscFunctionBegin;

    for (PetscInt i = 0; i < Np; ++i) {
        work.dense_rows[i] = row_offset + elem_pos + i;
        work.dense_cols[i] = col_offset + elem_pos + i;
    }

    const PetscInt nvals = Np * Np;
    for (PetscInt k = 0; k < nvals; ++k) {
        work.dense_vals[k] = alpha * Ke[k];
    }

    PetscCall(MatSetValues(
        A,
        Np, work.dense_rows.data(),
        Np, work.dense_cols.data(),
        work.dense_vals.data(),
        ADD_VALUES));

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode createTripleProductTensor1D(
    const unsigned int N_Order,
    TripleProductTensor1D &G1D)
{
    PetscFunctionBegin;

    const unsigned int n1 = N_Order + 1;

    G1D.resize(N_Order);

    // GLL interpolation nodes
    Vec ri = JacobiGL(0, 0, N_Order);
    // Quadrature order for exact triple products
    const unsigned int Q = static_cast<unsigned int>(std::ceil((3.0 * N_Order + 3.0))/2.0);
        //2.0* factor 2 to be safe

    Vec Weights = nullptr, QuadraturePoints = nullptr;
    QuadraturePoints = JacobiGL_withWeights(0, 0, Q, Weights);

    PetscScalar *w  = nullptr;
    PetscScalar *qp = nullptr;

    PetscCall(VecGetArray(Weights, &w));
    PetscCall(VecGetArray(QuadraturePoints, &qp));

    // -------------------------------------------------------------------------
    // Precompute basis values L(a,p) = l_a(qp[p])
    // size = n1 x (Q+1)
    // -------------------------------------------------------------------------
    std::vector<PetscScalar> L(n1 * (Q + 1), 0.0);

    auto Lidx = [Q](unsigned int a, unsigned int p) -> unsigned int {
        return a * (Q + 1) + p;
    };

    for (unsigned int a = 0; a < n1; ++a)
    {
        for (unsigned int p = 0; p <= Q; ++p)
        {
            L[Lidx(a, p)] = LagrangePolynomial(ri, qp[p], a);
        }
    }

    // -------------------------------------------------------------------------
    // Build G(a,b,c) = sum_p w[p] * L(a,p)*L(b,p)*L(c,p)
    // -------------------------------------------------------------------------
    for (unsigned int a = 0; a < n1; ++a)
    {
        for (unsigned int b = 0; b < n1; ++b)
        {
            for (unsigned int c = 0; c < n1; ++c)
            {
                PetscScalar val = 0.0;
                for (unsigned int p = 0; p <= Q; ++p)
                {
                    val += w[p]
                        * L[Lidx(a, p)]
                        * L[Lidx(b, p)]
                        * L[Lidx(c, p)];
                }
                G1D(a, b, c) = val;
            }
        }
    }

    PetscCall(VecRestoreArray(Weights, &w));
    PetscCall(VecRestoreArray(QuadraturePoints, &qp));
    PetscCall(VecDestroy(&ri));
    PetscCall(VecDestroy(&Weights));
    PetscCall(VecDestroy(&QuadraturePoints));

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
unsigned int numberOfNonzerosFromOrder(const unsigned int &Order)
{
  if (Order==0) return 7;
  unsigned int n = Order+1;
  return n*n*n+3*n*n; // (Order+1)^3 (interior) + 3 (Order+1)^2 (6 faces)
}
/*--------------------------------------------------------------------------*/
PetscErrorCode create_Matrices_Cuboids_Inertial(
  const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
  const std::vector<std::unique_ptr<Boundary>> &List_Of_Boundaries,
  const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
  unsigned int N_Nodes,
  unsigned int N_Order,
  InertialRawMatrices &raw,
  const RotationParameters &rotation)
{
  PetscFunctionBegin;
  // Number of nonzeros per row for mass matrix
  unsigned int Np1 = (N_Order+1)*(N_Order+1)*(N_Order+1);
  unsigned int NonzerosE = numberOfNonzerosFromOrder(N_Order);
  MatCreateSeqAIJ(PETSC_COMM_SELF, 3*N_Nodes, N_Nodes, NonzerosE, NULL, &raw.E);
  MatCreateSeqAIJ(PETSC_COMM_SELF, N_Nodes, 3*N_Nodes, 3*NonzerosE, NULL, &raw.ET);
  MatCreateSeqAIJ(PETSC_COMM_SELF, N_Nodes, N_Nodes, NonzerosE, NULL, &raw.Edx);
  MatCreateSeqAIJ(PETSC_COMM_SELF, N_Nodes, N_Nodes, NonzerosE, NULL, &raw.Edy);
  MatCreateSeqAIJ(PETSC_COMM_SELF, N_Nodes, N_Nodes, NonzerosE, NULL, &raw.Edz);
  MatCreateSeqAIJ(PETSC_COMM_SELF, N_Nodes, N_Nodes, NonzerosE, NULL, &raw.ETdx);
  MatCreateSeqAIJ(PETSC_COMM_SELF, N_Nodes, N_Nodes, NonzerosE, NULL, &raw.ETdy);
  MatCreateSeqAIJ(PETSC_COMM_SELF, N_Nodes, N_Nodes, NonzerosE, NULL, &raw.ETdz);
  MatCreateSeqAIJ(PETSC_COMM_SELF, 3*N_Nodes, 3*N_Nodes, Np1, NULL, &raw.invM);
  MatCreateSeqAIJ(PETSC_COMM_SELF, N_Nodes, N_Nodes, Np1, NULL, &raw.invM_small);
  MatCreateSeqAIJ(PETSC_COMM_SELF, 3*N_Nodes, 3*N_Nodes, Np1, NULL, &raw.M);
  MatCreateSeqAIJ(PETSC_COMM_SELF, N_Nodes, N_Nodes, Np1, NULL, &raw.M_small);
  MatCreateSeqAIJ(PETSC_COMM_SELF, 3*N_Nodes, 3*N_Nodes, 2*Np1, NULL, &raw.ConstantRotational);
  MatCreateSeqAIJ(PETSC_COMM_SELF, N_Nodes, 3*N_Nodes, 2*Np1, NULL, &raw.ConstantRotationalx);
  MatCreateSeqAIJ(PETSC_COMM_SELF, N_Nodes, 3*N_Nodes, 2*Np1, NULL, &raw.ConstantRotationaly);
  MatCreateSeqAIJ(PETSC_COMM_SELF, N_Nodes, 3*N_Nodes, 2*Np1, NULL, &raw.ConstantRotationalz);

  // Weak DG curl matrix C = M*CURL.
  // This is assembled in the broken-strong form:
  //   (phi, CURL_h v) = sum_K (phi, curl v)_K
  //                    - sum_F <avg(phi), n x (v^- - v^+)>_F
  // for interior/periodic faces. For impermeable slip walls there is no
  // additional face correction in this broken-strong form.
  //
  // NOTE: InertialRawMatrices must contain:
  //     Mat Curlx;  // N_Nodes   x 3*N_Nodes, x-component of C = M*CURL
  //     Mat Curly;  // N_Nodes   x 3*N_Nodes, y-component of C = M*CURL
  //     Mat Curlz;  // N_Nodes   x 3*N_Nodes, z-component of C = M*CURL
  //     Mat Curl;   // 3*N_Nodes x 3*N_Nodes, stacked [Curlx; Curly; Curlz]
  //
  // The row nonzero estimates below are conservative. You may tune them later.
  MatCreateSeqAIJ(PETSC_COMM_SELF, N_Nodes,   3*N_Nodes, 6*NonzerosE, NULL, &raw.Curlx);
  MatCreateSeqAIJ(PETSC_COMM_SELF, N_Nodes,   3*N_Nodes, 6*NonzerosE, NULL, &raw.Curly);
  MatCreateSeqAIJ(PETSC_COMM_SELF, N_Nodes,   3*N_Nodes, 6*NonzerosE, NULL, &raw.Curlz);
  MatCreateSeqAIJ(PETSC_COMM_SELF, 3*N_Nodes, 3*N_Nodes, 9*NonzerosE, NULL, &raw.Curl);

  std::cout << "\n \033[1;32m Start Elemental Calculations\033[0m \n" << std::endl;
  // Assumption: order (and number of nodes) on each element, in each direction is constant
  unsigned int Np = List_Of_Elements.front()->get_Number_Of_Nodes();

  // Gauss-Lobatto quadrature of order n is exact for polynomials up to order 2n-3
  //mass matrix: \int phi_i phi_j dx. Integrand degree = 2 N_Order -> ceil((2 N_Order + 3)/2)
  unsigned int Order_Gaussian_Quadrature = (N_Order+3); // 2* Factor 2 to be safe

  Vec ri, Weights, QuadraturePoints;
  ri = JacobiGL(0, 0, N_Order);
  QuadraturePoints = JacobiGL_withWeights(0, 0, Order_Gaussian_Quadrature, Weights);
  PetscScalar *w, *qp;
  VecGetArray(Weights, &w);
  VecGetArray(QuadraturePoints, &qp);

  // Precompute
  const unsigned int n1 = N_Order + 1;
  const unsigned int nq = Order_Gaussian_Quadrature + 1;

  std::vector<double> L1D(nq * n1, 0.0);
  std::vector<double> dL1D(nq * n1, 0.0);

  auto idx1D = [n1](unsigned int qpt, unsigned int a) -> unsigned int {
      return qpt * n1 + a;
  };

  auto addCurlFaceBlock =
      [&](PetscInt rowNode,
          PetscInt colNode,
          PetscScalar coeff,
          PetscScalar faceMass,
          double nx,
          double ny,
          double nz)
  {
      // Adds coeff * <phi, n x v>_F to:
      //   raw.Curl  : full stacked operator C = M*CURL
      //   raw.Curlx : x-component rows of C
      //   raw.Curly : y-component rows of C
      //   raw.Curlz : z-component rows of C
      //
      // If n x v = (-nz*v_y + ny*v_z,
      //              nz*v_x - nx*v_z,
      //             -ny*v_x + nx*v_y),
      // then the 3x3 component block is:
      //
      // [  0   -nz    ny ] * coeff*faceMass
      // [  nz   0    -nx ]
      // [ -ny   nx    0  ]
      const PetscScalar a = coeff * faceMass;

      const PetscInt row_x_full = rowNode;
      const PetscInt row_y_full = N_Nodes + rowNode;
      const PetscInt row_z_full = 2*N_Nodes + rowNode;

      const PetscInt row_component = rowNode;

      const PetscInt col_x = colNode;
      const PetscInt col_y = N_Nodes + colNode;
      const PetscInt col_z = 2*N_Nodes + colNode;

      // x-component: -nz*v_y + ny*v_z
      PetscCallAbort(PETSC_COMM_SELF, MatSetValue(raw.Curl,  row_x_full,  col_y, -nz*a, ADD_VALUES));
      PetscCallAbort(PETSC_COMM_SELF, MatSetValue(raw.Curl,  row_x_full,  col_z,  ny*a, ADD_VALUES));
      PetscCallAbort(PETSC_COMM_SELF, MatSetValue(raw.Curlx, row_component, col_y, -nz*a, ADD_VALUES));
      PetscCallAbort(PETSC_COMM_SELF, MatSetValue(raw.Curlx, row_component, col_z,  ny*a, ADD_VALUES));

      // y-component: nz*v_x - nx*v_z
      PetscCallAbort(PETSC_COMM_SELF, MatSetValue(raw.Curl,  row_y_full,  col_x,  nz*a, ADD_VALUES));
      PetscCallAbort(PETSC_COMM_SELF, MatSetValue(raw.Curl,  row_y_full,  col_z, -nx*a, ADD_VALUES));
      PetscCallAbort(PETSC_COMM_SELF, MatSetValue(raw.Curly, row_component, col_x,  nz*a, ADD_VALUES));
      PetscCallAbort(PETSC_COMM_SELF, MatSetValue(raw.Curly, row_component, col_z, -nx*a, ADD_VALUES));

      // z-component: -ny*v_x + nx*v_y
      PetscCallAbort(PETSC_COMM_SELF, MatSetValue(raw.Curl,  row_z_full,  col_x, -ny*a, ADD_VALUES));
      PetscCallAbort(PETSC_COMM_SELF, MatSetValue(raw.Curl,  row_z_full,  col_y,  nx*a, ADD_VALUES));
      PetscCallAbort(PETSC_COMM_SELF, MatSetValue(raw.Curlz, row_component, col_x, -ny*a, ADD_VALUES));
      PetscCallAbort(PETSC_COMM_SELF, MatSetValue(raw.Curlz, row_component, col_y,  nx*a, ADD_VALUES));
  };

  for (unsigned int p = 0; p < nq; ++p)
  {
      for (unsigned int a = 0; a < n1; ++a)
      {
          L1D[idx1D(p, a)]  = LagrangePolynomial(ri, qp[p], a);
          dL1D[idx1D(p, a)] = (N_Order > 0) ? LagrangePolynomialDeriv(ri, qp[p], a) : 0.0;
      }
  }

   for (auto e = List_Of_Elements.begin(); e < List_Of_Elements.end(); e++)
   {
       unsigned int pos = (*e)->get_pos();

       Mat M_Elemental;
       Mat invM_Elemental;
       MatCreateSeqAIJ(PETSC_COMM_SELF, Np, Np, Np, NULL, &M_Elemental);

      double J, drdx, drdy, drdz, dsdx, dsdy, dsdz, dtdx, dtdy, dtdz, x, y, z;
      Calculate_Jacobian_Cuboid((*e), List_Of_Vertices, 0.0, 0.0, 0.0, J, drdx, drdy, drdz, dsdx, dsdy, dsdz, dtdx, dtdy, dtdz, x, y, z);

      (*e)->set_detJ(J); // affine cuboids, these are constant.

       for (unsigned int k = 1; k <= Np; k++)
       {
           unsigned int alpha = (k-1)%(N_Order+1);            // r
           unsigned int kk = (k-1)/(N_Order+1);
           unsigned int beta = (kk)%(N_Order+1);              // s
           unsigned int zeta = (k-1)/((N_Order+1)*(N_Order+1));    // t
           for (unsigned int l = 1; l <= Np; l++)
           {
               //std::cout << "k = " << k << ", l = " << l << std::endl;
               unsigned int gamma = (l-1)%(N_Order+1);
               unsigned int ll = (l-1)/(N_Order+1);
               unsigned int delta = (ll)%(N_Order+1);
               unsigned int epsilon = (l-1)/((N_Order+1)*(N_Order+1));

               double value_ex = 0.0, value_ey = 0.0, value_ez = 0.0, value_m = 0.0;

               for (unsigned int p = 0; p <= Order_Gaussian_Quadrature; p++)
               {
                   // double L_alpha = LagrangePolynomial(ri, qp[p], alpha);
                   // double L_gamma = LagrangePolynomial(ri, qp[p], gamma);
                   // Lookup table:
                   const double L_alpha = L1D[idx1D(p, alpha)];
                   const double L_gamma = L1D[idx1D(p, gamma)];
                   for (unsigned int q = 0; q <= Order_Gaussian_Quadrature; q++)
                   {
                       // double L_beta  = LagrangePolynomial(ri, qp[q], beta);
                       // double L_delta = LagrangePolynomial(ri, qp[q], delta);
                       const double L_beta  = L1D[idx1D(q, beta)];
                       const double L_delta = L1D[idx1D(q, delta)];
                       for (unsigned int r = 0; r <= Order_Gaussian_Quadrature; r++)
                       {
                           // double L_zeta    = LagrangePolynomial(ri, qp[r], zeta);
                           // double L_epsilon = LagrangePolynomial(ri, qp[r], epsilon);
                           const double L_zeta    = L1D[idx1D(r, zeta)];
                           const double L_epsilon = L1D[idx1D(r, epsilon)];
                           //double J, drdx, drdy, drdz, dsdx, dsdy, dsdz, dtdx, dtdy, dtdz, x, y, z;
                           //Calculate_Jacobian_Cuboid((*e), List_Of_Vertices, qp[p], qp[q], qp[r], J, drdx, drdy, drdz, dsdx, dsdy, dsdz, dtdx, dtdy, dtdz, x, y, z);
                           // Perhaps rewrite code, the only part that depends on the element is this function
// std::cout << "pos = " << pos
//           << ", J = " << J
//           << ", drdx = " << drdx
//           << ", drdy = " << drdy
//           << ", drdz = " << drdz
//           << ", dsdx = " << dsdx
//           << ", dsdy = " << dsdy
//           << ", dsdz = " << dsdz
//           << ", dtdx = " << dtdx
//           << ", dtdy = " << dtdy
//           << ", dtdz = " << dtdz
//           << std::endl;
                           value_m  += w[p]*L_alpha*L_gamma * w[q]*L_beta*L_delta * w[r]*L_zeta*L_epsilon * J;
                           if (N_Order > 0)
                           {
                               // double dL_gamma   = LagrangePolynomialDeriv(ri, qp[p], gamma);
                               // double dL_delta   = LagrangePolynomialDeriv(ri, qp[q], delta);
                               // double dL_epsilon = LagrangePolynomialDeriv(ri, qp[r], epsilon);
                               const double dL_gamma   = dL1D[idx1D(p, gamma)];
                               const double dL_delta   = dL1D[idx1D(q, delta)];
                               const double dL_epsilon = dL1D[idx1D(r, epsilon)];
                               value_ex += w[p]*w[q]*w[r] * L_alpha*L_beta*L_zeta * (drdx*dL_gamma * L_delta * L_epsilon + dsdx * L_gamma * dL_delta * L_epsilon + dtdx * L_gamma * L_delta * dL_epsilon) * J;
                               value_ey += w[p]*w[q]*w[r] * L_alpha*L_beta*L_zeta * (drdy*dL_gamma * L_delta * L_epsilon + dsdy * L_gamma * dL_delta * L_epsilon + dtdy * L_gamma * L_delta * dL_epsilon) * J;
                               value_ez += w[p]*w[q]*w[r] * L_alpha*L_beta*L_zeta * (drdz*dL_gamma * L_delta * L_epsilon + dsdz * L_gamma * dL_delta * L_epsilon + dtdz * L_gamma * L_delta * dL_epsilon) * J;
                           }
                       }
                   }
               }
               if (value_m != 0.0)
               {
               MatSetValue(M_Elemental, (k-1), (l-1), value_m, ADD_VALUES);
               MatSetValue(raw.M_small, pos+(k-1), pos+(l-1), value_m, ADD_VALUES);
               MatSetValue(raw.M, pos+(k-1), pos+(l-1), value_m, ADD_VALUES);
               MatSetValue(raw.M, N_Nodes+pos+(k-1), N_Nodes+pos+(l-1), value_m, ADD_VALUES);
               MatSetValue(raw.M, 2*N_Nodes+pos+(k-1), 2*N_Nodes+pos+(l-1), value_m, ADD_VALUES);
               if (rotation.include_rotation)
               {
               // Assumes f1, f2 and f3 are constant
                 MatSetValue(raw.ConstantRotational, pos+(k-1), N_Nodes+pos+(l-1), value_m*-rotation.f3/rotation.RossbyNumber, ADD_VALUES);
                 MatSetValue(raw.ConstantRotational, pos+(k-1), 2*N_Nodes+pos+(l-1), value_m*rotation.f2/rotation.RossbyNumber, ADD_VALUES);
                 MatSetValue(raw.ConstantRotationalx, pos+(k-1), N_Nodes+pos+(l-1), value_m*-rotation.f3/rotation.RossbyNumber, ADD_VALUES);
                 MatSetValue(raw.ConstantRotationalx, pos+(k-1), 2*N_Nodes+pos+(l-1), value_m*rotation.f2/rotation.RossbyNumber, ADD_VALUES);
                 MatSetValue(raw.ConstantRotational, N_Nodes+pos+(k-1), pos+(l-1), value_m*rotation.f3/rotation.RossbyNumber, ADD_VALUES);
                 MatSetValue(raw.ConstantRotational, N_Nodes+pos+(k-1), 2*N_Nodes+pos+(l-1), value_m*-rotation.f1/rotation.RossbyNumber, ADD_VALUES);
                 MatSetValue(raw.ConstantRotationaly, pos+(k-1), pos+(l-1), value_m*rotation.f3/rotation.RossbyNumber, ADD_VALUES);
                 MatSetValue(raw.ConstantRotationaly, pos+(k-1), 2*N_Nodes+pos+(l-1), value_m*-rotation.f1/rotation.RossbyNumber, ADD_VALUES);
                 MatSetValue(raw.ConstantRotational, 2*N_Nodes+pos+(k-1), pos+(l-1), value_m*-rotation.f2/rotation.RossbyNumber, ADD_VALUES);
                 MatSetValue(raw.ConstantRotational, 2*N_Nodes+pos+(k-1), N_Nodes+pos+(l-1), value_m*rotation.f1/rotation.RossbyNumber, ADD_VALUES);
                 MatSetValue(raw.ConstantRotationalz, pos+(k-1), pos+(l-1), value_m*-rotation.f2/rotation.RossbyNumber, ADD_VALUES);
                 MatSetValue(raw.ConstantRotationalz, pos+(k-1), N_Nodes+pos+(l-1), value_m*rotation.f1/rotation.RossbyNumber, ADD_VALUES);
               }
               }
               if (N_Order > 0)
               {
               MatSetValue(raw.E, pos+(k-1), pos+(l-1), value_ex, ADD_VALUES);
               MatSetValue(raw.Edx, pos+(k-1), pos+(l-1), value_ex, ADD_VALUES);
               MatSetValue(raw.E, N_Nodes+pos+(k-1), pos+(l-1), value_ey, ADD_VALUES);
               MatSetValue(raw.Edy, pos+(k-1), pos+(l-1), value_ey, ADD_VALUES);
               MatSetValue(raw.E, 2*N_Nodes+pos+(k-1), pos+(l-1), value_ez, ADD_VALUES);
               MatSetValue(raw.Edz, pos+(k-1), pos+(l-1), value_ez, ADD_VALUES);
               MatSetValue(raw.ET, pos+(l-1), pos+(k-1), -value_ex, ADD_VALUES);
               MatSetValue(raw.ETdx, pos+(l-1), pos+(k-1), -value_ex, ADD_VALUES);
               MatSetValue(raw.ET, pos+(l-1), N_Nodes+pos+(k-1), -value_ey, ADD_VALUES);
               MatSetValue(raw.ETdy, pos+(l-1), pos+(k-1), -value_ey, ADD_VALUES);
               MatSetValue(raw.ET, pos+(l-1), 2*N_Nodes+pos+(k-1), -value_ez, ADD_VALUES);
               MatSetValue(raw.ETdz, pos+(l-1), pos+(k-1), -value_ez, ADD_VALUES);

               // Volume part of C = M*CURL in broken-strong form:
               //   (phi, curl v)_K.
               //
               // curl(v)_x = d_y v_z - d_z v_y
               // curl(v)_y = d_z v_x - d_x v_z
               // curl(v)_z = d_x v_y - d_y v_x
               const PetscInt row_x = pos + (k-1);
               const PetscInt row_y = N_Nodes + pos + (k-1);
               const PetscInt row_z = 2*N_Nodes + pos + (k-1);

               const PetscInt col_x = pos + (l-1);
               const PetscInt col_y = N_Nodes + pos + (l-1);
               const PetscInt col_z = 2*N_Nodes + pos + (l-1);

               MatSetValue(raw.Curl, row_x, col_y, -value_ez, ADD_VALUES);
               MatSetValue(raw.Curl, row_x, col_z,  value_ey, ADD_VALUES);
               MatSetValue(raw.Curlx, pos+(k-1), col_y, -value_ez, ADD_VALUES);
               MatSetValue(raw.Curlx, pos+(k-1), col_z,  value_ey, ADD_VALUES);

               MatSetValue(raw.Curl, row_y, col_x,  value_ez, ADD_VALUES);
               MatSetValue(raw.Curl, row_y, col_z, -value_ex, ADD_VALUES);
               MatSetValue(raw.Curly, pos+(k-1), col_x,  value_ez, ADD_VALUES);
               MatSetValue(raw.Curly, pos+(k-1), col_z, -value_ex, ADD_VALUES);

               MatSetValue(raw.Curl, row_z, col_x, -value_ey, ADD_VALUES);
               MatSetValue(raw.Curl, row_z, col_y,  value_ex, ADD_VALUES);
               MatSetValue(raw.Curlz, pos+(k-1), col_x, -value_ey, ADD_VALUES);
               MatSetValue(raw.Curlz, pos+(k-1), col_y,  value_ex, ADD_VALUES);
               }
           }
       }
       // Construction of Inverse Mass Matrix
       MatAssemblyBegin(M_Elemental, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(M_Elemental, MAT_FINAL_ASSEMBLY);

       invM_Elemental = Inverse_Matrix(M_Elemental);

       MatDestroy(&M_Elemental);
      std::vector<PetscScalar> invM_local_vec(Np * Np, 0.0);

      for (PetscInt i = 0; i < (PetscInt)Np; i++)
      {
          PetscInt ncols;
          const PetscInt *cols;
          const PetscScalar *vals_invM;

          PetscCall(MatGetRow(invM_Elemental, i, &ncols, &cols, &vals_invM));

          const PetscInt IndexI = pos + i;
          std::vector<PetscInt> GlobalIndexCol(ncols), GlobalIndexCol2(ncols), GlobalIndexCol3(ncols);

          for (PetscInt j = 0; j < ncols; j++)
          {
              invM_local_vec[i * Np + cols[j]] = vals_invM[j];

              GlobalIndexCol[j]  = cols[j] + pos;
              GlobalIndexCol2[j] = N_Nodes + cols[j] + pos;
              GlobalIndexCol3[j] = 2 * N_Nodes + cols[j] + pos;
          }

          PetscInt GlobalIndex[1]  = {IndexI};
          PetscInt GlobalIndex2[1] = {N_Nodes + i + pos};
          PetscInt GlobalIndex3[1] = {2 * N_Nodes + i + pos};

          PetscCall(MatSetValues(raw.invM_small, 1, GlobalIndex,  ncols, GlobalIndexCol.data(),  vals_invM, ADD_VALUES));
          PetscCall(MatSetValues(raw.invM,       1, GlobalIndex,  ncols, GlobalIndexCol.data(),  vals_invM, ADD_VALUES));
          PetscCall(MatSetValues(raw.invM,       1, GlobalIndex2, ncols, GlobalIndexCol2.data(), vals_invM, ADD_VALUES));
          PetscCall(MatSetValues(raw.invM,       1, GlobalIndex3, ncols, GlobalIndexCol3.data(), vals_invM, ADD_VALUES));

          PetscCall(MatRestoreRow(invM_Elemental, i, &ncols, &cols, &vals_invM));
      }

      (*e)->set_invM_local(std::move(invM_local_vec));
       MatDestroy(&invM_Elemental);
//                     /// Order 5, 9, 13, 17 give a nonzero difference
//                     /// Independent of Order_Gaussian_Quadrature

   }

   MatAssemblyBegin(raw.M, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(raw.M, MAT_FINAL_ASSEMBLY);
   MatAssemblyBegin(raw.M_small, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(raw.M_small, MAT_FINAL_ASSEMBLY);
   MatAssemblyBegin(raw.invM, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(raw.invM, MAT_FINAL_ASSEMBLY);
   MatAssemblyBegin(raw.invM_small, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(raw.invM_small, MAT_FINAL_ASSEMBLY);
   MatAssemblyBegin(raw.ConstantRotational, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(raw.ConstantRotational, MAT_FINAL_ASSEMBLY);
   MatAssemblyBegin(raw.ConstantRotationalx, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(raw.ConstantRotationalx, MAT_FINAL_ASSEMBLY);
   MatAssemblyBegin(raw.ConstantRotationaly, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(raw.ConstantRotationaly, MAT_FINAL_ASSEMBLY);
   MatAssemblyBegin(raw.ConstantRotationalz, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(raw.ConstantRotationalz, MAT_FINAL_ASSEMBLY);

   std::cout << "\033[1;32m Start Boundary Calculations\033[0m \n" << std::endl;
   for (auto f = List_Of_Boundaries.begin(); f < List_Of_Boundaries.end(); f++)
   {
     // These boundaries are quadrilaterals
     double Jacobian = (*f)->getJacobian();
     unsigned int Type_Boundary_Left = (*f)->getTypeLeft(); unsigned int Type_Boundary_Right = (*f)->getTypeRight();
     double theta = (*f)->get_theta();
     double nx = (*f)->get_nx(); double ny = (*f)->get_ny(); double nz = (*f)->get_nz();

     auto left = (*f)->getLeftElementID(); auto right = (*f)->getRightElementID();
     //std::cout << "left = " << left << ", right = " << right << std::endl;
     //std::cout << "nx = " << nx << ", ny = " << ny << ", nz = " << nz << std::endl;


     unsigned int posL = (*List_Of_Elements[left]).get_pos();
     unsigned int posR = (*List_Of_Elements[right]).get_pos();

     std::vector<unsigned int> Node_Numbers_On_Boundary_Left = (*List_Of_Elements[left]).get_nodes_on_boundary(Type_Boundary_Left);
     std::vector<unsigned int> Node_Numbers_On_Boundary_Right = (*List_Of_Elements[right]).get_nodes_on_boundary(Type_Boundary_Right);

// //             //std::cout << "Nodes face 0 = "<< std::endl;
// //             //std::vector<unsigned int> values0 = (*List_Of_Elements[left]).get_node_on_face0();
// //             //for (auto const& value : values0)
// //             //{
// //             //    std::cout << value << std::endl;
// //             //}
// //             //std::cout << std::endl;
// //             //std::cout << std::endl;
// //
// //             //std::cout << "Nodes face 1 = "<< std::endl;
// //             //std::vector<unsigned int> values1 = (*List_Of_Elements[left]).get_node_on_face1();
// //             //for (auto const& value : values1)
// //             //{
// //             //    std::cout << value << std::endl;
// //             //}
// //             //std::cout << std::endl;
// //             //std::cout << std::endl;
// //
  // std::cout << "Node Numbers = "<< std::endl;
  // for (auto l = Node_Numbers_On_Boundary_Left.begin(); l < Node_Numbers_On_Boundary_Left.end(); l++)
  // {
  //    std::cout << (*l) << std::endl;
  // }
  // std::cout << std::endl;
  // std::cout << std::endl;
  auto size_left = Node_Numbers_On_Boundary_Left.size();
  auto size_right = Node_Numbers_On_Boundary_Right.size();

  // Curl face corrections are assembled only for faces with two physical traces:
  // interior faces and periodic face pairs.
  //
  // For impermeable slip walls, no additional face correction is added in the
  // broken-strong curl form. If your mesh represents a wall with a ghost element
  // or a mirrored right element, replace this condition by your actual boundary
  // classification, e.g.
  //
  //     const bool assembleCurlFaceCorrection = (*f)->isInterior() || (*f)->isPeriodic();
  //
  // The default below assumes a physical wall has no distinct right trace.
  const bool assembleCurlFaceCorrection = (right != left);

   // GLL
   for (unsigned int k = 1; k < size_left+1; k++)
   {
      //std::cout << "k = " <<  k << std::endl;
      unsigned int alpha = (k-1)%(N_Order+1);
      unsigned int beta = (k-1)/(N_Order+1);
      for (unsigned int l = 1; l < size_left+1; l++)
      {
           //std::cout << "l = " <<  l << std::endl;
           unsigned int gamma = (l-1)%(N_Order+1);
           unsigned int delta = (l-1)/(N_Order+1);
//                     //std::cout << alpha << " " << beta << " " << gamma << " " << delta << std::endl;
//                     // E Matrix
           double value_e = 0.0;
           double faceMass = 0.0;
//                     //std::cout << alpha << " " << beta << " " << gamma << " " << delta << std::endl;
           for (unsigned int p = 0; p <= Order_Gaussian_Quadrature; p++) //Order_Gaussian_Quadrature_L
           {
               //double L_alpha = LagrangePolynomial(ri, qp[p], alpha);
               //double L_gamma = LagrangePolynomial(ri, qp[p], gamma);
               const double L_alpha = L1D[idx1D(p, alpha)];
               const double L_gamma = L1D[idx1D(p, gamma)];
               for (unsigned int q = 0; q <= Order_Gaussian_Quadrature; q++) //Order_Gaussian_Quadrature_L
               {
                   //double L_beta  = LagrangePolynomial(ri, qp[q], beta);
                   //double L_delta = LagrangePolynomial(ri, qp[q], delta);
                   const double L_beta  = L1D[idx1D(q, beta)];
                   const double L_delta = L1D[idx1D(q, delta)];
                   const double s = w[p]*L_alpha*L_beta * w[q]*L_gamma*L_delta * Jacobian;
                   faceMass += s;
                   value_e += (1.0-theta) * s;
               }
           }
           //std::cout << "GLL: " << value_e << std::endl;
           unsigned int i = k - 1;
           unsigned int j = l - 1;
           if (nx != 0.0)
           {
           MatSetValue(raw.E,  posL+Node_Numbers_On_Boundary_Left[i], posL+Node_Numbers_On_Boundary_Left[j], -nx*value_e, ADD_VALUES);
           MatSetValue(raw.ET, posL+Node_Numbers_On_Boundary_Left[j], posL+Node_Numbers_On_Boundary_Left[i], nx*value_e, ADD_VALUES);
           MatSetValue(raw.Edx,  posL+Node_Numbers_On_Boundary_Left[i], posL+Node_Numbers_On_Boundary_Left[j], -nx*value_e, ADD_VALUES);
           MatSetValue(raw.ETdx, posL+Node_Numbers_On_Boundary_Left[j], posL+Node_Numbers_On_Boundary_Left[i], nx*value_e, ADD_VALUES);
           }
           if (ny != 0.0)
           {
           MatSetValue(raw.E,  N_Nodes+posL+Node_Numbers_On_Boundary_Left[i], posL+Node_Numbers_On_Boundary_Left[j], -ny*value_e, ADD_VALUES);
           MatSetValue(raw.ET, posL+Node_Numbers_On_Boundary_Left[j], N_Nodes+posL+Node_Numbers_On_Boundary_Left[i], ny*value_e, ADD_VALUES);
           MatSetValue(raw.Edy,  posL+Node_Numbers_On_Boundary_Left[i], posL+Node_Numbers_On_Boundary_Left[j], -ny*value_e, ADD_VALUES);
           MatSetValue(raw.ETdy, posL+Node_Numbers_On_Boundary_Left[j], posL+Node_Numbers_On_Boundary_Left[i], ny*value_e, ADD_VALUES);
           }
           if (nz != 0.0)
           {
           MatSetValue(raw.E,  2*N_Nodes+posL+Node_Numbers_On_Boundary_Left[i], posL+Node_Numbers_On_Boundary_Left[j], -nz*value_e, ADD_VALUES);
           MatSetValue(raw.ET, posL+Node_Numbers_On_Boundary_Left[j], 2*N_Nodes+posL+Node_Numbers_On_Boundary_Left[i], nz*value_e, ADD_VALUES);
           MatSetValue(raw.Edz,  posL+Node_Numbers_On_Boundary_Left[i], posL+Node_Numbers_On_Boundary_Left[j], -nz*value_e, ADD_VALUES);
           MatSetValue(raw.ETdz, posL+Node_Numbers_On_Boundary_Left[j], posL+Node_Numbers_On_Boundary_Left[i], nz*value_e, ADD_VALUES);
           }

           if (assembleCurlFaceCorrection)
           {
               addCurlFaceBlock(posL+Node_Numbers_On_Boundary_Left[i],
                                posL+Node_Numbers_On_Boundary_Left[j],
                                -0.5, faceMass, nx, ny, nz); // LL
           }
       }
   }
           // GLR
           for (unsigned int k = 1; k < size_left+1; k++)
           {
//                 //std::cout << "k = " <<  k << std::endl;
               unsigned int alpha = (k-1)%(N_Order+1);
               unsigned int beta = (k-1)/(N_Order+1);
               for (unsigned int l = 1; l < size_right+1; l++)
               {
                   //std::cout << "l = " <<  l << std::endl;
                   unsigned int gamma = (l-1)%(N_Order+1);
                   unsigned int delta = (l-1)/(N_Order+1);
                   // E Matrix
                   double value_e = 0.0;
                   double faceMass = 0.0;
//                     //std::cout << alpha << " " << beta << " " << gamma << " " << delta << std::endl;
                   for (unsigned int p = 0; p <= Order_Gaussian_Quadrature; p++)
                   {
                       //double L_alpha = LagrangePolynomial(ri, qp[p], alpha);
                       //double L_gamma = LagrangePolynomial(ri, qp[p], gamma);
                       const double L_alpha = L1D[idx1D(p, alpha)];
                       const double L_gamma = L1D[idx1D(p, gamma)];
                       for (unsigned int q = 0; q <= Order_Gaussian_Quadrature; q++)
                       {
                           //double L_beta  = LagrangePolynomial(ri, qp[q], beta);
                           //double L_delta = LagrangePolynomial(ri, qp[q], delta);
                           const double L_beta  = L1D[idx1D(q, beta)];
                           const double L_delta = L1D[idx1D(q, delta)];
                           const double s = w[p]*L_alpha*L_beta * w[q]*L_gamma*L_delta * Jacobian;
                           faceMass += s;
                           value_e += -(1.0-theta) * s;
                       }
                   }
                   //std::cout << "GLR: " << value_e << std::endl;
                   unsigned int i = k - 1;
                   unsigned int j = l - 1;
                   if (nx != 0.0)
                   {
                   MatSetValue(raw.E,  posL+Node_Numbers_On_Boundary_Left[i],  posR+Node_Numbers_On_Boundary_Right[j], -nx*value_e, ADD_VALUES);
                   MatSetValue(raw.ET, posR+Node_Numbers_On_Boundary_Right[j], posL+Node_Numbers_On_Boundary_Left[i], nx*value_e, ADD_VALUES);
                   MatSetValue(raw.Edx,  posL+Node_Numbers_On_Boundary_Left[i],  posR+Node_Numbers_On_Boundary_Right[j], -nx*value_e, ADD_VALUES);
                   MatSetValue(raw.ETdx, posR+Node_Numbers_On_Boundary_Right[j], posL+Node_Numbers_On_Boundary_Left[i], nx*value_e, ADD_VALUES);
                   }
                   if (ny != 0.0)
                   {
                   MatSetValue(raw.E,  N_Nodes+posL+Node_Numbers_On_Boundary_Left[i],  posR+Node_Numbers_On_Boundary_Right[j], -ny*value_e, ADD_VALUES);
                   MatSetValue(raw.ET, posR+Node_Numbers_On_Boundary_Right[j],  N_Nodes+posL+Node_Numbers_On_Boundary_Left[i], ny*value_e, ADD_VALUES);
                   MatSetValue(raw.Edy,  posL+Node_Numbers_On_Boundary_Left[i],  posR+Node_Numbers_On_Boundary_Right[j], -ny*value_e, ADD_VALUES);
                   MatSetValue(raw.ETdy, posR+Node_Numbers_On_Boundary_Right[j],  posL+Node_Numbers_On_Boundary_Left[i], ny*value_e, ADD_VALUES);
                   }
                   if (nz != 0.0)
                   {
                   MatSetValue(raw.E,  2*N_Nodes+posL+Node_Numbers_On_Boundary_Left[i],  posR+Node_Numbers_On_Boundary_Right[j], -nz*value_e, ADD_VALUES);
                   MatSetValue(raw.ET, posR+Node_Numbers_On_Boundary_Right[j],  2*N_Nodes+posL+Node_Numbers_On_Boundary_Left[i], nz*value_e, ADD_VALUES);
                   MatSetValue(raw.Edz,  posL+Node_Numbers_On_Boundary_Left[i],  posR+Node_Numbers_On_Boundary_Right[j], -nz*value_e, ADD_VALUES);
                   MatSetValue(raw.ETdz, posR+Node_Numbers_On_Boundary_Right[j],  posL+Node_Numbers_On_Boundary_Left[i], nz*value_e, ADD_VALUES);
                   }

                   if (assembleCurlFaceCorrection)
                   {
                       addCurlFaceBlock(posL+Node_Numbers_On_Boundary_Left[i],
                                        posR+Node_Numbers_On_Boundary_Right[j],
                                        +0.5, faceMass, nx, ny, nz); // LR
                   }
               }
           }
           // GRL
           for (unsigned int k = 1; k < size_right+1; k++)
           {
               //std::cout << "k = " <<  k << std::endl;
               unsigned int alpha = (k-1)%(N_Order+1);
               unsigned int beta = (k-1)/(N_Order+1);
               for (unsigned int l = 1; l < size_left+1; l++)
               {
                   //std::cout << "l = " <<  l << std::endl;
                   unsigned int gamma = (l-1)%(N_Order+1);
                   unsigned int delta = (l-1)/(N_Order+1);
                   //std::cout << alpha << " " << beta << " " << gamma << " " << delta << std::endl;
                   // E Matrix
                   double value_e = 0.0;
                   double faceMass = 0.0;
                   for (unsigned int p = 0; p <= Order_Gaussian_Quadrature; p++)
                   {
                       //double L_alpha = LagrangePolynomial(ri, qp[p], alpha);
                       //double L_gamma = LagrangePolynomial(ri, qp[p], gamma);
                       const double L_alpha = L1D[idx1D(p, alpha)];
                       const double L_gamma = L1D[idx1D(p, gamma)];
                       for (unsigned int q = 0; q <= Order_Gaussian_Quadrature; q++)
                       {
                           //double L_beta  = LagrangePolynomial(ri, qp[q], beta);
                           //double L_delta = LagrangePolynomial(ri, qp[q], delta);
                           const double L_beta  = L1D[idx1D(q, beta)];
                           const double L_delta = L1D[idx1D(q, delta)];
                           const double s = w[p]*L_alpha*L_beta * w[q]*L_gamma*L_delta * Jacobian;
                           faceMass += s;
                           value_e += theta * s;
                       }
                   }
                   //std::cout << "GRL: " << value_e << std::endl;
                   unsigned int i = k - 1;
                   unsigned int j = l - 1;
                   if (nx != 0.0)
                   {
                   MatSetValue(raw.E,  posR+Node_Numbers_On_Boundary_Right[i], posL+Node_Numbers_On_Boundary_Left[j], -nx*value_e, ADD_VALUES);
                   MatSetValue(raw.ET, posL+Node_Numbers_On_Boundary_Left[j],  posR+Node_Numbers_On_Boundary_Right[i], nx*value_e, ADD_VALUES);
                   MatSetValue(raw.Edx,  posR+Node_Numbers_On_Boundary_Right[i], posL+Node_Numbers_On_Boundary_Left[j], -nx*value_e, ADD_VALUES);
                   MatSetValue(raw.ETdx, posL+Node_Numbers_On_Boundary_Left[j],  posR+Node_Numbers_On_Boundary_Right[i], nx*value_e, ADD_VALUES);
                   }
                   if (ny != 0.0)
                   {
                   MatSetValue(raw.E,  N_Nodes+posR+Node_Numbers_On_Boundary_Right[i], posL+Node_Numbers_On_Boundary_Left[j], -ny*value_e, ADD_VALUES);
                   MatSetValue(raw.ET, posL+Node_Numbers_On_Boundary_Left[j],  N_Nodes+posR+Node_Numbers_On_Boundary_Right[i], ny*value_e, ADD_VALUES);
                   MatSetValue(raw.Edy,  posR+Node_Numbers_On_Boundary_Right[i], posL+Node_Numbers_On_Boundary_Left[j], -ny*value_e, ADD_VALUES);
                   MatSetValue(raw.ETdy, posL+Node_Numbers_On_Boundary_Left[j], posR+Node_Numbers_On_Boundary_Right[i], ny*value_e, ADD_VALUES);
                   }
                   if (nz != 0.0)
                   {
                   MatSetValue(raw.E,  2*N_Nodes+posR+Node_Numbers_On_Boundary_Right[i], posL+Node_Numbers_On_Boundary_Left[j], -nz*value_e, ADD_VALUES);
                   MatSetValue(raw.ET, posL+Node_Numbers_On_Boundary_Left[j],  2*N_Nodes+posR+Node_Numbers_On_Boundary_Right[i], nz*value_e, ADD_VALUES);
                   MatSetValue(raw.Edz,  posR+Node_Numbers_On_Boundary_Right[i], posL+Node_Numbers_On_Boundary_Left[j], -nz*value_e, ADD_VALUES);
                   MatSetValue(raw.ETdz, posL+Node_Numbers_On_Boundary_Left[j],  posR+Node_Numbers_On_Boundary_Right[i], nz*value_e, ADD_VALUES);
                   }

                   if (assembleCurlFaceCorrection)
                   {
                       addCurlFaceBlock(posR+Node_Numbers_On_Boundary_Right[i],
                                        posL+Node_Numbers_On_Boundary_Left[j],
                                        -0.5, faceMass, nx, ny, nz); // RL
                   }
               }
           }
           // GRR
           for (unsigned int k = 1; k < size_right+1; k++)
           {
               unsigned int alpha = (k-1)%(N_Order+1);
               unsigned int beta = (k-1)/(N_Order+1);
               for (unsigned int l = 1; l < size_right+1; l++)
               {
                   unsigned int gamma = (l-1)%(N_Order+1);
                   unsigned int delta = (l-1)/(N_Order+1);
                   //std::cout << alpha << " " << beta << " " << gamma << " " << delta << std::endl;
                   // E Matrix
                   double value_e = 0.0;
                   double faceMass = 0.0;
                   for (unsigned int p = 0; p <= Order_Gaussian_Quadrature; p++)
                   {
                       //double L_alpha = LagrangePolynomial(ri, qp[p], alpha);
                       //double L_gamma = LagrangePolynomial(ri, qp[p], gamma);
                       const double L_alpha = L1D[idx1D(p, alpha)];
                       const double L_gamma = L1D[idx1D(p, gamma)];
                       for (unsigned int q = 0; q <= Order_Gaussian_Quadrature; q++)
                       {
                           //double L_beta  = LagrangePolynomial(ri, qp[q], beta);
                           //double L_delta = LagrangePolynomial(ri, qp[q], delta);
                           const double L_beta  = L1D[idx1D(q, beta)];
                           const double L_delta = L1D[idx1D(q, delta)];
                           const double s = w[p]*L_alpha*L_beta * w[q]*L_gamma*L_delta * Jacobian;
                           faceMass += s;
                           value_e += -theta * s;
                       }
                   }
                   //std::cout << "GRR: " << value_e << std::endl;
                   unsigned int i = k - 1;
                   unsigned int j = l - 1;
                   //std::cout << nx << std::endl;
                   if (nx != 0.0)
                   {
                   MatSetValue(raw.E,  posR+Node_Numbers_On_Boundary_Right[i], posR+Node_Numbers_On_Boundary_Right[j], -nx*value_e, ADD_VALUES);
                   MatSetValue(raw.ET, posR+Node_Numbers_On_Boundary_Right[j], posR+Node_Numbers_On_Boundary_Right[i], nx*value_e, ADD_VALUES);
                   MatSetValue(raw.Edx,  posR+Node_Numbers_On_Boundary_Right[i], posR+Node_Numbers_On_Boundary_Right[j], -nx*value_e, ADD_VALUES);
                   MatSetValue(raw.ETdx, posR+Node_Numbers_On_Boundary_Right[j], posR+Node_Numbers_On_Boundary_Right[i], nx*value_e, ADD_VALUES);
                   }
                   if (ny != 0.0)
                   {
                   MatSetValue(raw.E,  N_Nodes+posR+Node_Numbers_On_Boundary_Right[i], posR+Node_Numbers_On_Boundary_Right[j], -ny*value_e, ADD_VALUES);
                   MatSetValue(raw.ET, posR+Node_Numbers_On_Boundary_Right[j], N_Nodes+posR+Node_Numbers_On_Boundary_Right[i], ny*value_e, ADD_VALUES);
                   MatSetValue(raw.Edy,  posR+Node_Numbers_On_Boundary_Right[i], posR+Node_Numbers_On_Boundary_Right[j], -ny*value_e, ADD_VALUES);
                   MatSetValue(raw.ETdy, posR+Node_Numbers_On_Boundary_Right[j], posR+Node_Numbers_On_Boundary_Right[i], ny*value_e, ADD_VALUES);
                   }
                   if (nz != 0.0)
                   {
                   MatSetValue(raw.E,  2*N_Nodes+posR+Node_Numbers_On_Boundary_Right[i], posR+Node_Numbers_On_Boundary_Right[j], -nz*value_e, ADD_VALUES);
                   MatSetValue(raw.ET, posR+Node_Numbers_On_Boundary_Right[j], 2*N_Nodes+posR+Node_Numbers_On_Boundary_Right[i], nz*value_e, ADD_VALUES);
                   MatSetValue(raw.Edz,  posR+Node_Numbers_On_Boundary_Right[i], posR+Node_Numbers_On_Boundary_Right[j], -nz*value_e, ADD_VALUES);
                   MatSetValue(raw.ETdz, posR+Node_Numbers_On_Boundary_Right[j], posR+Node_Numbers_On_Boundary_Right[i], nz*value_e, ADD_VALUES);
                   }

                   if (assembleCurlFaceCorrection)
                   {
                       addCurlFaceBlock(posR+Node_Numbers_On_Boundary_Right[i],
                                        posR+Node_Numbers_On_Boundary_Right[j],
                                        +0.5, faceMass, nx, ny, nz); // RR
                   }
               }
           }
  }
  VecDestroy(&ri);
  VecRestoreArray(QuadraturePoints, &qp);
  VecRestoreArray(Weights, &w);
  VecDestroy(&Weights);
  VecDestroy(&QuadraturePoints);

   MatAssemblyBegin(raw.E, MAT_FINAL_ASSEMBLY);  MatAssemblyEnd(raw.E, MAT_FINAL_ASSEMBLY); MatAssemblyBegin(raw.ET, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(raw.ET, MAT_FINAL_ASSEMBLY);
   MatAssemblyBegin(raw.Edx, MAT_FINAL_ASSEMBLY);  MatAssemblyEnd(raw.Edx, MAT_FINAL_ASSEMBLY); MatAssemblyBegin(raw.ETdx, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(raw.ETdx, MAT_FINAL_ASSEMBLY);
   MatAssemblyBegin(raw.Edy, MAT_FINAL_ASSEMBLY);  MatAssemblyEnd(raw.Edy, MAT_FINAL_ASSEMBLY); MatAssemblyBegin(raw.ETdy, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(raw.ETdy, MAT_FINAL_ASSEMBLY);
   MatAssemblyBegin(raw.Edz, MAT_FINAL_ASSEMBLY);  MatAssemblyEnd(raw.Edz, MAT_FINAL_ASSEMBLY); MatAssemblyBegin(raw.ETdz, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(raw.ETdz, MAT_FINAL_ASSEMBLY);
   MatAssemblyBegin(raw.Curlx, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(raw.Curlx, MAT_FINAL_ASSEMBLY);
   MatAssemblyBegin(raw.Curly, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(raw.Curly, MAT_FINAL_ASSEMBLY);
   MatAssemblyBegin(raw.Curlz, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(raw.Curlz, MAT_FINAL_ASSEMBLY);
   MatAssemblyBegin(raw.Curl,  MAT_FINAL_ASSEMBLY); MatAssemblyEnd(raw.Curl,  MAT_FINAL_ASSEMBLY);
   PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode computeDivergenceDiagnostics(
    const Vec Velocity,
    const Mat DIV,
    DiagnosticsWorkspace &work,
    double &div_max,
    double &div_l2)
{
    PetscFunctionBegin;
    // work.div = DIV * Velocity
    PetscCall(MatMult(DIV, Velocity, work.div));

    PetscReal ninf, n2;
    PetscCall(VecNorm(work.div, NORM_INFINITY, &ninf));
    PetscCall(VecNorm(work.div, NORM_2, &n2));

    div_max = static_cast<double>(ninf);
    div_l2  = static_cast<double>(n2);

    PetscFunctionReturn(0);
}

PetscErrorCode computeHamiltonian(
    const Mat M,
    const Vec Velocity,
    DiagnosticsWorkspace &work,
    double &H)
{
    PetscFunctionBegin;
    // work.MAV = M * V
    PetscCall(MatMult(M, Velocity, work.MAV));

    PetscScalar val;
    PetscCall(VecDot(Velocity, work.MAV, &val));

    H = 0.5 * PetscRealPart(val);

    PetscFunctionReturn(0);
}
PetscErrorCode ApplyVelocityLaplacianBlockDiag(
    Mat L,
    Vec V,
    Vec LV,
    IS isu,
    IS isv,
    IS isw)
{
    PetscFunctionBeginUser;

    Vec Vu  = nullptr;
    Vec Vv  = nullptr;
    Vec Vw  = nullptr;
    Vec LVu = nullptr;
    Vec LVv = nullptr;
    Vec LVw = nullptr;

    PetscCall(VecGetSubVector(V,  isu, &Vu));
    PetscCall(VecGetSubVector(V,  isv, &Vv));
    PetscCall(VecGetSubVector(V,  isw, &Vw));

    PetscCall(VecGetSubVector(LV, isu, &LVu));
    PetscCall(VecGetSubVector(LV, isv, &LVv));
    PetscCall(VecGetSubVector(LV, isw, &LVw));

    PetscCall(MatMult(L, Vu, LVu));
    PetscCall(MatMult(L, Vv, LVv));
    PetscCall(MatMult(L, Vw, LVw));

    PetscCall(VecRestoreSubVector(V,  isu, &Vu));
    PetscCall(VecRestoreSubVector(V,  isv, &Vv));
    PetscCall(VecRestoreSubVector(V,  isw, &Vw));

    PetscCall(VecRestoreSubVector(LV, isu, &LVu));
    PetscCall(VecRestoreSubVector(LV, isv, &LVv));
    PetscCall(VecRestoreSubVector(LV, isw, &LVw));

    PetscFunctionReturn(0);
}
PetscErrorCode computeMidpointDissipation(
    const Vec Velocity_n,      // V^n
    const Vec Velocity_np1,    // V^{n+1}
    const Mat M,               // full 3N x 3N mass matrix
    const Mat Laplacian,       // scalar N x N viscous operator
    IS isu,
    IS isv,
    IS isw,
    DiagnosticsWorkspace &work,
    double &Ehalf)
{
    PetscFunctionBeginUser;

    // work.Vmid = 0.5 * (V^n + V^{n+1})
    PetscCall(VecCopy(Velocity_n, work.Vmid));
    PetscCall(VecAXPY(work.Vmid, 1.0, Velocity_np1));
    PetscCall(VecScale(work.Vmid, 0.5));

    // work.AV = blockdiag(L,L,L) * Vmid
    PetscCall(ApplyVelocityLaplacianBlockDiag(
        Laplacian,
        work.Vmid,
        work.AV,
        isu,
        isv,
        isw));

    // work.MAV = M * blockdiag(L,L,L) * Vmid
    PetscCall(MatMult(M, work.AV, work.MAV));

    // Ehalf = - Vmid^T M A_nu Vmid
    PetscScalar val;
    PetscCall(VecDot(work.Vmid, work.MAV, &val));

    Ehalf = -PetscRealPart(val);

    PetscFunctionReturn(0);
}

PetscErrorCode computeMidpointHelicityDissipation(
    const Vec Velocity_n,
    const Vec Velocity_np1,
    const Mat M,
    const Mat CURL,
    const Mat Laplacian,
    const RotationParameters &rot,
    IS isu,
    IS isv,
    IS isw,
    DiagnosticsWorkspace &work,
    double &Shalf)
{
    PetscFunctionBeginUser;

    // work.Vmid = 0.5 * (V^n + V^{n+1})
    PetscCall(VecCopy(Velocity_n, work.Vmid));
    PetscCall(VecAXPY(work.Vmid, 1.0, Velocity_np1));
    PetscCall(VecScale(work.Vmid, 0.5));

    // work.omega = CURL * Vmid
    PetscCall(MatMult(CURL, work.Vmid, work.omega));

    // Absolute-vorticity convention consistent with computeHelicity:
    //
    // h_a = 0.5 * V^T M (CURL V + 2/Ro Frot)
    //
    // Hence the viscous helicity dissipation uses
    // omega_a = CURL Vmid + 2/Ro Frot.
    if (rot.include_rotation)
    {
        PetscCheck(rot.RossbyNumber != 0.0,
                   PETSC_COMM_SELF,
                   PETSC_ERR_ARG_OUTOFRANGE,
                   "Rossby number must be nonzero when rotation is enabled.");

        PetscCall(VecAXPY(
            work.omega,
            1.0 / rot.RossbyNumber, // Changed 2.0 to 1.0
            work.Frot));
    }

    // work.AV = blockdiag(L,L,L) * Vmid
    PetscCall(ApplyVelocityLaplacianBlockDiag(
        Laplacian,
        work.Vmid,
        work.AV,
        isu,
        isv,
        isw));

    // work.MAV = M * A_nu Vmid
    PetscCall(MatMult(M, work.AV, work.MAV));

    // Shalf = - omega_a^T M A_nu Vmid
    PetscScalar val = 0.0;
    PetscCall(VecDot(work.omega, work.MAV, &val));

    Shalf = -PetscRealPart(val);

    PetscFunctionReturn(0);
}
PetscErrorCode computeForcingPowerInput(
    const Vec Velocity_n,
    const Vec Velocity_np1,
    const Vec ForceHalf,
    const Mat M,
    const PetscReal Fo,
    DiagnosticsWorkspace &work,
    double &W_unscaled,
    double &power_input)
{
    PetscFunctionBeginUser;

    W_unscaled = 0.0;
    power_input = 0.0;

    if (!ForceHalf)
    {
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // Vmid = 0.5 * (Velocity_n + Velocity_np1)
    PetscCall(VecWAXPY(work.Vmid, 1.0, Velocity_n, Velocity_np1));
    PetscCall(VecScale(work.Vmid, 0.5));

    // tmp3N = M * B^{n+1/2}
    PetscCall(MatMult(M, ForceHalf, work.tmp3N));

    PetscScalar dot = 0.0;
    PetscCall(VecDot(work.Vmid, work.tmp3N, &dot));

    W_unscaled = PetscRealPart(dot);
    power_input = W_unscaled / (Fo * Fo);

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode computeHelicityPowerInput(
    const Vec Velocity_n,
    const Vec Velocity_np1,
    const Vec ForceHalf,
    const Mat M,
    const Mat CURL,
    const RotationParameters &rot,
    const PetscReal Fo,
    DiagnosticsWorkspace &work,
    double &Womega_unscaled,
    double &helicity_power_input)
{
    PetscFunctionBeginUser;

    Womega_unscaled = 0.0;
    helicity_power_input = 0.0;

    if (!ForceHalf)
    {
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // Vmid = 0.5 * (V^n + V^{n+1})
    PetscCall(VecCopy(Velocity_n, work.Vmid));
    PetscCall(VecAXPY(work.Vmid, 1.0, Velocity_np1));
    PetscCall(VecScale(work.Vmid, 0.5));

    // omega_a = CURL Vmid + (1/Ro) F
    PetscCall(MatMult(CURL, work.Vmid, work.omega));

    if (rot.include_rotation)
    {
        PetscCheck(rot.RossbyNumber != 0.0,
                   PETSC_COMM_SELF,
                   PETSC_ERR_ARG_OUTOFRANGE,
                   "Rossby number must be nonzero when rotation is enabled.");

        PetscCall(VecAXPY(work.omega, 1.0 / rot.RossbyNumber, work.Frot));
    }

    // tmp3N = M * B^{n+1/2}
    PetscCall(MatMult(M, ForceHalf, work.tmp3N));

    PetscScalar dot = 0.0;
    PetscCall(VecDot(work.omega, work.tmp3N, &dot));

    Womega_unscaled = PetscRealPart(dot);
    helicity_power_input = Womega_unscaled / (Fo * Fo);

    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode computeHelicitySignSplitNodal(
    const Vec Velocity,
    const Mat CURL,
    const RotationParameters &rot,
    IS isu,
    IS isv,
    IS isw,
    DiagnosticsWorkspace &work,
    double &helicity_pos,
    double &helicity_neg,
    double &helicity_unsigned,
    double &helicity_signed_nodal,
    double &helicity_cancellation_ratio)
{
    PetscFunctionBeginUser;

    helicity_pos = 0.0;
    helicity_neg = 0.0;
    helicity_unsigned = 0.0;
    helicity_signed_nodal = 0.0;
    helicity_cancellation_ratio = 0.0;
    // q = 0.5 CURL V + (1/Ro) F
    // This is the local absolute-helicity density vector.
    PetscCall(MatMult(CURL, Velocity, work.omega));
    PetscCall(VecScale(work.omega, 0.5));

    if (rot.include_rotation)
    {
        PetscCheck(rot.RossbyNumber != 0.0,
                   PETSC_COMM_SELF,
                   PETSC_ERR_ARG_OUTOFRANGE,
                   "Rossby number must be nonzero when rotation is enabled.");

        PetscCall(VecAXPY(work.omega, 1.0 / rot.RossbyNumber, work.Frot));
    }

    Vec Vu = nullptr, Vv = nullptr, Vw = nullptr;
    Vec Qu = nullptr, Qv = nullptr, Qw = nullptr;
    Vec Wu = nullptr, Wv = nullptr, Ww = nullptr;

    PetscCall(VecGetSubVector(Velocity, isu, &Vu));
    PetscCall(VecGetSubVector(Velocity, isv, &Vv));
    PetscCall(VecGetSubVector(Velocity, isw, &Vw));

    PetscCall(VecGetSubVector(work.omega, isu, &Qu));
    PetscCall(VecGetSubVector(work.omega, isv, &Qv));
    PetscCall(VecGetSubVector(work.omega, isw, &Qw));

    PetscCall(VecGetSubVector(work.Mdiag, isu, &Wu));
    PetscCall(VecGetSubVector(work.Mdiag, isv, &Wv));
    PetscCall(VecGetSubVector(work.Mdiag, isw, &Ww));

    const PetscScalar *vu, *vv, *vw;
    const PetscScalar *qu, *qv, *qw;
    const PetscScalar *wu;

    PetscCall(VecGetArrayRead(Vu, &vu));
    PetscCall(VecGetArrayRead(Vv, &vv));
    PetscCall(VecGetArrayRead(Vw, &vw));

    PetscCall(VecGetArrayRead(Qu, &qu));
    PetscCall(VecGetArrayRead(Qv, &qv));
    PetscCall(VecGetArrayRead(Qw, &qw));

    PetscCall(VecGetArrayRead(Wu, &wu));

    PetscInt nloc = 0;
    PetscCall(VecGetLocalSize(Vu, &nloc));

    double local_pos = 0.0;
    double local_neg = 0.0;

    for (PetscInt i = 0; i < nloc; ++i)
    {
        const double weight = PetscRealPart(wu[i]);

        const double density =
            PetscRealPart(qu[i] * vu[i]
                        + qv[i] * vv[i]
                        + qw[i] * vw[i]);

        const double contribution = weight * density;

        if (contribution >= 0.0)
        {
            local_pos += contribution;
        }
        else
        {
            local_neg += contribution;
        }
    }

    PetscCall(VecRestoreArrayRead(Vu, &vu));
    PetscCall(VecRestoreArrayRead(Vv, &vv));
    PetscCall(VecRestoreArrayRead(Vw, &vw));

    PetscCall(VecRestoreArrayRead(Qu, &qu));
    PetscCall(VecRestoreArrayRead(Qv, &qv));
    PetscCall(VecRestoreArrayRead(Qw, &qw));

    PetscCall(VecRestoreArrayRead(Wu, &wu));

    PetscCall(VecRestoreSubVector(Velocity, isu, &Vu));
    PetscCall(VecRestoreSubVector(Velocity, isv, &Vv));
    PetscCall(VecRestoreSubVector(Velocity, isw, &Vw));

    PetscCall(VecRestoreSubVector(work.omega, isu, &Qu));
    PetscCall(VecRestoreSubVector(work.omega, isv, &Qv));
    PetscCall(VecRestoreSubVector(work.omega, isw, &Qw));

    PetscCall(VecRestoreSubVector(work.Mdiag, isu, &Wu));
    PetscCall(VecRestoreSubVector(work.Mdiag, isv, &Wv));
    PetscCall(VecRestoreSubVector(work.Mdiag, isw, &Ww));

    double local_vals[2] = {local_pos, local_neg};
    double global_vals[2] = {0.0, 0.0};

    PetscCallMPI(MPI_Allreduce(
        local_vals,
        global_vals,
        2,
        MPI_DOUBLE,
        MPI_SUM,
        PetscObjectComm((PetscObject)Velocity)));

    helicity_pos = global_vals[0];
    helicity_neg = global_vals[1];

    helicity_signed_nodal = helicity_pos + helicity_neg;
    helicity_unsigned = helicity_pos - helicity_neg;

    helicity_cancellation_ratio =
        std::abs(helicity_signed_nodal) / std::max(helicity_unsigned, PETSC_SMALL);

    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode computeHelicitySignSplitQuadrature(
    const Vec Velocity,
    const Mat CURL,
    const RotationParameters &rot,
    const std::vector<std::unique_ptr<Vertex>>  &List_Of_Vertices,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    const unsigned int N_Nodes,
    const unsigned int N_Order,
    DiagnosticsWorkspace &work,
    double &helicity_pos,
    double &helicity_neg,
    double &helicity_unsigned,
    double &helicity_signed_quad,
    double &helicity_cancellation_ratio)
{
    PetscFunctionBeginUser;

    helicity_pos = 0.0;
    helicity_neg = 0.0;
    helicity_unsigned = 0.0;
    helicity_signed_quad = 0.0;
    helicity_cancellation_ratio = 0.0;

    PetscCheck(work.omega,
               PetscObjectComm((PetscObject)Velocity),
               PETSC_ERR_ARG_WRONGSTATE,
               "work.omega is null.");

    // omega_h^r = CURL V
    PetscCall(MatMult(CURL, Velocity, work.omega));

    // Same quadrature choice as in create_Matrices_Cuboids_Inertial2.
    const unsigned int Order_Gaussian_Quadrature = N_Order + 3;
    const unsigned int n1 = N_Order + 1;
    const unsigned int nq = Order_Gaussian_Quadrature + 1;

    Vec ri = nullptr;
    Vec Weights = nullptr;
    Vec QuadraturePoints = nullptr;

    ri = JacobiGL(0, 0, N_Order);
    QuadraturePoints = JacobiGL_withWeights(
        0, 0, Order_Gaussian_Quadrature, Weights);

    const PetscScalar *w = nullptr;
    const PetscScalar *qp = nullptr;

    PetscCall(VecGetArrayRead(Weights, &w));
    PetscCall(VecGetArrayRead(QuadraturePoints, &qp));

    std::vector<double> L1D(nq * n1, 0.0);

    auto idx1D = [n1](unsigned int qpt, unsigned int a) -> unsigned int
    {
        return qpt * n1 + a;
    };

    for (unsigned int p = 0; p < nq; ++p)
    {
        for (unsigned int a = 0; a < n1; ++a)
        {
            L1D[idx1D(p, a)] =
                LagrangePolynomial(ri, qp[p], a);
        }
    }

    const PetscScalar *Varr = nullptr;
    const PetscScalar *Oarr = nullptr;

    PetscCall(VecGetArrayRead(Velocity, &Varr));
    PetscCall(VecGetArrayRead(work.omega, &Oarr));

    double local_pos = 0.0;
    double local_neg = 0.0;

    for (const auto &elem_ptr : List_Of_Elements)
    {
        const auto &elem = elem_ptr;

        const unsigned int pos = elem->get_pos();
        const unsigned int Np  = elem->get_Number_Of_Nodes();

        double J = 0.0;
        double drdx = 0.0, drdy = 0.0, drdz = 0.0;
        double dsdx = 0.0, dsdy = 0.0, dsdz = 0.0;
        double dtdx = 0.0, dtdy = 0.0, dtdz = 0.0;
        double x = 0.0, y = 0.0, z = 0.0;

        // For affine cuboids, J is constant.
        Calculate_Jacobian_Cuboid(
            elem,
            List_Of_Vertices,
            0.0, 0.0, 0.0,
            J,
            drdx, drdy, drdz,
            dsdx, dsdy, dsdz,
            dtdx, dtdy, dtdz,
            x, y, z);

        for (unsigned int p = 0; p < nq; ++p)
        {
            for (unsigned int q = 0; q < nq; ++q)
            {
                for (unsigned int r = 0; r < nq; ++r)
                {
                    double u = 0.0;
                    double v = 0.0;
                    double ww = 0.0;

                    double ox = 0.0;
                    double oy = 0.0;
                    double oz = 0.0;

                    for (unsigned int k = 0; k < Np; ++k)
                    {
                        const unsigned int alpha =
                            k % (N_Order + 1);

                        const unsigned int kk =
                            k / (N_Order + 1);

                        const unsigned int beta =
                            kk % (N_Order + 1);

                        const unsigned int zeta =
                            k / ((N_Order + 1) * (N_Order + 1));

                        const double phi =
                            L1D[idx1D(p, alpha)] *
                            L1D[idx1D(q, beta)]  *
                            L1D[idx1D(r, zeta)];

                        const PetscInt iu = static_cast<PetscInt>(pos + k);
                        const PetscInt iv = static_cast<PetscInt>(N_Nodes + pos + k);
                        const PetscInt iw = static_cast<PetscInt>(2 * N_Nodes + pos + k);

                        u  += phi * PetscRealPart(Varr[iu]);
                        v  += phi * PetscRealPart(Varr[iv]);
                        ww += phi * PetscRealPart(Varr[iw]);

                        ox += phi * PetscRealPart(Oarr[iu]);
                        oy += phi * PetscRealPart(Oarr[iv]);
                        oz += phi * PetscRealPart(Oarr[iw]);
                    }

                    if (rot.include_rotation)
                    {
                        PetscCheck(rot.RossbyNumber != 0.0,
                                   PETSC_COMM_SELF,
                                   PETSC_ERR_ARG_OUTOFRANGE,
                                   "Rossby number must be nonzero when rotation is enabled.");

                        ox = 0.5 * ox + rot.f1 / rot.RossbyNumber;
                        oy = 0.5 * oy + rot.f2 / rot.RossbyNumber;
                        oz = 0.5 * oz + rot.f3 / rot.RossbyNumber;
                    }
                    else
                    {
                        ox = 0.5 * ox;
                        oy = 0.5 * oy;
                        oz = 0.5 * oz;
                    }

                    const double density =
                        ox * u + oy * v + oz * ww;

                    const double dV =
                        PetscRealPart(w[p]) *
                        PetscRealPart(w[q]) *
                        PetscRealPart(w[r]) *
                        J;

                    const double contribution = density * dV;

                    if (contribution >= 0.0)
                    {
                        local_pos += contribution;
                    }
                    else
                    {
                        local_neg += contribution;
                    }
                }
            }
        }
    }

    PetscCall(VecRestoreArrayRead(Velocity, &Varr));
    PetscCall(VecRestoreArrayRead(work.omega, &Oarr));

    PetscCall(VecRestoreArrayRead(Weights, &w));
    PetscCall(VecRestoreArrayRead(QuadraturePoints, &qp));

    PetscCall(VecDestroy(&ri));
    PetscCall(VecDestroy(&Weights));
    PetscCall(VecDestroy(&QuadraturePoints));

    double local_vals[2] = {local_pos, local_neg};
    double global_vals[2] = {0.0, 0.0};

    PetscCallMPI(MPI_Allreduce(
        local_vals,
        global_vals,
        2,
        MPI_DOUBLE,
        MPI_SUM,
        PetscObjectComm((PetscObject)Velocity)));

    helicity_pos = global_vals[0];
    helicity_neg = global_vals[1];

    helicity_signed_quad = helicity_pos + helicity_neg;
    helicity_unsigned = helicity_pos - helicity_neg;

    helicity_cancellation_ratio =
        std::abs(helicity_signed_quad) /
        std::max(helicity_unsigned, PETSC_SMALL);

    PetscFunctionReturn(0);
}
PetscErrorCode computeCurlBoundaryPairing(
    const Vec U,
    const Vec V,
    const Mat CurlBoundary,
    Vec work,
    double &pairing)
{
    PetscFunctionBeginUser;

    PetscCall(MatMult(
        CurlBoundary,
        V,
        work));

    PetscScalar value = 0.0;

    PetscCall(VecDot(
        U,
        work,
        &value));

    pairing = PetscRealPart(value);

    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode recordDiagnosticsAtAcceptedStep(
    TimeDiagnostics &diag,
    AppCtx &user,
    const Vec ForceHalf,
    const std::vector<std::unique_ptr<Vertex>>  &List_Of_Vertices,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    const unsigned int N_Order,
    const double t,
    const double Delta_t)
{
    PetscFunctionBegin;

    double Hnp1       = 0.0;
    double helicity   = 0.0;
    double divmax     = 0.0;
    double divl2      = 0.0;
    double Ehalf      = 0.0;
    double Shalf      = 0.0;
    double W_unscaled = 0.0;
    double power_input = 0.0;
    double Womega_unscaled = 0.0;
    double helicity_power_input = 0.0;

    double helicity_pos = 0.0;
    double helicity_neg = 0.0;
    double helicity_unsigned = 0.0;
    double helicity_signed_nodal = 0.0;
    double helicity_cancellation_ratio = 0.0;

    PetscCall(computeHamiltonian(
        user.ops.M,
        user.Velocity,
        user.diagWork,
        Hnp1));

    PetscCall(computeHelicity(
        user.ops.M,
        user.ops.CURL,
        user.Velocity,
        user.rot,
        user.diagWork,
        helicity));

        PetscCall(computeHelicitySignSplitQuadrature(
            user.Velocity,
            user.ops.CURL,
            user.rot,
            List_Of_Vertices,
            List_Of_Elements,
            user.N_Nodes,
            N_Order,
            user.diagWork,
            helicity_pos,
            helicity_neg,
            helicity_unsigned,
            helicity_signed_nodal,
            helicity_cancellation_ratio));

    PetscCall(computeDivergenceDiagnostics(
        user.Velocity,
        user.ops.DIV,
        user.diagWork,
        divmax,
        divl2));

    PetscCall(computeMidpointDissipation(
        user.Velocity_n,
        user.Velocity,
        user.ops.M,
        user.ops.Laplacian,
        user.isu,
        user.isv,
        user.isw,
        user.diagWork,
        Ehalf));

    PetscCall(computeMidpointHelicityDissipation(
        user.Velocity_n,
        user.Velocity,
        user.ops.M,
        user.ops.CURL,
        user.ops.Laplacian,
        user.rot,
        user.isu,
        user.isv,
        user.isw,
        user.diagWork,
        Shalf));

    PetscCall(computeForcingPowerInput(
        user.Velocity_n,
        user.Velocity,
        ForceHalf,
        user.ops.M,
        user.opt.Fo,
        user.diagWork,
        W_unscaled,
        power_input));
    PetscCall(computeHelicityPowerInput(
        user.Velocity_n,
        user.Velocity,
        ForceHalf,
        user.ops.M,
        user.ops.CURL,
        user.rot,
        user.opt.Fo,
        user.diagWork,
        Womega_unscaled,
        helicity_power_input));

    AbsoluteHelicityBoundaryDiagnostics hb;

    PetscCall(computeAbsoluteHelicityBoundaryDiagnostics(
        user,
        ForceHalf,
        List_Of_Elements,
        hb));

    const double H0 = diag.H.empty() ? Hnp1 : diag.H.front();
    const double helicity0 = diag.helicity.empty() ? helicity : diag.helicity.front();
    const double Hn = diag.H.empty() ? Hnp1 : diag.H.back();

    const double H_drift = Hnp1 - H0;
    const double helicity_drift = helicity - helicity0;

    const double nu = user.opt.viscous ? (1.0 / user.Re) : 0.0;
    // const double energy_balance_residual =
    //     (Hnp1 - Hn) / Delta_t + nu * Ehalf;

    const double energy_balance_residual =
        (Hnp1 - Hn) / Delta_t + nu * Ehalf - power_input;

    const double hn = diag.helicity.empty() ? helicity : diag.helicity.back();
    const double helicity_balance_residual =
        (helicity - hn) / Delta_t + nu * Shalf - helicity_power_input - hb.total;
    // General balance:
    // dh/dt = -nu*S + P + B - D_cg.
    //
    // Therefore:
    // R = dh/dt + nu*S - P - B = -D_cg.
    const double helicity_balance_residual_corrected =
          helicity_balance_residual
        + hb.curl_grad_defect;

        // const double helicity_residual_forced_no_boundary =
        //     (helicity - hn) / Delta_t + nu * Shalf - helicity_power_input;

        const double previous_energy_residual_cumulative =
            diag.energy_balance_residual_cumulative.empty()
            ? 0.0
            : diag.energy_balance_residual_cumulative.back();

        const double energy_balance_residual_cumulative =
            previous_energy_residual_cumulative
            + Delta_t * energy_balance_residual;

        const double previous_boundary_cumulative =
            diag.helicity_boundary_flux_cumulative.empty()
            ? 0.0
            : diag.helicity_boundary_flux_cumulative.back();

        const double helicity_boundary_flux_cumulative =
            previous_boundary_cumulative
            + Delta_t * hb.total;

        const double previous_helicity_residual_cumulative =
            diag.helicity_balance_residual_cumulative.empty()
            ? 0.0
            : diag.helicity_balance_residual_cumulative.back();

        const double helicity_balance_residual_cumulative =
            previous_helicity_residual_cumulative
            + Delta_t * helicity_balance_residual;

        const double previous_corrected_cumulative =
            diag.helicity_balance_residual_corrected_cumulative.empty()
            ? 0.0
            : diag.helicity_balance_residual_corrected_cumulative.back();

        const double helicity_balance_residual_corrected_cumulative =
            previous_corrected_cumulative
            + Delta_t * helicity_balance_residual_corrected;


    diag.time_full.push_back(t);
    diag.H.push_back(Hnp1);
    diag.H_drift.push_back(H_drift);
    diag.helicity.push_back(helicity);
    diag.helicity_drift.push_back(helicity_drift);
    diag.div_max.push_back(divmax);
    diag.div_l2.push_back(divl2);

    diag.time_half.push_back(t - 0.5 * Delta_t);
    diag.energy_dissipation.push_back(Ehalf);
    diag.energy_power_unscaled.push_back(W_unscaled);
    diag.energy_power_input.push_back(power_input);
    diag.energy_balance_residual.push_back(energy_balance_residual);

    diag.helicity_dissipation.push_back(Shalf);
    //diag.helicity_balance_residual.push_back(helicity_balance_residual);
    diag.helicity_power_unscaled.push_back(Womega_unscaled);
    diag.helicity_power_input.push_back(helicity_power_input);
    diag.helicity_positive.push_back(helicity_pos);
    diag.helicity_negative.push_back(helicity_neg);
    diag.helicity_unsigned.push_back(helicity_unsigned);
    diag.helicity_signed_nodal.push_back(helicity_signed_nodal);
    diag.helicity_cancellation_ratio.push_back(helicity_cancellation_ratio);

    diag.energy_balance_residual_cumulative.push_back(
        energy_balance_residual_cumulative);

    diag.helicity_boundary_nonlinear.push_back(
        hb.nonlinear);

    diag.helicity_boundary_pressure_background.push_back(
        hb.pressure_background);

    diag.helicity_boundary_pressure_pairing.push_back(
        hb.pressure_pairing);

    diag.helicity_boundary_viscous.push_back(
        hb.viscous);

    diag.helicity_boundary_forcing.push_back(
        hb.forcing);

    diag.helicity_boundary_flux.push_back(
        hb.total);

    diag.helicity_boundary_flux_cumulative.push_back(
        helicity_boundary_flux_cumulative);

    diag.helicity_curl_grad_defect.push_back(
        hb.curl_grad_defect);
        diag.helicity_curl_grad_p_inf.push_back(
            hb.curl_grad_p_inf);

        diag.helicity_curl_grad_p_l2.push_back(
            hb.curl_grad_p_l2);

        diag.helicity_curl_grad_p_mass.push_back(
            hb.curl_grad_p_mass);

    diag.helicity_balance_residual.push_back(
        helicity_balance_residual);

    diag.helicity_balance_residual_corrected.push_back(
        helicity_balance_residual_corrected);

    diag.helicity_balance_residual_cumulative.push_back(
        helicity_balance_residual_cumulative);

    diag.helicity_balance_residual_corrected_cumulative.push_back(
        helicity_balance_residual_corrected_cumulative);



    PetscPrintf(
        PETSC_COMM_WORLD,
        "Helicity balance: "
        "S = %.6e, Pha = %.6e, "
        "Bnl = %.6e, Bp-bg = %.6e, "
        "Bp-pair = %.6e, Bnu = %.6e, "
        "Bf = %.6e, Btotal = %.6e, "
        "Dcg = %.6e, Rha = %.6e, "
        "Rha-corr = %.6e\n",
        Shalf,
        helicity_power_input,
        hb.nonlinear,
        hb.pressure_background,
        hb.pressure_pairing,
        hb.viscous,
        hb.forcing,
        hb.total,
        hb.curl_grad_defect,
        helicity_balance_residual,
        helicity_balance_residual_corrected);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode saveDiagnosticsCSV(
    const std::string &filename,
    const TimeDiagnostics &diag)
{
    PetscFunctionBegin;

    std::ofstream file(filename);
    if (!file.is_open())
    {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
                "Could not open diagnostics file");
    }

    file << std::setprecision(16);

    // file << "step,time,H,H_drift,helicity,helicity_drift,"
    //      << "div_max,div_l2,"
    //      << "time_half,energy_dissipation,energy_power_unscaled,"
    //      << "energy_power_input,energy_balance_residual,"
    //      << "helicity_dissipation,helicity_balance_residual\n";

   file << "step,time,H,H_drift,helicity,helicity_drift,"
        << "helicity_positive,helicity_negative,helicity_unsigned,"
        << "helicity_signed_nodal,helicity_cancellation_ratio,"
        << "div_max,div_l2,"
        << "time_half,energy_dissipation,energy_power_unscaled,"
        << "energy_power_input,energy_balance_residual,"
        << "energy_balance_residual_cumulative,"
        << "helicity_dissipation,"
        << "helicity_power_unscaled,"
        << "helicity_power_input,"
        << "helicity_boundary_nonlinear,"
        << "helicity_boundary_pressure_background,"
        << "helicity_boundary_pressure_pairing,"
        << "helicity_boundary_viscous,"
        << "helicity_boundary_forcing,"
        << "helicity_boundary_flux,"
        << "helicity_boundary_flux_cumulative,"
        << "helicity_curl_grad_defect,"
        << "helicity_curl_p_inf,"
        << "helicity_curl_p_l2,"
        << "helicity_curl_p_mass,"
        << "helicity_balance_residual,"
        << "helicity_balance_residual_corrected,"
        << "helicity_balance_residual_cumulative,"
        << "helicity_balance_residual_corrected_cumulative\n";

    const std::size_t n_full = diag.time_full.size();

    for (std::size_t i = 0; i < n_full; ++i)
    {
        file << i << ","
             << diag.time_full[i] << ","
             << diag.H[i] << ","
             << diag.H_drift[i] << ","
             << diag.helicity[i] << ","
             << diag.helicity_drift[i] << ","
             << diag.helicity_positive[i] << ","
             << diag.helicity_negative[i] << ","
             << diag.helicity_unsigned[i] << ","
             << diag.helicity_signed_nodal[i] << ","
             << diag.helicity_cancellation_ratio[i] << ","
             << diag.div_max[i] << ","
             << diag.div_l2[i] << ",";

         if (i == 0)
         {
             for (int k = 0; k < 23; ++k)
             {
                 file << ",";
             }
             file << "\n";
         }
        else
        {
            const std::size_t j = i - 1;
            file << diag.time_half[j] << ","
                 << diag.energy_dissipation[j] << ","
                 << diag.energy_power_unscaled[j] << ","
                 << diag.energy_power_input[j] << ","
                 << diag.energy_balance_residual[j] << ","
                 << diag.energy_balance_residual_cumulative[j] << ","
                 << diag.helicity_dissipation[j] << ","
                 << diag.helicity_power_unscaled[j] << ","
                 << diag.helicity_power_input[j] << ","
                 << diag.helicity_boundary_nonlinear[j] << ","
                 << diag.helicity_boundary_pressure_background[j] << ","
                 << diag.helicity_boundary_pressure_pairing[j] << ","
                 << diag.helicity_boundary_viscous[j] << ","
                 << diag.helicity_boundary_forcing[j] << ","
                 << diag.helicity_boundary_flux[j] << ","
                 << diag.helicity_boundary_flux_cumulative[j] << ","
                 << diag.helicity_curl_grad_defect[j] << ","
                 << diag.helicity_curl_grad_p_inf[j] << ","
                 << diag.helicity_curl_grad_p_l2[j] << ","
                 << diag.helicity_curl_grad_p_mass[j] << ","
                 << diag.helicity_balance_residual[j] << ","
                 << diag.helicity_balance_residual_corrected[j] << ","
                 << diag.helicity_balance_residual_cumulative[j] << ","
                 << diag.helicity_balance_residual_corrected_cumulative[j]
                 << "\n";
        }
    }

    file.close();

    PetscFunctionReturn(PETSC_SUCCESS);
}
/*--------------------------------------------------------------------------*/
void compute_Divergence_Velocity(const Vec &Velocity, const double &N_Nodes, const Mat &DIV)
{
    Vec RHS;
	  VecCreateSeq(PETSC_COMM_SELF, N_Nodes, &RHS);
    MatMult(DIV, Velocity, RHS);

    PetscInt a(10);
    PetscReal max;
    VecMax(RHS, &a, &max);
    std::cout << "Max Divergence = " << std::setprecision (20) <<max<< std::endl;
    VecDestroy(&RHS);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode FormMatrixLinearPart(Mat Alinear, const Mat &GRAD, const Mat &DIVdx, const Mat &DIVdy, const Mat &DIVdz, const Mat &ConstantRotationalMat, const Mat &ConstantRotationalMatx, const Mat &ConstantRotationalMaty, const Mat &ConstantRotationalMatz, const Mat &Laplacian, const double &Delta_t, const unsigned int &N_Nodes, const double &viscous, const double &Re, const PetscBool &include_viscous_divergence_correction)
{
  PetscFunctionBegin;
  // construct time-independent part of Matrix A(X^*,X^n)
  // (linear part of A)
  // if viscous == 1, include viscous terms (through Lap)

  //
  //             [ I-Dt/2*Lap  Dt/2*CR     Dt/2*CR    -Dt*GRADdx ]
  // Alinear =   [ Dt/2*CR     I-Dt/2*Lap  Dt/2*CR    -Dt*GRADdy ]
  //             [ Dt/2*CR     Dt/2*CR     I-Dt/2*LP -Dt*GRADdz ]
  //             [ PU          PV          PW          Lap       ]

  // CR = ConstantRotationalMat, Block Diagonal part of CR is zero
  // PU = DIVdx*CRx, PV = DIVdy*CRy, PW = DIVdz*CRz, Lap = DIV*GRAD

  // PetscInt* nnzAlinear = new PetscInt[4*N_Nodes];
  // // 1 = diag; 2*Np = CR;
  // for (unsigned int i = 0; i < N_Nodes; i++)
  // {
  //   nnzAlinear[i] = 1 + 2*Np;
  //   nnzAlinear[N_Nodes+i] = 1 + 2*Np;
  //   nnzAlinear[2*N_Nodes+i] = 1 + 2*Np;
  // }

  //MatCreateSeqAIJ(PETSC_COMM_SELF, 4*N_Nodes, 4*N_Nodes, 10*6*Np+1+9*Np*Np,  NULL, &Alinear);
  //MatSetOption(Alinear, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);

  //delete[] nnzAlinear;

  // Insert diagonal (for U, V, and W) into Alinear
  // Note: We do not want DIV*V^n+1 in the pressure equation => not solvable
  // We do add DIV*V^n in the pressure equation (see FormRHS) to make the method error correcting

	for (unsigned int i = 0; i<N_Nodes;++i)
	{
    MatSetValue(Alinear, i,           i,           1, ADD_VALUES);
    MatSetValue(Alinear, N_Nodes+i,   N_Nodes+i,   1, ADD_VALUES);
    MatSetValue(Alinear, 2*N_Nodes+i, 2*N_Nodes+i, 1, ADD_VALUES);
  }
  // Insert ConstantRotationalMat into Alinear
 	double dummy=0;
	const PetscInt* cols;
	const PetscScalar* values;
	PetscInt numberOfNonZeros;
  if (ConstantRotationalMat) {
	for (unsigned int i = 0; i<3*N_Nodes;++i)
	{
        MatGetRow(ConstantRotationalMat, i, &numberOfNonZeros, &cols, &values);
        for (int j=0;j<numberOfNonZeros;++j)
        {
            dummy = (values[j]);
            //if (dummy!=0)
            {
				          MatSetValue(Alinear, i, cols[j], Delta_t/2.0*dummy, 	ADD_VALUES);
            }
        }
        MatRestoreRow(ConstantRotationalMat, i, &numberOfNonZeros, &cols, &values);
	}
  // Add PU into Alinear
    Mat A_PU;
    MatMatMult(DIVdx, ConstantRotationalMatx, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &A_PU);
  	for (unsigned int i = 0; i<N_Nodes;++i)
  	{
          MatGetRow(A_PU, i, &numberOfNonZeros, &cols, &values);
          for (int j=0;j<numberOfNonZeros;++j)
          {
              dummy = (values[j]);
              //if (dummy!=0)
              {
  				          MatSetValue(Alinear, 3*N_Nodes+i, cols[j], Delta_t/2.0*dummy, 	ADD_VALUES);
              }
          }
          MatRestoreRow(A_PU, i, &numberOfNonZeros, &cols, &values);
  	}
    MatDestroy(&A_PU);
    // Add PV into Alinear
    Mat A_PV;
    MatMatMult(DIVdy, ConstantRotationalMaty, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &A_PV);
  	for (unsigned int i = 0; i<N_Nodes;++i)
  	{
          MatGetRow(A_PV, i, &numberOfNonZeros, &cols, &values);
          for (int j=0;j<numberOfNonZeros;++j)
          {
              dummy = (values[j]);
              //if (dummy!=0)
              {
  				          MatSetValue(Alinear, 3*N_Nodes+i, cols[j], Delta_t/2.0*dummy, 	ADD_VALUES);
              }
          }
          MatRestoreRow(A_PV, i, &numberOfNonZeros, &cols, &values);
  	}
    MatDestroy(&A_PV);
    // Add PW into Alinear
    Mat A_PW;
    MatMatMult(DIVdz, ConstantRotationalMatz, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &A_PW);
  	for (unsigned int i = 0; i<N_Nodes;++i)
  	{
          MatGetRow(A_PW, i, &numberOfNonZeros, &cols, &values);
          for (int j=0;j<numberOfNonZeros;++j)
          {
              dummy = (values[j]);
              //if (dummy!=0)
              {
  				          MatSetValue(Alinear, 3*N_Nodes+i, cols[j], Delta_t/2.0*dummy, 	ADD_VALUES);
              }
          }
          MatRestoreRow(A_PW, i, &numberOfNonZeros, &cols, &values);
  	}
    MatDestroy(&A_PW);
}
  // Insert GRAD matrix into Alinear
	for (unsigned int i = 0; i<3*N_Nodes;++i)
	{
        MatGetRow(GRAD, i, &numberOfNonZeros, &cols, &values);
        for (int j=0;j<numberOfNonZeros;++j)
        {
            dummy = (values[j]);
            //if (dummy!=0)
            {
				          MatSetValue(Alinear, i, 3*N_Nodes+cols[j], -Delta_t*dummy, 	ADD_VALUES);
            }
        }
        MatRestoreRow(GRAD, i, &numberOfNonZeros, &cols, &values);
	}
  // Add Laplacian into Alinear
	for (unsigned int i = 0; i<N_Nodes;++i)
	{
        MatGetRow(Laplacian, i, &numberOfNonZeros, &cols, &values);
        for (int j=0;j<numberOfNonZeros;++j)
        {
            dummy = (values[j]);
            //if (dummy!=0)
            {
				          MatSetValue(Alinear, 3*N_Nodes+i, 3*N_Nodes+cols[j], -Delta_t*dummy, 	ADD_VALUES);
            }
        }
        MatRestoreRow(Laplacian, i, &numberOfNonZeros, &cols, &values);
	}
  if (viscous == 1)
  {
	for (unsigned int i = 0; i<N_Nodes;++i)
	{
    MatGetRow(Laplacian, i, &numberOfNonZeros, &cols, &values);
    for (int j=0;j<numberOfNonZeros;++j)
    {
        dummy = (values[j]);
        MatSetValue(Alinear, i,           cols[j],           -Delta_t/(2.0*Re)*dummy, 	ADD_VALUES);
        MatSetValue(Alinear, N_Nodes+i,   N_Nodes+cols[j],   -Delta_t/(2.0*Re)*dummy, 	ADD_VALUES);
        MatSetValue(Alinear, 2*N_Nodes+i, 2*N_Nodes+cols[j], -Delta_t/(2.0*Re)*dummy, 	ADD_VALUES);
    }
    MatRestoreRow(Laplacian, i, &numberOfNonZeros, &cols, &values);
  }
  if (include_viscous_divergence_correction)
  {
    std::cout << "Viscous Divergence Correction!" << std::endl;
  Mat A_PU_viscous;
  MatMatMult(DIVdx, Laplacian, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &A_PU_viscous);
  for (unsigned int i = 0; i<N_Nodes;++i)
  {
        MatGetRow(A_PU_viscous, i, &numberOfNonZeros, &cols, &values);
        for (int j=0;j<numberOfNonZeros;++j)
        {
            dummy = (values[j]);
            {
                  MatSetValue(Alinear, 3*N_Nodes+i, cols[j], -Delta_t/(2.0*Re)*dummy, 	ADD_VALUES);
            }
        }
        MatRestoreRow(A_PU_viscous, i, &numberOfNonZeros, &cols, &values);
  }
  MatDestroy(&A_PU_viscous);
  Mat A_PV_viscous;
  MatMatMult(DIVdy, Laplacian, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &A_PV_viscous);
  for (unsigned int i = 0; i<N_Nodes;++i)
  {
        MatGetRow(A_PV_viscous, i, &numberOfNonZeros, &cols, &values);
        for (int j=0;j<numberOfNonZeros;++j)
        {
            dummy = (values[j]);
            {
                  MatSetValue(Alinear, 3*N_Nodes+i, N_Nodes+cols[j], -Delta_t/(2.0*Re)*dummy, 	ADD_VALUES);
            }
        }
        MatRestoreRow(A_PV_viscous, i, &numberOfNonZeros, &cols, &values);
  }
  MatDestroy(&A_PV_viscous);
  Mat A_PW_viscous;
  MatMatMult(DIVdz, Laplacian, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &A_PW_viscous);
  for (unsigned int i = 0; i<N_Nodes;++i)
  {
        MatGetRow(A_PW_viscous, i, &numberOfNonZeros, &cols, &values);
        for (int j=0;j<numberOfNonZeros;++j)
        {
            dummy = (values[j]);
            {
                  MatSetValue(Alinear, 3*N_Nodes+i, 2*N_Nodes+cols[j], -Delta_t/(2.0*Re)*dummy, 	ADD_VALUES);
            }
        }
        MatRestoreRow(A_PW_viscous, i, &numberOfNonZeros, &cols, &values);
  }
  MatDestroy(&A_PW_viscous);
}
}
PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
void Calculate_Jacobian_Cuboid(const std::unique_ptr<Element> &Element, const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices, const double &r_p, const double &s_p, const double &t_p, double &det_J, double &drdx, double &drdy, double &drdz, double &dsdx, double &dsdy, double &dsdz, double &dtdx, double &dtdy, double &dtdz, double &x, double &y, double &z)
{
        double x0 = List_Of_Vertices[Element->getVertex_V1()]->getxCoordinate();
        double y0 = List_Of_Vertices[Element->getVertex_V1()]->getyCoordinate();
        double z0 = List_Of_Vertices[Element->getVertex_V1()]->getzCoordinate();
        double x1 = List_Of_Vertices[Element->getVertex_V2()]->getxCoordinate();
        double y1 = List_Of_Vertices[Element->getVertex_V2()]->getyCoordinate();
        double z1 = List_Of_Vertices[Element->getVertex_V2()]->getzCoordinate();
        double x2 = List_Of_Vertices[Element->getVertex_V3()]->getxCoordinate();
        double y2 = List_Of_Vertices[Element->getVertex_V3()]->getyCoordinate();
        double z2 = List_Of_Vertices[Element->getVertex_V3()]->getzCoordinate();
        double x3 = List_Of_Vertices[Element->getVertex_V4()]->getxCoordinate();
        double y3 = List_Of_Vertices[Element->getVertex_V4()]->getyCoordinate();
        double z3 = List_Of_Vertices[Element->getVertex_V4()]->getzCoordinate();
        double x4 = List_Of_Vertices[Element->getVertex_V5()]->getxCoordinate();
        double y4 = List_Of_Vertices[Element->getVertex_V5()]->getyCoordinate();
        double z4 = List_Of_Vertices[Element->getVertex_V5()]->getzCoordinate();
        double x5 = List_Of_Vertices[Element->getVertex_V6()]->getxCoordinate();
        double y5 = List_Of_Vertices[Element->getVertex_V6()]->getyCoordinate();
        double z5 = List_Of_Vertices[Element->getVertex_V6()]->getzCoordinate();
        double x6 = List_Of_Vertices[Element->getVertex_V7()]->getxCoordinate();
        double y6 = List_Of_Vertices[Element->getVertex_V7()]->getyCoordinate();
        double z6 = List_Of_Vertices[Element->getVertex_V7()]->getzCoordinate();
        double x7 = List_Of_Vertices[Element->getVertex_V8()]->getxCoordinate();
        double y7 = List_Of_Vertices[Element->getVertex_V8()]->getyCoordinate();
        double z7 = List_Of_Vertices[Element->getVertex_V8()]->getzCoordinate();

        x = (1.0-r_p)*(1.0-s_p)*(1.0-t_p)*x0 + (1.0+r_p)*(1.0-s_p)*(1.0-t_p)*x1 + (1.0+r_p)*(1.0+s_p)*(1.0-t_p)*x2 + (1.0-r_p)*(1.0+s_p)*(1.0-t_p)*x3 + (1.0-r_p)*(1.0-s_p)*(1.0+t_p)*x4 + (1.0+r_p)*(1.0-s_p)*(1.0+t_p)*x5 + (1.0+r_p)*(1.0+s_p)*(1.0+t_p)*x6 + (1.0-r_p)*(1.0+s_p)*(1.0+t_p)*x7;
        y = (1.0-r_p)*(1.0-s_p)*(1.0-t_p)*y0 + (1.0+r_p)*(1.0-s_p)*(1.0-t_p)*y1 + (1.0+r_p)*(1.0+s_p)*(1.0-t_p)*y2 + (1.0-r_p)*(1.0+s_p)*(1.0-t_p)*y3 + (1.0-r_p)*(1.0-s_p)*(1.0+t_p)*y4 + (1.0+r_p)*(1.0-s_p)*(1.0+t_p)*y5 + (1.0+r_p)*(1.0+s_p)*(1.0+t_p)*y6 + (1.0-r_p)*(1.0+s_p)*(1.0+t_p)*y7;
        z = (1.0-r_p)*(1.0-s_p)*(1.0-t_p)*z0 + (1.0+r_p)*(1.0-s_p)*(1.0-t_p)*z1 + (1.0+r_p)*(1.0+s_p)*(1.0-t_p)*z2 + (1.0-r_p)*(1.0+s_p)*(1.0-t_p)*z3 + (1.0-r_p)*(1.0-s_p)*(1.0+t_p)*z4 + (1.0+r_p)*(1.0-s_p)*(1.0+t_p)*z5 + (1.0+r_p)*(1.0+s_p)*(1.0+t_p)*z6 + (1.0-r_p)*(1.0+s_p)*(1.0+t_p)*z7;

        x = x/8.0;
        y = y/8.0;
        z = z/8.0;

        double dxdr = (-x0+x1+x2-x3-x4+x5+x6-x7)/8.0 + (x0-x1+x2-x3+x4-x5+x6-x7)/8.0*s_p + (x0-x1-x2+x3-x4+x5+x6-x7)/8.0*t_p + (-x0+x1-x2+x3+x4-x5+x6-x7)/8.0*s_p*t_p;
        double dxds = (-x0-x1+x2+x3-x4-x5+x6+x7)/8.0 + (x0-x1+x2-x3+x4-x5+x6-x7)/8.0*r_p + (x0+x1-x2-x3-x4-x5+x6+x7)/8.0*t_p + (-x0+x1-x2+x3+x4-x5+x6-x7)/8.0*r_p*t_p;
        double dxdt = (-x0-x1-x2-x3+x4+x5+x6+x7)/8.0 + (x0+x1-x2-x3-x4-x5+x6+x7)/8.0*s_p + (x0-x1-x2+x3-x4+x5+x6-x7)/8.0*r_p + (-x0+x1-x2+x3+x4-x5+x6-x7)/8.0*r_p*s_p;
        double dydr = (-y0+y1+y2-y3-y4+y5+y6-y7)/8.0 + (y0-y1+y2-y3+y4-y5+y6-y7)/8.0*s_p + (y0-y1-y2+y3-y4+y5+y6-y7)/8.0*t_p + (-y0+y1-y2+y3+y4-y5+y6-y7)/8.0*s_p*t_p;
        double dyds = (-y0-y1+y2+y3-y4-y5+y6+y7)/8.0 + (y0-y1+y2-y3+y4-y5+y6-y7)/8.0*r_p + (y0+y1-y2-y3-y4-y5+y6+y7)/8.0*t_p + (-y0+y1-y2+y3+y4-y5+y6-y7)/8.0*r_p*t_p;
        double dydt = (-y0-y1-y2-y3+y4+y5+y6+y7)/8.0 + (y0+y1-y2-y3-y4-y5+y6+y7)/8.0*s_p + (y0-y1-y2+y3-y4+y5+y6-y7)/8.0*r_p + (-y0+y1-y2+y3+y4-y5+y6-y7)/8.0*r_p*s_p;
        double dzdr = (-z0+z1+z2-z3-z4+z5+z6-z7)/8.0 + (z0-z1+z2-z3+z4-z5+z6-z7)/8.0*s_p + (z0-z1-z2+z3-z4+z5+z6-z7)/8.0*t_p + (-z0+z1-z2+z3+z4-z5+z6-z7)/8.0*s_p*t_p;
        double dzds = (-z0-z1+z2+z3-z4-z5+z6+z7)/8.0 + (z0-z1+z2-z3+z4-z5+z6-z7)/8.0*r_p + (z0+z1-z2-z3-z4-z5+z6+z7)/8.0*t_p + (-z0+z1-z2+z3+z4-z5+z6-z7)/8.0*r_p*t_p;
        double dzdt = (-z0-z1-z2-z3+z4+z5+z6+z7)/8.0 + (z0+z1-z2-z3-z4-z5+z6+z7)/8.0*s_p + (z0-z1-z2+z3-z4+z5+z6-z7)/8.0*r_p + (-z0+z1-z2+z3+z4-z5+z6-z7)/8.0*r_p*s_p;

        // std::cout << "dxdr = " << dxdr << std::endl;
        // std::cout << "dxds = " << dxds << std::endl;
        // std::cout << "dxdt = " << dxdt << std::endl;
        // std::cout << "dydr = " << dydr << std::endl;
        // std::cout << "dyds = " << dyds << std::endl;
        // std::cout << "dydt = " << dydt << std::endl;
        // std::cout << "dzdr = " << dzdr << std::endl;
        // std::cout << "dzds = " << dzds << std::endl;
        // std::cout << "dzdt = " << dzdt << std::endl;
        det_J = dzdt*(dxdr*dyds-dxds*dydr)-dzds*(dxdt*dydr-dxdr*dydt)+dzdr*(dxds*dydt-dxdt*dyds);

        //std::cout << det_J << std::endl;
        //Element->set_detJ(det_J*8.0);
        //std::cout << "dxdr*dyds-dxds*dydr = " << dxdr*dyds-dxds*dydr << std::endl;
        drdx = (dyds*dzdt-dydt*dzds)/det_J;
        drdy = -(dxds*dzdt-dxdt*dzds)/det_J;
        drdz = (dxds*dydt-dxdt*dyds)/det_J;
        dsdx = -(dydr*dzdt-dxdt*dydr)/det_J;
        dsdy = (dxdr*dzdt-dxdt*dzdr)/det_J;
        dsdz = -(dxdr*dydt-dxdt*dydr)/det_J;
        dtdx = (dydr*dzds-dyds*dzdr)/det_J;
        dtdy = -(dxdr*dzds-dxds*dzdr)/det_J;
        dtdz = (dxdr*dyds-dxds*dydr)/det_J;
        //std::cout << "det_Jacobian = " << det_J << std::endl;
}
/*--------------------------------------------------------------------------*/
PetscErrorCode DestroyDiscreteOperators(DiscreteOperators &ops)
{
    PetscFunctionBeginUser;

    PetscCall(MatDestroy(&ops.M));
    PetscCall(MatDestroy(&ops.M_small));
    PetscCall(MatDestroy(&ops.invM));
    PetscCall(MatDestroy(&ops.invM_small));

    PetscCall(MatDestroy(&ops.GRAD));

    PetscCall(MatDestroy(&ops.DIV));
    PetscCall(MatDestroy(&ops.DIVdx));
    PetscCall(MatDestroy(&ops.DIVdy));
    PetscCall(MatDestroy(&ops.DIVdz));

    PetscCall(MatDestroy(&ops.CURL));
    PetscCall(MatDestroy(&ops.CURLdx));
    PetscCall(MatDestroy(&ops.CURLdy));
    PetscCall(MatDestroy(&ops.CURLdz));
    PetscCall(MatDestroy(&ops.CurlBoundary));
    PetscCall(MatDestroy(&ops.Laplacian));

    PetscCall(MatDestroy(&ops.ConstantRotationalMat));
    PetscCall(MatDestroy(&ops.ConstantRotationalMatx));
    PetscCall(MatDestroy(&ops.ConstantRotationalMaty));
    PetscCall(MatDestroy(&ops.ConstantRotationalMatz));

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode DestroyAppCtx(AppCtx &user)
{
    PetscFunctionBeginUser;

    PetscCall(ISDestroy(&user.isu));
    PetscCall(ISDestroy(&user.isv));
    PetscCall(ISDestroy(&user.isw));
    PetscCall(ISDestroy(&user.isp));
    PetscCall(ISDestroy(&user.isvel));

    PetscCall(VecDestroy(&user.Velocity));
    PetscCall(VecDestroy(&user.Velocity_U));
    PetscCall(VecDestroy(&user.Velocity_V));
    PetscCall(VecDestroy(&user.Velocity_W));

    PetscCall(VecDestroy(&user.Velocity_n));
    PetscCall(VecDestroy(&user.Velocity_k));
    PetscCall(VecDestroy(&user.Vsum));

    PetscCall(VecDestroy(&user.ForceHalf));
    PetscCall(VecDestroy(&user.WaveAttractorForceShape));

    PetscCall(MatDestroy(&user.Alinear));
    PetscCall(MatDestroy(&user.A));

    PetscCall(VecDestroy(&user.blinear));
    PetscCall(VecDestroy(&user.bnonlinear));
    PetscCall(VecDestroy(&user.b));

    PetscCall(VecDestroy(&user.X));

    PetscCall(VecDestroy(&user.OmegaStar));
    PetscCall(VecDestroy(&user.OmegaStar_x));
    PetscCall(VecDestroy(&user.OmegaStar_y));
    PetscCall(VecDestroy(&user.OmegaStar_z));

    PetscCall(DestroyDiscreteOperators(user.ops));

    PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------*/
PetscErrorCode createDiagnosticsWorkspace(
    const PetscInt N_Nodes,
    const Vec VelocityTemplate,
    DiagnosticsWorkspace &work)
{
    PetscFunctionBegin;

    PetscCall(VecCreateSeq(PETSC_COMM_SELF, N_Nodes, &work.div));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, N_Nodes, &work.tmpN));

    PetscCall(VecDuplicate(VelocityTemplate, &work.Vmid));
    PetscCall(VecDuplicate(VelocityTemplate, &work.Mdiag));
    PetscCall(VecZeroEntries(work.Mdiag));
    PetscCall(VecDuplicate(VelocityTemplate, &work.AV));
    PetscCall(VecDuplicate(VelocityTemplate, &work.MAV));
    PetscCall(VecDuplicate(VelocityTemplate, &work.err));
    PetscCall(VecDuplicate(VelocityTemplate, &work.tmp3N));

    PetscCall(VecDuplicate(VelocityTemplate, &work.boundaryWork));
    PetscCall(VecDuplicate(VelocityTemplate, &work.Nrelative));
    PetscCall(VecDuplicate(VelocityTemplate, &work.Nrotation));
    PetscCall(VecDuplicate(VelocityTemplate, &work.Ntotal));
    PetscCall(VecZeroEntries(work.boundaryWork));
    PetscCall(VecZeroEntries(work.Nrelative));
    PetscCall(VecZeroEntries(work.Nrotation));
    PetscCall(VecZeroEntries(work.Ntotal));

    PetscCall(VecDuplicate(VelocityTemplate, &work.omega));
    PetscCall(VecDuplicate(VelocityTemplate, &work.Frot));
    PetscCall(VecDuplicate(VelocityTemplate, &work.Momega));
    PetscCall(VecDuplicate(VelocityTemplate, &work.gradP));
    PetscCall(VecDuplicate(VelocityTemplate, &work.curlGradP));

    PetscFunctionReturn(0);
}
PetscErrorCode destroyDiagnosticsWorkspace(DiagnosticsWorkspace &work)
{
    PetscFunctionBegin;

    PetscCall(VecDestroy(&work.div));
    PetscCall(VecDestroy(&work.Vmid));
    PetscCall(VecDestroy(&work.boundaryWork));
    PetscCall(VecDestroy(&work.Nrelative));
    PetscCall(VecDestroy(&work.Nrotation));
    PetscCall(VecDestroy(&work.Ntotal));
    PetscCall(VecDestroy(&work.Mdiag));
    PetscCall(VecDestroy(&work.AV));
    PetscCall(VecDestroy(&work.MAV));
    PetscCall(VecDestroy(&work.tmpN));
    PetscCall(VecDestroy(&work.tmp3N));

    PetscCall(VecDestroy(&work.omega));
    PetscCall(VecDestroy(&work.Momega));
    PetscCall(VecDestroy(&work.Frot));
    PetscCall(VecDestroy(&work.gradP));
    PetscCall(VecDestroy(&work.curlGradP));

    PetscFunctionReturn(0);
}
PetscErrorCode initializeRotationVector(
    Vec Frot,
    IS isu,
    IS isv,
    IS isw,
    const RotationParameters &rot)
{
    Vec Fu = nullptr, Fv = nullptr, Fw = nullptr;

    PetscFunctionBeginUser;

    PetscCall(VecZeroEntries(Frot));

    PetscCall(VecGetSubVector(Frot, isu, &Fu));
    PetscCall(VecGetSubVector(Frot, isv, &Fv));
    PetscCall(VecGetSubVector(Frot, isw, &Fw));

    PetscCall(VecSet(Fu, rot.f1));
    PetscCall(VecSet(Fv, rot.f2));
    PetscCall(VecSet(Fw, rot.f3));

    PetscCall(VecRestoreSubVector(Frot, isu, &Fu));
    PetscCall(VecRestoreSubVector(Frot, isv, &Fv));
    PetscCall(VecRestoreSubVector(Frot, isw, &Fw));

    PetscFunctionReturn(PETSC_SUCCESS);
}
// ============================================================
// Mass-norm diagnostics and fixed-lambda Beltrami filter
// ============================================================
PetscErrorCode printDivergenceNorms(
    const char *label,
    Vec Velocity,
    const DiscreteOperators &ops,
    PetscInt N_Nodes)
{
    PetscFunctionBegin;

    Vec divV = nullptr;

    PetscCall(VecCreateSeq(PETSC_COMM_SELF, N_Nodes, &divV));
    PetscCall(MatMult(ops.DIV, Velocity, divV));

    PetscReal norm2 = 0.0;
    PetscReal normInf = 0.0;

    PetscCall(VecNorm(divV, NORM_2, &norm2));
    PetscCall(VecNorm(divV, NORM_INFINITY, &normInf));

    PetscPrintf(PETSC_COMM_SELF,
        "\n============================================================\n"
        "Divergence diagnostic: %s\n"
        "------------------------------------------------------------\n"
        "||DIV V||_2   = %.16e\n"
        "||DIV V||_inf = %.16e\n"
        "============================================================\n\n",
        label,
        (double)norm2,
        (double)normInf);

    PetscCall(VecDestroy(&divV));

    PetscFunctionReturn(0);
}

PetscErrorCode setupFieldSplitKSP(
    KSP ksp,
    Mat Amat,
    Mat Pmat,
    IS isvel,
    IS isp,
    PetscReal ksp_rtol,
    PetscReal ksp_atol)
{
    PetscFunctionBegin;

    PC pc;

    PetscCall(KSPSetType(ksp, KSPGMRES));
    PetscCall(KSPSetTolerances(
        ksp,
        ksp_rtol,
        ksp_atol,
        1e30,
        PETSC_DEFAULT));

    PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));

    PetscCall(KSPSetOperators(ksp, Amat, Pmat));

    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCFIELDSPLIT));
    PetscCall(PCFieldSplitSetType(pc, PC_COMPOSITE_MULTIPLICATIVE));

    PetscCall(PCFieldSplitSetIS(pc, "vel", isvel));
    PetscCall(PCFieldSplitSetIS(pc, "p",   isp));

    PetscCall(KSPSetReusePreconditioner(ksp, PETSC_TRUE));

    /*
       Let PETSc create the sub-KSP objects.
    */
    PetscCall(KSPSetUp(ksp));

    PetscInt nsplit = 0;
    KSP *subksp = nullptr;

    PetscCall(PCFieldSplitGetSubKSP(pc, &nsplit, &subksp));

    PetscCheck(
        nsplit == 2,
        PETSC_COMM_SELF,
        PETSC_ERR_ARG_WRONG,
        "Expected exactly 2 field splits: velocity and pressure.");

    /*
       Split ordering follows insertion order:
       0 = "vel"
       1 = "p"
    */

    // Velocity split
    {
        PC subpc;
        PetscCall(KSPSetType(subksp[0], KSPPREONLY));
        PetscCall(KSPGetPC(subksp[0], &subpc));
        PetscCall(PCSetType(subpc, PCILU));
        PetscCall(PCFactorSetLevels(subpc, 0));
    }

    // Pressure split
    {
        PC subpc;
        PetscCall(KSPSetType(subksp[1], KSPPREONLY));
        PetscCall(KSPGetPC(subksp[1], &subpc));
        PetscCall(PCSetType(subpc, PCILU));
        PetscCall(PCFactorSetLevels(subpc, 0));
        //PetscCall(PCFactorSetLevels(subpc, 1));
    }

    PetscCall(PetscFree(subksp));

    PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------------*/
// Wave attractor
/*--------------------------------------------------------------------------*/
PetscReal waveAttractorOmega()
{
    return std::acos(0.5 * std::sqrt(2.0));
}

PetscReal waveAttractorPeriod()
{
    return 2.0 * PETSC_PI / waveAttractorOmega();
}
PetscReal waveAttractorForcingOffTime(const RunOptions &opt)
{
    return opt.wave_attractor_forcing_off_periods * waveAttractorPeriod();
}
PetscBool isWaveAttractorForcingOffStep(
    const PetscReal t,
    const PetscReal Delta_t,
    const RunOptions &opt)
{
    if (opt.wave_attractor_turn_off_forcing != PETSC_TRUE)
    {
        return PETSC_FALSE;
    }

    const PetscReal toff = waveAttractorForcingOffTime(opt);
    const PetscReal eps_time =
        10.0 * PETSC_MACHINE_EPSILON * PetscMax(1.0, PetscAbsReal(toff));

    const PetscBool is_step =
        ((PetscAbsReal(t - toff) <= 0.5 * Delta_t + eps_time) &&
         (t >= toff - eps_time))
            ? PETSC_TRUE
            : PETSC_FALSE;

    return is_step;
}
PetscErrorCode printWaveAttractorForcingConfiguration(const RunOptions &opt)
{
    PetscFunctionBeginUser;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        " wave attractor forcing off = %d\n"
        " forcing off periods        = %.6e\n"
        " forcing off time           = %.16e\n"
        " forcing period             = %.16e\n",
        (int)opt.wave_attractor_turn_off_forcing,
        (double)opt.wave_attractor_forcing_off_periods,
        (double)waveAttractorForcingOffTime(opt),
        (double)waveAttractorPeriod()));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode initializeWaveAttractorForceShape(
    AppCtx &user,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements)
{
    PetscFunctionBeginUser;

    PetscCheck(
        user.ForceHalf != nullptr,
        PETSC_COMM_SELF,
        PETSC_ERR_ARG_WRONGSTATE,
        "ForceHalf must be allocated before initializing the force shape.");

    PetscCheck(
        user.N_Nodes > 0 && user.Np > 0,
        PETSC_COMM_SELF,
        PETSC_ERR_ARG_WRONGSTATE,
        "N_Nodes and Np must be initialized before constructing the force.");

    /*
       Allocate the persistent spatial shape once.
    */
    if (user.WaveAttractorForceShape == nullptr)
    {
        PetscCall(VecDuplicate(
            user.ForceHalf,
            &user.WaveAttractorForceShape));
    }

    PetscCall(VecZeroEntries(
        user.WaveAttractorForceShape));

    /*
       Raw spatial force:

           B_0 = 0.5 * (-y, x, 0)^T.
    */
    for (const auto &elem_ptr : List_Of_Elements)
    {
        const Element &K = *elem_ptr;

        const PetscInt pos =
            static_cast<PetscInt>(K.get_pos());

        const std::vector<double> x =
            K.get_node_coordinates_x();

        const std::vector<double> y =
            K.get_node_coordinates_y();

        PetscCheck(
            static_cast<PetscInt>(x.size()) >=
                static_cast<PetscInt>(user.Np) &&
            static_cast<PetscInt>(y.size()) >=
                static_cast<PetscInt>(user.Np),
            PETSC_COMM_SELF,
            PETSC_ERR_ARG_SIZ,
            "Element coordinate arrays contain fewer than Np entries.");

        for (PetscInt a = 0;
             a < static_cast<PetscInt>(user.Np);
             ++a)
        {
            const PetscInt iu =
                pos + a;

            const PetscInt iv =
                static_cast<PetscInt>(user.N_Nodes) + pos + a;

            /*
               The vector was zeroed, so the z component remains zero.
            */
            PetscCall(VecSetValue(
                user.WaveAttractorForceShape,
                iu,
                -0.5 * y[a],
                INSERT_VALUES));

            PetscCall(VecSetValue(
                user.WaveAttractorForceShape,
                iv,
                0.5 * x[a],
                INSERT_VALUES));
        }
    }

    PetscCall(VecAssemblyBegin(
        user.WaveAttractorForceShape));

    PetscCall(VecAssemblyEnd(
        user.WaveAttractorForceShape));

    PetscCall(printDivergenceNorms(
        "raw wave-attractor force shape",
        user.WaveAttractorForceShape,
        user.ops,
        user.N_Nodes));

    /*
       Use the same DIV, GRAD and scalar Laplacian as for the
       divergence-free velocity projection.
    */
    PetscCall(projectVectorToDiscreteDivergenceFree(
        static_cast<PetscInt>(user.N_Nodes),
        user.WaveAttractorForceShape,
        user.ops.DIV,
        user.ops.GRAD,
        user.ops.Laplacian));

    PetscCall(printDivergenceNorms(
        "projected wave-attractor force shape",
        user.WaveAttractorForceShape,
        user.ops,
        user.N_Nodes));

    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode projectVectorToDiscreteDivergenceFree(
    PetscInt N_Nodes,
    Vec Velocity, // can be general Field
    Mat DIV,
    Mat GRAD,
    Mat Laplacian)
{
    PetscFunctionBeginUser;

    const PetscReal abstol = 1e-12;

    Vec rhsP  = nullptr;
    Vec P     = nullptr;
    Vec gradP = nullptr;

    KSP ksp = nullptr;
    PC  pc  = nullptr;
    MatNullSpace pressureNullSpace = nullptr;

    PetscCall(VecCreateSeq(PETSC_COMM_SELF, N_Nodes, &rhsP));
    PetscCall(VecDuplicate(rhsP, &P));
    PetscCall(VecDuplicate(Velocity, &gradP));

    /*
       rhsP = DIV Velocity.
       We solve

           LaplacianClean P = DIV Velocity

       and then project

           Velocity <- Velocity - GRAD P.

       Without pressure pinning, LaplacianClean is singular.
       The pressure is determined only up to a constant.
    */
    PetscCall(MatMult(DIV, Velocity, rhsP));

    PetscReal divBeforeInf = 0.0;
    PetscReal divBefore2   = 0.0;

    PetscCall(VecNorm(rhsP, NORM_INFINITY, &divBeforeInf));
    PetscCall(VecNorm(rhsP, NORM_2, &divBefore2));

    if (divBeforeInf > abstol)
    {
        /*
           Constant pressure nullspace.

           PETSC_TRUE means: the constant vector is in the nullspace.
           This is the correct nullspace for periodic/Neumann pressure
           Poisson problems.
        */
        PetscCall(MatNullSpaceCreate(
            PETSC_COMM_SELF,
            PETSC_TRUE,
            0,
            nullptr,
            &pressureNullSpace));

        PetscCall(MatSetNullSpace(Laplacian, pressureNullSpace));
        PetscCall(MatSetTransposeNullSpace(Laplacian, pressureNullSpace));

        /*
           Compatibility condition.

           For the singular solve to be solvable, rhsP must be orthogonal
           to the constant nullspace. In exact arithmetic this should hold
           for a periodic divergence, but in floating point it is safer to
           enforce it explicitly.
        */
        PetscCall(MatNullSpaceRemove(pressureNullSpace, rhsP));

        PetscCall(VecSet(P, 0.0));

        PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
        PetscCall(KSPSetOperators(ksp, Laplacian, Laplacian));

        /*
           Do not use KSPPREONLY + PCLU here.
           The matrix is singular.

           GMRES is robust even if LaplacianClean is mildly nonsymmetric.
           If you have verified symmetry and positive semidefiniteness,
           KSPCG or KSPMINRES may also be appropriate.
        */
        PetscCall(KSPSetType(ksp, KSPGMRES));
        PetscCall(KSPGetPC(ksp, &pc));
        //PetscCall(PCSetType(pc, PCJACOBI));
        PetscCall(PCSetType(pc, PCILU));
        PetscCall(PCFactorSetLevels(pc,0));

        PetscCall(KSPSetTolerances(
            ksp,
            1e-12,        // relative tolerance
            1e-14,        // absolute tolerance
            PETSC_DEFAULT,
            1000));

        PetscCall(KSPSetFromOptions(ksp));

        PetscCall(KSPSolve(ksp, rhsP, P));

        KSPConvergedReason reason;
        PetscCall(KSPGetConvergedReason(ksp, &reason));

        if (reason < 0)
        {
            PetscInt its = 0;
            PetscReal rnorm = 0.0;
            const char *reasonStr = nullptr;

            PetscCall(KSPGetIterationNumber(ksp, &its));
            PetscCall(KSPGetResidualNorm(ksp, &rnorm));
            PetscCall(KSPGetConvergedReasonString(ksp, &reasonStr));

            PetscPrintf(PETSC_COMM_SELF,
                "Discrete divergence projection failed: reason = %d (%s), "
                "iterations = %d, residual = %.16e\n",
                (int)reason,
                reasonStr ? reasonStr : "unknown",
                (int)its,
                (double)rnorm);

            SETERRQ(PETSC_COMM_SELF,
                    PETSC_ERR_CONV_FAILED,
                    "Discrete divergence projection failed.");
        }

        /*
           Optional: fix representative by forcing mean(P) = 0.
           This does not change GRAD P, assuming constants are exactly in
           the nullspace of GRAD.
        */
        PetscScalar sumP = 0.0;
        PetscCall(VecSum(P, &sumP));
        PetscCall(VecShift(P, -sumP / (PetscScalar)N_Nodes));

        PetscCall(MatMult(GRAD, P, gradP));
        PetscCall(VecAXPY(Velocity, -1.0, gradP));

        PetscCall(KSPDestroy(&ksp));
        PetscCall(MatNullSpaceDestroy(&pressureNullSpace));
    }

    PetscCall(MatMult(DIV, Velocity, rhsP));

    PetscReal divAfterInf = 0.0;
    PetscReal divAfter2   = 0.0;

    PetscCall(VecNorm(rhsP, NORM_INFINITY, &divAfterInf));
    PetscCall(VecNorm(rhsP, NORM_2, &divAfter2));

    PetscPrintf(PETSC_COMM_SELF,
        "\033[1;32m"
        "  Velocity projected to discrete divergence-free space using nullspace pressure solve.\n"
        "    before: ||DIV V||_inf = %.16e, ||DIV V||_2 = %.16e\n"
        "    after : ||DIV V||_inf = %.16e, ||DIV V||_2 = %.16e"
        "\033[0m\n",
        (double)divBeforeInf,
        (double)divBefore2,
        (double)divAfterInf,
        (double)divAfter2);

    PetscCall(VecDestroy(&rhsP));
    PetscCall(VecDestroy(&P));
    PetscCall(VecDestroy(&gradP));

    PetscFunctionReturn(0);
}

PetscErrorCode updateWaveAttractorForceHalf(
    AppCtx &user,
    PetscReal tn,
    PetscBool *include_force)
{
    PetscFunctionBeginUser;

    PetscCheck(
        include_force != nullptr,
        PETSC_COMM_SELF,
        PETSC_ERR_ARG_WRONGSTATE,
        "include_force has not been allocated.");

    PetscCheck(
        user.ForceHalf != nullptr,
        PETSC_COMM_SELF,
        PETSC_ERR_ARG_WRONGSTATE,
        "ForceHalf has not been allocated.");

    PetscCheck(
        user.WaveAttractorForceShape != nullptr,
        PETSC_COMM_SELF,
        PETSC_ERR_ARG_WRONGSTATE,
        "WaveAttractorForceShape has not been initialized.");

    *include_force = PETSC_FALSE;

    PetscCall(VecZeroEntries(user.ForceHalf));

    /*
       Check whether forcing has already been switched off.
    */
    if (user.opt.wave_attractor_turn_off_forcing == PETSC_TRUE)
    {
        const PetscReal toff =
            waveAttractorForcingOffTime(user.opt);

        const PetscReal eps_time =
            10.0 * PETSC_MACHINE_EPSILON *
            PetscMax(1.0, PetscAbsReal(toff));

        if (tn >= toff - eps_time)
        {
            PetscFunctionReturn(PETSC_SUCCESS);
        }
    }

    const PetscReal tnp1 =
        tn + user.Delta_t;

    const PetscReal omega =
        waveAttractorOmega();
    /*
       Same temporal discretization as in the original implementation:

           a^{n+1/2}
             = 0.5 [cos(omega t_n)
                    + cos(omega t_{n+1})].
    */
    const PetscReal temporal_factor =
        0.5 *
        (
            PetscCosReal(omega * tn) +
            PetscCosReal(omega * tnp1)
        );

    PetscCall(VecCopy(
        user.WaveAttractorForceShape,
        user.ForceHalf));

    PetscCall(VecScale(
        user.ForceHalf,
        temporal_factor));

    *include_force = PETSC_TRUE;

    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode writeWaveAttractorForcingOffSnapshot(
    AppCtx &user,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    const Vec ForceHalf,
    const PetscReal t)
{
    PetscFunctionBeginUser;

    PetscCheck(ForceHalf != nullptr,
               PETSC_COMM_SELF,
               PETSC_ERR_ARG_NULL,
               "ForceHalf must be non-null when writing the wave-attractor forcing-off snapshot.");

    Vec Vmid          = nullptr;
    Vec relOmega      = nullptr;
    Vec absOmega      = nullptr;
    Vec AVmid         = nullptr;
    Vec relOmegaState = nullptr;
    Vec absOmegaState = nullptr;

    PetscCall(VecDuplicate(user.Velocity, &Vmid));
    PetscCall(VecDuplicate(user.Velocity, &relOmega));
    PetscCall(VecDuplicate(user.Velocity, &absOmega));
    PetscCall(VecDuplicate(user.Velocity, &AVmid));
    PetscCall(VecDuplicate(user.Velocity, &relOmegaState));
    PetscCall(VecDuplicate(user.Velocity, &absOmegaState));

    // ------------------------------------------------------------
    // Midpoint velocity: V^{n+1/2} = 0.5 * (V^n + V^{n+1})
    // ------------------------------------------------------------
    PetscCall(VecCopy(user.Velocity_n, Vmid));
    PetscCall(VecAXPY(Vmid, 1.0, user.Velocity));
    PetscCall(VecScale(Vmid, 0.5));

    // ------------------------------------------------------------
    // Relative vorticity at midpoint and at accepted state
    // ------------------------------------------------------------
    PetscCall(MatMult(user.ops.CURL, Vmid, relOmega));
    PetscCall(MatMult(user.ops.CURL, user.Velocity, relOmegaState));

    // ------------------------------------------------------------
    // Absolute vorticity:
    //
    // omega^a = omega^r + (1/Ro) F.
    //
    // Important:
    // Do not use 2/Ro here. The factor 2 only appears if h_a is
    // evaluated as 0.5 * V^T M (CURL V + 2/Ro F).
    // ------------------------------------------------------------
    PetscCall(VecCopy(relOmega, absOmega));
    PetscCall(VecCopy(relOmegaState, absOmegaState));

    if (user.rot.include_rotation == PETSC_TRUE)
    {
        PetscCheck(user.rot.RossbyNumber != 0.0,
                   PETSC_COMM_SELF,
                   PETSC_ERR_ARG_OUTOFRANGE,
                   "Rossby number must be nonzero when rotation is enabled.");

        PetscCall(VecAXPY(absOmega,
                          1.0 / user.rot.RossbyNumber,
                          user.diagWork.Frot));

        PetscCall(VecAXPY(absOmegaState,
                          1.0 / user.rot.RossbyNumber,
                          user.diagWork.Frot));
    }

    // ------------------------------------------------------------
    // Viscous operator on midpoint velocity:
    //
    // AVmid = A^nu V^{n+1/2}
    //
    // Here A^nu is represented as blockdiag(L,L,L), where L is the
    // scalar viscous/Laplacian operator.
    // ------------------------------------------------------------
    PetscCall(ApplyVelocityLaplacianBlockDiag(
        user.ops.Laplacian,
        Vmid,
        AVmid,
        user.isu,
        user.isv,
        user.isw));

    // ------------------------------------------------------------
    // Construct output filename.
    // ------------------------------------------------------------
    std::ostringstream filename;
    filename << "snapshot_WA_forcing_off_"
             << "Order" << user.opt.Order
             << "_nx" << user.opt.Nel_x
             << "_ny" << user.opt.Nel_y
             << "_nz" << user.opt.Nel_z
             << "_Fo" << std::setprecision(6) << user.opt.Fo
             << "_Re" << std::setprecision(6) << user.Re
             << "_t" << std::fixed << std::setprecision(3) << t
             << ".csv";

    std::ofstream file(filename.str());

    PetscCheck(file.is_open(),
               PETSC_COMM_SELF,
               PETSC_ERR_FILE_OPEN,
               "Could not open wave-attractor snapshot output file.");

    file << std::scientific << std::setprecision(16);

    file
        << "element,local_node,global_node,"
        << "x,y,z,"
        << "u,v,w,"
        << "u_mid,v_mid,w_mid,"
        << "B_x,B_y,B_z,"
        << "rel_omega_x,rel_omega_y,rel_omega_z,"
        << "abs_omega_x,abs_omega_y,abs_omega_z,"
        << "rel_omega_state_x,rel_omega_state_y,rel_omega_state_z,"
        << "abs_omega_state_x,abs_omega_state_y,abs_omega_state_z,"
        << "energy_density,"
        << "energy_density_mid,"
        << "absolute_helicity_density,"
        << "absolute_helicity_density_mid,"
        << "AnuV_mid_x,AnuV_mid_y,AnuV_mid_z,"
        << "E_density_proxy,"
        << "S_density_proxy,"
        << "W_density,"
        << "Womega_density,"
        << "Fo_minus_2_W_density,"
        << "Fo_minus_2_Womega_density\n";

    // ------------------------------------------------------------
    // Access vector arrays.
    // ------------------------------------------------------------
    const PetscScalar *Varr         = nullptr;
    const PetscScalar *Vmidarr      = nullptr;
    const PetscScalar *Barr         = nullptr;
    const PetscScalar *relarr       = nullptr;
    const PetscScalar *absarr       = nullptr;
    const PetscScalar *relStateArr  = nullptr;
    const PetscScalar *absStateArr  = nullptr;
    const PetscScalar *AVarr        = nullptr;
    const PetscScalar *Farr         = nullptr;

    PetscCall(VecGetArrayRead(user.Velocity,       &Varr));
    PetscCall(VecGetArrayRead(Vmid,                &Vmidarr));
    PetscCall(VecGetArrayRead(ForceHalf,           &Barr));
    PetscCall(VecGetArrayRead(relOmega,            &relarr));
    PetscCall(VecGetArrayRead(absOmega,            &absarr));
    PetscCall(VecGetArrayRead(relOmegaState,       &relStateArr));
    PetscCall(VecGetArrayRead(absOmegaState,       &absStateArr));
    PetscCall(VecGetArrayRead(AVmid,               &AVarr));
    PetscCall(VecGetArrayRead(user.diagWork.Frot,  &Farr));

    const double rotfac =
        (user.rot.include_rotation == PETSC_TRUE)
            ? 1.0 / user.rot.RossbyNumber
            : 0.0;

    const double Fo2inv =
        1.0 / (user.opt.Fo * user.opt.Fo);

    // ------------------------------------------------------------
    // Nodal output.
    //
    // The density fields E_density_proxy, S_density_proxy,
    // W_density and Womega_density are pointwise visualization
    // proxies. The exact discrete balance-law terms are the
    // corresponding mass-weighted global forms.
    // ------------------------------------------------------------
    for (const auto &elem_ptr : List_Of_Elements)
    {
        const Element &K = *elem_ptr;

        const PetscInt elem = static_cast<PetscInt>(K.getID());
        const PetscInt pos  = static_cast<PetscInt>(K.get_pos());

        const std::vector<double> x = K.get_node_coordinates_x();
        const std::vector<double> y = K.get_node_coordinates_y();
        const std::vector<double> z = K.get_node_coordinates_z();

        for (PetscInt a = 0; a < user.Np; ++a)
        {
            const PetscInt i  = pos + a;
            const PetscInt iu = i;
            const PetscInt iv = user.N_Nodes + i;
            const PetscInt iw = 2 * user.N_Nodes + i;

            // Accepted state V^{n+1}
            const double u = PetscRealPart(Varr[iu]);
            const double v = PetscRealPart(Varr[iv]);
            const double w = PetscRealPart(Varr[iw]);

            // Midpoint state V^{n+1/2}
            const double um = PetscRealPart(Vmidarr[iu]);
            const double vm = PetscRealPart(Vmidarr[iv]);
            const double wm = PetscRealPart(Vmidarr[iw]);

            // Unscaled body force B^{n+1/2}
            const double bx = PetscRealPart(Barr[iu]);
            const double by = PetscRealPart(Barr[iv]);
            const double bz = PetscRealPart(Barr[iw]);

            // Relative vorticity at midpoint
            const double ox = PetscRealPart(relarr[iu]);
            const double oy = PetscRealPart(relarr[iv]);
            const double oz = PetscRealPart(relarr[iw]);

            // Absolute vorticity at midpoint
            const double ax = PetscRealPart(absarr[iu]);
            const double ay = PetscRealPart(absarr[iv]);
            const double az = PetscRealPart(absarr[iw]);

            // Relative vorticity at accepted state
            const double osx = PetscRealPart(relStateArr[iu]);
            const double osy = PetscRealPart(relStateArr[iv]);
            const double osz = PetscRealPart(relStateArr[iw]);

            // Absolute vorticity at accepted state
            const double asx = PetscRealPart(absStateArr[iu]);
            const double asy = PetscRealPart(absStateArr[iv]);
            const double asz = PetscRealPart(absStateArr[iw]);

            // A^nu V^{n+1/2}
            const double avx = PetscRealPart(AVarr[iu]);
            const double avy = PetscRealPart(AVarr[iv]);
            const double avz = PetscRealPart(AVarr[iw]);

            // Rotation vector coefficients F
            const double fx = PetscRealPart(Farr[iu]);
            const double fy = PetscRealPart(Farr[iv]);
            const double fz = PetscRealPart(Farr[iw]);

            const double energy_density =
                0.5 * (u*u + v*v + w*w);

            const double energy_density_mid =
                0.5 * (um*um + vm*vm + wm*wm);

            // h_a density at accepted state:
            // 0.5 V · omega^r + (1/Ro) V · F.
            const double absolute_helicity_density =
                0.5 * (u*osx + v*osy + w*osz)
                + rotfac * (u*fx + v*fy + w*fz);

            // h_a density at midpoint:
            // 0.5 Vmid · omega^r_mid + (1/Ro) Vmid · F.
            const double absolute_helicity_density_mid =
                0.5 * (um*ox + vm*oy + wm*oz)
                + rotfac * (um*fx + vm*fy + wm*fz);

            // Local visualization proxies for the balance-law terms.
            const double E_density_proxy =
                -(um*avx + vm*avy + wm*avz);

            const double S_density_proxy =
                -(ax*avx + ay*avy + az*avz);

            const double W_density =
                um*bx + vm*by + wm*bz;

            const double Womega_density =
                ax*bx + ay*by + az*bz;

            const double Fo_minus_2_W_density =
                Fo2inv * W_density;

            const double Fo_minus_2_Womega_density =
                Fo2inv * Womega_density;

            file
                << elem << ","
                << a << ","
                << i << ","
                << x[a] << ","
                << y[a] << ","
                << z[a] << ","
                << u << ","
                << v << ","
                << w << ","
                << um << ","
                << vm << ","
                << wm << ","
                << bx << ","
                << by << ","
                << bz << ","
                << ox << ","
                << oy << ","
                << oz << ","
                << ax << ","
                << ay << ","
                << az << ","
                << osx << ","
                << osy << ","
                << osz << ","
                << asx << ","
                << asy << ","
                << asz << ","
                << energy_density << ","
                << energy_density_mid << ","
                << absolute_helicity_density << ","
                << absolute_helicity_density_mid << ","
                << avx << ","
                << avy << ","
                << avz << ","
                << E_density_proxy << ","
                << S_density_proxy << ","
                << W_density << ","
                << Womega_density << ","
                << Fo_minus_2_W_density << ","
                << Fo_minus_2_Womega_density
                << "\n";
        }
    }

    PetscCall(VecRestoreArrayRead(user.Velocity,       &Varr));
    PetscCall(VecRestoreArrayRead(Vmid,                &Vmidarr));
    PetscCall(VecRestoreArrayRead(ForceHalf,           &Barr));
    PetscCall(VecRestoreArrayRead(relOmega,            &relarr));
    PetscCall(VecRestoreArrayRead(absOmega,            &absarr));
    PetscCall(VecRestoreArrayRead(relOmegaState,       &relStateArr));
    PetscCall(VecRestoreArrayRead(absOmegaState,       &absStateArr));
    PetscCall(VecRestoreArrayRead(AVmid,               &AVarr));
    PetscCall(VecRestoreArrayRead(user.diagWork.Frot,  &Farr));

    file.close();

    // ------------------------------------------------------------
    // Also compute exact global mass-weighted quantities for the
    // balance-law terms at the snapshot time.
    // ------------------------------------------------------------

    double H = 0.0;
    double div_max = 0.0;
    double div_l2 = 0.0;

    PetscCall(computeHamiltonian(
        user.ops.M,
        user.Velocity,
        user.diagWork,
        H));

    PetscCall(computeDivergenceDiagnostics(
        user.Velocity,
        user.ops.DIV,
        user.diagWork,
        div_max,
        div_l2));

    // E = - Vmid^T M A^nu Vmid
    PetscCall(MatMult(user.ops.M, AVmid, user.diagWork.MAV));

    PetscScalar Edot = 0.0;
    PetscCall(VecDot(Vmid, user.diagWork.MAV, &Edot));
    const double E = -PetscRealPart(Edot);

    // S = - omega_a^T M A^nu Vmid
    PetscScalar Sdot = 0.0;
    PetscCall(VecDot(absOmega, user.diagWork.MAV, &Sdot));
    const double S = -PetscRealPart(Sdot);

    // W = Vmid^T M B^{n+1/2}
    PetscCall(MatMult(user.ops.M, ForceHalf, user.diagWork.tmp3N));

    PetscScalar Wdot = 0.0;
    PetscCall(VecDot(Vmid, user.diagWork.tmp3N, &Wdot));
    const double W = PetscRealPart(Wdot);

    // Womega = omega_a^T M B^{n+1/2}
    PetscScalar WomegaDot = 0.0;
    PetscCall(VecDot(absOmega, user.diagWork.tmp3N, &WomegaDot));
    const double Womega = PetscRealPart(WomegaDot);

    const double Fo_minus_2_W =
        Fo2inv * W;

    const double Fo_minus_2_Womega =
        Fo2inv * Womega;

    // Absolute helicity at accepted state.
    //
    // h_a = 0.5 V^T M CURL V + (1/Ro) V^T M F.
    //
    // We compute this explicitly to avoid confusing omega^a with the
    // compact helicity-evaluation vector CURL V + 2/Ro F.
    double ha = 0.0;
    {
        Vec MrelOmegaState = nullptr;
        Vec MFrot = nullptr;

        PetscCall(VecDuplicate(user.Velocity, &MrelOmegaState));
        PetscCall(VecDuplicate(user.Velocity, &MFrot));

        PetscCall(MatMult(user.ops.M, relOmegaState, MrelOmegaState));

        PetscScalar hrelDot = 0.0;
        PetscCall(VecDot(user.Velocity, MrelOmegaState, &hrelDot));

        double hrot = 0.0;
        if (user.rot.include_rotation == PETSC_TRUE)
        {
            PetscCall(MatMult(user.ops.M, user.diagWork.Frot, MFrot));

            PetscScalar hrotDot = 0.0;
            PetscCall(VecDot(user.Velocity, MFrot, &hrotDot));

            hrot = rotfac * PetscRealPart(hrotDot);
        }

        ha = 0.5 * PetscRealPart(hrelDot) + hrot;

        PetscCall(VecDestroy(&MrelOmegaState));
        PetscCall(VecDestroy(&MFrot));
    }

    std::ostringstream global_filename;
    global_filename << "snapshot_WA_forcing_off_global_"
                    << "Order" << user.opt.Order
                    << "_nx" << user.opt.Nel_x
                    << "_ny" << user.opt.Nel_y
                    << "_nz" << user.opt.Nel_z
                    << "_Fo" << std::setprecision(6) << user.opt.Fo
                    << "_Re" << std::setprecision(6) << user.Re
                    << "_t" << std::fixed << std::setprecision(3) << t
                    << ".csv";

    std::ofstream global_file(global_filename.str());

    PetscCheck(global_file.is_open(),
               PETSC_COMM_SELF,
               PETSC_ERR_FILE_OPEN,
               "Could not open wave-attractor global snapshot output file.");

    global_file << std::scientific << std::setprecision(16);

    global_file
        << "time,H,ha,E,S,W,Womega,Fo_minus_2_W,Fo_minus_2_Womega,"
        << "div_inf,div_l2\n";

    global_file
        << t << ","
        << H << ","
        << ha << ","
        << E << ","
        << S << ","
        << W << ","
        << Womega << ","
        << Fo_minus_2_W << ","
        << Fo_minus_2_Womega << ","
        << div_max << ","
        << div_l2 << "\n";

    global_file.close();

    PetscPrintf(PETSC_COMM_WORLD,
        "Wave-attractor forcing-off snapshot written at t = %.16e\n"
        "  fields: %s\n"
        "  global: %s\n"
        "  H = %.16e, ha = %.16e, E = %.16e, S = %.16e, W = %.16e, Womega = %.16e\n",
        (double)t,
        filename.str().c_str(),
        global_filename.str().c_str(),
        H,
        ha,
        E,
        S,
        W,
        Womega);

    PetscCall(VecDestroy(&Vmid));
    PetscCall(VecDestroy(&relOmega));
    PetscCall(VecDestroy(&absOmega));
    PetscCall(VecDestroy(&AVmid));
    PetscCall(VecDestroy(&relOmegaState));
    PetscCall(VecDestroy(&absOmegaState));

    PetscFunctionReturn(PETSC_SUCCESS);
}

/*--------------------------------------------------------------------------*/
/* Wave-attractor probe time-series output                                  */
/*--------------------------------------------------------------------------*/

namespace
{

WaveAttractorProbe makeWaveAttractorProbe(
    const std::string &label,
    const PetscReal x,
    const PetscReal y,
    const PetscReal z)
{
    WaveAttractorProbe probe;

    probe.label = label;

    probe.x_target = x;
    probe.y_target = y;
    probe.z_target = z;

    return probe;
}

/*--------------------------------------------------------------------------*/
std::vector<WaveAttractorProbe> defaultWaveAttractorProbes()
{
    std::vector<WaveAttractorProbe> probes;

    /*
     * Initial coordinates selected from the Fo = 20, DG(2),
     * 32 x 32 x 4 wave-attractor snapshot.
     *
     * There are:
     *   - two beam-centre probes;
     *   - one off-centre probe for each beam;
     *   - four reflection-region probes;
     *   - one probe outside the main beams.
     *
     * All target coordinates are mapped to the nearest available DG
     * coordinate. If several DG nodes share that coordinate, their values
     * are averaged.
     */

    const PetscReal zprobe = 0.5;

    // Main beam on the left.
    probes.push_back(makeWaveAttractorProbe(
        "left_beam_center",
        0.234375, 0.500000, zprobe));

    probes.push_back(makeWaveAttractorProbe(
        "left_beam_off_center",
        0.296875, 0.500000, zprobe));

    // Main beam on the right.
    probes.push_back(makeWaveAttractorProbe(
        "right_beam_center",
        0.734375, 0.500000, zprobe));

    probes.push_back(makeWaveAttractorProbe(
        "right_beam_off_center",
        0.796875, 0.500000, zprobe));

    // Reflection regions, located slightly inside the solid boundaries.
    probes.push_back(makeWaveAttractorProbe(
        "bottom_left_reflection",
        0.031250, 0.031250, zprobe));

    probes.push_back(makeWaveAttractorProbe(
        "top_left_reflection",
        0.468750, 0.968750, zprobe));

    probes.push_back(makeWaveAttractorProbe(
        "bottom_right_reflection",
        0.531250, 0.031250, zprobe));

    probes.push_back(makeWaveAttractorProbe(
        "top_right_reflection",
        0.968750, 0.968750, zprobe));

    // Region between the two main beams.
    probes.push_back(makeWaveAttractorProbe(
        "outside_main_beams",
        0.500000, 0.500000, zprobe));

    return probes;
}

/*--------------------------------------------------------------------------*/
PetscErrorCode mapProbeToNearestDGCoordinate(
    WaveAttractorProbe &probe,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements)
{
    PetscFunctionBeginUser;

    PetscReal best_dist2 = PETSC_MAX_REAL;

    /*
     * First pass:
     * find the nearest physical DG coordinate.
     */
    for (std::size_t e = 0; e < List_Of_Elements.size(); ++e)
    {
        const Element &element = *List_Of_Elements[e];

        const PetscInt pos =
            static_cast<PetscInt>(element.get_pos());

        const std::vector<double> x =
            element.get_node_coordinates_x();

        const std::vector<double> y =
            element.get_node_coordinates_y();

        const std::vector<double> z =
            element.get_node_coordinates_z();

        PetscCheck(
            x.size() == y.size() && x.size() == z.size(),
            PETSC_COMM_SELF,
            PETSC_ERR_ARG_SIZ,
            "Inconsistent element coordinate-array sizes.");

        for (std::size_t a = 0; a < x.size(); ++a)
        {
            const PetscReal dx =
                static_cast<PetscReal>(x[a]) - probe.x_target;

            const PetscReal dy =
                static_cast<PetscReal>(y[a]) - probe.y_target;

            const PetscReal dz =
                static_cast<PetscReal>(z[a]) - probe.z_target;

            const PetscReal dist2 =
                dx*dx + dy*dy + dz*dz;

            if (dist2 < best_dist2)
            {
                best_dist2 = dist2;

                probe.x_actual = static_cast<PetscReal>(x[a]);
                probe.y_actual = static_cast<PetscReal>(y[a]);
                probe.z_actual = static_cast<PetscReal>(z[a]);

                probe.element_id =
                    static_cast<PetscInt>(e);

                probe.local_node =
                    static_cast<PetscInt>(a);

                probe.representative_global_node =
                    pos + static_cast<PetscInt>(a);
            }
        }
    }

    PetscCheck(
        probe.representative_global_node >= 0,
        PETSC_COMM_SELF,
        PETSC_ERR_ARG_WRONGSTATE,
        "Could not map wave-attractor probe %s to a DG node.",
        probe.label.c_str());

    /*
     * Second pass:
     * collect every DG copy at the selected physical coordinate.
     *
     * This handles duplicated element-face nodes in the DG discretization.
     */
    probe.global_nodes.clear();

    const PetscReal coordinate_tolerance = 1.0e-12;

    for (std::size_t e = 0; e < List_Of_Elements.size(); ++e)
    {
        const Element &element = *List_Of_Elements[e];

        const PetscInt pos =
            static_cast<PetscInt>(element.get_pos());

        const std::vector<double> x =
            element.get_node_coordinates_x();

        const std::vector<double> y =
            element.get_node_coordinates_y();

        const std::vector<double> z =
            element.get_node_coordinates_z();

        for (std::size_t a = 0; a < x.size(); ++a)
        {
            const PetscReal dx =
                PetscAbsReal(
                    static_cast<PetscReal>(x[a])
                    - probe.x_actual);

            const PetscReal dy =
                PetscAbsReal(
                    static_cast<PetscReal>(y[a])
                    - probe.y_actual);

            const PetscReal dz =
                PetscAbsReal(
                    static_cast<PetscReal>(z[a])
                    - probe.z_actual);

            if (dx <= coordinate_tolerance &&
                dy <= coordinate_tolerance &&
                dz <= coordinate_tolerance)
            {
                probe.global_nodes.push_back(
                    pos + static_cast<PetscInt>(a));
            }
        }
    }

    PetscCheck(
        !probe.global_nodes.empty(),
        PETSC_COMM_SELF,
        PETSC_ERR_ARG_WRONGSTATE,
        "No DG nodes found for mapped probe coordinate.");

    PetscFunctionReturn(PETSC_SUCCESS);
}

/*--------------------------------------------------------------------------*/
PetscErrorCode createWaveAttractorProbeFilename(
    AppCtx &user)
{
    PetscFunctionBeginUser;

    char filename[PETSC_MAX_PATH_LEN];

    std::snprintf(
        filename,
        sizeof(filename),
        "probes_WA_Order%d_nx%d_ny%d_nz%d_t%d_P%d_"
        "Fo%.6g_Re%.6g_Ro%.6g_divcorr%d_projcurl%d.csv",
        static_cast<int>(user.opt.Order),
        static_cast<int>(user.opt.Nel_x),
        static_cast<int>(user.opt.Nel_y),
        static_cast<int>(user.opt.Nel_z),
        static_cast<int>(user.opt.steps_per_period),
        static_cast<int>(user.opt.periods),
        static_cast<double>(user.opt.Fo),
        static_cast<double>(user.Re),
        static_cast<double>(user.rot.RossbyNumber),
        static_cast<int>(
            user.opt.include_divergence_correction),
        static_cast<int>(0));

    user.wa_probe_recorder.filename = filename;

    PetscFunctionReturn(PETSC_SUCCESS);
}

} // end anonymous namespace

/*--------------------------------------------------------------------------*/
PetscErrorCode initializeWaveAttractorProbeRecorder(
    AppCtx &user,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements)
{
    PetscFunctionBeginUser;

    WaveAttractorProbeRecorder &recorder =
        user.wa_probe_recorder;

    PetscCheck(
        recorder.initialized != PETSC_TRUE,
        PETSC_COMM_SELF,
        PETSC_ERR_ARG_WRONGSTATE,
        "Wave-attractor probe recorder was initialized twice.");

    recorder.probes = defaultWaveAttractorProbes();

    PetscCall(createWaveAttractorProbeFilename(user));

    /*
     * Map targets to the nearest physical coordinates and collect
     * coincident DG nodes.
     */
    PetscPrintf(
        PETSC_COMM_WORLD,
        "\n"
        "============================================================\n"
        " Wave-attractor probe configuration\n"
        "============================================================\n");

    for (std::size_t p = 0;
         p < recorder.probes.size();
         ++p)
    {
        WaveAttractorProbe &probe =
            recorder.probes[p];

        PetscCall(mapProbeToNearestDGCoordinate(
            probe,
            List_Of_Elements));

        const PetscReal dx =
            probe.x_actual - probe.x_target;

        const PetscReal dy =
            probe.y_actual - probe.y_target;

        const PetscReal dz =
            probe.z_actual - probe.z_target;

        const PetscReal distance =
            PetscSqrtReal(dx*dx + dy*dy + dz*dz);

        PetscPrintf(
            PETSC_COMM_WORLD,
            " Probe %d: %s\n"
            "   target       = (%.8f, %.8f, %.8f)\n"
            "   actual       = (%.8f, %.8f, %.8f)\n"
            "   distance     = %.8e\n"
            "   DG copies    = %d\n"
            "   representative global node = %d\n",
            static_cast<int>(p),
            probe.label.c_str(),
            static_cast<double>(probe.x_target),
            static_cast<double>(probe.y_target),
            static_cast<double>(probe.z_target),
            static_cast<double>(probe.x_actual),
            static_cast<double>(probe.y_actual),
            static_cast<double>(probe.z_actual),
            static_cast<double>(distance),
            static_cast<int>(probe.global_nodes.size()),
            static_cast<int>(
                probe.representative_global_node));
    }

    /*
     * One reusable CURL workspace. This avoids allocating and destroying
     * a 3N vector every timestep.
     */
    PetscCall(VecDuplicate(
        user.Velocity,
        &recorder.relative_omega));

    PetscCall(VecZeroEntries(
        recorder.relative_omega));

    recorder.file = std::fopen(
        recorder.filename.c_str(),
        "w");

    PetscCheck(
        recorder.file != nullptr,
        PETSC_COMM_SELF,
        PETSC_ERR_FILE_OPEN,
        "Could not open wave-attractor probe file: %s",
        recorder.filename.c_str());

    /*
     * Keep rotation parameters in every row. This ensures that the
     * post-processing comparison uses exactly the same f and Ro as
     * the PETSc simulation.
     */
    std::fprintf(
        recorder.file,
        "step,time,period,forcing_active,"
        "probe_id,probe_label,"
        "x_target,y_target,z_target,"
        "x_actual,y_actual,z_actual,"
        "representative_global_node,dg_copy_count,"
        "Ro,f1,f2,f3,"
        "u,v,w,speed,"
        "rel_omega_x,rel_omega_y,rel_omega_z,"
        "kinetic_energy_density,"
        "absolute_helicity_density,"
        "nonlinear_term_magnitude,"
        "linear_rotation_term_magnitude,"
        "nonlinearity_ratio\n");

    std::fflush(recorder.file);

    recorder.initialized = PETSC_TRUE;

    PetscPrintf(
        PETSC_COMM_WORLD,
        " Probe time series file: %s\n"
        " Probe values will be recorded at every accepted timestep.\n"
        "============================================================\n\n",
        recorder.filename.c_str());

    PetscFunctionReturn(PETSC_SUCCESS);
}

/*--------------------------------------------------------------------------*/
PetscErrorCode appendWaveAttractorProbeSamples(
    AppCtx &user,
    const PetscInt step,
    const PetscReal t)
{
    PetscFunctionBeginUser;

    WaveAttractorProbeRecorder &recorder =
        user.wa_probe_recorder;

    PetscCheck(
        recorder.initialized == PETSC_TRUE,
        PETSC_COMM_SELF,
        PETSC_ERR_ARG_WRONGSTATE,
        "Wave-attractor probe recorder has not been initialized.");

    PetscCheck(
        recorder.file != nullptr,
        PETSC_COMM_SELF,
        PETSC_ERR_FILE_OPEN,
        "Wave-attractor probe output file is not open.");

    PetscCheck(
        recorder.relative_omega != nullptr,
        PETSC_COMM_SELF,
        PETSC_ERR_ARG_WRONGSTATE,
        "Wave-attractor relative-vorticity workspace is null.");

    PetscCheck(
        user.rot.RossbyNumber != 0.0,
        PETSC_COMM_SELF,
        PETSC_ERR_ARG_OUTOFRANGE,
        "Rossby number must be nonzero for probe diagnostics.");

    /*
     * Relative vorticity at the accepted state:
     *
     *     omega^r = CURL V.
     */
    PetscCall(MatMult(
        user.ops.CURL,
        user.Velocity,
        recorder.relative_omega));

    const PetscScalar *velocity_array = nullptr;
    const PetscScalar *omega_array = nullptr;

    PetscCall(VecGetArrayRead(
        user.Velocity,
        &velocity_array));

    PetscCall(VecGetArrayRead(
        recorder.relative_omega,
        &omega_array));

    const PetscReal invRo =
        user.rot.include_rotation == PETSC_TRUE
        ? 1.0 / user.rot.RossbyNumber
        : 0.0;

    const PetscReal f1 =
        static_cast<PetscReal>(user.rot.f1);

    const PetscReal f2 =
        static_cast<PetscReal>(user.rot.f2);

    const PetscReal f3 =
        static_cast<PetscReal>(user.rot.f3);

    const PetscReal period =
        t / waveAttractorPeriod();

    PetscBool forcing_active = PETSC_TRUE;

    if (user.opt.wave_attractor_turn_off_forcing ==
        PETSC_TRUE)
    {
        const PetscReal forcing_off_time =
            waveAttractorForcingOffTime(user.opt);

        forcing_active =
            t <= forcing_off_time
            ? PETSC_TRUE
            : PETSC_FALSE;
    }

    for (std::size_t p = 0;
         p < recorder.probes.size();
         ++p)
    {
        const WaveAttractorProbe &probe =
            recorder.probes[p];

        PetscReal u = 0.0;
        PetscReal v = 0.0;
        PetscReal w = 0.0;

        PetscReal omega_x = 0.0;
        PetscReal omega_y = 0.0;
        PetscReal omega_z = 0.0;

        const PetscReal number_of_copies =
            static_cast<PetscReal>(
                probe.global_nodes.size());

        for (const PetscInt node :
             probe.global_nodes)
        {
            const PetscInt iu = node;
            const PetscInt iv =
                static_cast<PetscInt>(user.N_Nodes)
                + node;

            const PetscInt iw =
                2 * static_cast<PetscInt>(user.N_Nodes)
                + node;

            u += PetscRealPart(
                velocity_array[iu]);

            v += PetscRealPart(
                velocity_array[iv]);

            w += PetscRealPart(
                velocity_array[iw]);

            omega_x += PetscRealPart(
                omega_array[iu]);

            omega_y += PetscRealPart(
                omega_array[iv]);

            omega_z += PetscRealPart(
                omega_array[iw]);
        }

        u /= number_of_copies;
        v /= number_of_copies;
        w /= number_of_copies;

        omega_x /= number_of_copies;
        omega_y /= number_of_copies;
        omega_z /= number_of_copies;

        const PetscReal speed2 =
            u*u + v*v + w*w;

        const PetscReal speed =
            PetscSqrtReal(speed2);

        const PetscReal kinetic_energy_density =
            0.5 * speed2;

        /*
         * Absolute-helicity density consistent with
         *
         *   h_a =
         *      1/2 V . omega^r
         *      + Ro^{-1} V . f.
         */
        const PetscReal absolute_helicity_density =
            0.5 * (
                u*omega_x
              + v*omega_y
              + w*omega_z)
            + invRo * (
                u*f1
              + v*f2
              + w*f3);

        /*
         * Nonlinear relative-vorticity term:
         *
         *     omega^r x V.
         */
        const PetscReal nonlinear_x =
            omega_y*w - omega_z*v;

        const PetscReal nonlinear_y =
            omega_z*u - omega_x*w;

        const PetscReal nonlinear_z =
            omega_x*v - omega_y*u;

        const PetscReal nonlinear_magnitude =
            PetscSqrtReal(
                nonlinear_x*nonlinear_x
              + nonlinear_y*nonlinear_y
              + nonlinear_z*nonlinear_z);

        /*
         * Linear rotational term:
         *
         *     Ro^{-1} f x V.
         */
        const PetscReal linear_x =
            invRo * (f2*w - f3*v);

        const PetscReal linear_y =
            invRo * (f3*u - f1*w);

        const PetscReal linear_z =
            invRo * (f1*v - f2*u);

        const PetscReal linear_magnitude =
            PetscSqrtReal(
                linear_x*linear_x
              + linear_y*linear_y
              + linear_z*linear_z);

        /*
         * Avoid an artificially large ratio where the denominator
         * is effectively zero.
         */
        PetscReal nonlinearity_ratio =
            std::numeric_limits<PetscReal>::quiet_NaN();

        const PetscReal denominator_tolerance =
            100.0 * PETSC_MACHINE_EPSILON;

        if (linear_magnitude >
            denominator_tolerance)
        {
            nonlinearity_ratio =
                nonlinear_magnitude /
                linear_magnitude;
        }

        std::fprintf(
            recorder.file,
            "%d,"
            "%.17e,%.17e,%d,"
            "%d,%s,"
            "%.17e,%.17e,%.17e,"
            "%.17e,%.17e,%.17e,"
            "%d,%d,"
            "%.17e,%.17e,%.17e,%.17e,"
            "%.17e,%.17e,%.17e,%.17e,"
            "%.17e,%.17e,%.17e,"
            "%.17e,"
            "%.17e,"
            "%.17e,"
            "%.17e,"
            "%.17e\n",
            static_cast<int>(step),
            static_cast<double>(t),
            static_cast<double>(period),
            static_cast<int>(forcing_active),
            static_cast<int>(p),
            probe.label.c_str(),
            static_cast<double>(probe.x_target),
            static_cast<double>(probe.y_target),
            static_cast<double>(probe.z_target),
            static_cast<double>(probe.x_actual),
            static_cast<double>(probe.y_actual),
            static_cast<double>(probe.z_actual),
            static_cast<int>(
                probe.representative_global_node),
            static_cast<int>(
                probe.global_nodes.size()),
            static_cast<double>(
                user.rot.RossbyNumber),
            static_cast<double>(f1),
            static_cast<double>(f2),
            static_cast<double>(f3),
            static_cast<double>(u),
            static_cast<double>(v),
            static_cast<double>(w),
            static_cast<double>(speed),
            static_cast<double>(omega_x),
            static_cast<double>(omega_y),
            static_cast<double>(omega_z),
            static_cast<double>(
                kinetic_energy_density),
            static_cast<double>(
                absolute_helicity_density),
            static_cast<double>(
                nonlinear_magnitude),
            static_cast<double>(
                linear_magnitude),
            static_cast<double>(
                nonlinearity_ratio));
    }

    PetscCall(VecRestoreArrayRead(
        user.Velocity,
        &velocity_array));

    PetscCall(VecRestoreArrayRead(
        recorder.relative_omega,
        &omega_array));

    /*
     * Flushing every timestep is acceptable here because only nine
     * probe rows are written. It also preserves almost all samples if
     * a long simulation terminates early.
     */
    std::fflush(recorder.file);

    PetscFunctionReturn(PETSC_SUCCESS);
}

/*--------------------------------------------------------------------------*/
PetscErrorCode destroyWaveAttractorProbeRecorder(
    AppCtx &user)
{
    PetscFunctionBeginUser;

    WaveAttractorProbeRecorder &recorder =
        user.wa_probe_recorder;

    if (recorder.file != nullptr)
    {
        std::fflush(recorder.file);
        std::fclose(recorder.file);
        recorder.file = nullptr;
    }

    PetscCall(VecDestroy(
        &recorder.relative_omega));

    recorder.probes.clear();
    recorder.filename.clear();
    recorder.initialized = PETSC_FALSE;

    PetscFunctionReturn(PETSC_SUCCESS);
}
