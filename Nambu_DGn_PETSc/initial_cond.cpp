#include "initial_cond.hpp"

#include <cmath>
#include <functional>
#include <vector>

#include "Legendre_Gauss_Lobatto.hpp"
#include "HIGW.hpp"
#include "test_cases.hpp"

namespace
{

using PointwiseVelocityFunction = std::function<PetscErrorCode(
    PetscScalar x,
    PetscScalar y,
    PetscScalar z,
    PetscScalar t,
    PetscScalar &u,
    PetscScalar &v,
    PetscScalar &w)>;

PetscErrorCode refreshVelocityComponentsFromFullVelocity(
    unsigned int N_Nodes,
    Vec Velocity,
    Vec Velocity_U,
    Vec Velocity_V,
    Vec Velocity_W);

PetscErrorCode compute_ExactSolutionPointwise_Generic(
    Vec Exact_Solution,
    Vec Velocity,
    Vec Velocity_U,
    Vec Velocity_V,
    Vec Velocity_W,
    const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    unsigned int N_Nodes,
    unsigned int N_Order,
    PetscScalar t,
    const PointwiseVelocityFunction &pointwise_velocity,
    const char *label)
{
    PetscFunctionBeginUser;

    PetscPrintf(PETSC_COMM_SELF,
                "Computing pointwise exact velocity: %s, t = %.16e\n",
                label,
                (double)t);

    PetscCall(VecZeroEntries(Velocity_U));
    PetscCall(VecZeroEntries(Velocity_V));
    PetscCall(VecZeroEntries(Velocity_W));
    PetscCall(VecZeroEntries(Velocity));

    if (Exact_Solution)
    {
        PetscCall(VecZeroEntries(Exact_Solution));
    }

    Vec ri = JacobiGL(0, 0, N_Order);

    PetscScalar *rnodes = nullptr;
    PetscCall(VecGetArray(ri, &rnodes));

    const unsigned int n1 = N_Order + 1;
    const unsigned int Np = n1 * n1 * n1;

    for (auto e = List_Of_Elements.begin(); e < List_Of_Elements.end(); ++e)
    {
        const unsigned int pos = (*e)->get_pos();
        const unsigned int Np_elem = (*e)->get_Number_Of_Nodes();

        PetscCheck(Np_elem == Np,
                   PETSC_COMM_SELF,
                   PETSC_ERR_ARG_SIZ,
                   "Element node count does not match N_Order-based Np.");

        for (unsigned int a = 0; a < Np; ++a)
        {
            const unsigned int alpha = a % n1;
            const unsigned int aa    = a / n1;
            const unsigned int beta  = aa % n1;
            const unsigned int zeta  = a / (n1 * n1);

            const PetscScalar r = rnodes[alpha];
            const PetscScalar s = rnodes[beta];
            const PetscScalar q = rnodes[zeta];

            double J;
            double drdx, drdy, drdz;
            double dsdx, dsdy, dsdz;
            double dtdx, dtdy, dtdz;
            double x, y, z;

            Calculate_Jacobian_Cuboid(
                (*e),
                List_Of_Vertices,
                r,
                s,
                q,
                J,
                drdx,
                drdy,
                drdz,
                dsdx,
                dsdy,
                dsdz,
                dtdx,
                dtdy,
                dtdz,
                x,
                y,
                z);

            PetscScalar u = 0.0;
            PetscScalar v = 0.0;
            PetscScalar w = 0.0;

              PetscCall(pointwise_velocity(x, y, z, t, u, v, w));

            const PetscInt gi = static_cast<PetscInt>(pos + a);

            PetscCall(VecSetValue(Velocity_U, gi, u, INSERT_VALUES));
            PetscCall(VecSetValue(Velocity_V, gi, v, INSERT_VALUES));
            PetscCall(VecSetValue(Velocity_W, gi, w, INSERT_VALUES));

            PetscCall(VecSetValue(Velocity, gi, u, INSERT_VALUES));
            PetscCall(VecSetValue(Velocity, static_cast<PetscInt>(N_Nodes) + gi, v, INSERT_VALUES));
            PetscCall(VecSetValue(Velocity, static_cast<PetscInt>(2 * N_Nodes) + gi, w, INSERT_VALUES));

            if (Exact_Solution)
            {
                PetscCall(VecSetValue(Exact_Solution, gi, u, INSERT_VALUES));
                PetscCall(VecSetValue(Exact_Solution, static_cast<PetscInt>(N_Nodes) + gi, v, INSERT_VALUES));
                PetscCall(VecSetValue(Exact_Solution, static_cast<PetscInt>(2 * N_Nodes) + gi, w, INSERT_VALUES));
            }
        }
    }

    PetscCall(VecAssemblyBegin(Velocity_U));
    PetscCall(VecAssemblyEnd(Velocity_U));
    PetscCall(VecAssemblyBegin(Velocity_V));
    PetscCall(VecAssemblyEnd(Velocity_V));
    PetscCall(VecAssemblyBegin(Velocity_W));
    PetscCall(VecAssemblyEnd(Velocity_W));
    PetscCall(VecAssemblyBegin(Velocity));
    PetscCall(VecAssemblyEnd(Velocity));

    if (Exact_Solution)
    {
        PetscCall(VecAssemblyBegin(Exact_Solution));
        PetscCall(VecAssemblyEnd(Exact_Solution));
    }

    PetscCall(VecRestoreArray(ri, &rnodes));
    PetscCall(VecDestroy(&ri));

    PetscFunctionReturn(0);
}


PetscErrorCode compute_ExactSolutionProjection_Generic(
    Vec Exact_Solution,
    Vec Velocity,
    Vec Velocity_U,
    Vec Velocity_V,
    Vec Velocity_W,
    const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    unsigned int N_Nodes,
    unsigned int N_Order,
    PetscScalar t,
    const PointwiseVelocityFunction &pointwise_velocity,
    const char *label)
{
    PetscFunctionBeginUser;

    PetscPrintf(PETSC_COMM_SELF,
                "Computing L2-projected exact velocity: %s, t = %.16e\n",
                label,
                (double)t);

    PetscCall(VecZeroEntries(Velocity_U));
    PetscCall(VecZeroEntries(Velocity_V));
    PetscCall(VecZeroEntries(Velocity_W));
    PetscCall(VecZeroEntries(Velocity));

    if (Exact_Solution)
    {
        PetscCall(VecZeroEntries(Exact_Solution));
    }

    Vec ri = JacobiGL(0, 0, N_Order);
    Vec Weights = nullptr;
    Vec QuadraturePoints = nullptr;

    // Reduced quadrature order.
    //
    // Old choice:
    //   Order_Gaussian_Quadrature = 2 * (N_Order + 3)
    //
    // New choice:
    //   Order_Gaussian_Quadrature = N_Order + 4
    //
    // This is much cheaper in 3D. For smooth trigonometric exact solutions,
    // this should be sufficient for projection accuracy. Verify once against
    // the old setting if using this in final publication tables.
    const unsigned int Order_Gaussian_Quadrature = 2 * (N_Order + 3);//N_Order + 4;

    QuadraturePoints = JacobiGL_withWeights(0, 0, Order_Gaussian_Quadrature, Weights);

    PetscScalar *wq1D = nullptr;
    PetscScalar *qp = nullptr;

    PetscCall(VecGetArray(Weights, &wq1D));
    PetscCall(VecGetArray(QuadraturePoints, &qp));

    const unsigned int n1 = N_Order + 1;
    const unsigned int Np = n1 * n1 * n1;
    const unsigned int nq = Order_Gaussian_Quadrature + 1;
    const unsigned int Nq3 = nq * nq * nq;

    std::vector<double> L1D(nq * n1, 0.0);

    auto idx1D = [n1](unsigned int qpt, unsigned int a) -> unsigned int
    {
        return qpt * n1 + a;
    };

    auto idxQ = [nq](unsigned int p, unsigned int q, unsigned int r) -> unsigned int
    {
        return (p * nq + q) * nq + r;
    };

    auto idxBasis = [n1](unsigned int alpha, unsigned int beta, unsigned int zeta) -> unsigned int
    {
        return alpha + n1 * (beta + n1 * zeta);
    };

    // Precompute all 1D Lagrange basis values at quadrature points.
    for (unsigned int p = 0; p < nq; ++p)
    {
        for (unsigned int a = 0; a < n1; ++a)
        {
            L1D[idx1D(p, a)] = LagrangePolynomial(ri, qp[p], a);
        }
    }

    for (auto e = List_Of_Elements.begin(); e < List_Of_Elements.end(); ++e)
    {
        const unsigned int pos = (*e)->get_pos();
        const unsigned int Np_elem = (*e)->get_Number_Of_Nodes();

        PetscCheck(Np_elem == Np,
                   PETSC_COMM_SELF,
                   PETSC_ERR_ARG_SIZ,
                   "Element node count does not match N_Order-based Np.");

        const std::vector<PetscScalar> &invM_local = (*e)->get_invM_local();

        PetscCheck(invM_local.size() == Np * Np,
                   PETSC_COMM_SELF,
                   PETSC_ERR_ARG_SIZ,
                   "invM_local has incorrect size.");

        std::vector<PetscScalar> bU(Np, 0.0), bV(Np, 0.0), bW(Np, 0.0);
        std::vector<PetscScalar> cU(Np, 0.0), cV(Np, 0.0), cW(Np, 0.0);

        // ---------------------------------------------------------------------
        // Precompute geometry, quadrature weights, and exact velocity once per
        // quadrature point.
        //
        // This removes the old factor Np from:
        //   - Calculate_Jacobian_Cuboid
        //   - pointwise_velocity
        // ---------------------------------------------------------------------

        std::vector<PetscScalar> qWeightJ(Nq3, 0.0);
        std::vector<PetscScalar> qU(Nq3, 0.0);
        std::vector<PetscScalar> qV(Nq3, 0.0);
        std::vector<PetscScalar> qW(Nq3, 0.0);

        for (unsigned int p = 0; p < nq; ++p)
        {
            const PetscScalar rp = qp[p];

            for (unsigned int q = 0; q < nq; ++q)
            {
                const PetscScalar sq = qp[q];

                for (unsigned int r = 0; r < nq; ++r)
                {
                    const PetscScalar tr = qp[r];
                    const unsigned int iq = idxQ(p, q, r);

                    double J;
                    double drdx, drdy, drdz;
                    double dsdx, dsdy, dsdz;
                    double dtdx, dtdy, dtdz;
                    double x, y, z;

                    Calculate_Jacobian_Cuboid(
                        (*e),
                        List_Of_Vertices,
                        rp,
                        sq,
                        tr,
                        J,
                        drdx,
                        drdy,
                        drdz,
                        dsdx,
                        dsdy,
                        dsdz,
                        dtdx,
                        dtdy,
                        dtdz,
                        x,
                        y,
                        z);

                    PetscScalar u_exact = 0.0;
                    PetscScalar v_exact = 0.0;
                    PetscScalar w_exact = 0.0;

                    PetscCall(pointwise_velocity(
                        x,
                        y,
                        z,
                        t,
                        u_exact,
                        v_exact,
                        w_exact));

                    qWeightJ[iq] = wq1D[p] * wq1D[q] * wq1D[r] * J;
                    qU[iq] = u_exact;
                    qV[iq] = v_exact;
                    qW[iq] = w_exact;
                }
            }
        }

        // ---------------------------------------------------------------------
        // Quadrature-major projection RHS assembly:
        //
        //   b_k = integral phi_k u_exact dx
        //
        // Old structure:
        //   for k:
        //       for q:
        //           evaluate geometry and exact velocity
        //
        // New structure:
        //   for q:
        //       evaluate/reuse exact velocity once
        //       for k:
        //           accumulate b_k
        // ---------------------------------------------------------------------

        for (unsigned int p = 0; p < nq; ++p)
        {
            for (unsigned int q = 0; q < nq; ++q)
            {
                for (unsigned int r = 0; r < nq; ++r)
                {
                    const unsigned int iq = idxQ(p, q, r);

                    const PetscScalar weightJ = qWeightJ[iq];
                    const PetscScalar u_exact = qU[iq];
                    const PetscScalar v_exact = qV[iq];
                    const PetscScalar w_exact = qW[iq];

                    for (unsigned int zeta = 0; zeta < n1; ++zeta)
                    {
                        const PetscScalar L_zeta = L1D[idx1D(r, zeta)];

                        for (unsigned int beta = 0; beta < n1; ++beta)
                        {
                            const PetscScalar L_beta_zeta =
                                L1D[idx1D(q, beta)] * L_zeta;

                            for (unsigned int alpha = 0; alpha < n1; ++alpha)
                            {
                                const unsigned int k =
                                    idxBasis(alpha, beta, zeta);

                                const PetscScalar phi_k =
                                    L1D[idx1D(p, alpha)] * L_beta_zeta;

                                const PetscScalar factor = weightJ * phi_k;

                                bU[k] += factor * u_exact;
                                bV[k] += factor * v_exact;
                                bW[k] += factor * w_exact;
                            }
                        }
                    }
                }
            }
        }

        // Apply local inverse mass matrix:
        //
        //   c = M^{-1} b
        //
        // Keep this unchanged from the original implementation.
        for (unsigned int i = 0; i < Np; ++i)
        {
            PetscScalar sumU = 0.0;
            PetscScalar sumV = 0.0;
            PetscScalar sumW = 0.0;

            for (unsigned int j = 0; j < Np; ++j)
            {
                const PetscScalar mij = invM_local[i * Np + j];

                sumU += mij * bU[j];
                sumV += mij * bV[j];
                sumW += mij * bW[j];
            }

            cU[i] = sumU;
            cV[i] = sumV;
            cW[i] = sumW;
        }

        for (unsigned int n = 0; n < Np; ++n)
        {
            const PetscInt gi = static_cast<PetscInt>(pos + n);

            PetscCall(VecSetValue(Velocity_U, gi, cU[n], INSERT_VALUES));
            PetscCall(VecSetValue(Velocity_V, gi, cV[n], INSERT_VALUES));
            PetscCall(VecSetValue(Velocity_W, gi, cW[n], INSERT_VALUES));

            PetscCall(VecSetValue(Velocity, gi, cU[n], INSERT_VALUES));
            PetscCall(VecSetValue(Velocity, static_cast<PetscInt>(N_Nodes) + gi, cV[n], INSERT_VALUES));
            PetscCall(VecSetValue(Velocity, static_cast<PetscInt>(2 * N_Nodes) + gi, cW[n], INSERT_VALUES));

            if (Exact_Solution)
            {
                PetscCall(VecSetValue(Exact_Solution, gi, cU[n], INSERT_VALUES));
                PetscCall(VecSetValue(Exact_Solution, static_cast<PetscInt>(N_Nodes) + gi, cV[n], INSERT_VALUES));
                PetscCall(VecSetValue(Exact_Solution, static_cast<PetscInt>(2 * N_Nodes) + gi, cW[n], INSERT_VALUES));
            }
        }
    }

    PetscCall(VecAssemblyBegin(Velocity_U));
    PetscCall(VecAssemblyEnd(Velocity_U));

    PetscCall(VecAssemblyBegin(Velocity_V));
    PetscCall(VecAssemblyEnd(Velocity_V));

    PetscCall(VecAssemblyBegin(Velocity_W));
    PetscCall(VecAssemblyEnd(Velocity_W));

    PetscCall(VecAssemblyBegin(Velocity));
    PetscCall(VecAssemblyEnd(Velocity));

    if (Exact_Solution)
    {
        PetscCall(VecAssemblyBegin(Exact_Solution));
        PetscCall(VecAssemblyEnd(Exact_Solution));
    }

    PetscCall(VecRestoreArray(Weights, &wq1D));
    PetscCall(VecRestoreArray(QuadraturePoints, &qp));

    PetscCall(VecDestroy(&Weights));
    PetscCall(VecDestroy(&QuadraturePoints));
    PetscCall(VecDestroy(&ri));

    PetscFunctionReturn(0);
}

PetscErrorCode setExactVelocity_Generic(
    Vec FullState,
    Vec Velocity,
    Vec Velocity_U,
    Vec Velocity_V,
    Vec Velocity_W,
    const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    unsigned int N_Nodes,
    unsigned int N_Order,
    ExactVelocitySampling sampling,
    PetscScalar t,
    const PointwiseVelocityFunction &pointwise_velocity,
    const char *label,
    Mat DIV,
    Mat GRAD,
    Mat LaplacianClean)
{
    PetscFunctionBeginUser;

    if (sampling == ExactVelocitySampling::L2Projection)
    {
        PetscCall(compute_ExactSolutionProjection_Generic(
            nullptr,
            Velocity,
            Velocity_U,
            Velocity_V,
            Velocity_W,
            List_Of_Vertices,
            List_Of_Elements,
            N_Nodes,
            N_Order,
            t,
            pointwise_velocity,
            label));
    }
    else if (sampling == ExactVelocitySampling::Pointwise)
    {
        PetscCall(compute_ExactSolutionPointwise_Generic(
            nullptr,
            Velocity,
            Velocity_U,
            Velocity_V,
            Velocity_W,
            List_Of_Vertices,
            List_Of_Elements,
            N_Nodes,
            N_Order,
            t,
            pointwise_velocity,
            label));
    }
    else
    {
        SETERRQ(PETSC_COMM_SELF,
                PETSC_ERR_ARG_WRONG,
                "Unknown ExactVelocitySampling mode.");
    }

    PetscCall(projectVelocityToDiscreteDivergenceFree(
        N_Nodes,
        Velocity,
        Velocity_U,
        Velocity_V,
        Velocity_W,
        DIV,
        GRAD,
        LaplacianClean));

    if (FullState)
    {
        PetscCall(fillFullStateFromVelocityComponents(
            N_Nodes,
            FullState,
            Velocity_U,
            Velocity_V,
            Velocity_W));
    }

    PetscFunctionReturn(0);
}

} // namespace

// -----------------------------------------------------------------------------
// Pointwise exact velocity fields
// -----------------------------------------------------------------------------

PetscErrorCode exactVelocity_BoostedRotatingEuler_Pointwise(
    PetscScalar x,
    PetscScalar y,
    PetscScalar z,
    PetscScalar t,
    PetscScalar Ro,
    PetscScalar W,
    PetscScalar &u,
    PetscScalar &v,
    PetscScalar &w)
{
    PetscFunctionBeginUser;

    PetscCheck(Ro > 0.0,
               PETSC_COMM_SELF,
               PETSC_ERR_ARG_OUTOFRANGE,
               "Rossby number Ro must be positive. Got %.16e",
               (double)Ro);

    const PetscScalar theta =
        2.0 * PETSC_PI * (x + y + z - W * t)
        + (std::sqrt(3.0) / (3.0 * Ro)) * t;

    const PetscScalar c = std::cos(theta);
    const PetscScalar s = std::sin(theta);

    u = (1.0 / (2.0 * PETSC_PI)) * (std::sqrt(3.0) * c + 3.0 * s);
    v = (1.0 / (2.0 * PETSC_PI)) * (std::sqrt(3.0) * c - 3.0 * s);
    w = -(std::sqrt(3.0) / PETSC_PI) * c + W;

    PetscFunctionReturn(0);
}

PetscErrorCode exactVelocity_DecayingNavierStokes_Pointwise(
    PetscScalar x,
    PetscScalar y,
    PetscScalar z,
    PetscScalar t,
    PetscScalar Re,
    PetscScalar Uboost,
    PetscScalar Vboost,
    PetscScalar Wboost,
    PetscScalar &u,
    PetscScalar &v,
    PetscScalar &w)
{
    PetscFunctionBeginUser;

    PetscCheck(Re > 0.0,
               PETSC_COMM_SELF,
               PETSC_ERR_ARG_OUTOFRANGE,
               "Reynolds number Re must be positive. Got %.16e",
               (double)Re);

    const PetscScalar xi   = x - Uboost * t;
    const PetscScalar eta  = y - Vboost * t;
    const PetscScalar zeta = z - Wboost * t;

    const PetscScalar decay =
        std::exp(-12.0 * PETSC_PI * PETSC_PI * t / Re);

    const PetscScalar A =
        4.0 * std::sqrt(2.0) / (3.0 * std::sqrt(3.0));

    const PetscScalar u_fluc = A *
        (
            std::sin(2.0 * PETSC_PI * xi + PETSC_PI / 6.0) *
            std::sin(2.0 * PETSC_PI * eta + PETSC_PI / 3.0) *
            std::sin(2.0 * PETSC_PI * zeta)
            +
            std::cos(2.0 * PETSC_PI * zeta + PETSC_PI / 6.0) *
            std::cos(2.0 * PETSC_PI * xi   + PETSC_PI / 3.0) *
            std::sin(2.0 * PETSC_PI * eta)
        ) * decay;

    const PetscScalar v_fluc = A *
        (
            std::sin(2.0 * PETSC_PI * eta  + PETSC_PI / 6.0) *
            std::sin(2.0 * PETSC_PI * zeta + PETSC_PI / 3.0) *
            std::sin(2.0 * PETSC_PI * xi)
            +
            std::cos(2.0 * PETSC_PI * xi  + PETSC_PI / 6.0) *
            std::cos(2.0 * PETSC_PI * eta + PETSC_PI / 3.0) *
            std::sin(2.0 * PETSC_PI * zeta)
        ) * decay;

    const PetscScalar w_fluc = A *
        (
            std::sin(2.0 * PETSC_PI * zeta + PETSC_PI / 6.0) *
            std::sin(2.0 * PETSC_PI * xi   + PETSC_PI / 3.0) *
            std::sin(2.0 * PETSC_PI * eta)
            +
            std::cos(2.0 * PETSC_PI * eta  + PETSC_PI / 6.0) *
            std::cos(2.0 * PETSC_PI * zeta + PETSC_PI / 3.0) *
            std::sin(2.0 * PETSC_PI * xi)
        ) * decay;

    u = Uboost + u_fluc;
    v = Vboost + v_fluc;
    w = Wboost + w_fluc;

    PetscFunctionReturn(0);
}

PetscErrorCode exactVelocity_SteadyRotating2D3CTaylorGreenWalls_Pointwise(
    PetscScalar x,
    PetscScalar y,
    PetscScalar z,
    PetscScalar t,
    PetscScalar &u,
    PetscScalar &v,
    PetscScalar &w)
{
    PetscFunctionBeginUser;

    (void)z;
    (void)t;

    const PetscScalar sx = std::sin(PETSC_PI * x);
    const PetscScalar cx = std::cos(PETSC_PI * x);
    const PetscScalar sy = std::sin(2.0 * PETSC_PI * y);
    const PetscScalar cy = std::cos(2.0 * PETSC_PI * y);

    // // psi = sin(pi x) sin(2 pi y)
    // u = std::sin(2.0 * PETSC_PI * x)*std::sin(2.0 * PETSC_PI * y);//2.0 * PETSC_PI * sx * cy;  // psi_y
    // v = std::cos(2.0 * PETSC_PI * x)*std::cos(2.0 * PETSC_PI * y);//-1.0 * PETSC_PI * cx * sy;  // -psi_x
    // w = 0.0;// sx * sy;                   // psi

    // // psi = sin(pi x) sin(2 pi y)
    // u = 2.0 * PETSC_PI * sx * cy;  // psi_y
    // v = -1.0 * PETSC_PI * cx * sy;  // -psi_x
    // w = sx * sy;                   // psi

    // psi = sin(pi x) sin(2 pi y)
    u = PETSC_PI * std::sin(PETSC_PI * x) * std::cos(PETSC_PI * y);  // psi_y
    v = -PETSC_PI * std::cos(PETSC_PI * x) * std::sin(PETSC_PI * y);  // -psi_x
    w = std::sin(PETSC_PI * x) * std::sin(PETSC_PI * y);                   // psi

    PetscFunctionReturn(0);
}


PetscErrorCode exactVelocity_TaylorGreenFreeSlipBox_Pointwise(
    PetscScalar x,
    PetscScalar y,
    PetscScalar z,
    PetscScalar t,
    PetscScalar Re,
    PetscScalar &u,
    PetscScalar &v,
    PetscScalar &w)
{
    PetscFunctionBeginUser;

    (void)z;

    PetscCheck(Re > 0.0,
               PETSC_COMM_SELF,
               PETSC_ERR_ARG_OUTOFRANGE,
               "Reynolds number Re must be positive. Got %.16e",
               (double)Re);

    const PetscScalar k = 2.0 * PETSC_PI;
    const PetscScalar decay = std::exp(-2.0 * k * k * t / Re);

    u =  std::sin(k * x) * std::cos(k * y) * decay;
    v = -std::cos(k * x) * std::sin(k * y) * decay;
    w =  0.0;

    PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// High-level exact/reference velocity constructors
// -----------------------------------------------------------------------------

PetscErrorCode setExactVelocity_BoostedRotatingEuler(
    Vec FullState,
    Vec Velocity,
    Vec Velocity_U,
    Vec Velocity_V,
    Vec Velocity_W,
    const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    unsigned int N_Nodes,
    unsigned int N_Order,
    ExactVelocitySampling sampling,
    PetscScalar t,
    PetscScalar Ro,
    PetscScalar W,
    Mat DIV,
    Mat GRAD,
    Mat LaplacianClean)
{
    PetscFunctionBeginUser;

    PetscCall(setExactVelocity_Generic(
        FullState,
        Velocity,
        Velocity_U,
        Velocity_V,
        Velocity_W,
        List_Of_Vertices,
        List_Of_Elements,
        N_Nodes,
        N_Order,
        sampling,
        t,
        [Ro, W](PetscScalar x,
                PetscScalar y,
                PetscScalar z,
                PetscScalar t,
                PetscScalar &u,
                PetscScalar &v,
                PetscScalar &w) -> PetscErrorCode
        {
            PetscFunctionBeginUser;
            PetscCall(exactVelocity_BoostedRotatingEuler_Pointwise(x, y, z, t, Ro, W, u, v, w));
            PetscFunctionReturn(0);
        },
        "BoostedRotatingEuler",
        DIV,
        GRAD,
        LaplacianClean));

    PetscFunctionReturn(0);
}

PetscErrorCode setExactVelocity_DecayingNavierStokes(
    Vec FullState,
    Vec Velocity,
    Vec Velocity_U,
    Vec Velocity_V,
    Vec Velocity_W,
    const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    unsigned int N_Nodes,
    unsigned int N_Order,
    ExactVelocitySampling sampling,
    PetscScalar t,
    PetscScalar Re,
    PetscScalar Uboost,
    PetscScalar Vboost,
    PetscScalar Wboost,
    Mat DIV,
    Mat GRAD,
    Mat LaplacianClean)
{
    PetscFunctionBeginUser;

    PetscCall(setExactVelocity_Generic(
        FullState,
        Velocity,
        Velocity_U,
        Velocity_V,
        Velocity_W,
        List_Of_Vertices,
        List_Of_Elements,
        N_Nodes,
        N_Order,
        sampling,
        t,
        [Re, Uboost, Vboost, Wboost](
             PetscScalar x,
             PetscScalar y,
             PetscScalar z,
             PetscScalar t,
             PetscScalar &u,
             PetscScalar &v,
             PetscScalar &w) -> PetscErrorCode
        {
            PetscFunctionBeginUser;
            PetscCall(exactVelocity_DecayingNavierStokes_Pointwise(
                x, y, z, t, Re, Uboost, Vboost, Wboost, u, v, w));
            PetscFunctionReturn(0);
        },
        "DecayingNavierStokes",
        DIV,
        GRAD,
        LaplacianClean));

    PetscFunctionReturn(0);
}

PetscErrorCode setExactVelocity_SteadyRotating2D3CTaylorGreenWalls(
    Vec FullState,
    Vec Velocity,
    Vec Velocity_U,
    Vec Velocity_V,
    Vec Velocity_W,
    const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    unsigned int N_Nodes,
    unsigned int N_Order,
    ExactVelocitySampling sampling,
    PetscScalar t,
    Mat DIV,
    Mat GRAD,
    Mat LaplacianClean)
{
    PetscFunctionBeginUser;

    PetscCall(setExactVelocity_Generic(
        FullState,
        Velocity,
        Velocity_U,
        Velocity_V,
        Velocity_W,
        List_Of_Vertices,
        List_Of_Elements,
        N_Nodes,
        N_Order,
        sampling,
        t,
        [](PetscScalar x,
           PetscScalar y,
           PetscScalar z,
           PetscScalar t,
           PetscScalar &u,
           PetscScalar &v,
           PetscScalar &w) -> PetscErrorCode
        {
            PetscFunctionBeginUser;
            PetscCall(exactVelocity_SteadyRotating2D3CTaylorGreenWalls_Pointwise(
                x, y, z, t, u, v, w));
            PetscFunctionReturn(0);
        },
        "SteadyRotating2D3CTaylorGreenWalls",
        DIV,
        GRAD,
        LaplacianClean));

    PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Public pointwise interpolation wrappers
// -----------------------------------------------------------------------------

PetscErrorCode compute_ExactSolutionPointwise_BoostedRotatingEuler(
    Vec Exact_Solution,
    Vec Velocity,
    Vec Velocity_U,
    Vec Velocity_V,
    Vec Velocity_W,
    const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    unsigned int N_Nodes,
    unsigned int N_Order,
    PetscScalar t,
    PetscScalar Ro,
    PetscScalar W)
{
    PetscFunctionBeginUser;

    PetscCall(compute_ExactSolutionPointwise_Generic(
        Exact_Solution,
        Velocity,
        Velocity_U,
        Velocity_V,
        Velocity_W,
        List_Of_Vertices,
        List_Of_Elements,
        N_Nodes,
        N_Order,
        t,
        [Ro, W](PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar t,
                PetscScalar &u, PetscScalar &v, PetscScalar &w) -> PetscErrorCode
        {
            PetscFunctionBeginUser;
            PetscCall(exactVelocity_BoostedRotatingEuler_Pointwise(x, y, z, t, Ro, W, u, v, w));
            PetscFunctionReturn(0);
        },
        "BoostedRotatingEuler / pointwise"));

    PetscFunctionReturn(0);
}

PetscErrorCode compute_ExactSolutionPointwise_DecayingNavierStokes(
    Vec Exact_Solution,
    Vec Velocity,
    Vec Velocity_U,
    Vec Velocity_V,
    Vec Velocity_W,
    const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    unsigned int N_Nodes,
    unsigned int N_Order,
    PetscScalar t,
    PetscScalar Re,
    PetscScalar Uboost,
    PetscScalar Vboost,
    PetscScalar Wboost)
{
    PetscFunctionBeginUser;

    PetscCall(compute_ExactSolutionPointwise_Generic(
        Exact_Solution,
        Velocity,
        Velocity_U,
        Velocity_V,
        Velocity_W,
        List_Of_Vertices,
        List_Of_Elements,
        N_Nodes,
        N_Order,
        t,
        [Re, Uboost, Vboost, Wboost](
             PetscScalar x,
             PetscScalar y,
             PetscScalar z,
             PetscScalar t,
             PetscScalar &u,
             PetscScalar &v,
             PetscScalar &w) -> PetscErrorCode
        {
            PetscFunctionBeginUser;
            PetscCall(exactVelocity_DecayingNavierStokes_Pointwise(
                x, y, z, t, Re, Uboost, Vboost, Wboost, u, v, w));
            PetscFunctionReturn(0);
        },
        "DecayingNavierStokes / pointwise"));

    PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Public L2 projection wrappers
// -----------------------------------------------------------------------------

PetscErrorCode compute_ExactSolutionProjection_BoostedRotatingEuler(
    Vec Exact_Solution,
    Vec Velocity,
    Vec Velocity_U,
    Vec Velocity_V,
    Vec Velocity_W,
    const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    unsigned int N_Nodes,
    unsigned int N_Order,
    PetscScalar t,
    PetscScalar Ro,
    PetscScalar W)
{
    PetscFunctionBeginUser;

    PetscCall(compute_ExactSolutionProjection_Generic(
        Exact_Solution,
        Velocity,
        Velocity_U,
        Velocity_V,
        Velocity_W,
        List_Of_Vertices,
        List_Of_Elements,
        N_Nodes,
        N_Order,
        t,
        [Ro, W](PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar t,
                PetscScalar &u, PetscScalar &v, PetscScalar &w) -> PetscErrorCode
        {
            PetscFunctionBeginUser;
            PetscCall(exactVelocity_BoostedRotatingEuler_Pointwise(x, y, z, t, Ro, W, u, v, w));
            PetscFunctionReturn(0);
        },
        "BoostedRotatingEuler / L2 projection"));

    PetscFunctionReturn(0);
}

PetscErrorCode compute_ExactSolutionProjection_DecayingNavierStokes(
    Vec Exact_Solution,
    Vec Velocity,
    Vec Velocity_U,
    Vec Velocity_V,
    Vec Velocity_W,
    const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    unsigned int N_Nodes,
    unsigned int N_Order,
    PetscScalar t,
    PetscScalar Re,
    PetscScalar Uboost,
    PetscScalar Vboost,
    PetscScalar Wboost)
{
    PetscFunctionBeginUser;

    PetscCall(compute_ExactSolutionProjection_Generic(
        Exact_Solution,
        Velocity,
        Velocity_U,
        Velocity_V,
        Velocity_W,
        List_Of_Vertices,
        List_Of_Elements,
        N_Nodes,
        N_Order,
        t,
        [Re, Uboost, Vboost, Wboost](
             PetscScalar x,
             PetscScalar y,
             PetscScalar z,
             PetscScalar t,
             PetscScalar &u,
             PetscScalar &v,
             PetscScalar &w) -> PetscErrorCode
        {
            PetscFunctionBeginUser;
            PetscCall(exactVelocity_DecayingNavierStokes_Pointwise(
                x, y, z, t, Re, Uboost, Vboost, Wboost, u, v, w));
            PetscFunctionReturn(0);
        },
        "DecayingNavierStokes / L2 projection"));

    PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Discrete divergence projection
// -----------------------------------------------------------------------------

PetscErrorCode projectVelocityToDiscreteDivergenceFree(
    unsigned int N_Nodes,
    Vec Velocity,
    Vec Velocity_U,
    Vec Velocity_V,
    Vec Velocity_W,
    Mat DIV,
    Mat GRAD,
    Mat LaplacianClean)
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

        PetscCall(MatSetNullSpace(LaplacianClean, pressureNullSpace));
        PetscCall(MatSetTransposeNullSpace(LaplacianClean, pressureNullSpace));

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
        PetscCall(KSPSetOperators(ksp, LaplacianClean, LaplacianClean));

        /*
           Do not use KSPPREONLY + PCLU here.
           The matrix is singular.

           GMRES is robust even if LaplacianClean is mildly nonsymmetric.
           If you have verified symmetry and positive semidefiniteness,
           KSPCG or KSPMINRES may also be appropriate.
        */
        PetscCall(KSPSetType(ksp, KSPGMRES));
        PetscCall(KSPGetPC(ksp, &pc));
        PetscCall(PCSetType(pc, PCJACOBI));

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
                "Velocity divergence projection failed: reason = %d (%s), "
                "iterations = %d, residual = %.16e\n",
                (int)reason,
                reasonStr ? reasonStr : "unknown",
                (int)its,
                (double)rnorm);

            SETERRQ(PETSC_COMM_SELF,
                    PETSC_ERR_CONV_FAILED,
                    "Velocity divergence projection failed.");
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

    PetscCall(refreshVelocityComponentsFromFullVelocity(
        N_Nodes,
        Velocity,
        Velocity_U,
        Velocity_V,
        Velocity_W));

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

// -----------------------------------------------------------------------------
// Diagnostics / utilities
// -----------------------------------------------------------------------------

PetscErrorCode compute_Divergence_Velocity(
    Vec Velocity,
    unsigned int N_Nodes,
    Mat DIV)
{
    PetscFunctionBeginUser;

    Vec RHS = nullptr;

    PetscCall(VecCreateSeq(PETSC_COMM_SELF, N_Nodes, &RHS));
    PetscCall(MatMult(DIV, Velocity, RHS));

    PetscReal normInf = 0.0;
    PetscReal norm2   = 0.0;

    PetscCall(VecNorm(RHS, NORM_INFINITY, &normInf));
    PetscCall(VecNorm(RHS, NORM_2, &norm2));

    PetscPrintf(PETSC_COMM_SELF,
                "Divergence diagnostics: ||DIV V||_inf = %.16e, ||DIV V||_2 = %.16e\n",
                (double)normInf,
                (double)norm2);

    PetscCall(VecDestroy(&RHS));

    PetscFunctionReturn(0);
}

PetscErrorCode fillFullStateFromVelocityComponents(
    unsigned int N_Nodes,
    Vec FullState,
    Vec Velocity_U,
    Vec Velocity_V,
    Vec Velocity_W)
{
    PetscFunctionBeginUser;

    PetscCheck(FullState != nullptr,
               PETSC_COMM_SELF,
               PETSC_ERR_ARG_NULL,
               "FullState is null.");

    PetscCall(VecZeroEntries(FullState));

    IS isu = nullptr;
    IS isv = nullptr;
    IS isw = nullptr;

    Vec Ublock = nullptr;
    Vec Vblock = nullptr;
    Vec Wblock = nullptr;

    PetscCall(ISCreateStride(PETSC_COMM_SELF, N_Nodes, 0, 1, &isu));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, N_Nodes, N_Nodes, 1, &isv));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, N_Nodes, 2 * N_Nodes, 1, &isw));

    PetscCall(VecGetSubVector(FullState, isu, &Ublock));
    PetscCall(VecGetSubVector(FullState, isv, &Vblock));
    PetscCall(VecGetSubVector(FullState, isw, &Wblock));

    PetscCall(VecCopy(Velocity_U, Ublock));
    PetscCall(VecCopy(Velocity_V, Vblock));
    PetscCall(VecCopy(Velocity_W, Wblock));

    PetscCall(VecRestoreSubVector(FullState, isu, &Ublock));
    PetscCall(VecRestoreSubVector(FullState, isv, &Vblock));
    PetscCall(VecRestoreSubVector(FullState, isw, &Wblock));

    PetscCall(ISDestroy(&isu));
    PetscCall(ISDestroy(&isv));
    PetscCall(ISDestroy(&isw));

    PetscFunctionReturn(0);
}

namespace
{

PetscErrorCode refreshVelocityComponentsFromFullVelocity(
    unsigned int N_Nodes,
    Vec Velocity,
    Vec Velocity_U,
    Vec Velocity_V,
    Vec Velocity_W)
{
    PetscFunctionBeginUser;

    IS isx = nullptr;
    IS isy = nullptr;
    IS isz = nullptr;

    Vec U  = nullptr;
    Vec V  = nullptr;
    Vec Wv = nullptr;

    PetscCall(ISCreateStride(PETSC_COMM_SELF, N_Nodes, 0, 1, &isx));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, N_Nodes, N_Nodes, 1, &isy));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, N_Nodes, 2 * N_Nodes, 1, &isz));

    PetscCall(VecGetSubVector(Velocity, isx, &U));
    PetscCall(VecGetSubVector(Velocity, isy, &V));
    PetscCall(VecGetSubVector(Velocity, isz, &Wv));

    PetscCall(VecCopy(U,  Velocity_U));
    PetscCall(VecCopy(V,  Velocity_V));
    PetscCall(VecCopy(Wv, Velocity_W));

    PetscCall(VecRestoreSubVector(Velocity, isx, &U));
    PetscCall(VecRestoreSubVector(Velocity, isy, &V));
    PetscCall(VecRestoreSubVector(Velocity, isz, &Wv));

    PetscCall(ISDestroy(&isx));
    PetscCall(ISDestroy(&isy));
    PetscCall(ISDestroy(&isz));

    PetscFunctionReturn(0);
}

} // namespace


PetscErrorCode compute_MassInnerProduct(
    Mat M,
    Vec x,
    Vec y,
    PetscScalar *value)
{
    PetscFunctionBeginUser;

    Vec My = nullptr;

    PetscCall(VecDuplicate(y, &My));

    PetscCall(MatMult(M, y, My));
    PetscCall(VecDot(x, My, value));

    PetscCall(VecDestroy(&My));

    PetscFunctionReturn(0);
}

PetscErrorCode compute_MassNorm(
    Mat M,
    Vec x,
    PetscReal *norm)
{
    PetscFunctionBeginUser;

    PetscScalar dot = 0.0;

    PetscCall(compute_MassInnerProduct(M, x, x, &dot));

    *norm = std::sqrt(PetscMax(0.0, PetscRealPart(dot)));

    PetscFunctionReturn(0);
}

PetscErrorCode setExactVelocityForTestCase(
    const TestCase &tc,
    Vec FullState,
    Vec Velocity,
    Vec Velocity_U,
    Vec Velocity_V,
    Vec Velocity_W,
    const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    unsigned int N_Nodes,
    unsigned int N_Order,
    ExactVelocitySampling sampling,
    PetscScalar t,
    Mat DIV,
    Mat GRAD,
    Mat Laplacian)
{
    PetscFunctionBeginUser;

    switch (tc.id)
    {
        case ProblemID::BoostedRotatingEuler:
        {
            PetscCall(setExactVelocity_BoostedRotatingEuler(
                FullState,
                Velocity,
                Velocity_U,
                Velocity_V,
                Velocity_W,
                List_Of_Vertices,
                List_Of_Elements,
                N_Nodes,
                N_Order,
                sampling,
                t,
                tc.Ro,
                tc.W,
                DIV,
                GRAD,
                Laplacian));
            break;
        }

        case ProblemID::DecayingNavierStokes:
        {
          PetscCall(setExactVelocity_DecayingNavierStokes(
              FullState,
              Velocity,
              Velocity_U,
              Velocity_V,
              Velocity_W,
              List_Of_Vertices,
              List_Of_Elements,
              N_Nodes,
              N_Order,
              sampling,
              t,
              tc.Re,
              tc.U,
              tc.V,
              tc.W,
              DIV,
              GRAD,
              Laplacian));
            break;
        }

        case ProblemID::SteadyRotating2D3CTaylorGreenWalls:
        {
            PetscCall(setExactVelocity_SteadyRotating2D3CTaylorGreenWalls(
                FullState,
                Velocity,
                Velocity_U,
                Velocity_V,
                Velocity_W,
                List_Of_Vertices,
                List_Of_Elements,
                N_Nodes,
                N_Order,
                sampling,
                t,
                DIV,
                GRAD,
                Laplacian));
            break;
        }

        default:
        {
            SETERRQ(PETSC_COMM_SELF,
                    PETSC_ERR_SUP,
                    "setExactVelocityForTestCase is not implemented for this test case.");
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}
