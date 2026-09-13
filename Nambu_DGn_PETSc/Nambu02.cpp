static char help[] = "TO DO\n \n\n";

#include <petscksp.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <petscviewerhdf5.h>
#include "initial_cond.hpp"
#include "mesh_gen.hpp"
#include "Legendre_Gauss_Lobatto.hpp"
#include "HIGW.hpp"
#include "Elements.hpp"
#include "options.hpp"
#include <chrono>
//#include <slepceps.h>
//#include <slepcsys.h>
#include "test_cases.hpp"


int main(int argc,char **args)
{
  PetscCall(SlepcInitialize(&argc, &args, nullptr, help));


AppCtx user;
PetscCall(ReadOptions(user.opt));

TestCase tc;
PetscCall(makeTestCase(user.opt, tc));
PetscCall(applyTestCaseToUser(tc, user));
PetscCall(printTestCaseBanner(tc, user));
if (tc.id == ProblemID::InertialWaveAttractor)
{
    PetscCall(printWaveAttractorForcingConfiguration(user.opt));
}

PetscInt N_Petsc = user.opt.Order;
PetscScalar theta = user.opt.theta;

auto t1 = std::chrono::high_resolution_clock::now();

std::vector<std::unique_ptr<Element>> List_Of_Elements;
std::vector<std::unique_ptr<Boundary>> List_Of_Boundaries;
std::vector<std::unique_ptr<Vertex>>   List_Of_Vertices;

unsigned int N_Elements_total, N_Vertices_total;

create_Mesh_Cuboid(
    List_Of_Vertices,
    List_Of_Elements,
    N_Elements_total,
    N_Vertices_total,
    user.opt.Nel_x,
    user.opt.Nel_y,
    user.opt.Nel_z);

    if (tc.periodic_x && tc.periodic_y && tc.periodic_z)
    {
        Connect_3DPeriodic(
            user.opt.Nel_x,
            user.opt.Nel_y,
            user.opt.Nel_z,
            N_Elements_total,
            List_Of_Boundaries);
    }
    else if (tc.id == ProblemID::InertialWaveAttractor)
    {
        // Solid walls in x,y; periodic in z.
        Connect_3D(
            user.opt.Nel_x,
            user.opt.Nel_y,
            user.opt.Nel_z,
            N_Elements_total,
            List_Of_Boundaries,
            1, 1, 0);
    }
    else if (tc.id == ProblemID::SteadyRotating2D3CTaylorGreenWalls)
    {
    // Solid walls in x,y; periodic in z.
    Connect_3D(
        user.opt.Nel_x,
        user.opt.Nel_y,
        user.opt.Nel_z,
        N_Elements_total,
        List_Of_Boundaries,// This overwrites the options setting on walls: periodic (0,0,0) reported
        1, 1, 0);
      }
    else
    {
        Connect_3D(
            user.opt.Nel_x,
            user.opt.Nel_y,
            user.opt.Nel_z,
            N_Elements_total,
            List_Of_Boundaries,
            0, 0, 0);
    }

const PetscInt Nsteps =
    static_cast<PetscInt>(std::llround(user.EndTime / user.Delta_t));

PetscPrintf(PETSC_COMM_WORLD,
    "EndTime = %.16e, Nsteps = %d, Delta_t = %.16e\n",
    (double)user.EndTime, (int)Nsteps, (double)user.Delta_t);

//print(List_Of_Vertices);
//print(List_Of_Elements);
//print(List_Of_Boundaries);

set_Order_Polynomials_Uniform(List_Of_Elements, N_Petsc);
set_theta_Uniform(List_Of_Boundaries, theta);
set_Node_Coordinates_Uniform(List_Of_Elements, List_Of_Vertices, N_Petsc);
Calculate_CuboidFaceNormals(List_Of_Elements, List_Of_Boundaries, List_Of_Vertices);
unsigned int N_Nodes = get_Number_Of_Nodes(List_Of_Elements); // Total number of nodes
unsigned int Np = List_Of_Elements.front()->get_Number_Of_Nodes(); // Number of nodes per element
std::cout << "Total Number of Nodes = " << N_Nodes << std::endl;
unsigned int N_Elements = List_Of_Elements.size();
std::cout << "Total Number of Elements = " << N_Elements << std::endl;
user.Np = Np; user.N_Nodes = N_Nodes;

PetscCall(ISCreateStride(PETSC_COMM_SELF, user.N_Nodes,     0,                1, &user.isu));
PetscCall(ISCreateStride(PETSC_COMM_SELF, user.N_Nodes,     user.N_Nodes,     1, &user.isv));
PetscCall(ISCreateStride(PETSC_COMM_SELF, user.N_Nodes,     2*user.N_Nodes,   1, &user.isw));
PetscCall(ISCreateStride(PETSC_COMM_SELF, user.N_Nodes,     3*user.N_Nodes,   1, &user.isp));
PetscCall(ISCreateStride(PETSC_COMM_SELF, 3*user.N_Nodes,   0,                1, &user.isvel));

PetscCall(build_DiscreteOperators_Inertial(
    List_Of_Vertices,
    List_Of_Boundaries,
    List_Of_Elements,
    user.N_Nodes,
    N_Petsc,
    user.rot,
    user.ops));

Vec Initial_Condition, Velocity, Velocity_U, Velocity_V, Velocity_W;
PetscCall(VecCreateSeq(PETSC_COMM_SELF, 4 * user.N_Nodes, &Initial_Condition));
PetscCall(VecCreateSeq(PETSC_COMM_SELF, 3 * user.N_Nodes, &Velocity));
PetscCall(VecCreateSeq(PETSC_COMM_SELF,     user.N_Nodes, &Velocity_U));
PetscCall(VecCreateSeq(PETSC_COMM_SELF,     user.N_Nodes, &Velocity_V));
PetscCall(VecCreateSeq(PETSC_COMM_SELF,     user.N_Nodes, &Velocity_W));


const ExactVelocitySampling exact_sampling =
    ExactVelocitySampling::L2Projection; // Pointwise;

if (tc.has_exact_solution)
{
    PetscCall(setExactVelocityForTestCase(
        tc,
        Initial_Condition,
        Velocity,
        Velocity_U,
        Velocity_V,
        Velocity_W,
        List_Of_Vertices,
        List_Of_Elements,
        user.N_Nodes,
        N_Petsc,
        exact_sampling,
        0.0,
        user.ops.DIV,
        user.ops.GRAD,
        user.ops.Laplacian));
}
else if (tc.id == ProblemID::InertialWaveAttractor)
{
    PetscCall(VecZeroEntries(Initial_Condition));
    PetscCall(VecZeroEntries(Velocity));
    PetscCall(VecZeroEntries(Velocity_U));
    PetscCall(VecZeroEntries(Velocity_V));
    PetscCall(VecZeroEntries(Velocity_W));
}
else
{
    SETERRQ(PETSC_COMM_SELF,
            PETSC_ERR_SUP,
            "No initialization implemented for this non-exact test case.");
}


PetscCall(compute_Divergence_Velocity(
    Velocity,
    user.N_Nodes,
    user.ops.DIV));

std::cout << "\033[1;32m Initial Condition Computed \033[0m" << std::endl;
user.Velocity = Velocity; user.Velocity_U = Velocity_U; user.Velocity_V = Velocity_V; user.Velocity_W = Velocity_W;

PetscCall(VecDuplicate(user.Velocity, &user.ForceHalf));
PetscCall(VecZeroEntries(user.ForceHalf));
if (tc.id == ProblemID::InertialWaveAttractor)
{
    PetscCall(initializeWaveAttractorForceShape(
        user,
        List_Of_Elements));
}

Vec Xn;
Vec *X_components;
PetscMalloc1(4, &X_components); // allocates an array of memory
// X_components[0] = U^n, X_components[1] = V, X_components[2] = W, X_components[3] = P = 0
VecDuplicate(Velocity_U, &X_components[0]); VecDuplicate(Velocity_V, &X_components[1]); VecDuplicate(Velocity_W, &X_components[2]);
VecCopy(Velocity_U, X_components[0]); VecCopy(Velocity_V, X_components[1]); VecCopy(Velocity_W, X_components[2]);
VecDuplicate(X_components[0], &X_components[3]); VecZeroEntries(X_components[3]);
VecConcatenate(4, X_components, &Xn, NULL);
VecDestroy(&X_components[0]); VecDestroy(&X_components[1]);
VecDestroy(&X_components[2]); VecDestroy(&X_components[3]);
PetscCall(PetscFree(X_components));

Vec rhs_tmp_vel = nullptr;
Vec rhs_bvel_for_div = nullptr;

PetscCall(VecDuplicate(user.Velocity, &rhs_tmp_vel));
PetscCall(VecDuplicate(user.Velocity, &rhs_bvel_for_div));

auto &ops = user.ops;
std::cout << "\033[1;32m Start Computation Sparsity Pattern \033[0m" << std::endl;
Mat Apattern;
PetscCall(CreateFullPatternMatrix(&Apattern, List_Of_Elements, user));

PetscCall(MatDuplicate(Apattern, MAT_DO_NOT_COPY_VALUES, &user.A));
PetscCall(MatDuplicate(Apattern, MAT_DO_NOT_COPY_VALUES, &user.Alinear));
//PetscCall(MatDuplicate(Apattern, MAT_DO_NOT_COPY_VALUES, &user.Anonlinear));

// PetscCall(MatDuplicate(Apattern, MAT_COPY_VALUES, &user.A));
// PetscCall(MatDuplicate(Apattern, MAT_COPY_VALUES, &user.Alinear));
// PetscCall(MatDuplicate(Apattern, MAT_COPY_VALUES, &user.Anonlinear));

PetscCall(MatZeroEntries(user.A));
PetscCall(MatZeroEntries(user.Alinear));
//PetscCall(MatZeroEntries(user.Anonlinear));

PetscCall(MatDestroy(&Apattern));
/*--------------------------------------------------------------------------*/
// Temporarily disabled:
PetscCall(MatSetOption(user.A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
PetscCall(MatSetOption(user.Alinear, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
//PetscCall(MatSetOption(user.Anonlinear, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
PetscCall(MatSetOption(user.A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
PetscCall(MatSetOption(user.Alinear, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
//PetscCall(MatSetOption(user.Anonlinear, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
// Temporarily turned on
// PetscCall(MatSetOption(user.A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE));
// PetscCall(MatSetOption(user.Alinear, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE));
// PetscCall(MatSetOption(user.Anonlinear, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE));
// PetscCall(MatSetOption(user.A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
// PetscCall(MatSetOption(user.Alinear, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
// PetscCall(MatSetOption(user.Anonlinear, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

PetscCall(MatSetOption(user.A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
/*--------------------------------------------------------------------------*/
std::cout << "\033[1;32m Start Computation Linear Part Matrix A \033[0m" << std::endl;
PetscCall(FormMatrixLinearPart(user.Alinear,
                     ops.GRAD,
                     ops.DIVdx, ops.DIVdy, ops.DIVdz,
                     ops.ConstantRotationalMat,
                     ops.ConstantRotationalMatx, ops.ConstantRotationalMaty, ops.ConstantRotationalMatz,
                     ops.Laplacian,
                     user.Delta_t, user.N_Nodes, user.opt.viscous, user.Re, user.opt.include_viscous_divergence_correction));
PetscCall(MatAssemblyBegin(user.Alinear,MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(user.Alinear,MAT_FINAL_ASSEMBLY));
std::cout << "\033[1;32m Computation Linear Part Matrix A Done \033[0m" << std::endl;

user.picardCache.resize(static_cast<PetscInt>(List_Of_Elements.size()),
                        static_cast<PetscInt>(user.Np));

// Set tolerances
PetscBool run_success = PETSC_TRUE;
PetscBool ksp_failed = PETSC_FALSE;
PetscBool picard_failed = PETSC_FALSE;

PetscReal toler = 1e-8;//1e-10;//1e-12;
PetscReal picard_abs = toler;
PetscReal picard_rel = toler;
PetscInt  picard_maxit = 200;
if (!user.opt.nonlinear) picard_maxit = 1;

PetscReal ksp_rtol = toler;
PetscReal ksp_atol = toler;

std::cout << "\033[1;32m Tolerance = " <<  toler << "\033[0m" << std::endl;

PetscCall(VecCreateSeq(PETSC_COMM_SELF, 3*user.N_Nodes, &user.OmegaStar));
PetscCall(VecCreateSeq(PETSC_COMM_SELF,   user.N_Nodes, &user.OmegaStar_x));
PetscCall(VecCreateSeq(PETSC_COMM_SELF,   user.N_Nodes, &user.OmegaStar_y));
PetscCall(VecCreateSeq(PETSC_COMM_SELF,   user.N_Nodes, &user.OmegaStar_z));

KSP ksp;
PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));

setupFieldSplitKSP(
    ksp,
  user.A,
user.Alinear,
user.isvel,
user.isp,
ksp_rtol,
ksp_atol);


std::cout << "\033[1;32m Initialize Time Loop \033[0m" << std::endl;

double t = 0.0;

PetscCall(VecDuplicate(user.Velocity, &user.Velocity_n));
PetscCall(VecDuplicate(user.Velocity_n, &user.Velocity_k));
PetscCall(VecDuplicate(user.Velocity_n, &user.Vsum));
PetscCall(VecDuplicate(Xn, &user.X));
PetscCall(VecCreateSeq(PETSC_COMM_SELF, 4 * user.N_Nodes, &user.blinear));
PetscCall(VecZeroEntries(user.blinear));
PetscCall(VecCreateSeq(PETSC_COMM_SELF, 4 * user.N_Nodes, &user.bnonlinear));
PetscCall(VecZeroEntries(user.bnonlinear));
PetscCall(VecCreateSeq(PETSC_COMM_SELF, 4 * user.N_Nodes, &user.b));
PetscCall(VecZeroEntries(user.b));

PetscCall(initializePicardWorkspace(
    user.picardWork,
    List_Of_Elements,
    user.N_Nodes,
    user.Np));

Vec Solution            = nullptr;
Vec Solution_Velocity   = nullptr;
Vec Solution_Velocity_U = nullptr;
Vec Solution_Velocity_V = nullptr;
Vec Solution_Velocity_W = nullptr;

PetscCall(VecCreateSeq(PETSC_COMM_SELF, 4 * user.N_Nodes, &Solution));
PetscCall(VecCreateSeq(PETSC_COMM_SELF, 3 * user.N_Nodes, &Solution_Velocity));
PetscCall(VecCreateSeq(PETSC_COMM_SELF,     user.N_Nodes, &Solution_Velocity_U));
PetscCall(VecCreateSeq(PETSC_COMM_SELF,     user.N_Nodes, &Solution_Velocity_V));
PetscCall(VecCreateSeq(PETSC_COMM_SELF,     user.N_Nodes, &Solution_Velocity_W));


Vec diff = nullptr;
//PetscCall(VecDuplicate(user.X, &diff));
PetscCall(VecDuplicate(user.Velocity, &diff));

TimeDiagnostics diag;
std::cout << "Nsteps = " << Nsteps << std::endl;
std::cout << "EndTime = " << user.EndTime << std::endl;
std::cout << "Delta_t = " << user.Delta_t << std::endl;

const double t0 = 0.0;
double H0           = 0.0;
double helicity0    = 0.0;
double divmax0      = 0.0;
double divl20       = 0.0;
double helicity_pos0 = 0.0;
double helicity_neg0 = 0.0;
double helicity_unsigned0 = 0.0;
double helicity_signed_nodal0 = 0.0;
double helicity_cancellation_ratio0 = 0.0;

PetscCall(createDiagnosticsWorkspace(
    user.N_Nodes,
    user.Velocity,
    user.diagWork));
PetscCall(computeHamiltonian(
    user.ops.M,
    user.Velocity,
    user.diagWork,
    H0));
PetscCall(computeDivergenceDiagnostics(
    user.Velocity,
    user.ops.DIV,
    user.diagWork,
    divmax0,
    divl20));
PetscCall(initializeRotationVector(
    user.diagWork.Frot,
    user.isu,
    user.isv,
    user.isw,
    user.rot));
PetscCall(computeHelicity(
    user.ops.M,
    user.ops.CURL,
    user.Velocity,
    user.rot,
    user.diagWork,
    helicity0));
PetscCall(computeHelicitySignSplitNodal(
    user.Velocity,
    user.ops.CURL,
    user.rot,
    user.isu,
    user.isv,
    user.isw,
    user.diagWork,
    helicity_pos0,
    helicity_neg0,
    helicity_unsigned0,
    helicity_signed_nodal0,
    helicity_cancellation_ratio0));

PetscPrintf(PETSC_COMM_WORLD,
    "Initial diagnostics: t = %.6e, H = %.16e, h = %.16e, ||DIV V||_inf = %.6e\n",
    t0, H0, helicity0, divmax0);

if (tc.id == ProblemID::InertialWaveAttractor &&
    user.opt.wa_write_probe_timeseries == PETSC_TRUE)
{
    PetscCall(initializeWaveAttractorProbeRecorder(
        user,
        List_Of_Elements));

    // Record the initial state at t = 0 as sample step 0.
    PetscCall(appendWaveAttractorProbeSamples(
        user,
        0,
        0.0));
}

diag.time_full.reserve(Nsteps + 1);
diag.H.reserve(Nsteps + 1);
diag.H_drift.reserve(Nsteps + 1);
diag.helicity.reserve(Nsteps + 1);
diag.helicity_drift.reserve(Nsteps + 1);
diag.div_max.reserve(Nsteps + 1);
diag.div_l2.reserve(Nsteps + 1);


diag.helicity_positive.reserve(Nsteps + 1);
diag.helicity_negative.reserve(Nsteps + 1);
diag.helicity_unsigned.reserve(Nsteps + 1);
diag.helicity_signed_nodal.reserve(Nsteps + 1);
diag.helicity_cancellation_ratio.reserve(Nsteps + 1);

diag.time_half.reserve(Nsteps);
diag.energy_dissipation.reserve(Nsteps);
diag.energy_balance_residual.reserve(Nsteps);
diag.helicity_dissipation.reserve(Nsteps);
diag.helicity_balance_residual.reserve(Nsteps);
diag.energy_power_unscaled.reserve(Nsteps);
diag.energy_power_input.reserve(Nsteps);
diag.helicity_power_unscaled.reserve(Nsteps);
diag.helicity_power_input.reserve(Nsteps);


diag.energy_balance_residual_cumulative.reserve(Nsteps);
diag.helicity_boundary_nonlinear.reserve(Nsteps);
diag.helicity_boundary_pressure_background.reserve(Nsteps);
diag.helicity_boundary_pressure_pairing.reserve(Nsteps);
diag.helicity_boundary_viscous.reserve(Nsteps);
diag.helicity_boundary_forcing.reserve(Nsteps);
diag.helicity_boundary_flux.reserve(Nsteps);
diag.helicity_boundary_flux_cumulative.reserve(Nsteps);
diag.helicity_curl_grad_defect.reserve(Nsteps);
diag.helicity_curl_grad_p_inf.reserve(Nsteps);
diag.helicity_curl_grad_p_l2.reserve(Nsteps);
diag.helicity_curl_grad_p_mass.reserve(Nsteps);
diag.helicity_balance_residual_corrected.reserve(Nsteps);
diag.helicity_balance_residual_cumulative.reserve(Nsteps);
diag.helicity_balance_residual_corrected_cumulative.reserve(Nsteps);

diag.time_full.push_back(t0);
diag.H.push_back(H0);
diag.div_max.push_back(divmax0);
diag.div_l2.push_back(divl20);
diag.helicity.push_back(helicity0);
diag.H_drift.push_back(0.0);
diag.helicity_drift.push_back(0.0);

diag.helicity_positive.push_back(helicity_pos0);
diag.helicity_negative.push_back(helicity_neg0);
diag.helicity_unsigned.push_back(helicity_unsigned0);
diag.helicity_signed_nodal.push_back(helicity_signed_nodal0);
diag.helicity_cancellation_ratio.push_back(helicity_cancellation_ratio0);

Vec Velocity_nm1 = nullptr;
PetscCall(VecDuplicate(user.Velocity, &Velocity_nm1));

// At t = 0, V^{n-1} is unavailable.
// Initialize it equal to V^n so extrapolation reduces to V^n at the first step.
PetscCall(VecCopy(user.Velocity, Velocity_nm1));

// Lumped/nodal weights from diagonal of M.
// For exact sign splitting, prefer an element-quadrature implementation.
PetscCall(MatGetDiagonal(user.ops.M, user.diagWork.Mdiag));

std::cout << "\033[1;32m Start Time Loop \033[0m" << std::endl;

PetscBool include_force = PETSC_FALSE;
if (tc.id == ProblemID::InertialWaveAttractor)
{
    std::cout << "user.opt.Fo = " << user.opt.Fo << std::endl;
    PetscCheck(user.opt.Fo > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
    "Wave attractor forcing uses amplitude 1/Fo^2, so Fo must be positive. "
    " Use a large Fo for weak forcing, e.g. -Fo 40");
}
for (PetscInt step = 0; step < Nsteps; ++step) //
{
    t = (step + 1) * user.Delta_t;
    user.t = t;
    PR(t);

    // ------------------------------------------------------------
    // 1. Set Velocity_n = accepted velocity from previous timestep
    // ------------------------------------------------------------
    PetscCall(VecCopy(user.Velocity, user.Velocity_n));


    if (tc.id == ProblemID::InertialWaveAttractor)
    {
        const PetscReal tn =
            t - user.Delta_t;

        PetscCall(updateWaveAttractorForceHalf(
            user,
            tn,
            &include_force));

            // Note: for nonpolynomial forcing: first perform L2 projection, just like initial velocity

            //             PetscCall(printDivergenceNorms(
            //                 "DIV B",
            //                 user.ForceHalf,
            //                 user.ops,
            //                 user.N_Nodes));
            // pause();
    }
    else
    {
        include_force = PETSC_FALSE;
    }
    // ------------------------------------------------------------
    // 2. Build linear RHS once this timestep
    // ------------------------------------------------------------

    const PetscReal force_scale = (include_force == PETSC_TRUE) ? 1.0/(user.opt.Fo * user.opt.Fo) : 1.0;
    PetscCall(formRHSLinear(
        user.blinear,
        user.Velocity_n,
        include_force ? user.ForceHalf : nullptr,
        user.ops,
        user.rot,
        user.Delta_t,
        user.Re,
        force_scale,
        user.opt.viscous,
        include_force,
        user.opt.include_divergence_correction,
        user.isvel,
        user.isp,
        user.isu,
        user.isv,
        user.isw,
        rhs_tmp_vel,
        rhs_bvel_for_div));

        // ------------------------------------------------------------
        // 3. Initial Picard guess.
        //
        // Default:
        //   V_guess = V^n
        //
        // Optional extrapolated guess:
        //   V_guess = 2 V^n - V^{n-1}
        //
        // We also put this velocity guess into the velocity block of user.X,
        // so that the Picard convergence difference X^{k+1} - X^k is measured
        // relative to the same initial guess that is used to freeze omega.
        // ------------------------------------------------------------
        PetscCall(VecCopy(Xn, user.X));  // keep previous pressure as pressure guess

        if (user.opt.picard_extrapolate_initial_guess && step > 0)
        {
            // Velocity_k <- 2 Velocity_n - Velocity_nm1
            PetscCall(VecCopy(user.Velocity_n, user.Velocity_k));
            PetscCall(VecScale(user.Velocity_k, 2.0));
            PetscCall(VecAXPY(user.Velocity_k, -1.0, Velocity_nm1));
        }
        else
        {
            // First timestep, or extrapolation disabled:
            // Velocity_k <- Velocity_n
            PetscCall(VecCopy(user.Velocity_n, user.Velocity_k));
        }

        // Put extrapolated velocity guess into the velocity block of user.X.
        {
            Vec Xvel_guess = nullptr;
            PetscCall(VecGetSubVector(user.X, user.isvel, &Xvel_guess));
            PetscCall(VecCopy(user.Velocity_k, Xvel_guess));
            PetscCall(VecRestoreSubVector(user.X, user.isvel, &Xvel_guess));
        }

        PetscInt picard_it = 0;
        PetscBool converged = PETSC_FALSE;

    while (picard_it < picard_maxit)
    {
        ++picard_it;

if (user.opt.nonlinear)
{
PetscCall(computeFrozenOmega(
    user.ops,
    user.Velocity_k,
    user.Velocity_n,
    user.N_Nodes,
    user.Vsum,
    user.OmegaStar,
    user.OmegaStar_x,
    user.OmegaStar_y,
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

PetscCall(formMatrixPicardTotalFromCache(
    user.A,
    user.Alinear,
    user.picardCache,
    user.picardWork,
    List_Of_Elements,
    user.ops,
    user.N_Nodes,
    user.Np,
    user.Delta_t));

PetscCall(formRHSPicardDirectFromCache(
    user.bnonlinear,
    user.picardCache,
    user.picardWork,
    user.Velocity_n,
    user.ops,
    user.N_Nodes,
    user.Np,
    user.Delta_t));

        // --------------------------------------------------------
        // 6. Assemble total matrix and RHS
        // --------------------------------------------------------
        //PetscCall(assembleTotalMatrix(user.A, user.Alinear, user.Anonlinear));
        PetscCall(assembleTotalRHS(user.b, user.blinear, user.bnonlinear));
}
else
{
  PetscCall(MatCopy(user.Alinear, user.A, SAME_NONZERO_PATTERN));
  PetscCall(VecCopy(user.blinear, user.b));
}

        // --------------------------------------------------------
        // 7. Solve A X = b
        // --------------------------------------------------------

        if (user.opt.use_variable_ksp_tol_picard)
        {
            PetscReal ksp_rtol_this = user.opt.ksp_rtol_picard_tight;

            if (picard_it < user.opt.ksp_rtol_picard_tight_after)
            {
                const PetscReal alpha =
                    (PetscReal)(picard_it - 1) /
                    (PetscReal)PetscMax(1, user.opt.ksp_rtol_picard_tight_after - 1);

                // Geometric interpolation from loose to tight:
                // rtol = loose^(1-alpha) * tight^alpha
                ksp_rtol_this =
                    PetscExpReal((1.0 - alpha) * PetscLogReal(user.opt.ksp_rtol_picard_loose)
                               + alpha        * PetscLogReal(user.opt.ksp_rtol_picard_tight));
            }

            PetscCall(KSPSetTolerances(
                ksp,
                ksp_rtol_this,
                PETSC_CURRENT,
                PETSC_CURRENT,
                PETSC_CURRENT));

            if (step < 3)
            {
                PetscPrintf(PETSC_COMM_WORLD,
                    "  Picard iter %d: KSP rtol = %.3e\n",
                    (int)picard_it,
                    (double)ksp_rtol_this);
            }
        }

        //PetscCall(KSPSetOperators(ksp, user.A, user.A));
        PetscCall(KSPSolve(ksp, user.b, user.X));

        KSPConvergedReason ksp_reason;
        PetscCall(KSPGetConvergedReason(ksp, &ksp_reason));
        if (ksp_reason < 0)
        {
            PetscPrintf(PETSC_COMM_WORLD,
                "\nKSP DIVERGENCE detected at timestep %d, t = %.16e, Picard iter %d\n",
                (int)step, (double)t, (int)picard_it);
            PetscPrintf(PETSC_COMM_WORLD,
                "KSP diverged with reason %d\n\n",
                (int)ksp_reason);

                run_success = PETSC_FALSE;
                ksp_failed = PETSC_TRUE;
            break;
        }
        // --------------------------------------------------------
        // 8. Extract velocity part from X and update Velocity_k
        // --------------------------------------------------------

            Vec Vnew = nullptr;
            PetscCall(VecGetSubVector(user.X, user.isvel, &Vnew));

            // --------------------------------------------------------
            // 9. Check Picard convergence
            // --------------------------------------------------------
            PetscReal diff_norm, xold_norm;
            PetscCall(VecCopy(user.Velocity_k, diff));
            PetscCall(VecAXPY(diff, -1.0, Vnew));
            PetscCall(VecNorm(diff, NORM_2, &diff_norm));
            PetscCall(VecNorm(user.Velocity_k, NORM_2, &xold_norm));

            PetscCall(VecCopy(Vnew, user.Velocity_k));
            PetscCall(VecRestoreSubVector(user.X, user.isvel, &Vnew));


        if (!user.opt.nonlinear)
        {
          converged = PETSC_TRUE;
          break;
        }

        PetscPrintf(PETSC_COMM_WORLD,
            "  Picard iter %d: abs = %.6e, rel = %.6e\n",
            (int)picard_it,
            (double)diff_norm,
            (double)(xold_norm > 0 ? diff_norm / xold_norm : diff_norm));

        if (diff_norm < picard_abs ||
            (xold_norm > 0 && diff_norm / xold_norm < picard_rel))
        {
            converged = PETSC_TRUE;
            break;
        }
    }

    if (ksp_failed)
    {
        break;  // exits the outer time-step loop
    }

    if (!converged)
    {
        PetscPrintf(PETSC_COMM_WORLD,
            "\nPicard did not converge at timestep %d, t = %.16e\n\n",
            (int)step, (double)t);

        run_success = PETSC_FALSE;
        picard_failed = PETSC_TRUE;
        break;
    }

    // ------------------------------------------------------------
    // 10. Accept solution: Xn <- X, Velocity <- velocity(X)
    // ------------------------------------------------------------

    {
        Vec Velocity_star_U = nullptr, Velocity_star_V = nullptr, Velocity_star_W = nullptr;
        Vec Velocity_star = nullptr, Pressure_star = nullptr; // used to subtract mean pressure

        PetscCall(VecGetSubVector(user.X, user.isu, &Velocity_star_U));
        PetscCall(VecGetSubVector(user.X, user.isv, &Velocity_star_V));
        PetscCall(VecGetSubVector(user.X, user.isw, &Velocity_star_W));
        PetscCall(VecGetSubVector(user.X, user.isvel, &Velocity_star));

        PetscCall(VecCopy(Velocity_star,   user.Velocity));
        PetscCall(VecCopy(Velocity_star_U, user.Velocity_U));
        PetscCall(VecCopy(Velocity_star_V, user.Velocity_V));
        PetscCall(VecCopy(Velocity_star_W, user.Velocity_W));

        PetscCall(VecRestoreSubVector(user.X, user.isu, &Velocity_star_U));
        PetscCall(VecRestoreSubVector(user.X, user.isv, &Velocity_star_V));
        PetscCall(VecRestoreSubVector(user.X, user.isw, &Velocity_star_W));
        PetscCall(VecRestoreSubVector(user.X, user.isvel, &Velocity_star));

        // // pressure mean subtraction
        // PetscCall(VecGetSubVector(user.X, user.isp, &Pressure_star));
        // PetscScalar sum_p;
        // PetscCall(VecSum(Pressure_star, &sum_p));
        // PetscScalar mean_p = sum_p / (PetscScalar)user.N_Nodes;
        // PetscCall(VecShift(Pressure_star, -mean_p));
        // PetscCall(VecRestoreSubVector(user.X, user.isp, &Pressure_star));

    }
    if (user.opt.enable_diagnostics)
{
  PetscCall(recordDiagnosticsAtAcceptedStep(
    diag,
    user,
    include_force ? user.ForceHalf : nullptr,
    List_Of_Vertices,
    List_Of_Elements,
    N_Petsc,
    t,
    user.Delta_t));
}

if (tc.id == ProblemID::InertialWaveAttractor &&
    user.opt.wa_write_probe_timeseries == PETSC_TRUE)
{
    PetscCall(appendWaveAttractorProbeSamples(
        user,
        step + 1,
        t));
}
if (tc.id == ProblemID::InertialWaveAttractor &&
    user.opt.wa_write_snapshot_at_forcing_off == PETSC_TRUE &&
    user.wa_snapshot_written != PETSC_TRUE &&
    isWaveAttractorForcingOffStep(t, user.Delta_t, user.opt))
{
    PetscCall(writeWaveAttractorForcingOffSnapshot(
        user,
        List_Of_Elements,
        user.ForceHalf,
        t));

    user.wa_snapshot_written = PETSC_TRUE;
}

    // Store old accepted velocity for next timestep extrapolation.
    // At this point:
    //   user.Velocity_n = V^n
    //   user.Velocity   = V^{n+1}
    PetscCall(VecCopy(user.Velocity_n, Velocity_nm1));

    // Store accepted full state.
    PetscCall(VecCopy(user.X, Xn));


}

    PetscCall(printDivergenceNorms(
        "state after timeloop",
        user.Velocity,
        user.ops,
        user.N_Nodes));
compute_Divergence_Velocity(user.Velocity, user.N_Nodes, user.ops.DIV);

PetscCall(VecDestroy(&diff));

auto t5 = std::chrono::high_resolution_clock::now();

std::cout << "Solve took " ;
std::cout << std::chrono::duration_cast<std::chrono::seconds>(t5-t1).count() << " seconds ";
std::cout << std::chrono::duration_cast<std::chrono::minutes>(t5-t1).count() << " minutes ";
std::cout << std::chrono::duration_cast<std::chrono::hours>(t5-t1).count()   << " hours\n";

double Hend, helicityend;
PetscCall(computeHamiltonian(
    user.ops.M,
    user.Velocity,
    user.diagWork,
    Hend));

PetscCall(computeHelicity(
    user.ops.M,
    user.ops.CURL,
    user.Velocity,
    user.rot,
    user.diagWork,
    helicityend));

std::cout << "\033[1;32m Difference in Energy = " << Hend-H0 << "\033[0m" << std::endl;
std::cout << "\033[1;32m Difference in helicity = " << helicityend-helicity0 << "\033[0m" << std::endl;

if (run_success)
{

KSPGetTolerances(ksp, &ksp_rtol, &ksp_atol, NULL, NULL);
//char ksp_tol_str[16];
//sprint(ksp_tol_str, "%.0e", ksp_rtol);
int ksp_exp = (int)(-log10(ksp_rtol));
std::cout << "ksp_exp = " << ksp_exp << std::endl;
std::cout << "t = " << t << std::endl;
char filename[PETSC_MAX_PATH_LEN];

if (tc.id == ProblemID::BoostedRotatingEuler)
{
    std::snprintf(filename, sizeof(filename),
        "output_Nonlinear_RotatingEuler_%02d_%03d_%03d_%03d_%04d_%03d_ksp1em%02d.m",
        (int)N_Petsc,
        (int)user.opt.Nel_x,
        (int)user.opt.Nel_y,
        (int)user.opt.Nel_z,
        (int)user.opt.steps_per_period,
        (int)user.opt.periods,
        ksp_exp);
}
else if (tc.id == ProblemID::DecayingNavierStokes)
{
    std::snprintf(filename, sizeof(filename),
        "output_DecayingNavierStokes_U%.2f_V%.2f_W%.2f_%02d_%03d_%03d_%03d_%04d_%03d_%.3f_ksp1em%02d.m",
        (double)tc.U,
        (double)tc.V,
        (double)tc.W,
        (int)N_Petsc,
        (int)user.opt.Nel_x,
        (int)user.opt.Nel_y,
        (int)user.opt.Nel_z,
        (int)user.opt.steps_per_period,
        (int)user.opt.periods,
        (double)user.EndTime,
        ksp_exp);
}
else if (tc.id == ProblemID::SteadyRotating2D3CTaylorGreenWalls)
{
    std::snprintf(filename, sizeof(filename),
        "output_SteadyRotating2D3CTaylorGreenWalls_%02d_%03d_%03d_%03d_%04d_%03d_%.3f_ksp1em%02d.m",
        (int)N_Petsc,
        (int)user.opt.Nel_x,
        (int)user.opt.Nel_y,
        (int)user.opt.Nel_z,
        (int)user.opt.steps_per_period,
        (int)user.opt.periods,
        (double)user.EndTime,
        ksp_exp);
}
else if (tc.id == ProblemID::InertialWaveAttractor)
{
    std::snprintf(filename, sizeof(filename),
        "output_InertialWaveAttractor_%02d_%03d_%03d_%03d_%04d_%03d_%.3f_Fo%.2f_ksp1em%02d  _divcorr%01d_visccorr%01d.m",
        (int)N_Petsc,
        (int)user.opt.Nel_x,
        (int)user.opt.Nel_y,
        (int)user.opt.Nel_z,
        (int)user.opt.steps_per_period,
        (int)user.opt.periods,
        (double)user.EndTime,
        (double)user.opt.Fo,
        ksp_exp,
        (int)user.opt.include_divergence_correction,
        (int)user.opt.include_viscous_divergence_correction);
}
else
{
    SETERRQ(PETSC_COMM_SELF,
            PETSC_ERR_SUP,
            "Output filename is not implemented for this test case.");
}

std::cout << "Viscous = " << user.opt.viscous << std::endl;

double ErrorVel = 0.0;
if (tc.has_exact_solution)
{
    PetscCall(setExactVelocityForTestCase(
        tc,
        Solution,
        Solution_Velocity,
        Solution_Velocity_U,
        Solution_Velocity_V,
        Solution_Velocity_W,
        List_Of_Vertices,
        List_Of_Elements,
        user.N_Nodes,
        N_Petsc,
        exact_sampling,
        t,
        user.ops.DIV,
        user.ops.GRAD,
        user.ops.Laplacian));

    PetscCall(compute_Divergence_Velocity(
        Solution_Velocity,
        user.N_Nodes,
        user.ops.DIV));

    Vec err, tmp;
    PetscScalar dot;
    PetscCall(VecDuplicate(user.Velocity, &err));
    PetscCall(VecDuplicate(user.Velocity, &tmp));

    PetscCall(VecWAXPY(err, -1.0, Solution_Velocity, user.Velocity));
    PetscCall(MatMult(user.ops.M, err, tmp));
    PetscCall(VecDot(err, tmp, &dot));

    ErrorVel = std::sqrt(PetscRealPart(dot));
    std::cout << "\033[1;32m ErrorVel = "
              << std::scientific << std::setprecision(4)
              << ErrorVel << "\033[0m" << std::endl;

    PetscCall(VecDestroy(&err));
    PetscCall(VecDestroy(&tmp));
}
else if (tc.id == ProblemID::InertialWaveAttractor)
{
    std::cout << "\033[1;32m No exact velocity: skipping ErrorVel for wave attractor. \033[0m"
              << std::endl;
}

if (user.opt.enable_diagnostics)
{
  char filename[256];

std::snprintf(filename, sizeof(filename),
    "%s_%02d_%03d_%03d_%03d_%04d_%03d_%.3f_Fo%.4f_ksp1em%02d_divcorr%01d_visccorr%01d.csv",
    tc.diagnostics_prefix.c_str(),
    (int)N_Petsc,
    (int)user.opt.Nel_x,
    (int)user.opt.Nel_y,
    (int)user.opt.Nel_z,
    (int)user.opt.steps_per_period,
    (int)user.opt.periods,
    (double)user.EndTime,
    (double)user.opt.Fo,
    ksp_exp,
    (int)user.opt.include_divergence_correction,
    (int)user.opt.include_viscous_divergence_correction);


// Call writer
PetscCall(saveDiagnosticsCSV(filename, diag));
}
Vec Xcoord, Ycoord, Zcoord;
PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.N_Nodes, &Xcoord));
PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.N_Nodes, &Ycoord));
PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.N_Nodes, &Zcoord));

    //char szFileName[255] = {0};
    //std::string store_solution = "Solution/Solutions/Coordinates_n"+std::to_string(Number_Of_Elements_Petsc)+"x"+std::to_string(Number_Of_Elements_Petsc)+"N"+std::to_string(N_Petsc)+"Ts"+std::to_string(Number_Of_TimeSteps_In_One_Period)+".txt";
    //const char *store_solution_char = store_solution.c_str();
    //FILE *f = fopen(store_solution_char, "w");
    //fprintf(f, "n \t pos \t xCoor \t zCoor \t p value \n");
    for (auto k = List_Of_Elements.begin(); k < List_Of_Elements.end(); k++)
    {
        unsigned int Np = (*k)->get_Number_Of_Nodes();
        unsigned int pos = (*k)->get_pos();
        std::vector<double> xCoor, yCoor, zCoor;
        xCoor = (*k)->get_node_coordinates_x();
        yCoor = (*k)->get_node_coordinates_y();
        zCoor = (*k)->get_node_coordinates_z();
        for (unsigned int n = 0; n < Np; n++)
        {
            VecSetValue(Xcoord, pos + n, xCoor[n], INSERT_VALUES);
            VecSetValue(Ycoord, pos + n, yCoor[n], INSERT_VALUES);
            VecSetValue(Zcoord, pos + n, zCoor[n], INSERT_VALUES);
        }
    }
    //fclose(f);
    VecAssemblyBegin(Xcoord); VecAssemblyEnd(Xcoord);
    VecAssemblyBegin(Ycoord); VecAssemblyEnd(Ycoord);
    VecAssemblyBegin(Zcoord); VecAssemblyEnd(Zcoord);

PetscViewer viewerhdf5 = nullptr;
PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF, filename, &viewerhdf5));
PetscCall(PetscViewerPushFormat(viewerhdf5, PETSC_VIEWER_ASCII_MATLAB));
if (tc.has_exact_solution)
{
PetscViewerASCIIPrintf(viewerhdf5, "Error Vel = %.16e;\n", ErrorVel);
}
PetscViewerASCIIPrintf(viewerhdf5, "Time = %.16e;\n", t);
PetscViewerASCIIPrintf(viewerhdf5, "Delta t = %.16e;\n", user.Delta_t);
PetscObjectSetName((PetscObject)Xcoord, "X"); VecView(Xcoord, viewerhdf5);
PetscObjectSetName((PetscObject)Ycoord, "Y"); VecView(Ycoord, viewerhdf5);
PetscObjectSetName((PetscObject)Zcoord, "Z"); VecView(Zcoord, viewerhdf5);
PetscObjectSetName((PetscObject)user.X, "Numerical"); VecView(user.X, viewerhdf5);
if (tc.has_exact_solution)
{
    PetscObjectSetName((PetscObject)Solution, "Exact");
    PetscCall(VecView(Solution, viewerhdf5));
}
PetscCall(PetscViewerDestroy(&viewerhdf5));

std::cout << "\033[1;32m Output Stored \033[0m" << std::endl;



    PetscCall(VecDestroy(&Xcoord)); PetscCall(VecDestroy(&Ycoord)); PetscCall(VecDestroy(&Zcoord));
}
PetscCall(VecDestroy(&rhs_tmp_vel)); PetscCall(VecDestroy(&rhs_bvel_for_div));

PetscCall(VecDestroy(&Velocity_nm1)); PetscCall(VecDestroy(&diff));
PetscCall(VecDestroy(&Initial_Condition)); PetscCall(VecDestroy(&Solution)); PetscCall(VecDestroy(&Solution_Velocity)); PetscCall(VecDestroy(&Solution_Velocity_U));
PetscCall(VecDestroy(&Solution_Velocity_V)); PetscCall(VecDestroy(&Solution_Velocity_W)); PetscCall(VecDestroy(&Xn));
PetscCall(destroyPicardWorkspace(user.picardWork)); PetscCall(destroyDiagnosticsWorkspace(user.diagWork));

PetscCall(KSPDestroy(&ksp));
PetscCall(destroyWaveAttractorProbeRecorder(user));
PetscCall(DestroyAppCtx(user));

auto t3 = std::chrono::high_resolution_clock::now();
std::cout << "Execution took " ;
std::cout << std::chrono::duration_cast<std::chrono::seconds>(t3-t1).count() << " seconds ";
std::cout << std::chrono::duration_cast<std::chrono::minutes>(t3-t1).count() << " minutes ";
std::cout << std::chrono::duration_cast<std::chrono::hours>(t3-t1).count()   << " hours\n";

PetscCall(SlepcFinalize());
return 0;
}
