#ifndef HIGW
#define HIGW

#include <petscksp.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <unordered_set>
#include <algorithm>
#include <vector>
#include <string>
#include <cstdio>

#include "Elements.hpp"
#include "Legendre_Gauss_Lobatto.hpp"
#include "initial_cond.hpp"
#include "options.hpp"
//#include "test_cases.hpp"
//#include "mesh_gen.hpp"
/*--------------------------------------------------------------------------*/
struct TripleProductTensor1D
{
    unsigned int N = 0;                 // polynomial order in 1D
    unsigned int n1 = 0;                // n1 = N+1
    std::vector<PetscScalar> G;         // size n1*n1*n1

    void resize(unsigned int N_in)
    {
        N  = N_in;
        n1 = N + 1;
        G.assign(n1 * n1 * n1, 0.0);
    }

    inline PetscScalar &operator()(unsigned int a,
                                   unsigned int b,
                                   unsigned int c)
    {
        return G[(a * n1 + b) * n1 + c];
    }

    inline const PetscScalar &operator()(unsigned int a,
                                         unsigned int b,
                                         unsigned int c) const
    {
        return G[(a * n1 + b) * n1 + c];
    }
};
/*--------------------------------------------------------------------------*/
struct InertialRawMatrices
{
    Mat E = nullptr, ET = nullptr;
    Mat Edx = nullptr, Edy = nullptr, Edz = nullptr;
    Mat ETdx = nullptr, ETdy = nullptr, ETdz = nullptr;

    Mat GRADdx = nullptr, GRADdy = nullptr, GRADdz = nullptr;
    Mat M = nullptr, M_small = nullptr;
    Mat invM = nullptr, invM_small = nullptr;

    Mat ConstantRotational = nullptr;
    Mat ConstantRotationalx = nullptr;
    Mat ConstantRotationaly = nullptr;
    Mat ConstantRotationalz = nullptr;

    // Curl matrices: C = M*CURL
    Mat Curl = nullptr;   // 3*N_Nodes x 3*N_Nodes
    Mat Curlx = nullptr;  // N_Nodes   x 3*N_Nodes
    Mat Curly = nullptr;  // N_Nodes   x 3*N_Nodes
    Mat Curlz = nullptr;  // N_Nodes   x 3*N_Nodes
    // symmetrized Curl matrices: Curl_sym = 0.5(Curl+Curl^T)
    Mat Curl_sym = nullptr;   // 3*N_Nodes x 3*N_Nodes
    Mat Curlx_sym = nullptr;  // N_Nodes   x 3*N_Nodes
    Mat Curly_sym = nullptr;  // N_Nodes   x 3*N_Nodes
    Mat Curlz_sym = nullptr;  // N_Nodes   x 3*N_Nodes
};
/*--------------------------------------------------------------------------*/
PetscErrorCode checkCurlSymmetry(Mat C, const char name[]);
/*--------------------------------------------------------------------------*/
struct DiscreteOperators
{
    Mat M = nullptr, M_small = nullptr;
    Mat invM = nullptr, invM_small = nullptr;

    Mat GRAD = nullptr;
    //Mat GRADdx = nullptr, GRADdy = nullptr, GRADdz = nullptr;

    Mat DIV = nullptr;
    Mat DIVdx = nullptr, DIVdy = nullptr, DIVdz = nullptr;

    Mat CURL = nullptr;
    Mat CURLdx = nullptr, CURLdy = nullptr, CURLdz = nullptr;
    Mat CurlBoundary = nullptr;  // C - C^T, where C = M*CURL
    // Mat CURL_sym = nullptr;
    // Mat CURLdx_sym = nullptr, CURLdy_sym = nullptr, CURLdz_sym = nullptr;

    Mat Laplacian = nullptr;

    Mat ConstantRotationalMat = nullptr;
    Mat ConstantRotationalMatx = nullptr;
    Mat ConstantRotationalMaty = nullptr;
    Mat ConstantRotationalMatz = nullptr;

    TripleProductTensor1D G1D; // new factorized 1D tensor
};
/*--------------------------------------------------------------------------*/
struct RotationParameters
{
    double f1 = 0.0;
    double f2 = 0.0;
    double f3 = 0.0;
    double RossbyNumber = 1.0;
    PetscBool include_rotation = PETSC_FALSE;
};
/*--------------------------------------------------------------------------*/
struct TimeDiagnostics
{
    // Full time levels: t^n
    std::vector<double> time_full;

    std::vector<double> H;                // Hamiltonian / kinetic energy H^n
    std::vector<double> H_drift;          // H^n - H^0

    std::vector<double> helicity;         // h^n, including rotation if computeHelicity does so
    std::vector<double> helicity_drift;   // h^n - h^0

    std::vector<double> div_max;          // ||DIV V^n||_inf
    std::vector<double> div_l2;           // ||DIV V^n||_2 or mass-weighted L2 depending on implementation

    // Half time levels: t^{n+1/2}
    std::vector<double> time_half;

    std::vector<double> energy_dissipation;        // E^{n+1/2}
    std::vector<double> energy_balance_residual;   // (H^{n+1}-H^n)/dt + Re^{-1} E^{n+1/2}
    std::vector<double> energy_balance_residual_cumulative;
    std::vector<double> energy_power_unscaled;     // W^{n+1/2}
    std::vector<double> energy_power_input;        // Fo^{-2} W^{n+1/2}

    std::vector<double> helicity_dissipation;
    std::vector<double> helicity_balance_residual;

    std::vector<double> helicity_power_unscaled;
    std::vector<double> helicity_power_input;

    std::vector<double> helicity_boundary_nonlinear;
    std::vector<double> helicity_boundary_pressure_background;
    std::vector<double> helicity_boundary_pressure_pairing;
    std::vector<double> helicity_boundary_viscous;
    std::vector<double> helicity_boundary_forcing;
    std::vector<double> helicity_boundary_flux;
    std::vector<double> helicity_boundary_flux_cumulative;

    std::vector<double> helicity_curl_grad_defect;
    std::vector<double> helicity_curl_grad_p_inf;
    std::vector<double> helicity_curl_grad_p_l2;
    std::vector<double> helicity_curl_grad_p_mass;

    std::vector<double> helicity_balance_residual_corrected;
    std::vector<double> helicity_balance_residual_cumulative;
    std::vector<double> helicity_balance_residual_corrected_cumulative;

    //std::vector<double> helicity_residual;
    //std::vector<double> helicity_residual_forced_no_boundary;

    std::vector<double> helicity_positive;
    std::vector<double> helicity_negative;
    std::vector<double> helicity_unsigned;
    std::vector<double> helicity_signed_nodal;
    std::vector<double> helicity_cancellation_ratio;

};
/*--------------------------------------------------------------------------*/
struct WaveAttractorProbe
{
    std::string label;

    PetscReal x_target = 0.0;
    PetscReal y_target = 0.0;
    PetscReal z_target = 0.0;

    PetscReal x_actual = 0.0;
    PetscReal y_actual = 0.0;
    PetscReal z_actual = 0.0;

    PetscInt element_id = -1;
    PetscInt local_node = -1;
    PetscInt representative_global_node = -1;

    // A physical coordinate may occur multiple times in a DG discretization,
    // for example on an element interface. All copies are averaged.
    std::vector<PetscInt> global_nodes;
};
/*--------------------------------------------------------------------------*/
struct WaveAttractorProbeRecorder
{
    std::vector<WaveAttractorProbe> probes;

    // Reused workspace:
    // relative_omega = CURL * Velocity.
    Vec relative_omega = nullptr;

    std::string filename;
    FILE *file = nullptr;

    PetscBool initialized = PETSC_FALSE;
};
/*--------------------------------------------------------------------------*/
struct DiagnosticsWorkspace
{
    Vec div = nullptr;
    Vec Vmid = nullptr;
    Vec Mdiag = nullptr;
    Vec AV = nullptr;
    Vec MAV = nullptr;
    Vec tmpN = nullptr;

    Vec err = nullptr;
    Vec tmp3N = nullptr;

    // helicity diagnostics
    Vec omega = nullptr;  // Size 3 N_t, omega = CURL V
    Vec Momega = nullptr; // Size 3 N_t, M omega

    Vec boundaryWork = nullptr;
    Vec Nrelative = nullptr;
    Vec Nrotation = nullptr;
    Vec Ntotal = nullptr;

    Vec Frot = nullptr; // Size 3 N_t, coefficients of f = (f1, f2, f3)
    // CURL GRAD P diagnostic
    Vec gradP = nullptr; // Size 3 N_t
    Vec curlGradP = nullptr; // Size 3 N_t

};
/*--------------------------------------------------------------------------*/
struct AbsoluteHelicityBoundaryDiagnostics
{
    // This contains both:
    //   -(CURL Vmid) x Vmid
    // and
    //   -(1/Ro) f x Vmid.
    double nonlinear = 0.0;

    double pressure_background = 0.0;
    double pressure_pairing = 0.0;

    double viscous = 0.0;
    double forcing = 0.0;

    double total = 0.0;

    // Scalar balance-law contribution:
     // (CURL GRAD P)^T M Vmid
     double curl_grad_defect = 0.0;

     // Norms of CURL GRAD P itself:
     double curl_grad_p_inf = 0.0;
     double curl_grad_p_l2 = 0.0;
     double curl_grad_p_mass = 0.0;
};
/*--------------------------------------------------------------------------*/
struct PicardWorkspace
{
    // element / dof mapping
    std::vector<PetscInt> node_to_elem_idx;
    std::vector<PetscInt> elem_pos_list;

    // reusable dense local buffers
    std::vector<PetscScalar> Pe_x, Pe_y, Pe_z;
    std::vector<PetscScalar> u_loc, v_loc, w_loc;
    std::vector<PetscScalar> bu_loc, bv_loc, bw_loc;

    // reusable index sets for [U;V;W;P]
    IS isU = nullptr, isV = nullptr, isW = nullptr, isP = nullptr;

    // reusable temp vector for pressure RHS
    Vec tmpP = nullptr;
    PetscInt NElements = 0;

    // New reusable buffers for pressure-row nonlinear assembly
    std::vector<PetscInt> pressure_touched_elems;
    std::vector<PetscInt> pressure_touched_mark;
    std::vector<PetscScalar> pressure_accum;
    std::vector<PetscInt> pressure_cols_out;

    std::vector<PetscInt> dense_rows;
    std::vector<PetscInt> dense_cols;
    std::vector<PetscScalar> dense_vals;
};
/*--------------------------------------------------------------------------*/
struct PicardNonlinearCache
{
    std::vector<std::vector<PetscScalar>> MPx_by_elem;
    std::vector<std::vector<PetscScalar>> MPy_by_elem;
    std::vector<std::vector<PetscScalar>> MPz_by_elem;
    PetscInt Np = 0;
    PetscInt NElements = 0;

    void resize(PetscInt nElements, PetscInt np)
    {
        NElements = nElements;
        Np = np;
        MPx_by_elem.assign(nElements, std::vector<PetscScalar>(np * np, 0.0));
        MPy_by_elem.assign(nElements, std::vector<PetscScalar>(np * np, 0.0));
        MPz_by_elem.assign(nElements, std::vector<PetscScalar>(np * np, 0.0));
    }
};
struct AppCtx
{
    RunOptions opt;

    double Delta_t, Re, t;
    double EndTime;

    RotationParameters rot;

    unsigned int N_Nodes, Np;

    DiscreteOperators ops;

    IS isu = nullptr;
    IS isv = nullptr;
    IS isw = nullptr;
    IS isp = nullptr;
    IS isvel = nullptr;

    Vec Velocity = nullptr;
    Vec Velocity_U = nullptr;
    Vec Velocity_V = nullptr;
    Vec Velocity_W = nullptr;

    Vec Velocity_n = nullptr;   // full 3N vector at previous timestep
    Vec Velocity_k = nullptr;   // full 3N Picard iterate
    Vec Vsum = nullptr;

    Vec ForceHalf = nullptr;   // size 3*N_Nodes, known forcing at t^{n+1/2}
    // Time-independent projected spatial forcing:
    // P_DIV [0.5*(-y,x,0)^T]
    Vec WaveAttractorForceShape = nullptr;


    Mat Alinear = nullptr;
    Vec blinear = nullptr;
    Vec bnonlinear = nullptr;
    Mat A = nullptr;
    Vec b = nullptr;

    Vec X = nullptr;
    Vec OmegaStar = nullptr;     // size 3*N_Nodes
    Vec OmegaStar_x = nullptr;   // size N_Nodes
    Vec OmegaStar_y = nullptr;   // size N_Nodes
    Vec OmegaStar_z = nullptr;   // size N_Nodes

    PicardWorkspace picardWork;
    PicardNonlinearCache picardCache;

    DiagnosticsWorkspace diagWork;

    PetscBool wa_snapshot_written = PETSC_FALSE;
    WaveAttractorProbeRecorder wa_probe_recorder;
};
/*--------------------------------------------------------------------------*/
PetscErrorCode computeHelicity(
    Mat M,
    Mat CURL,
    Vec V,
    const RotationParameters &rot,
    DiagnosticsWorkspace &work,
    double &helicity);
/*--------------------------------------------------------------------------*/
PetscErrorCode DestroyDiscreteOperators(DiscreteOperators &ops);
PetscErrorCode DestroyAppCtx(AppCtx &user);
PetscErrorCode createDiagnosticsWorkspace(
    const PetscInt N_Nodes,
    const Vec VelocityTemplate,
    DiagnosticsWorkspace &work);
PetscErrorCode destroyDiagnosticsWorkspace(DiagnosticsWorkspace &work);
PetscErrorCode initializeRotationVector(
    Vec Frot,
    IS isu,
    IS isv,
    IS isw,
    const RotationParameters &rot);
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
    Vec bvel_for_div);                  // reusable full 3N vector
// /*--------------------------------------------------------------------------*/
PetscErrorCode buildLocalProductMatrixFromTensor(
    const TripleProductTensor1D &G1D,
    const PetscScalar *a_local,   // length Np
    PetscInt Np,
    PetscScalar J,
    PetscScalar *Pe);
/*--------------------------------------------------------------------------*/
PetscErrorCode computeDivergenceDiagnostics(
    const Vec Velocity,
    const Mat DIV,
    DiagnosticsWorkspace &work,
    double &div_max,
    double &div_l2);
PetscErrorCode computeHamiltonian(
    const Mat M,
    const Vec Velocity,
    DiagnosticsWorkspace &work,
    double &H);
PetscErrorCode ApplyVelocityLaplacianBlockDiag(
    Mat L,
    Vec V,
    Vec LV,
    IS isu,
    IS isv,
    IS isw);
PetscErrorCode computeMidpointDissipation(
    const Vec Velocity_n,      // V^n
    const Vec Velocity_np1,    // V^{n+1}
    const Mat M,               // full 3N x 3N mass matrix
    const Mat Laplacian,       // scalar N x N viscous operator
    IS isu,
    IS isv,
    IS isw,
    DiagnosticsWorkspace &work,
    double &Ehalf);
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
    double &Shalf);
PetscErrorCode computeForcingPowerInput(
    const Vec Velocity_n,
    const Vec Velocity_np1,
    const Vec ForceHalf,
    const Mat M,
    const PetscReal Fo,
    DiagnosticsWorkspace &work,
    double &W_unscaled,
    double &power_input);
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
    double &helicity_power_input);
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
    double &helicity_cancellation_ratio);
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
    double &helicity_cancellation_ratio);
PetscErrorCode computeCurlBoundaryPairing(
    const Vec U,
    const Vec V,
    const Mat CurlBoundary,
    Vec work,
    double &pairing);
PetscErrorCode recordDiagnosticsAtAcceptedStep(
    TimeDiagnostics &diag,
    AppCtx &user,
    const Vec ForceHalf,
    const std::vector<std::unique_ptr<Vertex>>  &List_Of_Vertices,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    const unsigned int N_Order,
    const double t,
    const double Delta_t);
PetscErrorCode saveDiagnosticsCSV(
    const std::string &filename,
    const TimeDiagnostics &diag);
/*--------------------------------------------------------------------------*/
PetscErrorCode denseMatMat(
    PetscInt Np,
    const PetscScalar *A,
    const PetscScalar *B,
    PetscScalar *C);
/*--------------------------------------------------------------------------*/
// PetscErrorCode denseMatMat_old(
//     PetscInt Np,
//     const PetscScalar *A,
//     const PetscScalar *B,
//     PetscScalar *C);
//
// PetscErrorCode denseMatMat_new(
//     PetscInt Np,
//     const PetscScalar *A,
//     const PetscScalar *B,
//     PetscScalar *C);
//
// PetscErrorCode benchmarkDenseMatMat(PetscInt Np, PetscInt nrepeat);
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
    PicardWorkspace &work);
/*--------------------------------------------------------------------------*/
PetscErrorCode denseMatVecAdd(
    PetscInt Np,
    const PetscScalar *A,   // Np x Np, row-major
    const PetscScalar *x,   // length Np
    PetscScalar alpha,
    PetscScalar *y) ;
/*--------------------------------------------------------------------------*/
// PetscErrorCode addDenseElementBlockValues(
//     Mat A,
//     PetscInt row_offset,
//     PetscInt col_offset,
//     PetscInt elem_pos,
//     PetscInt Np,
//     PetscScalar alpha,
//     const PetscScalar *Ke);
PetscErrorCode addDenseElementBlockValues(
    Mat A,
    PetscInt row_offset,
    PetscInt col_offset,
    PetscInt elem_pos,
    PetscInt Np,
    PetscScalar alpha,
    const PetscScalar *Ke,
    PicardWorkspace &work);
/*--------------------------------------------------------------------------*/
PetscErrorCode createTripleProductTensor1D(
    const unsigned int N_Order,
    TripleProductTensor1D &G1D);
/*--------------------------------------------------------------------------*/
PetscErrorCode CreateFullPatternMatrix(
    Mat *Apattern,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    const AppCtx &user);
PetscErrorCode insertNonlinearPressurePattern(
    Mat Apattern,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    const DiscreteOperators &ops,
    const PetscInt N_Nodes,
    const PetscInt Np);
PetscErrorCode insertPressureBlockPatternFromDivRow(
    Mat A,
    const Mat DIV,
    PetscInt div_row,
    PetscInt pressure_row,
    PetscInt target_col_offset,              // 0, N_Nodes, or 2*N_Nodes
    PetscInt Np,
    const std::vector<PetscInt> &node_to_elem_pos);
PetscErrorCode buildNodeToElemPos(
      const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
      PetscInt N_Nodes,
      PetscInt Np,
      std::vector<PetscInt> &node_to_elem_pos);
PetscErrorCode insertDenseElementBlockPattern(
    Mat A,
    PetscInt row_offset,
    PetscInt col_offset,
    PetscInt elem_pos,
    PetscInt Np);
PetscErrorCode insertNonlinearVelocityPattern(
    Mat Apattern,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    const PetscInt N_Nodes,
    const PetscInt Np);
/*--------------------------------------------------------------------------*/
PetscErrorCode assembleTotalRHS(Vec b, const Vec blinear, const Vec bnonlinear);
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
    Vec OmegaStar_z);
/*--------------------------------------------------------------------------*/
PetscErrorCode FormMatrixLinearPart(Mat Alinear, const Mat &GRAD, const Mat &DIVdx, const Mat &DIVdy, const Mat &DIVdz, const Mat &ConstantRotationalMat, const Mat &ConstantRotationalMatx, const Mat &ConstantRotationalMaty, const Mat &ConstantRotationalMatz, const Mat &Laplacian, const double &Delta_t, const unsigned int &N_Nodes, const double &viscous, const double &Re, const PetscBool &include_viscous_divergence_correction);
/*--------------------------------------------------------------------------*/
PetscErrorCode checkMatrixCommutator(
    Mat A,
    Mat B,
    const char *label);
PetscErrorCode build_DiscreteOperators_Inertial(
    const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
    const std::vector<std::unique_ptr<Boundary>> &List_Of_Boundaries,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    unsigned int N_Nodes,
    unsigned int N_Order,
    const RotationParameters &rotation,
    DiscreteOperators &ops);
PetscErrorCode create_Matrices_Cuboids_Inertial(
    const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
    const std::vector<std::unique_ptr<Boundary>> &List_Of_Boundaries,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    unsigned int N_Nodes,
    unsigned int N_Order,
    InertialRawMatrices &raw,
    const RotationParameters &rotation);
/*--------------------------------------------------------------------------*/
PetscErrorCode initializePicardWorkspace(
    PicardWorkspace &work,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    PetscInt N_Nodes,
    PetscInt Np);

PetscErrorCode destroyPicardWorkspace(PicardWorkspace &work);

PetscErrorCode buildPicardNonlinearCache(
    PicardNonlinearCache &cache,
    PicardWorkspace &work,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    const DiscreteOperators &ops,
    PetscInt Np,
    const Vec OmegaStar_x,
    const Vec OmegaStar_y,
    const Vec OmegaStar_z);
PetscErrorCode computeRelativeNonlinearFieldFromCache(
    const PicardNonlinearCache &cache,
    PicardWorkspace &work,
    const Vec Vmid,
    Vec Nrelative,
    const PetscInt N_Nodes,
    const PetscInt Np);
PetscErrorCode computeAbsoluteHelicityBoundaryDiagnostics(
    AppCtx &user,
    const Vec ForceHalf,
    const std::vector<std::unique_ptr<Element>>
        &List_Of_Elements,
    AbsoluteHelicityBoundaryDiagnostics &out);
PetscErrorCode addNonlinearVelocityAndPressureFromCache(
    Mat A,
    const PicardNonlinearCache &cache,
    PicardWorkspace &work,
    const DiscreteOperators &ops,
    PetscInt N_Nodes,
    PetscInt Np,
    PetscReal Delta_t);
PetscErrorCode formMatrixPicardTotalFromCache(
    Mat A,
    const Mat Alinear,
    const PicardNonlinearCache &cache,
    PicardWorkspace &work,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    const DiscreteOperators &ops,
    PetscInt N_Nodes,
    PetscInt Np,
    PetscReal Delta_t);
PetscErrorCode formRHSPicardDirectFromCache(
    Vec bnonlinear,
    const PicardNonlinearCache &cache,
    PicardWorkspace &work,
    const Vec Velocity_n,
    const DiscreteOperators &ops,
    PetscInt N_Nodes,
    PetscInt Np,
    PetscReal Delta_t);
/*--------------------------------------------------------------------------*/
//extern void FormMatrixNonlinearPart(Mat &Anonlinear, const Vec &Velocity, const Vec &Velocity_U, const Vec &Velocity_V, const Vec &Velocity_W, const Vec &Velocity_star, const Mat &GRAD, const Mat &DIVdx, const Mat &DIVdy, const Mat &DIVdz, const Mat &CURL, const Mat &CURLdx, const Mat &CURLdy, const Mat &CURLdz, const Mat &ConstantRotationalMat, const double &Delta_t, const unsigned int &N_Nodes, const std::vector<std::unique_ptr<Element>> &List_Of_Elements);
/*--------------------------------------------------------------------------*/
void Calculate_Jacobian_Cuboid(const std::unique_ptr<Element> &Element, const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices, const double &r_p, const double &s_p, const double &t_p, double &det_J, double &drdx, double &drdy, double &drdz, double &dsdx, double &dsdy, double &dsdz, double &dtdx, double &dtdy, double &dtdz, double &x, double &y, double &z);
/*--------------------------------------------------------------------------*/
void compute_Divergence_Velocity(const Vec &Initial_Condition, const double &N_Nodes, const Mat &DIV);
PetscErrorCode printMatrixNNZInfo(const Mat A, const std::string &name);
/*--------------------------------------------------------------------------*/
// EB //
/*--------------------------------------------------------------------------*/
void create_Matrices_Quadrilaterals_Inertial(const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices, const std::vector<std::unique_ptr<Boundary>> &List_Of_Boundaries, const std::vector<std::unique_ptr<Element>> &List_Of_Elements, const unsigned int &N_Nodes, const unsigned int &N_Order, Mat &E, Mat &ET, Mat &Edx, Mat &Edy, Mat &ETdx, Mat &ETdy, Mat &invM, Mat &invM_small, Mat &M, Mat &M_small, Mat &ConstantRotational, Mat &ConstantRotationalx, Mat &ConstantRotationaly, const double &f3, const double &RossbyNumber, const double &include_rotation);
PetscErrorCode create_Matrices_Cuboids_Inertial(
    const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
    const std::vector<std::unique_ptr<Boundary>> &List_Of_Boundaries,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    unsigned int N_Nodes,
    unsigned int N_Order,
    InertialRawMatrices &raw,
    const RotationParameters &rotation);
/*--------------------------------------------------------------------------*/


            // ============================================================
            // Mass-norm diagnostics and fixed-lambda Beltrami filter
            // ============================================================


            PetscErrorCode printDivergenceNorms(
                const char *label,
                Vec Velocity,
                const DiscreteOperators &ops,
                PetscInt N_Nodes);
/*--------------------------------------------------------------------------*/
PetscErrorCode setupFieldSplitKSP(
    KSP ksp,
    Mat Amat,
    Mat Pmat,
    IS isvel,
    IS isp,
    PetscReal ksp_rtol,
    PetscReal ksp_atol);
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
// Wave attractor
/*--------------------------------------------------------------------------*/
PetscReal waveAttractorOmega();

PetscReal waveAttractorPeriod();

PetscReal waveAttractorForcingOffTime(const RunOptions &opt);

PetscBool isWaveAttractorForcingOffStep(
    const PetscReal t,
    const PetscReal Delta_t,
    const RunOptions &opt);
PetscErrorCode printWaveAttractorForcingConfiguration(const RunOptions &opt);
PetscErrorCode initializeWaveAttractorForceShape(
    AppCtx &user,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements);
PetscErrorCode updateWaveAttractorForceHalf(
    AppCtx &user,
    PetscReal tn,
    PetscBool *include_force);
PetscErrorCode projectVectorToDiscreteDivergenceFree(
    PetscInt N_Nodes,
    Vec Field,
    Mat DIV,
    Mat GRAD,
    Mat Laplacian);
PetscErrorCode writeWaveAttractorForcingOffSnapshot(
    AppCtx &user,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements,
    const Vec ForceHalf,
    const PetscReal t);

PetscErrorCode initializeWaveAttractorProbeRecorder(
    AppCtx &user,
    const std::vector<std::unique_ptr<Element>> &List_Of_Elements);

PetscErrorCode appendWaveAttractorProbeSamples(
    AppCtx &user,
    const PetscInt step,
    const PetscReal t);

PetscErrorCode destroyWaveAttractorProbeRecorder(
    AppCtx &user);

#endif
