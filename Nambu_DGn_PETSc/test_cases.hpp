// test_cases.hpp
#pragma once

#include <petscksp.h>
#include <string>
#include <cmath>
#include "options.hpp"
#include "HIGW.hpp"

enum class ProblemID : PetscInt
{
    BoostedRotatingEuler               = 1,
    DecayingNavierStokes               = 2,
    TaylorGreenFreeSlipBox             = 3,
    Rotating2D3CChannel                = 4,
    SteadyRotating2D3CTaylorGreenWalls = 5,
    InertialWaveAttractor              = 6
};

inline PetscErrorCode getProblemID(PetscInt raw_id, ProblemID& id)
{
    PetscFunctionBeginUser;

    switch (raw_id)
    {
        case 1:
            id = ProblemID::BoostedRotatingEuler;
            break;

        case 2:
            id = ProblemID::DecayingNavierStokes;
            break;

        case 5:
            id = ProblemID::SteadyRotating2D3CTaylorGreenWalls;
            break;

        case 6:
            id = ProblemID::InertialWaveAttractor;
            break;

        default:
            SETERRQ(PETSC_COMM_SELF,
                    PETSC_ERR_ARG_OUTOFRANGE,
                    "Unknown problem_type = %d", (int)raw_id);
    }

    PetscFunctionReturn(0);
}

struct TestCase
{
    ProblemID id;

    std::string name;
    std::string short_name;

    PetscBool has_exact_solution = PETSC_TRUE;

    PetscBool periodic_x = PETSC_TRUE;
    PetscBool periodic_y = PETSC_TRUE;
    PetscBool periodic_z = PETSC_TRUE;

    PetscBool viscous = PETSC_FALSE;
    PetscBool include_rotation = PETSC_FALSE;
    PetscBool include_buoyancy = PETSC_FALSE;

    PetscScalar Re = 1.0;
    PetscScalar Ro = 1.0;
    PetscScalar Fo = 1.0;
    PetscScalar U  = 0.0;
    PetscScalar V  = 0.0;
    PetscScalar W  = 0.0;

    PetscScalar f1 = 0.0;
    PetscScalar f2 = 0.0;
    PetscScalar f3 = 1.0;

    PetscScalar EndTime = 1.0;
    PetscScalar Delta_t = 1.0;

    PetscBool project_initial_velocity = PETSC_TRUE;

    std::string output_prefix;
    std::string diagnostics_prefix;
};

inline PetscErrorCode makeTestCase(const RunOptions& opt, TestCase& tc)
{
    PetscFunctionBeginUser;

    PetscCall(getProblemID(opt.problem_type, tc.id));

    switch (tc.id)
    {
        case ProblemID::BoostedRotatingEuler:
        {
            tc.name       = "Boosted rotating propagating Euler flow";
            tc.short_name = "BoostedRotatingEuler";

            tc.output_prefix      = "output_BoostedRotatingEuler";
            tc.diagnostics_prefix = "diagnostics_BoostedRotatingEuler";

            tc.has_exact_solution = PETSC_TRUE;

            tc.viscous = PETSC_FALSE;
            tc.include_rotation = PETSC_TRUE;
            tc.include_buoyancy = PETSC_FALSE;

            tc.Re = 1.0;
            tc.Ro = 1.0;
            tc.Fo = opt.Fo;

            tc.f1 = 0.0;
            tc.f2 = 0.0;
            tc.f3 = 1.0;
            tc.W  = 2.0;

            tc.periodic_x = PETSC_TRUE;
            tc.periodic_y = PETSC_TRUE;
            tc.periodic_z = PETSC_TRUE;

            const PetscScalar omega_eff = std::sqrt(3.0) / (3.0*tc.Ro) - 2.0*PETSC_PI*tc.W;
            const PetscScalar period = 2.0 * PETSC_PI / PetscAbsScalar(omega_eff);
            const PetscInt nsteps = opt.periods * opt.steps_per_period;

            tc.EndTime = opt.periods * period;
            tc.Delta_t = tc.EndTime / static_cast<PetscScalar>(nsteps);

            tc.project_initial_velocity = PETSC_TRUE;

            break;
        }

        case ProblemID::DecayingNavierStokes:
        {
            tc.name       = "Decaying periodic Navier-Stokes flow";
            tc.short_name = "DecayingNavierStokes";

            tc.output_prefix      = "output_DecayingNavierStokes";
            tc.diagnostics_prefix = "diagnostics_DecayingNavierStokes";

            tc.has_exact_solution = PETSC_TRUE;

            tc.viscous = PETSC_TRUE;
            tc.include_rotation = PETSC_FALSE;
            tc.include_buoyancy = PETSC_FALSE;

            tc.Re = 1.0e3;
            tc.Ro = 1.0e10;
            tc.Fo = opt.Fo;

            tc.f1 = 0.0;
            tc.f2 = 0.0;
            tc.f3 = 0.0;

            // tc.U = 0.0;
            // tc.V = 0.0;
            // tc.W = 0.0;

            tc.U = 0.5;
            tc.V = PetscSqrtReal(2.0)/2.0;
            tc.W = PetscSqrtReal(3.0)/2.0;

            tc.periodic_x = PETSC_TRUE;
            tc.periodic_y = PETSC_TRUE;
            tc.periodic_z = PETSC_TRUE;

            const PetscScalar period = 0.1 * tc.Re / (12.0 * PETSC_PI * PETSC_PI);
            tc.EndTime = opt.periods * period;
            const PetscInt nsteps = opt.periods * opt.steps_per_period;
            tc.Delta_t = tc.EndTime / static_cast<PetscScalar>(nsteps);

            tc.project_initial_velocity = PETSC_TRUE;

            break;
        }

        case ProblemID::SteadyRotating2D3CTaylorGreenWalls:
        {
            tc.name       = "Steady rotating 2D3C Taylor-Green flow with impermeable walls";
            tc.short_name = "SteadyRotating2D3CTaylorGreenWalls";

            tc.output_prefix      = "output_SteadyRotating2D3CTaylorGreenWalls";
            tc.diagnostics_prefix = "diagnostics_SteadyRotating2D3CTaylorGreenWalls";

            tc.has_exact_solution = PETSC_TRUE;

            tc.viscous = PETSC_FALSE;
            tc.include_rotation = PETSC_TRUE;
            tc.include_buoyancy = PETSC_FALSE;

            tc.Re = 1.0;       // unused because viscous = false
            tc.Ro = 1.0;
            tc.Fo = opt.Fo;

            tc.f1 = 0.0;
            tc.f2 = 0.0;
            tc.f3 = 1.0;

            tc.U = 0.0;
            tc.V = 0.0;
            tc.W = 0.0;

            // Solid walls at x=0,1 and y=0,1; periodic in z.
            tc.periodic_x = PETSC_FALSE;//PETSC_TRUE;//
            tc.periodic_y = PETSC_FALSE;//PETSC_TRUE;//PETSC_FALSE;
            tc.periodic_z = PETSC_TRUE;

            // The solution is steady. Use "periods" only as a runtime multiplier.
            const PetscScalar pseudo_period = 1.0/(2.0*PETSC_PI);//0.01;//1.0;
            const PetscInt nsteps = opt.periods * opt.steps_per_period;

            tc.EndTime = opt.periods * pseudo_period;
            tc.Delta_t = tc.EndTime / static_cast<PetscScalar>(nsteps);

            tc.project_initial_velocity = PETSC_TRUE;

            break;
        }

        case ProblemID::InertialWaveAttractor:
        {
            tc.name       = "Nonlinear inertial wave attractor showcase";
            tc.short_name = "InertialWaveAttractor";

            tc.output_prefix      = "output_InertialWaveAttractor";
            tc.diagnostics_prefix = "diagnostics_InertialWaveAttractor";

            tc.has_exact_solution = PETSC_FALSE;

            tc.viscous = opt.viscous;
            tc.include_rotation = PETSC_TRUE;
            tc.include_buoyancy = PETSC_FALSE;

            tc.Re = 1.0e5;
            tc.Ro = 1.0;
            tc.Fo = opt.Fo;

            tc.f1 = -1.0/std::sqrt(10.0);
            tc.f2 = 3.0/std::sqrt(10.0);
            tc.f3 = opt.f3;

            tc.periodic_x = PETSC_FALSE;
            tc.periodic_y = PETSC_FALSE;
            tc.periodic_z = PETSC_FALSE;

    /*
       Temporary Stage-1 choice:
       interpret 'periods' as forcing/rotation periods and
       'steps_per_period' as temporal resolution.
    */
    const PetscScalar period = 8.0;
    const PetscInt nsteps = opt.periods * opt.steps_per_period;

    tc.EndTime = static_cast<PetscScalar>(opt.periods) * period;
    tc.Delta_t = tc.EndTime / static_cast<PetscScalar>(nsteps);

            tc.project_initial_velocity = PETSC_TRUE;

            break;
        }
    }

    PetscFunctionReturn(0);
}

inline PetscErrorCode applyTestCaseToUser(const TestCase& tc, AppCtx& user)
{
    PetscFunctionBeginUser;

    user.opt.viscous = tc.viscous;
    user.opt.Re = tc.Re;
    user.opt.Fo = tc.Fo;
    user.opt.include_rotation = tc.include_rotation;

    // AppCtx-level aliases used elsewhere
    user.Re       = tc.Re;
    user.EndTime  = tc.EndTime;
    user.Delta_t  = tc.Delta_t;
    user.t        = 0.0;

    user.rot.include_rotation = tc.include_rotation;
    user.rot.RossbyNumber = tc.Ro;
    user.rot.f1 = tc.f1;
    user.rot.f2 = tc.f2;
    user.rot.f3 = tc.f3;

    PetscFunctionReturn(0);
}

inline PetscErrorCode printTestCaseBanner(const TestCase& tc, const AppCtx& user)
{
    PetscFunctionBeginUser;

    PetscPrintf(PETSC_COMM_WORLD, "\n");
    PetscPrintf(PETSC_COMM_WORLD, "\033[1;34m============================================================\033[0m\n");
    PetscPrintf(PETSC_COMM_WORLD, "\033[1;34m Problem: %s\033[0m\n", tc.name.c_str());
    PetscPrintf(PETSC_COMM_WORLD, "\033[1;34m============================================================\033[0m\n");

    PetscPrintf(PETSC_COMM_WORLD, " short_name       = %s\n", tc.short_name.c_str());
    PetscPrintf(PETSC_COMM_WORLD, " has_exact        = %d\n", (int)tc.has_exact_solution);

    PetscPrintf(PETSC_COMM_WORLD, " periodic         = (%d, %d, %d)\n",
                (int)tc.periodic_x,
                (int)tc.periodic_y,
                (int)tc.periodic_z);

    PetscPrintf(PETSC_COMM_WORLD, " viscous          = %d\n", (int)user.opt.viscous);
    PetscPrintf(PETSC_COMM_WORLD, " Re               = %.16e\n", (double)user.opt.Re);
    PetscPrintf(PETSC_COMM_WORLD, " nonlinear        = %d\n", (int)user.opt.nonlinear);

    PetscPrintf(PETSC_COMM_WORLD, " rotation         = %d\n", (int)user.rot.include_rotation);
    PetscPrintf(PETSC_COMM_WORLD, " Ro               = %.16e\n", (double)user.rot.RossbyNumber);
    PetscPrintf(PETSC_COMM_WORLD, " f                = (%.6e, %.6e, %.6e)\n",
                (double)user.rot.f1,
                (double)user.rot.f2,
                (double)user.rot.f3);
    PetscPrintf(PETSC_COMM_WORLD, " boost velocity   = (%.16e, %.16e, %.16e)\n",
                (double)tc.U,
                (double)tc.V,
                (double)tc.W);

    PetscPrintf(PETSC_COMM_WORLD, " Fo               = %.16e\n", (double)user.opt.Fo);

    PetscPrintf(PETSC_COMM_WORLD, " EndTime          = %.16e\n", (double)user.EndTime);
    PetscPrintf(PETSC_COMM_WORLD, " Delta_t          = %.16e\n", (double)user.Delta_t);

    PetscPrintf(PETSC_COMM_WORLD, " rhs div correction     = %d\n",
                (int)user.opt.include_divergence_correction);

    PetscPrintf(PETSC_COMM_WORLD, " viscous div correction = %d\n",
                (int)user.opt.include_viscous_divergence_correction);
    PetscPrintf(PETSC_COMM_WORLD, "\033[1;34m============================================================\033[0m\n\n");

    PetscFunctionReturn(0);
}
