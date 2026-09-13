#pragma once

#include <petscsys.h>

// -----------------------------------------------------------------------------
// Run-time configuration options (parsed from command line)
// -----------------------------------------------------------------------------
struct RunOptions
{
    // Mesh
    PetscInt Nel_x = 4;
    PetscInt Nel_y = 4;
    PetscInt Nel_z = 4;

    // Discretization
    PetscInt Order = 0;

    // Time stepping
    PetscInt steps_per_period = 100;
    PetscInt periods = 1;

    // Problem selection
    PetscInt problem_type = 1;

    // Physics
    PetscBool viscous = PETSC_TRUE;
    PetscBool include_divergence_correction = PETSC_TRUE;
    PetscBool include_viscous_divergence_correction = PETSC_FALSE;
    PetscBool nonlinear = PETSC_TRUE;

    PetscScalar theta = 0.5;
    PetscScalar Fo    = 0.0;
    PetscScalar Re    = 1e5;

    PetscBool include_rotation = PETSC_FALSE;
    PetscScalar RossbyNumber = 1.0;
    PetscScalar f1 = 0.0;
    PetscScalar f2 = 0.0;
    PetscScalar f3 = 0.0;

    PetscBool enable_diagnostics = PETSC_FALSE;

    PetscBool picard_extrapolate_initial_guess = PETSC_FALSE;
    PetscBool use_variable_ksp_tol_picard = PETSC_FALSE;
    PetscReal ksp_rtol_picard_loose = 1e-5;
    PetscReal ksp_rtol_picard_tight = 1e-8;
    PetscInt  ksp_rtol_picard_tight_after = 6;

    // wave attractor
    PetscBool wave_attractor_turn_off_forcing = PETSC_TRUE;
    PetscReal wave_attractor_forcing_off_periods = 50.0;
    PetscBool wa_write_snapshot_at_forcing_off = PETSC_TRUE;
    // Record selected wave-attractor probes at every accepted timestep.
    PetscBool wa_write_probe_timeseries = PETSC_FALSE;
};

// -----------------------------------------------------------------------------
// Read options from PETSc command line
// -----------------------------------------------------------------------------
PetscErrorCode ReadOptions(RunOptions &opt);
