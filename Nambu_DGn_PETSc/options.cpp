#include "options.hpp"
#include <petsc.h>

PetscErrorCode ReadOptions(RunOptions &opt)
{
    PetscErrorCode ierr;

    PetscFunctionBegin;

    PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Simulation options", "");

    ierr = PetscOptionsInt("-nx", "Number of elements in x-direction",
                           "", opt.Nel_x, &opt.Nel_x, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsInt("-ny", "Number of elements in y-direction",
                           "", opt.Nel_y, &opt.Nel_y, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsInt("-nz", "Number of elements in z-direction",
                           "", opt.Nel_z, &opt.Nel_z, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsInt("-Order", "Polynomial order",
                           "", opt.Order, &opt.Order, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsInt("-t", "Time steps per period",
                           "", opt.steps_per_period, &opt.steps_per_period, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsInt("-P", "Number of periods",
                           "", opt.periods, &opt.periods, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsBool("-include_divergence_correction",
                            "Enable divergence correction", "",
                            opt.include_divergence_correction,
                            &opt.include_divergence_correction, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsBool("-include_viscous_divergence_correction",
                            "Enable viscous divergence correction", "",
                            opt.include_viscous_divergence_correction,
                            &opt.include_viscous_divergence_correction, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsInt("-Problem", "Problem type: 1 Euler, 2 decaying NS, 3 Steady Rotating Taylor Green Walls, 4 Wave Attractor",
                           "", opt.problem_type, &opt.problem_type, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsBool("-viscous", "Enable viscous term",
                            "", opt.viscous, &opt.viscous, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsBool("-nonlinear", "Enable nonlinear term v x curl(v)",
                            "", opt.nonlinear, &opt.nonlinear, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsBool("-include_rotation", "Enable rotation",
                            "", opt.include_rotation, &opt.include_rotation, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsScalar("-theta", "Theta parameter",
                              "", opt.theta, &opt.theta, NULL); CHKERRQ(ierr);

    // ierr = PetscOptionsScalar("-EndTime", "EndTime",
    //                           "", opt.EndTime, &opt.EndTime, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsScalar("-Fo", "Froude number",
                              "", opt.Fo, &opt.Fo, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsScalar("-Re", "Reynolds number",
                              "", opt.Re, &opt.Re, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsScalar("-Ro", "Rossby number",
                              "", opt.RossbyNumber, &opt.RossbyNumber, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsScalar("-f1", "Rotation vector component f1",
                              "", opt.f1, &opt.f1, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsScalar("-f2", "Rotation vector component f2",
                              "", opt.f2, &opt.f2, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsScalar("-f3", "Rotation vector component f3",
                              "", opt.f3, &opt.f3, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsBool("-diagn", "Error, Hamiltonian, divergence diagnostics",
                            "", opt.enable_diagnostics, &opt.enable_diagnostics, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsBool("-picard_extrapolate_initial_guess",
                            "Use V_guess = 2 V^n - V^{n-1} as first Picard frozen velocity",
                            "",
                            opt.picard_extrapolate_initial_guess,
                            &opt.picard_extrapolate_initial_guess, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-use_variable_ksp_tol_picard",
                            "Use looser KSP tolerances in early Picard iterations",
                            "",
                            opt.use_variable_ksp_tol_picard,
                            &opt.use_variable_ksp_tol_picard,
                            NULL); CHKERRQ(ierr);

    ierr = PetscOptionsReal("-ksp_rtol_picard_loose",
                            "Loose KSP relative tolerance for early Picard iterations",
                            "",
                            opt.ksp_rtol_picard_loose,
                            &opt.ksp_rtol_picard_loose,
                            NULL); CHKERRQ(ierr);

    ierr = PetscOptionsReal("-ksp_rtol_picard_tight",
                            "Tight KSP relative tolerance for late Picard iterations",
                            "",
                            opt.ksp_rtol_picard_tight,
                            &opt.ksp_rtol_picard_tight,
                            NULL); CHKERRQ(ierr);

    ierr = PetscOptionsInt("-ksp_rtol_picard_tight_after",
                           "Use tight KSP tolerance from this Picard iteration onward",
                           "",
                           opt.ksp_rtol_picard_tight_after,
                           &opt.ksp_rtol_picard_tight_after,
                           NULL); CHKERRQ(ierr);

   ierr = PetscOptionsBool("-wa_turn_off_forcing",
                           "Turn off wave-attractor forcing after prescribed number of forcing periods",
                           "",
                           opt.wave_attractor_turn_off_forcing,
                           &opt.wave_attractor_turn_off_forcing,
                           NULL); CHKERRQ(ierr);

   ierr = PetscOptionsScalar("-wa_forcing_off_periods",
                             "Number of forcing periods after which wave-attractor forcing is turned off",
                             "",
                             opt.wave_attractor_forcing_off_periods,
                             &opt.wave_attractor_forcing_off_periods,
                             NULL); CHKERRQ(ierr);

   ierr = PetscOptionsBool("-wa_write_snapshot_at_forcing_off",
                           "Write wave-attractor field snapshot at forcing shutoff time",
                           "",
                           opt.wa_write_snapshot_at_forcing_off,
                           &opt.wa_write_snapshot_at_forcing_off,
                           NULL); CHKERRQ(ierr);
   ierr = PetscOptionsBool(
       "-wa_write_probe_timeseries",
       "Write wave-attractor probe time series at every accepted timestep",
       "",
       opt.wa_write_probe_timeseries,
       &opt.wa_write_probe_timeseries,
       NULL); CHKERRQ(ierr);

    PetscOptionsEnd();

    PetscFunctionReturn(0);
}
