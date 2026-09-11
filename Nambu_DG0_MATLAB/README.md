# MATLAB DG(0) reference implementation

This folder contains a compact MATLAB reference implementation of the
centred DG(0) / finite-volume specialization of the Nambu DGFEM scheme.

## Run an example

Open either script below in MATLAB and press **Run**:

- `examples/example_rotating_inviscid_DG0.m`
- `examples/example_decaying_viscous_DG0.m`
- `examples/example_steady_rotating_taylor_green_walls_DG0.m`

The first example is the periodic propagating rotating inviscid test from
Section 5.2.1 of the paper. The second is the periodic decaying viscous
test from Section 5.2.2. The third is the steady rotating Taylor-Green
test with impermeable walls in x/y and periodicity in z from Section 5.2.3.

This MATLAB code is intended as a readable reference/verification
implementation. The numerical experiments in the paper were implemented
in C++ with PETSc and used a field-split ILU(0) preconditioner. This compact
MATLAB version omits that PETSc-specific preconditioning layer.
