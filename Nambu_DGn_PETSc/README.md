# Nambu DGFEM

C++/PETSc implementation of the structure-preserving Nambu discontinuous Galerkin method for three-dimensional rotating incompressible Navier–Stokes flow.

The code accompanies the paper:

**Structure-preserving Nambu discontinuous Galerkin method for rotating incompressible Navier–Stokes flow in periodic and wall-bounded domains**
Alexander M. van Oers and Onno Bokhove

## Requirements

* C++ compiler
* PETSc
* SLEPc

## Build
make

This creates the executable `Nambu02`.

## Example
A small test run for Problem 1 is:

./Nambu02 -Problem 1 -Order 2 -nx 4 -ny 4 -nz 4 -t 100 \
  -include_viscous_divergence_correction 0 \
  -nonlinear 1 \
  -include_divergence_correction 1 \
  -diagn 1

The main options are:

* `-Problem` — test case
* `-Order` — polynomial order
* `-nx`, `-ny`, `-nz` — number of elements
* `-t` — time steps per period
* `-P` — number of periods
* `-diagn` — enable diagnostics

The repository contains the three verification cases and the inertial-wave-attractor demonstrator presented in the paper.