#ifndef INITIAL_COND_HPP
#define INITIAL_COND_HPP

#include <petscksp.h>

#include <memory>
#include <vector>

#include "Elements.hpp"

struct TestCase;

enum class ExactVelocitySampling
{
    Pointwise,      // interpolate exact velocity at DG/LGL nodes
    L2Projection    // elementwise L2 projection into the DG space
};

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
    Mat LaplacianClean);

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
    Mat LaplacianClean);

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
    Mat LaplacianClean);

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
    PetscScalar &w);

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
    PetscScalar &w);

PetscErrorCode exactVelocity_SteadyRotating2D3CTaylorGreenWalls_Pointwise(
    PetscScalar x,
    PetscScalar y,
    PetscScalar z,
    PetscScalar t,
    PetscScalar &u,
    PetscScalar &v,
    PetscScalar &w);

// -----------------------------------------------------------------------------
// Pointwise interpolation wrappers
//
// These only fill the DG velocity vectors. They do not apply the divergence
// projection. For initial/reference states, prefer the setExactVelocity_* wrappers.
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
    PetscScalar W);

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
    PetscScalar Wboost);

// -----------------------------------------------------------------------------
// L2 projection wrappers
//
// These only fill the DG velocity vectors. They do not apply the divergence
// projection. For initial/reference states, prefer the setExactVelocity_* wrappers.
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
    PetscScalar W);

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
    PetscScalar Wboost);

// -----------------------------------------------------------------------------
// Discrete divergence projection
//
// Uses a temporary pinned copy of LaplacianClean. The input LaplacianClean is
// left untouched, so diagnostics based on LAP = DIV*GRAD remain meaningful.
// -----------------------------------------------------------------------------

PetscErrorCode projectVelocityToDiscreteDivergenceFree(
    unsigned int N_Nodes,
    Vec Velocity,
    Vec Velocity_U,
    Vec Velocity_V,
    Vec Velocity_W,
    Mat DIV,
    Mat GRAD,
    Mat LaplacianClean);

// -----------------------------------------------------------------------------
// Diagnostics / utilities
// -----------------------------------------------------------------------------

PetscErrorCode compute_Divergence_Velocity(
    Vec Velocity,
    unsigned int N_Nodes,
    Mat DIV);

PetscErrorCode fillFullStateFromVelocityComponents(
    unsigned int N_Nodes,
    Vec FullState,
    Vec Velocity_U,
    Vec Velocity_V,
    Vec Velocity_W);

PetscErrorCode compute_MassInnerProduct(
    Mat M,
    Vec x,
    Vec y,
    PetscScalar *value);

PetscErrorCode compute_MassNorm(
    Mat M,
    Vec x,
    PetscReal *norm);

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
    Mat Laplacian);

#endif
