#ifndef LGL_GRID
#define LGL_GRID

#include <petscksp.h>
#include <slepceps.h>
#include <slepcsys.h>

#include <iostream>
#include <limits>
#include <array>
#include <vector>
#include <chrono>
#include <memory>

#include "Elements.hpp"

/*--------------------------------------------------------------------------*/
#define NODETOL 1e-12
/*--------------------------------------------------------------------------*/

// 1D Jacobi / Gauss / Gauss-Lobatto
extern Vec JacobiGL(const double &alpha, const double &beta, const unsigned int &N);
/*--------------------------------------------------------------------------*/
extern Vec JacobiGL_withWeights(const double &alpha, const double &beta, const unsigned int &N, Vec &Weights);
/*--------------------------------------------------------------------------*/
extern Vec JacobiGQ(const double &alpha, const double &beta, const unsigned int &N);
/*--------------------------------------------------------------------------*/
extern Vec JacobiGQ_withWeights(const double &alpha, const double &beta, const unsigned int &N, Vec &Weights);
/*--------------------------------------------------------------------------*/
extern Vec JacobiP(const Vec &x, const double &alpha, const double &beta, const unsigned int &N);
// Purpose: Evaluate Jacobi Polynomial of type (alpha,beta) > -1
//          at points x for order N.
// Note   : They are normalized to be orthonormal.
/*--------------------------------------------------------------------------*/
extern std::vector<double> ComputeBarycentricWeights(const Vec &r);
/*--------------------------------------------------------------------------*/
// Compatibility wrappers: same call style as old code
extern double LagrangePolynomial(const Vec &r, const double &x, const unsigned int &i);
/*--------------------------------------------------------------------------*/
extern double LagrangePolynomialDeriv(const Vec &r, const double &x, const unsigned int &i);
/*--------------------------------------------------------------------------*/

// Fast overloads with precomputed barycentric weights
extern double LagrangePolynomial(const Vec &r, const std::vector<double> &baryWeights, const double &x, const unsigned int &i);
/*--------------------------------------------------------------------------*/
extern double LagrangePolynomialDeriv(const Vec &r, const std::vector<double> &baryWeights, const double &x, const unsigned int &i);
/*--------------------------------------------------------------------------*/

// Tensor-product quadrilateral / cuboid nodes
/*--------------------------------------------------------------------------*/
extern void set_Node_Coordinates_Uniform(std::vector<std::unique_ptr<Element>> &List_Of_Elements,
                                         const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
                                         const unsigned int &N);
/*--------------------------------------------------------------------------*/
extern void set_Node_Coordinates_Uniform_3D(std::vector<std::unique_ptr<Element>> &List_Of_Elements,
                                            const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
                                            const unsigned int &Nx,
                                            const unsigned int &Ny,
                                            const unsigned int &Nz);
/*--------------------------------------------------------------------------*/
extern void set_Node_Coordinates_Uniform_2D(std::vector<std::unique_ptr<Element>> &List_Of_Elements,
                                            const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices,
                                            const unsigned int &Nx,
                                            const unsigned int &Ny);
/*--------------------------------------------------------------------------*/
extern void NodesCuboid(const unsigned int &Nx,
                        const unsigned int &Ny,
                        const unsigned int &Nz,
                        std::vector<double> &x,
                        std::vector<double> &y,
                        std::vector<double> &z,
                        const unsigned int &Np);
/*--------------------------------------------------------------------------*/
extern void NodesQuadrilateral(const unsigned int &Nx,
                               const unsigned int &Ny,
                               std::vector<double> &x,
                               std::vector<double> &y,
                               const unsigned int &Np);
/*--------------------------------------------------------------------------*/

// Generic helpers
extern Mat Inverse_Matrix(const Mat &V);
/*--------------------------------------------------------------------------*/

#endif
