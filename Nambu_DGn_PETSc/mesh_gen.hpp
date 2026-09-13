#ifndef MESH_GEN
#define MESH_GEN

#include <petscksp.h>
#include <iostream>
#include <limits>
#include <algorithm>
#include <set>
#include <memory>

#include "Elements.hpp"

#define NODETOL 1e-12
/*--------------------------------------------------------------------------*/
// Structured mesh generators
/*--------------------------------------------------------------------------*/
void create_Mesh_Cuboid(std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices, std::vector<std::unique_ptr<Element>> &List_Of_Elements, unsigned int &N_Elements_total, unsigned int &N_Vertices_total, const unsigned int &Nel_x, const unsigned int &Nel_y, const unsigned int &Nel_z);
/*--------------------------------------------------------------------------*/
// Structured connectivity
/*--------------------------------------------------------------------------*/
void Connect_3DPeriodic(const unsigned int &Nel_x, const unsigned int &Nel_y, const unsigned int &Nel_z, const unsigned int &N_Elements_total, std::vector<std::unique_ptr<Boundary>> &List_Of_Boundaries);
void Connect_3D(const unsigned int &Nel_x, const unsigned int &Nel_y, const unsigned int &Nel_z, const unsigned int &N_Elements_total, std::vector<std::unique_ptr<Boundary>> &List_Of_Boundaries, const bool &Sx, const bool &Sy, const bool &Sz);
/*--------------------------------------------------------------------------*/
void set_Order_Polynomials_Uniform(std::vector<std::unique_ptr<Element>> &List_Of_Elements, const unsigned int &Nx, const unsigned int &Ny, const unsigned int &Nz);
void set_Order_Polynomials_Uniform(std::vector<std::unique_ptr<Element>> &List_Of_Elements, const unsigned int &N);
/*--------------------------------------------------------------------------*/
extern unsigned int get_Number_Of_Nodes(std::vector<std::unique_ptr<Element>> &List_Of_Elements);
/*--------------------------------------------------------------------------*/
void set_theta_Uniform(std::vector<std::unique_ptr<Boundary>> &List_Of_Boundaries2D, const double &theta);

void Calculate_CuboidFaceNormals(const std::vector<std::unique_ptr<Element>> &List_Of_Elements, std::vector<std::unique_ptr<Boundary>> &List_Of_Boundaries, const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices);
/*--------------------------------------------------------------------------*/
#endif
