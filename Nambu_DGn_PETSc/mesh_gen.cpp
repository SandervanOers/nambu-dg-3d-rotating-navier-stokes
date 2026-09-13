#include "mesh_gen.hpp"
/*--------------------------------------------------------------------------*/
void create_Mesh_Cuboid(std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices, std::vector<std::unique_ptr<Element>> &List_Of_Elements, unsigned int &N_Elements_total, unsigned int &N_Vertices_total, const unsigned int &Nel_x, const unsigned int &Nel_y, const unsigned int &Nel_z)
{
/* Input: Number of elements in each direction: Nel_x, Nel_y, Nel_z
 Output: List of Vertices: [IDv, x, y, z]
 Element to Vertex connectivity: EToV = [IDe, IDv1, IDv2, IDv3, ID_v4
                                              IDv5, IDv6, IDv7, IDv8];
*/
if (Nel_x == 0 || Nel_y == 0 || Nel_z == 0) {
    throw std::invalid_argument("create_Mesh_Cuboid: Nel_x, Nel_y, Nel_z must be positive.");
}
List_Of_Vertices.clear();
List_Of_Elements.clear();

N_Elements_total = Nel_x*Nel_y*Nel_z;
N_Vertices_total = (Nel_x+1)*(Nel_y+1)*(Nel_z+1);

// Step sizes
double dx = 1.0/Nel_x, dy = 1.0/Nel_y, dz = 1.0/Nel_z;

  std::cout << "\n";
  std::cout << " Generating mesh for 3D cuboid \n";
  std::cout << "  Spatial dimension = [0,1]^3 \n";
  std::cout << "  Number of elements = " << Nel_x << " x " << Nel_y << " x " << Nel_z << " = " << N_Elements_total << "\n";
  std::cout << "  Number of vertices = " << N_Vertices_total << "\n";

// Compute the coordinates of the Vertices and store the results in dense Matrix of size [N_Vertices_total x 3]

/* Vertex numbering
      y ^
        |
        |
        x--x--x--x--x--x (36) = (Nel_x+1)*(Nel_y+1)
    5   |  |  |  |  |  |
        x--x--x--x--x--x
    4   |  |  |  |  |  |
r       x--x--x--x--x--x
o   3   |  |  |  |  |  |
w       x--x--x--x--x--x
    2   |  |  |  |  |  |
        x--x--x--x--x--x 2*(Nel_x+1)
    1   |  |  |  |  |  |
        x--x--x--x--x--x --> x
      (1) 1  2  3  4  5 (Nel_x+1)
             column
*/
for (unsigned int i = 0; i < N_Vertices_total; i++)
  {
    unsigned int level  = (i/(Nel_x+1)/(Nel_y+1));
    unsigned int row    = ((i-(level)*(Nel_x+1)*(Nel_y+1))/(Nel_x+1));
    unsigned int column = i-(level)*(Nel_x+1)*(Nel_y+1)-(row)*(Nel_x+1);

    List_Of_Vertices.push_back(std::make_unique<Vertex3D>(i, column*dx, row*dy, level*dz));
  }
  std::cout << "  Vertex coordinates computed \n";

/*
Element - Vertex numbering:
            v
     3----------2
     |\     ^   |\
     | \    |   | \
     |  \   |   |  \
     |   7------+---6
     |   |  +-- |-- | -> u
     0---+---\--1   |
      \  |    \  \  |
       \ |     \  \ |
        \|      w  \|
         4----------5
  */

  for (unsigned int e = 0; e < N_Elements_total; e++)
    {
      unsigned int level  = ceil((double)(e+1)/(Nel_x)/(Nel_y));
      unsigned int row    = ceil(((double)(e+1)-(level-1)*(Nel_x)*(Nel_y))/(Nel_x));
      unsigned int column = e-(level-1)*(Nel_x)*(Nel_y)-(row-1)*(Nel_x);

      // lower bottom left vertex ( == 0)
      unsigned int LBL = (level-1)*(Nel_x+1)*(Nel_y+1)+(row-1)*(Nel_x+1)+column;
      //std::cout << e << " " << LBL << " " << LBL+1 << " " << LBL+1+Nel_x+1 << " " << LBL+Nel_x+1 << " " << LBL+(Nel_x+1)*(Nel_y+1) << " " << LBL+1+(Nel_x+1)*(Nel_y+1) << " " << LBL+1+Nel_x+1+(Nel_x+1)*(Nel_y+1) << " " << LBL+Nel_x+1+(Nel_x+1)*(Nel_y+1) <<  std::endl;
      List_Of_Elements.push_back(std::make_unique<Element3D>(e, LBL, LBL+1, LBL+1+Nel_x+1, LBL+Nel_x+1, LBL+(Nel_x+1)*(Nel_y+1), LBL+1+(Nel_x+1)*(Nel_y+1), LBL+1+Nel_x+1+(Nel_x+1)*(Nel_y+1), LBL+Nel_x+1+(Nel_x+1)*(Nel_y+1)));

      ////etov[8*e]   = LBL;
      ////etov[8*e+1] = LBL+1;
      ////etov[8*e+2] = LBL+1+Nel_x+1;
      ////etov[8*e+3] = LBL+Nel_x+1;
      ////etov[8*e+4] = LBL+(Nel_x+1)*(Nel_y+1);
      ////etov[8*e+5] = LBL+1+(Nel_x+1)*(Nel_y+1);
      ////etov[8*e+6] = LBL+1+Nel_x+1+(Nel_x+1)*(Nel_y+1);
      ////etov[8*e+7] = LBL+Nel_x+1+(Nel_x+1)*(Nel_y+1);

    }
    std::cout << "  Element to Vertex connectivity computed \n";
    std::cout << "  Grid computation done \n";
    std::cout << "\n";

}
/*--------------------------------------------------------------------------*/
void Connect_3DPeriodic(const unsigned int &Nel_x, const unsigned int &Nel_y, const unsigned int &Nel_z, const unsigned int &N_Elements_total, std::vector<std::unique_ptr<Boundary>> &List_Of_Boundaries)
{
  //unsigned int Nfaces = 6;

  if (Nel_x == 0 || Nel_y == 0 || Nel_z == 0) {
      throw std::invalid_argument("Connect_3DPeriodic: Nel_x, Nel_y, Nel_z must be positive.");
  }

  List_Of_Boundaries.clear();

  if (Nel_z == 1 || Nel_y == 1 || Nel_x == 1)
  {
    std::cout << "Warning: One of the dimensions has 1 element." << std::endl;
  }
  unsigned int ID_Boundary = 1;
   // Store 1 .. N_Elements_total in each column of EtoE
  PetscInt etoe[6];
  for (unsigned int e = 0; e < N_Elements_total; e++)
  {
    unsigned int level  = ceil((double)(e+1)/(Nel_x*Nel_y));
    unsigned int row    = ceil(((double)(e+1)-(level-1)*(Nel_x)*(Nel_y))/(Nel_x));
    unsigned int column = e+1-(level-1)*(Nel_x)*(Nel_y)-(row-1)*(Nel_x);

    // reference neighbour to element boundary
    // internal element
    etoe[0] = e-Nel_x*Nel_y; // z down
    etoe[1] = e+Nel_x*Nel_y; // z up
    etoe[2] = e-Nel_x;       // y down
    etoe[3] = e+Nel_x;       // y up
    etoe[4] = e-1;           // x left
    etoe[5] = e+1;           // x right

    if (level==1)      etoe[0] = e+Nel_x*Nel_y*Nel_z-Nel_x*Nel_y; // periodic in z
    if (level==Nel_z)  etoe[1] = e+Nel_x*Nel_y-Nel_x*Nel_y*Nel_z; // periodic in z
    if (row==1)        etoe[2] = e+Nel_x*Nel_y-Nel_x; // periodic in y
    if (row==Nel_y)    etoe[3] = e+Nel_x-Nel_x*Nel_y; // periodic in y
    if (column==1)     etoe[4] = e+Nel_x-1; // periodic in x
    if (column==Nel_x) etoe[5] = e+1-Nel_x; // periodic in x

    List_Of_Boundaries.push_back(std::make_unique<Boundary3D>(ID_Boundary, e, etoe[5], 5, 4)); ID_Boundary++; // boundary in x-direction
    List_Of_Boundaries.push_back(std::make_unique<Boundary3D>(ID_Boundary, e, etoe[3], 3, 2)); ID_Boundary++; // boundary in y-direction
    List_Of_Boundaries.push_back(std::make_unique<Boundary3D>(ID_Boundary, e, etoe[1], 1, 0)); ID_Boundary++; // boundary in z-direction

  }

  std::cout << "\033[1;32m  List Internal Boundaries computed\033[0m \n";
  std::cout << "\n";
}
/*--------------------------------------------------------------------------*/
void Connect_3D(const unsigned int &Nel_x, const unsigned int &Nel_y, const unsigned int &Nel_z, const unsigned int &N_Elements_total, std::vector<std::unique_ptr<Boundary>> &List_Of_Boundaries, const bool &Sx, const bool &Sy, const bool &Sz)
{

  if (Nel_x == 0 || Nel_y == 0 || Nel_z == 0) {
      throw std::invalid_argument("Connect_3D: Nel_x, Nel_y, Nel_z must be positive.");
  }
  List_Of_Boundaries.clear();

  if (Nel_z == 1 || Nel_y == 1 || Nel_x == 1)
  {
    std::cout << "  Warning: One of the dimensions has 1 element." << std::endl;
  }
  unsigned int ID_Boundary = 1;
   // Store 1 .. N_Elements_total in each column of EtoE
  PetscScalar etoe[6];
  // Loop over all Elements
  // one Element has 6 boundaries (6 faces of a cuboid)
  // store only upper 3 boundaries (x right, y up, z up)
  for (unsigned int e = 0; e < N_Elements_total; e++)
  {
    // where in the cube is the Element located
    unsigned int level  = ceil((double)(e+1)/(Nel_x)/(Nel_y));
    unsigned int row    = ceil(((double)(e+1)-(level-1)*(Nel_x)*(Nel_y))/(Nel_x));
    unsigned int column = e+1-(level-1)*(Nel_x)*(Nel_y)-(row-1)*(Nel_x);

    // reference neighbour to element boundary
    // internal element
    etoe[0] = e-Nel_x*Nel_y; // z down
    etoe[1] = e+Nel_x*Nel_y; // z up
    etoe[2] = e-Nel_x;       // y down
    etoe[3] = e+Nel_x;       // y up
    etoe[4] = e-1;           // x left
    etoe[5] = e+1;           // x right

    if (column==1)
    {
      if (Sx == 1)
      {
        // Solid walls in x-direction
        etoe[4] = e; // refer to element itself as neighbour
      }
      else
      {
        etoe[4] = e+Nel_x-1; // periodic in x
      }
    }
    if (column==Nel_x)
    {
      if (Sx == 1)
      {
        // Solid walls in x-direction
        etoe[5] = e; // refer to element itself as neighbour
      }
      else
      {
        etoe[5] = e+1-Nel_x; // periodic in x
      }
    }
    if (row==1)
    {
      if (Sy == 1)
      {
        // Solid walls in y-direction
        etoe[2] = e; // refer to element itself as neighbour
      }
      else
      {
        etoe[2] = e+Nel_x*Nel_y-Nel_x; // periodic in y
      }
    }
    if (row==Nel_y)
    {
      if (Sy == 1)
      {
        // Solid walls in y-direction
        etoe[3] = e; // refer to element itself as neighbour
      }
      else
      {
        etoe[3] = e+Nel_x-Nel_x*Nel_y; // periodic in y
      }
    }
    if (level==1)
    {
      if (Sz == 1)
      {
        // Solid walls in z-direction
        etoe[0] = e; // refer to element itself as neighbour
      }
      else
      {
        etoe[0] = e+Nel_x*Nel_y*Nel_z-Nel_x*Nel_y; // periodic in z
      }
    }
    if (level==Nel_z)
    {
      if (Sz == 1)
      {
        // Solid walls in z-direction
        etoe[1] = e; // refer to element itself as neighbour
      }
      else
      {
        etoe[1] = e+Nel_x*Nel_y-Nel_x*Nel_y*Nel_z; // periodic in z
      }
    }
    if (etoe[5] != e) {List_Of_Boundaries.push_back(std::make_unique<Boundary3D>(ID_Boundary, e, etoe[5], 5, 4)); ID_Boundary++;} // boundary in x-direction
    if (etoe[3] != e) {List_Of_Boundaries.push_back(std::make_unique<Boundary3D>(ID_Boundary, e, etoe[3], 3, 2)); ID_Boundary++;} // boundary in y-direction
    if (etoe[1] != e) {List_Of_Boundaries.push_back(std::make_unique<Boundary3D>(ID_Boundary, e, etoe[1], 1, 0)); ID_Boundary++;} // boundary in z-direction

  }
  std::cout << "\033[1;32mList Internal Boundaries computed\033[0m \n";
  std::cout << "\n";
}
/*--------------------------------------------------------------------------*/
void Calculate_CuboidFaceNormals(const std::vector<std::unique_ptr<Element>> &List_Of_Elements, std::vector<std::unique_ptr<Boundary>> &List_Of_Boundaries, const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices)
{
    for(auto f = List_Of_Boundaries.begin(); f < List_Of_Boundaries.end(); f++)
    {
        int left = (*f)->getLeftElementID();
        int left_face = (*f)->getTypeLeft();
        //int right = (*f)->getRightElementID();

        double x1, x2, x3, y1, y2, y3, z1, z2, z3;
        x1 = x2 = x3 = y1 = y2 = y3 = z1 = z2 = z3 = 0.0;
        // We read the vertices counterclockwise so that we obtain the outward normal
switch(left_face)
{
    case 0: // z down
        x1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V1()]->getxCoordinate();
        y1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V1()]->getyCoordinate();
        z1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V1()]->getzCoordinate();
        x2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V3()]->getxCoordinate();
        y2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V3()]->getyCoordinate();
        z2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V3()]->getzCoordinate();
        x3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V2()]->getxCoordinate();
        y3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V2()]->getyCoordinate();
        z3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V2()]->getzCoordinate();
        break;

    case 1: // z up
        x1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V5()]->getxCoordinate();
        y1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V5()]->getyCoordinate();
        z1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V5()]->getzCoordinate();
        x2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V6()]->getxCoordinate();
        y2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V6()]->getyCoordinate();
        z2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V6()]->getzCoordinate();
        x3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V7()]->getxCoordinate();
        y3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V7()]->getyCoordinate();
        z3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V7()]->getzCoordinate();
        break;

    case 2: // y down
        x1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V1()]->getxCoordinate();
        y1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V1()]->getyCoordinate();
        z1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V1()]->getzCoordinate();
        x2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V6()]->getxCoordinate();
        y2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V6()]->getyCoordinate();
        z2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V6()]->getzCoordinate();
        x3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V5()]->getxCoordinate();
        y3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V5()]->getyCoordinate();
        z3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V5()]->getzCoordinate();
        break;

    case 3: // y up
        x1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V4()]->getxCoordinate();
        y1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V4()]->getyCoordinate();
        z1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V4()]->getzCoordinate();
        x2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V8()]->getxCoordinate();
        y2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V8()]->getyCoordinate();
        z2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V8()]->getzCoordinate();
        x3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V7()]->getxCoordinate();
        y3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V7()]->getyCoordinate();
        z3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V7()]->getzCoordinate();
        break;

    case 4: // x left
        x1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V1()]->getxCoordinate();
        y1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V1()]->getyCoordinate();
        z1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V1()]->getzCoordinate();
        x2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V8()]->getxCoordinate();
        y2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V8()]->getyCoordinate();
        z2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V8()]->getzCoordinate();
        x3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V4()]->getxCoordinate();
        y3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V4()]->getyCoordinate();
        z3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V4()]->getzCoordinate();
        break;

    case 5: // x right
        x1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V2()]->getxCoordinate();
        y1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V2()]->getyCoordinate();
        z1 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V2()]->getzCoordinate();
        x2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V3()]->getxCoordinate();
        y2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V3()]->getyCoordinate();
        z2 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V3()]->getzCoordinate();
        x3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V7()]->getxCoordinate();
        y3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V7()]->getyCoordinate();
        z3 = List_Of_Vertices[List_Of_Elements[left]->getVertex_V7()]->getzCoordinate();
        break;

    default:
        std::cout << "Something went wrong in the calculation of the normals" << std::endl;
}

        double v1x = x2 - x1;
        double v1y = y2 - y1;
        double v1z = z2 - z1;
        double v2x = x3 - x2;
        double v2y = y3 - y2;
        double v2z = z3 - z2;

        double nx = v1y*v2z-v1z*v2y;
        double ny = v1z*v2x-v1x*v2z;
        double nz = v1x*v2y-v1y*v2x;

        // Normalize
        double length_n = sqrt(nx*nx+ny*ny+nz*nz);
        nx = nx/length_n;
        ny = ny/length_n;
        nz = nz/length_n;

        double Area = 2.0*2.0;
        double detJacobian = length_n/Area;
        
        (*f)->setJacobian(detJacobian);
        (*f)->set_nx(nx);
        (*f)->set_ny(ny);
        (*f)->set_nz(nz);
    }
}
/*--------------------------------------------------------------------------*/
void set_Order_Polynomials_Uniform(std::vector<std::unique_ptr<Element>> &List_Of_Elements, const unsigned int &N)
{
    unsigned int dimension = List_Of_Elements.front()->getDIM();
    for(auto i = List_Of_Elements.begin(); i < List_Of_Elements.end(); i++)
    {
        (*i)->set_Order_Of_Polynomials_x(N);
        (*i)->set_Order_Of_Polynomials_y(N);
        unsigned int Nnodes = (N+1)*(N+1);
        if (dimension == 3)
        {
          (*i)->set_Order_Of_Polynomials_z(N);
          Nnodes = Nnodes*(N+1);
        }
        (*i)->set_Number_Of_Nodes(Nnodes);
    }
}
/*--------------------------------------------------------------------------*/
extern unsigned int get_Number_Of_Nodes(std::vector<std::unique_ptr<Element>> &List_Of_Elements)
{
    unsigned int Number_Of_Nodes = 0;
    for(auto i = List_Of_Elements.begin(); i < List_Of_Elements.end(); i++)
    {
        (*i)->set_pos(Number_Of_Nodes);
        Number_Of_Nodes += (*i)->get_Number_Of_Nodes();
    }
    return Number_Of_Nodes;
}
/*--------------------------------------------------------------------------*/
void set_theta_Uniform(std::vector<std::unique_ptr<Boundary>> &List_Of_Boundaries, const double &theta)
{
    for(auto f = List_Of_Boundaries.begin(); f < List_Of_Boundaries.end(); f++)
    {
        (*f)->set_theta(theta);
    }
}
/*--------------------------------------------------------------------------*/
