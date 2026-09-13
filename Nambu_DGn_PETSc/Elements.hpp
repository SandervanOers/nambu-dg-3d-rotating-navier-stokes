#ifndef ELEMENTS
#define ELEMENTS

#include <petscksp.h>
#include <vector>
#include <iostream>
#include <memory>
/*--------------------------------------------------------------------------*/
#define PR(x) std::cout << #x " = " << x << "\n";
#define PRp(x) std::cout << #x " = " << std::setprecision (20) << x << "\n";
/*--------------------------------------------------------------------------*/
class Vertex
{
    protected:
        unsigned int ID;
        double xCoordinate;
        double yCoordinate;
        double zCoordinate;
        unsigned int DIM;
        Vertex(unsigned int IDg, unsigned int DIMg, double xC, double yC, double zC);

    public:
        virtual unsigned int getID() const;
        virtual unsigned int getDIM() const;
        virtual double getxCoordinate() const;
        virtual double getyCoordinate() const;
        virtual double getzCoordinate() const;

        virtual ~Vertex();

};
class Vertex1D: public Vertex
{
    public:
    Vertex1D(unsigned int IDg, double xC): Vertex{ IDg, 1, xC, 0.0, 0.0}
    {
    }
};
class Vertex2D: public Vertex
{
    public:
    Vertex2D(unsigned int IDg, double xC, double yC): Vertex{ IDg, 2, xC, yC, 0.0}
    {
    }

};
class Vertex3D: public Vertex
{
    public:
    Vertex3D(unsigned int IDg, double xC, double yC, double zC): Vertex{ IDg, 3, xC, yC, zC}
    {
    }
};
extern void print(const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices);
/*--------------------------------------------------------------------------*/
class Boundary
{
    protected:
        unsigned int ID;
        unsigned int DIM;
        double nx;
        double ny;
        double nz;
        int ID_Left_Element;
        int ID_Right_Element;
        unsigned int Type_Left;  // 0 1 2 3 4 5
        unsigned int Type_Right; // 0 1 2 3 4 5
        double theta;
        double Jacobian;
        Boundary(unsigned int IDg, unsigned int DIMg, int ID_El_L, int ID_El_R, int Type_L, int Type_R);

    public:
        virtual unsigned int getID() const;
        virtual unsigned int getDIM() const;
        virtual int getLeftElementID() const;
        virtual int getRightElementID() const;
        virtual unsigned int getTypeLeft() const;
        virtual unsigned int getTypeRight() const;
        virtual void setJacobian(double J);
        virtual double getJacobian() const;
        virtual void set_nx(double normalx);
        virtual void set_ny(double normaly);
        virtual void set_nz(double normalz);
        virtual double get_nx() const;
        virtual double get_ny() const;
        virtual double get_nz() const;
        virtual void set_theta(double T);
        virtual double get_theta() const;

        bool operator < (const Boundary& other) const {return ID < other.ID; }


        virtual ~Boundary();
};
class Boundary1D: public Boundary
{
    public:
        Boundary1D(unsigned int IDg, int ID_El_L, int ID_El_R, int Type_L, int Type_R): Boundary{ IDg, 1, ID_El_L, ID_El_R, Type_L, Type_R}
        {
        }
};
class Boundary2D: public Boundary
{
    public:
        Boundary2D(unsigned int IDg, int ID_El_L, int ID_El_R, int Type_L, int Type_R): Boundary{ IDg, 2, ID_El_L, ID_El_R, Type_L, Type_R}
        {
        }
};
class Boundary3D: public Boundary
{
    public:
        Boundary3D(unsigned int IDg, int ID_El_L, int ID_El_R, int Type_L, int Type_R): Boundary{ IDg, 3, ID_El_L, ID_El_R, Type_L, Type_R}
        {
        }
};
extern void print(const std::vector<std::unique_ptr<Boundary>> &List_Of_Boundaries);
/*--------------------------------------------------------------------------*/
      /* Gmsh Ordering
        Line:                 Line3:          Line4:

              v
              ^
              |
              |
        0-----+-----1 --> u   0----2----1     0---2---3---1

        Quadrangle:            Quadrangle8:            Quadrangle9:

              v
              ^
              |
        3-----------2          3-----6-----2           3-----6-----2
        |     |     |          |           |           |           |
        |     |     |          |           |           |           |
        |     +---- | --> u    7           5           7     8     5
        |           |          |           |           |           |
        |           |          |           |           |           |
        0-----------1          0-----4-----1           0-----4-----1

        Hexahedron:             Hexahedron20:          Hexahedron27:

               v
        3----------2            3----13----2           3----13----2
        |\     ^   |\           |\         |\          |\         |\
        | \    |   | \          | 15       | 14        |15    24  | 14
        |  \   |   |  \         9  \       11 \        9  \ 20    11 \
        |   7------+---6        |   7----19+---6       |   7----19+---6
        |   |  +-- |-- | -> u   |   |      |   |       |22 |  26  | 23|
        0---+---\--1   |        0---+-8----1   |       0---+-8----1   |
         \  |    \  \  |         \  17      \  18       \ 17    25 \  18
          \ |     \  \ |         10 |        12|        10 |  21    12|
           \|      w  \|           \|         \|          \|         \|
            4----------5            4----16----5           4----16----5 */

/*--------------------------------------------------------------------------*/
class Element
{
    protected:
        unsigned int ID;
        unsigned int Number_Of_Nodes;
        unsigned int DIM;
        unsigned int pos;
        double detJ;

        int ID_Vertex_V1;
        int ID_Vertex_V2;
        int ID_Vertex_V3;
        int ID_Vertex_V4;
        int ID_Vertex_V5;
        int ID_Vertex_V6;
        int ID_Vertex_V7;
        int ID_Vertex_V8;
        unsigned int Order_Of_Polynomials_x;
        unsigned int Order_Of_Polynomials_y;
        unsigned int Order_Of_Polynomials_z;
        std::vector<double> node_coordinates_x;
        std::vector<double> node_coordinates_y;
        std::vector<double> node_coordinates_z;
        std::vector<unsigned int> node_on_face0;
        std::vector<unsigned int> node_on_face1;
        std::vector<unsigned int> node_on_face2;
        std::vector<unsigned int> node_on_face3;
        std::vector<unsigned int> node_on_face4;
        std::vector<unsigned int> node_on_face5;

    public:
        Element(unsigned int IDg, unsigned int DIMg, int ID_V1, int ID_V2, int ID_V3, int ID_V4, int ID_V5, int ID_V6, int ID_V7, int ID_V8);
        unsigned int getID() const ;
        unsigned int get_pos() const;
        virtual unsigned int getDIM() const;
        void set_pos(unsigned int POS);
        void set_detJ(double J);
        double get_detJ() const;

        std::vector<PetscScalar> invM_local;

        int getVertex_V1() const;
        int getVertex_V2() const;
        int getVertex_V3() const;
        int getVertex_V4() const;
        int getVertex_V5() const;
        int getVertex_V6() const;
        int getVertex_V7() const;
        int getVertex_V8() const;
        void set_Order_Of_Polynomials_x(unsigned int N);
        void set_Order_Of_Polynomials_y(unsigned int N);
        void set_Order_Of_Polynomials_z(unsigned int N);
        unsigned int get_Order_Of_Polynomials_x() const;
        unsigned int get_Order_Of_Polynomials_y() const;
        unsigned int get_Order_Of_Polynomials_z() const;
        unsigned int get_Number_Of_Nodes() const;
        void set_Number_Of_Nodes(unsigned int Nnodes);
        void set_node_coordinates_x(double x);
        void set_node_coordinates_y(double y);
        void set_node_coordinates_z(double z);
        const std::vector<double>& get_node_coordinates_x() const;
        const std::vector<double>& get_node_coordinates_y() const;
        const std::vector<double>& get_node_coordinates_z() const;
        void set_node_on_face0(unsigned int k);
        void set_node_on_face1(unsigned int k);
        void set_node_on_face2(unsigned int k);
        void set_node_on_face3(unsigned int k);
        void set_node_on_face4(unsigned int k);
        void set_node_on_face5(unsigned int k);
        std::vector<unsigned int> get_node_on_face0() const;
        std::vector<unsigned int> get_node_on_face1() const;
        std::vector<unsigned int> get_node_on_face2() const;
        std::vector<unsigned int> get_node_on_face3() const;
        std::vector<unsigned int> get_node_on_face4() const;
        std::vector<unsigned int> get_node_on_face5() const;
        std::vector<unsigned int> get_nodes_on_boundary(unsigned int Type) const;

        void set_invM_local(const std::vector<PetscScalar> &invM);
        void set_invM_local(std::vector<PetscScalar> &&invM);
        const std::vector<PetscScalar> &get_invM_local() const;

    virtual ~Element();
};
class Element1D: public Element
{
public:
    Element1D(unsigned int IDg, int ID_V1, int ID_V2)
        : Element{ IDg, 1, ID_V1, ID_V2, -1, -1, -1, -1, -1, -1 }
    {
    }
};

class Element2D: public Element
{
public:
    Element2D(unsigned int IDg, int ID_V1, int ID_V2, int ID_V3, int ID_V4)
        : Element{ IDg, 2, ID_V1, ID_V2, ID_V3, ID_V4, -1, -1, -1, -1 }
    {
    }
};

class Element3D: public Element
{
public:
    Element3D(
        unsigned int IDg,
        int ID_V1,
        int ID_V2,
        int ID_V3,
        int ID_V4,
        int ID_V5,
        int ID_V6,
        int ID_V7,
        int ID_V8)
        : Element{ IDg, 3, ID_V1, ID_V2, ID_V3, ID_V4, ID_V5, ID_V6, ID_V7, ID_V8 }
    {
    }
};

extern void print(const std::vector<std::unique_ptr<Element>> &List_Of_Elements);

/*--------------------------------------------------------------------------*/
#endif
