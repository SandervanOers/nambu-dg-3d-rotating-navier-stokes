#include "Legendre_Gauss_Lobatto.hpp"

/*--------------------------------------------------------------------------*/
extern Vec JacobiGL(const double &alpha, const double &beta, const unsigned int &N)
{
// function [x] = JacobiGL(alpha,beta,N)
// Purpose: Compute the N'th order Gauss Lobatto quadrature
//          points, x, associated with the Jacobi polynomial,
//          of type (alpha,beta) > -1 ( <> -0.5).

    Vec x;
    VecCreateSeq(PETSC_COMM_WORLD, N+1,&x);
    if (N==0)
    {
        VecSetValue(x, 0, 0.0, INSERT_VALUES);
    }
    else if (N==1)
    {
        VecSetValue(x, 0, -1.0, INSERT_VALUES);
        VecSetValue(x, 1, 1.0, INSERT_VALUES);
    }
    else
    {
        VecSetValue(x, 0, -1.0, INSERT_VALUES);
        Vec xint;
        xint = JacobiGQ(alpha+1, beta+1, N-2);
        PetscScalar *array;
        VecGetArray(xint,&array);
        for (unsigned int i = 1; i < N; i++)
            VecSetValue(x, i, array[i-1], INSERT_VALUES);
        VecRestoreArray(xint, &array);
        VecSetValue(x, N, 1.0, INSERT_VALUES);
        VecDestroy(&xint);
    }
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);
    return x;
}
/*--------------------------------------------------------------------------*/
extern Vec JacobiGL_withWeights(const double &alpha, const double &beta, const unsigned int &N, Vec &Weights)
{
// function [x] = JacobiGL(alpha,beta,N)
// Purpose: Compute the N'th order Gauss Lobatto quadrature
//          points, x, and weights, w, associated with the Jacobi polynomial,
//          of type (alpha,beta) > -1 ( <> -0.5).

    Vec x;
    VecCreateSeq(PETSC_COMM_WORLD, N+1, &x);
    VecCreateSeq(PETSC_COMM_WORLD, N+1, &Weights);
    if (N==0)
    {
        VecSetValue(x, 0, 0.0, INSERT_VALUES);
        VecSetValue(Weights, 0, 2.0, INSERT_VALUES);
    }
    else if (N==1)
    {
        VecSetValue(x, 0, -1.0, INSERT_VALUES);
        VecSetValue(x, 1, 1.0, INSERT_VALUES);
        VecSetValue(Weights, 0, 1.0, INSERT_VALUES);
        VecSetValue(Weights, 1, 1.0, INSERT_VALUES);
    }
    else
    {
        VecSetValue(x, 0, -1.0, INSERT_VALUES);
        VecSetValue(Weights, 0, 2.0/(N+1)/(N), INSERT_VALUES);
        Vec xint;
        xint = JacobiGQ(alpha+1, beta+1, N-2);
        PetscScalar    *array;
        VecGetArray(xint,&array);
        Vec ww = JacobiP(xint, 0, 0, N);
        VecScale(ww, sqrt(2.0/(2.0*(N)+1.0)));
        PetscScalar    *warray;
        VecGetArray(ww,&warray);
        for (unsigned int i = 1; i < N; i++)
        {
            VecSetValue(x, i, array[i-1], INSERT_VALUES);
            double w = warray[i-1];
            w = w*w;
            w = 2.0/N/(N+1.0)/w;
            VecSetValue(Weights, i, w, INSERT_VALUES);
        }
        VecRestoreArray(xint, &array);
        VecRestoreArray(ww,&warray);
        VecDestroy(&ww);
        VecSetValue(x, N, 1.0, INSERT_VALUES);
        VecSetValue(Weights, N, 2.0/N/(N+1.0), INSERT_VALUES);
        VecDestroy(&xint);

    }
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);
    VecAssemblyBegin(Weights);
    VecAssemblyEnd(Weights);
    return x;
}
/*--------------------------------------------------------------------------*/
extern Vec JacobiGQ(const double &alpha, const double &beta, const unsigned int &N)
{
// function [x] = JacobiGQ(alpha,beta,N)
// Purpose: Compute the N'th order Gauss quadrature
//          points, x, associated with the Jacobi polynomial,
//          of type (alpha,beta) > -1 ( <> -0.5).
    Vec x;
    VecCreateSeq(PETSC_COMM_WORLD, N+1, &x);
    if (N==0)
    {
        VecSetValue(x, 0, -(alpha-beta)/(alpha+beta+2.0), INSERT_VALUES);
    }
    else
    {
        PetscScalar h1[N+1];
        PetscScalar a[(N+1)*(N+1)]={0};
        for (unsigned int j=0; j<=N; j++)
        {
            h1[j] = 2.0*j+alpha+beta;
        }
        for (unsigned int j = 0; j<=N; j++)
        {
            double value = -0.5*(alpha*alpha-beta*beta)/(h1[j]+2.0)/h1[j];
            a[j+(N+1)*j]= value;
            if (j < N)
            {
                double value1 = 2.0/(h1[j]+2.0)*sqrt((j+1.0)*(j+1.0+alpha+beta)*(j+1.0+alpha)*(j+1.0+beta)/(h1[j]+1.0)/(h1[j]+3.0));
                a[j+(N+1)*(j+1)] = value1;
                a[j+1+(N+1)*(j)] = value1;
            }
        }
        if ((alpha+beta) <= 10.0*std::numeric_limits<double>::epsilon())
            a[0] = 0.0;
        Mat J;
        MatCreate(PETSC_COMM_WORLD,&J);
        MatSetSizes(J, N+1, N+1, N+1, N+1);
        MatSetType(J, MATSEQDENSE);
        MatSeqDenseSetPreallocation(J,a);
        MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
        //MatView(J, PETSC_VIEWER_STDOUT_SELF);

        // Solve for eigenvalues
        EPS eps;

        EPSCreate(PETSC_COMM_WORLD,&eps);
        EPSSetOperators(eps,J,NULL);
        EPSSetProblemType(eps,EPS_HEP);
        EPSSetFromOptions(eps);
        EPSSetDimensions(eps, N+1, 2*(N+1), PETSC_DEFAULT);
        EPSSolve(eps);

        PetscReal      re;
        PetscScalar    kr,ki;
        Vec            xr,xi;
        PetscInt       i,nconv;

        MatCreateVecs(J,NULL,&xr);
        MatCreateVecs(J,NULL,&xi);

        EPSGetConverged(eps,&nconv);

        PetscScalar lam[nconv];
        if (nconv>0)
        {
            for (i=0;i<nconv;i++)
            {
                EPSGetEigenpair(eps,i,&kr,&ki,xr,xi);
                #if defined(PETSC_USE_COMPLEX)
                    re = PetscRealPart(kr);
                #else
                    re = kr;
                #endif
                lam[i] = (double)re;
            }
        }
        EPSDestroy(&eps);
        VecDestroy(&xr);
        VecDestroy(&xi);
        MatDestroy(&J);

        PetscSortReal(N+1,lam);
        PetscInt ix[N+1];
        for (unsigned int k=0;k<=N; k++)
            ix[k] = k;

        VecSetValues(x, N+1, ix, lam, INSERT_VALUES);
    }
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);

    //VecView(x, PETSC_VIEWER_STDOUT_SELF);
    return x;
}/*--------------------------------------------------------------------------*/
extern Vec JacobiGQ_withWeights(const double &alpha, const double &beta, const unsigned int &N, Vec &Weights)
{
    Vec x;
    VecCreateSeq(PETSC_COMM_WORLD, N+1, &x);
    VecCreateSeq(PETSC_COMM_WORLD, N+1, &Weights);
    if (N==0)
    {
        VecSetValue(x, 0, -(alpha-beta)/(alpha+beta+2.0), INSERT_VALUES);
        VecSetValue(Weights, 0, 2.0, INSERT_VALUES);
    }
    else
    {
        PetscScalar h1[N+1];
        PetscScalar a[(N+1)*(N+1)]={0};
        for (unsigned int j=0; j<=N; j++)
        {
            h1[j] = 2.0*j+alpha+beta;
        }
        for (unsigned int j = 0; j<=N; j++)
        {
            double value = -0.5*(alpha*alpha-beta*beta)/(h1[j]+2.0)/h1[j];
            a[j+(N+1)*j]= value;
            if (j < N)
            {
                double value1 = 2.0/(h1[j]+2.0)*sqrt((j+1.0)*(j+1.0+alpha+beta)*(j+1.0+alpha)*(j+1.0+beta)/(h1[j]+1.0)/(h1[j]+3.0));
                a[j+(N+1)*(j+1)] = value1;
                a[j+1+(N+1)*(j)] = value1;
            }
        }
        if ((alpha+beta) <= 10.0*std::numeric_limits<double>::epsilon())
            a[0] = 0.0;
        Mat J;
        MatCreate(PETSC_COMM_WORLD,&J);
        MatSetSizes(J, N+1, N+1, N+1, N+1);
        MatSetType(J, MATSEQDENSE);
        MatSeqDenseSetPreallocation(J,a);
        MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
        //MatView(J, PETSC_VIEWER_STDOUT_SELF);

        // Solve for eigenvalues and eigenvectors
        EPS eps;

        EPSCreate(PETSC_COMM_WORLD,&eps);
        EPSSetOperators(eps,J,NULL);
        EPSSetProblemType(eps,EPS_HEP);
        EPSSetFromOptions(eps);
        EPSSetDimensions(eps, N+1, 2*(N+1), PETSC_DEFAULT);
        EPSSetTarget(eps, -1.0);
        EPSSetWhichEigenpairs(eps, EPS_TARGET_REAL);
        EPSSolve(eps);

        PetscReal      re;
        PetscScalar    kr,ki;
        Vec            xr,xi;
        PetscInt       i,nconv;

        MatCreateVecs(J,NULL,&xr);
        MatCreateVecs(J,NULL,&xi);

        EPSGetConverged(eps,&nconv);

        PetscScalar lam[nconv];
        PetscScalar w[nconv];
        if (nconv>0)
        {
            for (i=0;i<nconv;i++)
            {
                EPSGetEigenpair(eps,i,&kr,&ki,xr,xi);
                #if defined(PETSC_USE_COMPLEX)
                    re = PetscRealPart(kr);
                #else
                    re = kr;
                #endif
                lam[i] = (double)re;
                PetscScalar *xa;
                VecGetArray(xr, &xa);
                w[i] = xa[0]*xa[0]*pow(2,(alpha+beta+1.0))/(alpha+beta+1.0)*tgamma(alpha+1.0)*tgamma(beta+1.0)/tgamma(alpha+beta+1.0);
                VecRestoreArray(xr, &xa);
            }
        }
        EPSDestroy(&eps);
        VecDestroy(&xr);
        VecDestroy(&xi);
        MatDestroy(&J);
        PetscInt ix[N+1];
        for (unsigned int k=0;k<=N; k++)
            ix[k] = k;

        VecSetValues(x, N+1, ix, lam, INSERT_VALUES);
        //VecSetValues(x, N+1, ix, w, INSERT_VALUES);
        VecSetValues(Weights, N+1, ix, w, INSERT_VALUES);
    }
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);
    VecAssemblyBegin(Weights);
    VecAssemblyEnd(Weights);

    //VecView(x, PETSC_VIEWER_STDOUT_SELF);
    return x;
}
/*--------------------------------------------------------------------------*/
extern Vec JacobiP(const Vec &x, const double &alpha, const double &beta, const unsigned int &N)
{
    PetscInt size_r = 0;
    VecGetSize(x, &size_r);

    Vec P;
    VecCreateSeq(PETSC_COMM_WORLD, size_r, &P);

    const PetscScalar *xa = nullptr;
    VecGetArrayRead(x, &xa);

    PetscScalar *pa = nullptr;
    VecGetArray(P, &pa);

    const double gamma0 =
        std::pow(2.0, alpha + beta + 1.0) /
        (alpha + beta + 1.0) *
        std::tgamma(alpha + 1.0) *
        std::tgamma(beta + 1.0) /
        std::tgamma(alpha + beta + 1.0);

    if (N == 0)
    {
        const double val = 1.0 / std::sqrt(gamma0);
        for (PetscInt k = 0; k < size_r; ++k)
        {
            pa[k] = val;
        }
    }
    else
    {
        std::vector<double> pnm1(size_r, 1.0 / std::sqrt(gamma0));
        std::vector<double> pn(size_r, 0.0);

        const double gamma1 = (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0) * gamma0;
        for (PetscInt k = 0; k < size_r; ++k)
        {
            pn[k] = (((alpha + beta + 2.0) * xa[k]) / 2.0 + (alpha - beta) / 2.0) / std::sqrt(gamma1);
        }

        if (N == 1)
        {
            for (PetscInt k = 0; k < size_r; ++k)
            {
                pa[k] = pn[k];
            }
        }
        else
        {
            double aold = 2.0 / (2.0 + alpha + beta) *
                          std::sqrt((alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0));

            std::vector<double> pnp1(size_r, 0.0);

            for (unsigned int n = 1; n <= N - 1; ++n)
            {
                const double h1 = 2.0 * static_cast<double>(n) + alpha + beta;
                const double anew =
                    2.0 / (h1 + 2.0) *
                    std::sqrt((n + 1.0) * (n + 1.0 + alpha + beta) *
                              (n + 1.0 + alpha) * (n + 1.0 + beta) /
                              ((h1 + 1.0) * (h1 + 3.0)));

                const double bnew = -(alpha * alpha - beta * beta) / (h1 * (h1 + 2.0));

                for (PetscInt k = 0; k < size_r; ++k)
                {
                    pnp1[k] = (-(aold) * pnm1[k] + (xa[k] - bnew) * pn[k]) / anew;
                }

                pnm1.swap(pn);
                pn.swap(pnp1);
                aold = anew;
            }

            for (PetscInt k = 0; k < size_r; ++k)
            {
                pa[k] = pn[k];
            }
        }
    }

    VecRestoreArray(P, &pa);
    VecRestoreArrayRead(x, &xa);

    VecAssemblyBegin(P);
    VecAssemblyEnd(P);
    return P;
}
/*--------------------------------------------------------------------------*/
extern void set_Node_Coordinates_Uniform(std::vector<std::unique_ptr<Element>> &List_Of_Elements, const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices, const unsigned int &N)
{
    unsigned int dimension = List_Of_Elements.front()->getDIM();
    if (dimension == 3)
    {
      set_Node_Coordinates_Uniform_3D(List_Of_Elements, List_Of_Vertices, N, N, N);
    }
    else if (dimension == 2)
    {
      set_Node_Coordinates_Uniform_2D(List_Of_Elements, List_Of_Vertices, N, N);
    }
}
/*--------------------------------------------------------------------------*/
void set_Node_Coordinates_Uniform_3D(std::vector<std::unique_ptr<Element>> &List_Of_Elements, const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices, const unsigned int &Nx, const unsigned int &Ny, const unsigned int &Nz)
{
    // total number of nodes
    unsigned int  Np = (Nx+1)*(Ny+1)*(Nz+1);
    std::vector<double> r_p(Np), s_p(Np), t_p(Np);
    NodesCuboid(Nx, Ny, Nz, r_p, s_p, t_p, Np);

    // Compute the physical location of each node on each element
    for(auto i = List_Of_Elements.begin(); i < List_Of_Elements.end(); i++)
    {
        double x0 = List_Of_Vertices[(*i)->getVertex_V1()]->getxCoordinate();
        double y0 = List_Of_Vertices[(*i)->getVertex_V1()]->getyCoordinate();
        double z0 = List_Of_Vertices[(*i)->getVertex_V1()]->getzCoordinate();
        double x1 = List_Of_Vertices[(*i)->getVertex_V2()]->getxCoordinate();
        double y1 = List_Of_Vertices[(*i)->getVertex_V2()]->getyCoordinate();
        double z1 = List_Of_Vertices[(*i)->getVertex_V2()]->getzCoordinate();
        double x2 = List_Of_Vertices[(*i)->getVertex_V3()]->getxCoordinate();
        double y2 = List_Of_Vertices[(*i)->getVertex_V3()]->getyCoordinate();
        double z2 = List_Of_Vertices[(*i)->getVertex_V3()]->getzCoordinate();
        double x3 = List_Of_Vertices[(*i)->getVertex_V4()]->getxCoordinate();
        double y3 = List_Of_Vertices[(*i)->getVertex_V4()]->getyCoordinate();
        double z3 = List_Of_Vertices[(*i)->getVertex_V4()]->getzCoordinate();
        double x4 = List_Of_Vertices[(*i)->getVertex_V5()]->getxCoordinate();
        double y4 = List_Of_Vertices[(*i)->getVertex_V5()]->getyCoordinate();
        double z4 = List_Of_Vertices[(*i)->getVertex_V5()]->getzCoordinate();
        double x5 = List_Of_Vertices[(*i)->getVertex_V6()]->getxCoordinate();
        double y5 = List_Of_Vertices[(*i)->getVertex_V6()]->getyCoordinate();
        double z5 = List_Of_Vertices[(*i)->getVertex_V6()]->getzCoordinate();
        double x6 = List_Of_Vertices[(*i)->getVertex_V7()]->getxCoordinate();
        double y6 = List_Of_Vertices[(*i)->getVertex_V7()]->getyCoordinate();
        double z6 = List_Of_Vertices[(*i)->getVertex_V7()]->getzCoordinate();
        double x7 = List_Of_Vertices[(*i)->getVertex_V8()]->getxCoordinate();
        double y7 = List_Of_Vertices[(*i)->getVertex_V8()]->getyCoordinate();
        double z7 = List_Of_Vertices[(*i)->getVertex_V8()]->getzCoordinate();

        for(unsigned int k = 0; k < Np; k++)
        {
            // All nodes
            double x = (1.0-r_p[k])*(1.0-s_p[k])*(1.0-t_p[k])*x0 + (1.0+r_p[k])*(1.0-s_p[k])*(1.0-t_p[k])*x1 + (1.0+r_p[k])*(1.0+s_p[k])*(1.0-t_p[k])*x2 + (1.0-r_p[k])*(1.0+s_p[k])*(1.0-t_p[k])*x3 + (1.0-r_p[k])*(1.0-s_p[k])*(1.0+t_p[k])*x4 + (1.0+r_p[k])*(1.0-s_p[k])*(1.0+t_p[k])*x5 + (1.0+r_p[k])*(1.0+s_p[k])*(1.0+t_p[k])*x6 + (1.0-r_p[k])*(1.0+s_p[k])*(1.0+t_p[k])*x7;
            double y = (1.0-r_p[k])*(1.0-s_p[k])*(1.0-t_p[k])*y0 + (1.0+r_p[k])*(1.0-s_p[k])*(1.0-t_p[k])*y1 + (1.0+r_p[k])*(1.0+s_p[k])*(1.0-t_p[k])*y2 + (1.0-r_p[k])*(1.0+s_p[k])*(1.0-t_p[k])*y3 + (1.0-r_p[k])*(1.0-s_p[k])*(1.0+t_p[k])*y4 + (1.0+r_p[k])*(1.0-s_p[k])*(1.0+t_p[k])*y5 + (1.0+r_p[k])*(1.0+s_p[k])*(1.0+t_p[k])*y6 + (1.0-r_p[k])*(1.0+s_p[k])*(1.0+t_p[k])*y7;
            double z = (1.0-r_p[k])*(1.0-s_p[k])*(1.0-t_p[k])*z0 + (1.0+r_p[k])*(1.0-s_p[k])*(1.0-t_p[k])*z1 + (1.0+r_p[k])*(1.0+s_p[k])*(1.0-t_p[k])*z2 + (1.0-r_p[k])*(1.0+s_p[k])*(1.0-t_p[k])*z3 + (1.0-r_p[k])*(1.0-s_p[k])*(1.0+t_p[k])*z4 + (1.0+r_p[k])*(1.0-s_p[k])*(1.0+t_p[k])*z5 + (1.0+r_p[k])*(1.0+s_p[k])*(1.0+t_p[k])*z6 + (1.0-r_p[k])*(1.0+s_p[k])*(1.0+t_p[k])*z7;
            x = x/8.0;
            y = y/8.0;
            z = z/8.0;

            //std::cout << "(x,y,z) = (" << x << "," << y << "," << z << ") : (r,s,p) = (" << r_p[k] << "," << s_p[k] << "," << t_p[k] << ")" << std::endl;
            // store node physical coordinates for each element
            (*i)->set_node_coordinates_x(x);
            (*i)->set_node_coordinates_y(y);
            (*i)->set_node_coordinates_z(z);

            // Boundary Nodes
            if (abs(t_p[k]+1.0) < NODETOL || Nz == 0)
            {
                (*i)->set_node_on_face0(k);
            }
            if (abs(t_p[k]-1.0) < NODETOL || Nz == 0)
            {
                (*i)->set_node_on_face1(k);
            }
            if (abs(s_p[k]+1.0) < NODETOL || Ny == 0)
            {
                (*i)->set_node_on_face2(k);
            }
            if (abs(s_p[k]-1.0) < NODETOL || Ny == 0)
            {
                (*i)->set_node_on_face3(k);
            }
            if (abs(r_p[k]+1.0) < NODETOL || Nx == 0)
            {
                (*i)->set_node_on_face4(k);
            }
            if (abs(r_p[k]-1.0) < NODETOL || Nx == 0)
            {
                (*i)->set_node_on_face5(k);
            }

          }
    }
}
/*--------------------------------------------------------------------------*/
void set_Node_Coordinates_Uniform_2D(std::vector<std::unique_ptr<Element>> &List_Of_Elements, const std::vector<std::unique_ptr<Vertex>> &List_Of_Vertices, const unsigned int &Nx, const unsigned int &Ny)
{
      // total number of nodes
      unsigned int  Np = (Nx+1)*(Ny+1);
      std::vector<double> r_p(Np), s_p(Np);
      NodesQuadrilateral(Nx, Ny, r_p, s_p, Np);

      // Compute the physical location of each node on each element
      for(auto i = List_Of_Elements.begin(); i < List_Of_Elements.end(); i++)
      {
        double x_v1 = List_Of_Vertices[(*i)->getVertex_V1()]->getxCoordinate();
        double y_v1 = List_Of_Vertices[(*i)->getVertex_V1()]->getyCoordinate();
        double x_v2 = List_Of_Vertices[(*i)->getVertex_V2()]->getxCoordinate();
        double y_v2 = List_Of_Vertices[(*i)->getVertex_V2()]->getyCoordinate();
        double x_v3 = List_Of_Vertices[(*i)->getVertex_V3()]->getxCoordinate();
        double y_v3 = List_Of_Vertices[(*i)->getVertex_V3()]->getyCoordinate();
        double x_v4 = List_Of_Vertices[(*i)->getVertex_V4()]->getxCoordinate();
        double y_v4 = List_Of_Vertices[(*i)->getVertex_V4()]->getyCoordinate();

        for(unsigned int k = 0; k < Np; k++)
        {
            double x = 0.25*((1.0-s_p[k]-r_p[k]+s_p[k]*r_p[k])*x_v1+(1.0-s_p[k]+r_p[k]-s_p[k]*r_p[k])*x_v2+(1.0+s_p[k]+r_p[k]+s_p[k]*r_p[k])*x_v3+(1.0+s_p[k]-r_p[k]-s_p[k]*r_p[k])*x_v4);
            double y = 0.25*((1.0-s_p[k]-r_p[k]+s_p[k]*r_p[k])*y_v1+(1.0-s_p[k]+r_p[k]-s_p[k]*r_p[k])*y_v2+(1.0+s_p[k]+r_p[k]+s_p[k]*r_p[k])*y_v3+(1.0+s_p[k]-r_p[k]-s_p[k]*r_p[k])*y_v4);
            // All nodes
            (*i)->set_node_coordinates_x(x);
            (*i)->set_node_coordinates_y(y);

            // Boundary nodes
            if (abs(s_p[k]+1.0) < NODETOL || Ny == 0)
            {
                (*i)->set_node_on_face0(k);
            }
            if (abs(s_p[k]-1.0) < NODETOL || Ny == 0)
            {
                (*i)->set_node_on_face1(k);
            }
            if (abs(r_p[k]+1.0) < NODETOL || Nx == 0)
            {
                (*i)->set_node_on_face2(k);
            }
            if (abs(r_p[k]-1.0) < NODETOL || Nx == 0)
            {
                (*i)->set_node_on_face3(k);
            }

        }
    }
}
/*---------get_Number_Of_Nodes-----------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/
void NodesCuboid(const unsigned int &Nx, const unsigned int &Ny, const unsigned int &Nz, std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, const unsigned int &Np)
{
    // function [x,y,z] = NodesCuboid(Nx, Ny, Nz, X, Y, Z);
    // Purpose  : Compute (x,y,z) nodes in cuboids for
    //             polynomial of order N

    if (Np==1)
    {
        x[0] = 0;
        y[0] = 0;
        z[0] = 0;
    }
    else
    {
        Vec rx = JacobiGL(0, 0, Nx);
        Vec ry = JacobiGL(0, 0, Ny);
        Vec rz = JacobiGL(0, 0, Nz);
        //std::cout << "rx = " << std::endl;
        //VecView(rx, PETSC_VIEWER_STDOUT_SELF);

        PetscScalar *rx_a, *ry_a, *rz_a;
        VecGetArray(rx, &rx_a);
        VecGetArray(ry, &ry_a);
        VecGetArray(rz, &rz_a);
        for (unsigned int i = 0; i < (Nz+1); i++)
        {
            for (unsigned int j = 0; j < (Ny+1); j++)
            {
                for (unsigned int k = 0; k < (Nx+1); k++)
                {
                x[i*(Ny+1)*(Nx+1)+j*(Nx+1)+k] = rx_a[k];
                y[i*(Ny+1)*(Nx+1)+j*(Nx+1)+k] = ry_a[j];
                z[i*(Ny+1)*(Nx+1)+j*(Nx+1)+k] = rz_a[i];
                }
            }
        }
        VecRestoreArray(rx, &rx_a);
        VecRestoreArray(ry, &ry_a);
        VecRestoreArray(rz, &rz_a);
        VecDestroy(&rx);
        VecDestroy(&ry);
        VecDestroy(&rz);
    }
}
/*--------------------------------------------------------------------------*/
void NodesQuadrilateral(const unsigned int &Nx, const unsigned int &Ny, std::vector<double> &x, std::vector<double> &y, const unsigned int &Np)
{
    // function [x,y] = NodesQuadrilateral(Nx, Ny, X, Y);
    // Purpose  : Compute (x,y) nodes in quadrilaterals for
    //             polynomial of order N

    if (Np==1)
    {
        x[0] = 0;
        y[0] = 0;
    }
    else
    {
        Vec rx = JacobiGL(0, 0, Nx);
        Vec ry = JacobiGL(0, 0, Ny);

        PetscScalar *rx_a, *ry_a;
        VecGetArray(rx, &rx_a);
        VecGetArray(ry, &ry_a);
        unsigned int i = 0;
        for (unsigned int j = 0; j < (Ny+1); j++)
        {
            for (unsigned int k = 0; k < (Nx+1); k++)
            {
                x[i*(Ny+1)*(Nx+1)+j*(Nx+1)+k] = rx_a[k];
                y[i*(Ny+1)*(Nx+1)+j*(Nx+1)+k] = ry_a[j];
            }
        }
        VecRestoreArray(rx, &rx_a);
        VecRestoreArray(ry, &ry_a);
        VecDestroy(&rx);
        VecDestroy(&ry);
    }
}
/*--------------------------------------------------------------------------*/
extern double LagrangePolynomial(const Vec &r, const double &x, const unsigned int &i)
{
    const std::vector<double> baryWeights = ComputeBarycentricWeights(r);
    return LagrangePolynomial(r, baryWeights, x, i);
}
/*--------------------------------------------------------------------------*/
extern double LagrangePolynomialDeriv(const Vec &r, const double &x, const unsigned int &i)
{
    const std::vector<double> baryWeights = ComputeBarycentricWeights(r);
    return LagrangePolynomialDeriv(r, baryWeights, x, i);
}
/*--------------------------------------------------------------------------*/
extern std::vector<double> ComputeBarycentricWeights(const Vec &r)
{
    PetscInt size_r = 0;
    VecGetSize(r, &size_r);

    const PetscScalar *ra = nullptr;
    VecGetArrayRead(r, &ra);

    std::vector<double> baryWeights(static_cast<std::size_t>(size_r), 1.0);

    for (PetscInt i = 0; i < size_r; ++i)
    {
        double denom = 1.0;
        for (PetscInt j = 0; j < size_r; ++j)
        {
            if (j != i)
            {
                denom *= (static_cast<double>(ra[i]) - static_cast<double>(ra[j]));
            }
        }
        baryWeights[static_cast<std::size_t>(i)] = 1.0 / denom;
    }

    VecRestoreArrayRead(r, &ra);
    return baryWeights;
}
/*--------------------------------------------------------------------------*/
extern double LagrangePolynomial(const Vec &r,
                                 const std::vector<double> &baryWeights,
                                 const double &x,
                                 const unsigned int &i)
{
    PetscInt size_r = 0;
    VecGetSize(r, &size_r);

    const PetscScalar *ra = nullptr;
    VecGetArrayRead(r, &ra);

    const PetscInt ii = static_cast<PetscInt>(i);

    // If x is exactly a node, return Kronecker delta
    for (PetscInt j = 0; j < size_r; ++j)
    {
        if (std::abs(x - static_cast<double>(ra[j])) < NODETOL)
        {
            VecRestoreArrayRead(r, &ra);
            return (j == ii) ? 1.0 : 0.0;
        }
    }

    double denom = 0.0;
    for (PetscInt j = 0; j < size_r; ++j)
    {
        denom += baryWeights[static_cast<std::size_t>(j)] /
                 (x - static_cast<double>(ra[j]));
    }

    const double numer =
        baryWeights[static_cast<std::size_t>(ii)] /
        (x - static_cast<double>(ra[ii]));

    VecRestoreArrayRead(r, &ra);
    return numer / denom;
}
/*--------------------------------------------------------------------------*/
extern double LagrangePolynomialDeriv(const Vec &r,
                                      const std::vector<double> &baryWeights,
                                      const double &x,
                                      const unsigned int &i)
{
    PetscInt size_r = 0;
    VecGetSize(r, &size_r);

    const PetscScalar *ra = nullptr;
    VecGetArrayRead(r, &ra);

    const PetscInt ii = static_cast<PetscInt>(i);

    // If x coincides with a node r_m, use the exact barycentric differentiation matrix formula
    for (PetscInt m = 0; m < size_r; ++m)
    {
        const double rm = static_cast<double>(ra[m]);

        if (std::abs(x - rm) < NODETOL)
        {
            double result = 0.0;

            if (ii != m)
            {
                result =
                    baryWeights[static_cast<std::size_t>(ii)] /
                    baryWeights[static_cast<std::size_t>(m)] /
                    (rm - static_cast<double>(ra[ii]));
            }
            else
            {
                for (PetscInt j = 0; j < size_r; ++j)
                {
                    if (j != m)
                    {
                        result -=
                            baryWeights[static_cast<std::size_t>(j)] /
                            baryWeights[static_cast<std::size_t>(m)] /
                            (rm - static_cast<double>(ra[j]));
                    }
                }
            }

            VecRestoreArrayRead(r, &ra);
            return result;
        }
    }

    // Generic x
    double S1 = 0.0;
    double S2 = 0.0;

    for (PetscInt j = 0; j < size_r; ++j)
    {
        const double xmrj = x - static_cast<double>(ra[j]);
        const double term = baryWeights[static_cast<std::size_t>(j)] / xmrj;
        S1 += term;
        S2 += term / xmrj;
    }

    const double li =
        (baryWeights[static_cast<std::size_t>(ii)] /
         (x - static_cast<double>(ra[ii]))) / S1;

    const double dli =
        li * (S2 / S1 - 1.0 / (x - static_cast<double>(ra[ii])));

    VecRestoreArrayRead(r, &ra);
    return dli;
}
/*--------------------------------------------------------------------------*/
// To be moved to another file:
/*--------------------------------------------------------------------------*/
extern Mat Inverse_Matrix(const Mat &V)
{
    // Calculate inverse Vandermonde Matrix
    Mat A, B, X;
    MatDuplicate(V,MAT_COPY_VALUES,&A);
    MatDuplicate(V,MAT_DO_NOT_COPY_VALUES,&X);
    MatDuplicate(V,MAT_DO_NOT_COPY_VALUES,&B);
    MatConvert(A, MATSEQDENSE, MAT_INPLACE_MATRIX, &A);
    MatConvert(B, MATSEQDENSE, MAT_INPLACE_MATRIX, &B);
    MatConvert(X, MATSEQDENSE, MAT_INPLACE_MATRIX, &X);

    MatShift(B, 1.0);
    MatOrderingType rtype = MATORDERINGNATURAL;
    IS row, col;
    MatGetOrdering(A, rtype, &row, &col);
    MatFactorInfo info;
    MatFactorInfoInitialize(&info);
    info.fill=1.0;
    info.dtcol=1.0;
    MatLUFactor(A, row, col, &info);
    MatMatSolve(A, B, X);
    //std::cout << "X = " << std::endl;
    //MatView(X, PETSC_VIEWER_STDOUT_SELF);
    // X is the inverse
    MatDestroy(&A);
    MatDestroy(&B);
    ISDestroy(&row);
    ISDestroy(&col);


    return X;
}
