function ops = build_operators_DG0(Nel, solidWalls)
%BUILD_OPERATORS_DG0 Centred DG(0) operators on [0,1]^3.
%
%   ops = build_operators_DG0(Nel, solidWalls)
%
% Nel        = [Nx Ny Nz]
% solidWalls = [Sx Sy Sz], with
%              0 = periodic direction
%              1 = impermeable walls at both ends of that direction
%
% This is the theta = 1/2 Cartesian DG(0) specialization of the operators
% in the Nambu DGFEM paper.
%
% Important for wall-bounded domains:
%   GRAD = [Gx; Gy; Gz]
%   DIV  = -GRAD'
%   CURL is constructed from the SAME Gx,Gy,Gz directional operators:
%
%       [  0  -Gz   Gy ]
%       [ Gz    0  -Gx ]
%       [-Gy   Gx    0 ]
%
% Consequently CURL*GRAD = 0 exactly on Cartesian tensor-product meshes,
% also when one or more coordinate directions have impermeable walls.
%
% Ordering:
%   global index = i + (j-1)Nx + (k-1)Nx*Ny,
% so x is the fastest-running index.

if nargin < 2
    solidWalls = [false false false];
end
solidWalls = logical(solidWalls(:).');

Nx = Nel(1);
Ny = Nel(2);
Nz = Nel(3);

hx = 1/Nx;
hy = 1/Ny;
hz = 1/Nz;

Dx1 = centred_derivative_1d(Nx, hx, solidWalls(1));
Dy1 = centred_derivative_1d(Ny, hy, solidWalls(2));
Dz1 = centred_derivative_1d(Nz, hz, solidWalls(3));

Ix = speye(Nx);
Iy = speye(Ny);
Iz = speye(Nz);

Gx = kron(Iz, kron(Iy, Dx1));
Gy = kron(Iz, kron(Dy1, Ix));
Gz = kron(Dz1, kron(Iy, Ix));

N = Nx*Ny*Nz;

GRAD = [Gx; Gy; Gz];
DIV = -GRAD';

Z = sparse(N,N);
CURL = [ Z,  -Gz,  Gy; ...
         Gz,   Z,  -Gx; ...
        -Gy,  Gx,   Z];

LAP = DIV*GRAD;

ops.N = N;
ops.Nel = Nel;
ops.h = [hx hy hz];
ops.volume = hx*hy*hz;
ops.solidWalls = solidWalls;

ops.Gx = Gx;
ops.Gy = Gy;
ops.Gz = Gz;
ops.GRAD = GRAD;
ops.DIV = DIV;
ops.CURL = CURL;
ops.LAP = LAP;

ops.boundaryFaces = build_boundary_faces(Nel, [hx hy hz], solidWalls);

end


function D = centred_derivative_1d(N, h, solidWall)
% theta = 1/2 directional DG(0) gradient.
%
% Periodic:
%   (p_{i+1}-p_{i-1})/(2h)
%
% Impermeable-wall direction:
%   interior as above, while at the first and last element the exterior
%   numerical normal flux is zero.  This gives
%
%   first: (p_2-p_1)/(2h)
%   last : (p_N-p_{N-1})/(2h)
%
% which is the compact matrix form of compute_GRADMatrix_sparse_central2.

if ~solidWall
    row = (1:N).';
    colPlus = [2:N 1].';
    colMinus = [N 1:N-1].';

    D = sparse([row; row], ...
               [colPlus; colMinus], ...
               [ones(N,1); -ones(N,1)]/(2*h), ...
               N, N);
    return;
end

ii = [];
jj = [];
vv = [];

if N == 1
    D = sparse(1,1);
    return;
end

% Left boundary element
ii = [ii; 1; 1];
jj = [jj; 1; 2];
vv = [vv; -1/(2*h); 1/(2*h)];

% Interior elements
for i = 2:N-1
    ii = [ii; i; i];
    jj = [jj; i-1; i+1];
    vv = [vv; -1/(2*h); 1/(2*h)];
end

% Right boundary element
ii = [ii; N; N];
jj = [jj; N-1; N];
vv = [vv; -1/(2*h); 1/(2*h)];

D = sparse(ii,jj,vv,N,N);

end


function faces = build_boundary_faces(Nel, h, solidWalls)
% One entry per physical boundary face of a DG(0) element.

Nx = Nel(1); Ny = Nel(2); Nz = Nel(3);
hx = h(1); hy = h(2); hz = h(3);

cellIndex = [];
normal = [];
area = [];

if solidWalls(1)
    [J,K] = ndgrid(1:Ny,1:Nz);

    left = 1 + (J(:)-1)*Nx + (K(:)-1)*Nx*Ny;
    right = Nx + (J(:)-1)*Nx + (K(:)-1)*Nx*Ny;

    cellIndex = [cellIndex; left; right];
    normal = [normal; ...
              repmat([-1 0 0],numel(left),1); ...
              repmat([ 1 0 0],numel(right),1)];
    area = [area; ...
            hy*hz*ones(numel(left),1); ...
            hy*hz*ones(numel(right),1)];
end

if solidWalls(2)
    [I,K] = ndgrid(1:Nx,1:Nz);

    front = I(:) + (K(:)-1)*Nx*Ny;
    back = I(:) + (Ny-1)*Nx + (K(:)-1)*Nx*Ny;

    cellIndex = [cellIndex; front; back];
    normal = [normal; ...
              repmat([0 -1 0],numel(front),1); ...
              repmat([0  1 0],numel(back),1)];
    area = [area; ...
            hx*hz*ones(numel(front),1); ...
            hx*hz*ones(numel(back),1)];
end

if solidWalls(3)
    [I,J] = ndgrid(1:Nx,1:Ny);

    bottom = I(:) + (J(:)-1)*Nx;
    top = I(:) + (J(:)-1)*Nx + (Nz-1)*Nx*Ny;

    cellIndex = [cellIndex; bottom; top];
    normal = [normal; ...
              repmat([0 0 -1],numel(bottom),1); ...
              repmat([0 0  1],numel(top),1)];
    area = [area; ...
            hx*hy*ones(numel(bottom),1); ...
            hx*hy*ones(numel(top),1)];
end

faces.cell = cellIndex;
faces.normal = normal;
faces.area = area;

end
