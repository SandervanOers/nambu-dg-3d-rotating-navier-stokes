function V = exact_decaying_viscous_DG0(Nel, t, Re, Ubar)
%EXACT_DECAYING_VISCOUS_DG0 DG(0) L2 projection of paper Eq. (193).
%
% Periodic domain [0,1]^3, Fo = Ro = Inf.
%
%   xi   = x - Ubar(1)*t
%   eta  = y - Ubar(2)*t
%   zeta = z - Ubar(3)*t
%
%   a(t) = 4*sqrt(2)/(3*sqrt(3)) * exp(-12*pi^2*t/Re)
%
% The returned coefficients are exact cell averages.

Nx = Nel(1);
Ny = Nel(2);
Nz = Nel(3);

hx = 1/Nx;
hy = 1/Ny;
hz = 1/Nz;

x = ((1:Nx)-0.5)/Nx;
y = ((1:Ny)-0.5)/Ny;
z = ((1:Nz)-0.5)/Nz;

[X,Y,Z] = ndgrid(x,y,z);

xi   = X - Ubar(1)*t;
eta  = Y - Ubar(2)*t;
zeta = Z - Ubar(3)*t;

a = 4*sqrt(2)/(3*sqrt(3)) * exp(-12*pi^2*t/Re);
projectionFactor = sinc_pi(hx) * sinc_pi(hy) * sinc_pi(hz);

uShape = ...
    sin(2*pi*xi + pi/6).*sin(2*pi*eta + pi/3).*sin(2*pi*zeta) + ...
    cos(2*pi*zeta + pi/6).*cos(2*pi*xi + pi/3).*sin(2*pi*eta);

vShape = ...
    sin(2*pi*eta + pi/6).*sin(2*pi*zeta + pi/3).*sin(2*pi*xi) + ...
    cos(2*pi*xi + pi/6).*cos(2*pi*eta + pi/3).*sin(2*pi*zeta);

wShape = ...
    sin(2*pi*zeta + pi/6).*sin(2*pi*xi + pi/3).*sin(2*pi*eta) + ...
    cos(2*pi*eta + pi/6).*cos(2*pi*zeta + pi/3).*sin(2*pi*xi);

U = Ubar(1) + projectionFactor*a*uShape;
Vv = Ubar(2) + projectionFactor*a*vShape;
W = Ubar(3) + projectionFactor*a*wShape;

V = [U(:); Vv(:); W(:)];

end


function y = sinc_pi(x)
if x == 0
    y = 1;
else
    y = sin(pi*x)/(pi*x);
end
end
