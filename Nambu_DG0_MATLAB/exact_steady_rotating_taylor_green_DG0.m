function V = exact_steady_rotating_taylor_green_DG0(Nel, ~)
%EXACT_STEADY_ROTATING_TAYLOR_GREEN_DG0
% DG(0) L2 projection of the steady rotating Taylor-Green solution
% with impermeable walls in x and y and periodicity in z.
%
% Paper Eq. (196):
%   psi = sin(pi*x) sin(pi*y)
%   u   =  pi sin(pi*x) cos(pi*y)
%   v   = -pi cos(pi*x) sin(pi*y)
%   w   =      sin(pi*x) sin(pi*y)
%
% The solution is independent of z.  Returned coefficients are exact
% element averages, not centre-point samples.

Nx = Nel(1);
Ny = Nel(2);
Nz = Nel(3);

hx = 1/Nx;
hy = 1/Ny;

x = ((1:Nx)-0.5)/Nx;
y = ((1:Ny)-0.5)/Ny;
z = ((1:Nz)-0.5)/Nz;

[X,Y,~] = ndgrid(x,y,z);

sx = sinc_half_wave(hx);
sy = sinc_half_wave(hy);
projectionFactor = sx*sy;

U = pi * projectionFactor .* sin(pi*X).*cos(pi*Y);
Vv = -pi * projectionFactor .* cos(pi*X).*sin(pi*Y);
W = projectionFactor .* sin(pi*X).*sin(pi*Y);

V = [U(:); Vv(:); W(:)];

end


function y = sinc_half_wave(h)
a = pi*h/2;
if a == 0
    y = 1;
else
    y = sin(a)/a;
end
end
