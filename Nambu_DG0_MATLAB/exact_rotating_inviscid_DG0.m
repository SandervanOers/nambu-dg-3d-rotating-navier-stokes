function V = exact_rotating_inviscid_DG0(Nel, t, Ro, W)
%EXACT_ROTATING_INVISCID_DG0 DG(0) L2 projection of paper Eq. (190).
%
% Periodic domain [0,1]^3, f = (0,0,1), Re = Fo = Inf.
%
%   phi = 2*pi*(x+y+z-W*t) + sqrt(3)/(3*Ro)*t
%
% The returned coefficients are exact cell averages, i.e. the DG(0)
% L2 projection Q_h v_exact rather than point samples at cell centres.

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

phi = 2*pi*(X + Y + Z - W*t) + sqrt(3)/(3*Ro)*t;

% Cell average of exp(i*2*pi*x) over one element.
projectionFactor = sinc_pi(hx) * sinc_pi(hy) * sinc_pi(hz);

U = projectionFactor/(2*pi) .* ...
    (sqrt(3)*cos(phi) + 3*sin(phi));

Vv = projectionFactor/(2*pi) .* ...
    (sqrt(3)*cos(phi) - 3*sin(phi));

Wv = -projectionFactor*sqrt(3)/pi .* cos(phi) + W;

V = [U(:); Vv(:); Wv(:)];

end


function y = sinc_pi(x)
if x == 0
    y = 1;
else
    y = sin(pi*x)/(pi*x);
end
end
