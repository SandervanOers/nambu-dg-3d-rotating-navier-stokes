function [V, info] = project_divergence_free(Vtilde, GRAD, DIV, LAP, tol)
%PROJECT_DIVERGENCE_FREE Discrete Helmholtz projection.
%
%   V = Vtilde - GRAD*phi,
%   LAP*phi = DIV*Vtilde.
%
% LAP is singular on a periodic domain because pressure is defined only up
% to its null space.  lsqminnorm selects a minimum-norm pressure potential.
% The velocity projection itself is unique up to gradient-null modes.

if nargin < 5
    tol = 1e-12;
end

rhs = DIV*Vtilde;
before = norm(rhs, inf);

% Avoid solving a singular system when the supplied DG(0) coefficients are
% already divergence-free to roundoff.
threshold = max(tol, 100*eps*max(1,norm(Vtilde,inf)));

if before <= threshold
    phi = zeros(size(LAP,1),1);
    V = Vtilde;
else
    phi = lsqminnorm(LAP, rhs, tol);
    V = Vtilde - GRAD*phi;
end

after = norm(DIV*V, inf);

info.before = before;
info.after = after;
info.phi = phi;

end
