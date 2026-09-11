function result = nambu_DG0(cfg)
%NAMBU_DG0 Compact MATLAB reference solver for the DG(0) Nambu method.
%
%   result = nambu_DG0(cfg)
%
% Required cfg fields
%   Nel        [Nx Ny Nz]
%   Re         Reynolds number (Inf for inviscid)
%   Ro         Rossby number (Inf for no background rotation)
%   f          1x3 background rotation vector
%   exact      function handle V = exact(t), returning 3*N DG coefficients
%
% Time integration: choose ONE of the following modes.
%
%   Period mode:
%     period       reference period
%     nPeriods     number of periods to simulate
%     nSteps       number of time steps PER period
%
%   Direct-time mode:
%     tEnd         final time
%     totalSteps   total number of time steps
%
% Optional cfg fields
%   solidWalls             [Sx Sy Sz], default [0 0 0]
%                          0 = periodic, 1 = impermeable walls
%   divergenceCorrection   a in the paper, default 1
%   nonlinear              true/false, default true
%   picardTol              default 1e-10
%   maxPicard              default 25
%   linearTol              default 1e-11
%   gmresRestart           default 50
%   gmresMaxOuter          default 20
%   projectionTol          default 1e-12
%   verbose                default true
%
% Scope
%   - Cartesian [0,1]^3
%   - centred theta = 1/2
%   - DG(0)
%   - periodic and coordinate-aligned impermeable-wall directions
%   - no body forcing
%   - viscosity supported for periodic examples
%
% For wall-bounded inviscid examples the discrete absolute-helicity wall
% contribution of Eq. (157) is also evaluated.

cfg = set_defaults(cfg);

Nel = cfg.Nel(:).';
if numel(Nel) ~= 3
    error('cfg.Nel must contain [Nx Ny Nz].');
end

ops = build_operators_DG0(Nel, cfg.solidWalls);
N = ops.N;

[dt, tEnd, totalSteps, timeMode] = resolve_time_grid(cfg);

if isinf(cfg.Re)
    nu = 0;
else
    nu = 1/cfg.Re;
end

if isinf(cfg.Ro)
    fOverRo = zeros(3*N,1);
else
    fOverRo = [cfg.f(1)*ones(N,1); ...
               cfg.f(2)*ones(N,1); ...
               cfg.f(3)*ones(N,1)] / cfg.Ro;
end

Ivel = speye(3*N);
LAP3 = kron(speye(3), ops.LAP);

% Initial DG coefficients followed by the discrete Helmholtz projection.
Vtilde = cfg.exact(0);
if numel(Vtilde) ~= 3*N
    error('cfg.exact(t) must return a column vector with 3*N entries.');
end
Vtilde = Vtilde(:);

[V, projectionInfo] = project_divergence_free( ...
    Vtilde, ops.GRAD, ops.DIV, ops.LAP, cfg.projectionTol);

P = zeros(N,1);

time = linspace(0, tEnd, totalSteps+1).';

energy = zeros(totalSteps+1,1);
helicity = zeros(totalSteps+1,1);
divergenceInf = zeros(totalSteps+1,1);
velocityError = zeros(totalSteps+1,1);

picardIterations = zeros(totalSteps,1);
nonlinearResidual = zeros(totalSteps,1);
relativeNonlinearResidual = zeros(totalSteps,1);
linearFlag = zeros(totalSteps,1);
linearIterations = zeros(totalSteps,1);

boundaryHelicityFlux = zeros(totalSteps,1);
helicityBalanceResidual = zeros(totalSteps,1);
cumulativeHelicityResidual = zeros(totalSteps+1,1);

[energy(1), helicity(1)] = invariants( ...
    V, ops.CURL, fOverRo, ops.volume);
divergenceInf(1) = norm(ops.DIV*V, inf);

Vexact = projected_exact(cfg.exact, 0, ops, cfg.projectionTol);
velocityError(1) = sqrt(ops.volume * sum((V - Vexact).^2));

if cfg.verbose
    fprintf('Nambu DG(0), centred Cartesian discretization\n');
    fprintf('Mesh                 : %d x %d x %d\n', ...
        Nel(1), Nel(2), Nel(3));
    fprintf('Time step            : %.6e\n', dt);
    if strcmp(timeMode,'period')
        fprintf('Reference period     : %.6e\n', cfg.period);
        fprintf('Number of periods    : %d\n', cfg.nPeriods);
        fprintf('Steps per period     : %d\n', cfg.nSteps);
    else
        fprintf('Final time           : %.6e\n', tEnd);
        fprintf('Total time steps     : %d\n', totalSteps);
    end
    fprintf('Solid walls [x y z]  : [%d %d %d]\n', cfg.solidWalls);
    fprintf('Initial DIV error    : %.3e\n', divergenceInf(1));
    fprintf('CURL*GRAD defect     : %.3e\n', ...
        norm(ops.CURL*ops.GRAD, inf));
end

for n = 1:totalSteps
    Vn = V;
    Vk = V;
    Pk = P;

    absResidual = Inf;
    relResidual = Inf;
    lastFlag = 0;
    lastLinearIterations = 0;

    for k = 1:cfg.maxPicard
        [A, b] = assemble_system( ...
            Vn, Vk, ops, Ivel, LAP3, dt, nu, fOverRo, cfg);

        x0 = [Vk; Pk];

        [x, flag, ~, iter] = gmres( ...
            A, b, cfg.gmresRestart, cfg.linearTol, ...
            cfg.gmresMaxOuter, [], [], x0);

        lastFlag = flag;
        lastLinearIterations = gmres_iteration_count( ...
            iter, cfg.gmresRestart);

        % The pressure operator is singular because pressure is defined
        % only up to its null space.  The coupled right-hand side is
        % consistent.  Use a minimum-norm fallback only if GMRES reports
        % non-convergence.
        if flag ~= 0
            x = lsqminnorm(A, b, cfg.linearTol);
        end

        Vnew = x(1:3*N);
        Pnew = x(3*N+1:end);
        Pnew = Pnew - mean(Pnew);

        % Evaluate the nonlinear algebraic residual with the newly
        % computed velocity.
        [Ares, bres] = assemble_system( ...
            Vn, Vnew, ops, Ivel, LAP3, dt, nu, fOverRo, cfg);

        r = Ares*[Vnew; Pnew] - bres;
        absResidual = norm(r);
        relResidual = absResidual / max(norm(bres), eps);

        Vk = Vnew;
        Pk = Pnew;

        if absResidual <= cfg.picardTol || relResidual <= cfg.picardTol
            break;
        end
    end

    if k == cfg.maxPicard && ...
            absResidual > cfg.picardTol && relResidual > cfg.picardTol
        warning('nambu_DG0:Picard', ...
            ['Picard iteration reached maxPicard at step %d ' ...
             '(relative residual %.3e).'], ...
            n, relResidual);
    end

    V = Vnew;
    P = Pnew;

    [energy(n+1), helicity(n+1)] = invariants( ...
        V, ops.CURL, fOverRo, ops.volume);

    divergenceInf(n+1) = norm(ops.DIV*V, inf);

    Vexact = projected_exact( ...
        cfg.exact, time(n+1), ops, cfg.projectionTol);

    velocityError(n+1) = sqrt( ...
        ops.volume * sum((V - Vexact).^2));

    % Wall-bounded inviscid absolute-helicity balance, Eq. (157).
    if any(cfg.solidWalls) && nu == 0
        Vmid = 0.5*(Vn + V);

        ZaMid = fOverRo;
        if cfg.nonlinear
            ZaMid = ZaMid + ops.CURL*Vmid;
        end

        Nmid = cross_right_matrix(ZaMid, N)*Vmid;

        % Eq. (157), with Re = Fo = Inf:
        %   B_ha = -(1/Ro) int P* n.f dS
        %          - 1/2 b_boundary(Nmid + GRAD P*, Vmid)
        Q = Nmid + ops.GRAD*P;

        boundaryHelicityFlux(n) = wall_helicity_flux( ...
            Q, Vmid, P, cfg, ops);

        helicityBalanceResidual(n) = ...
            (helicity(n+1)-helicity(n))/dt ...
            - boundaryHelicityFlux(n);

        cumulativeHelicityResidual(n+1) = ...
            helicity(n+1)-helicity(1) ...
            - dt*sum(boundaryHelicityFlux(1:n));
    end

    picardIterations(n) = k;
    nonlinearResidual(n) = absResidual;
    relativeNonlinearResidual(n) = relResidual;
    linearFlag(n) = lastFlag;
    linearIterations(n) = lastLinearIterations;

    if cfg.verbose && ...
            (n == 1 || mod(n, max(1,ceil(totalSteps/10))) == 0)

        fprintf(['step %4d/%4d, t = %.4e, DIV = %.3e, ' ...
                 'relR_nl = %.3e, Picard = %d\n'], ...
                 n, totalSteps, time(n+1), ...
                 divergenceInf(n+1), relResidual, k);
    end
end

result.U = V(1:N);
result.V = V(N+1:2*N);
result.W = V(2*N+1:3*N);
result.velocity = V;
result.P = P;

result.time = time;
result.dt = dt;
result.tEnd = tEnd;
result.totalSteps = totalSteps;
result.timeMode = timeMode;

if strcmp(timeMode,'period')
    result.period = cfg.period;
    result.nPeriods = cfg.nPeriods;
    result.stepsPerPeriod = cfg.nSteps;
end

result.energy = energy;
result.helicity = helicity;
result.divergenceInf = divergenceInf;
result.velocityError = velocityError;

result.picardIterations = picardIterations;
result.nonlinearResidual = nonlinearResidual;
result.relativeNonlinearResidual = relativeNonlinearResidual;
result.linearFlag = linearFlag;
result.linearIterations = linearIterations;

result.boundaryHelicityFlux = boundaryHelicityFlux;
result.helicityBalanceResidual = helicityBalanceResidual;
result.cumulativeHelicityResidual = cumulativeHelicityResidual;

result.initialProjection = projectionInfo;
result.Nel = Nel;
result.Re = cfg.Re;
result.Ro = cfg.Ro;
result.f = cfg.f;
result.solidWalls = cfg.solidWalls;
result.divergenceCorrection = cfg.divergenceCorrection;

end


function [A, b] = assemble_system( ...
        Vn, Vk, ops, Ivel, LAP3, dt, nu, fOverRo, cfg)
% Picard linearization of the implicit-midpoint equation.
%
% At Picard iterate Vk,
%
%   omegaHalf = f/(2 Ro) + 1/4 CURL(Vk + Vn)
%
% and C*V = V x omegaHalf.  Therefore
%
%   C*(V^{n+1}+V^n)
%
% is the Picard-linearized midpoint rotational term.

N = ops.N;

omegaHalf = 0.5*fOverRo;

if cfg.nonlinear
    omegaHalf = omegaHalf + 0.25*ops.CURL*(Vk + Vn);
end

C = cross_right_matrix(omegaHalf, N);

Lnonlinear = -dt*C;
Lviscous = -0.5*dt*nu*LAP3;
L = Lnonlinear + Lviscous;

rhsV = Vn ...
     + dt*C*Vn ...
     + 0.5*dt*nu*(LAP3*Vn);

Avel = Ivel + L;

targetDiv = ...
    (1-cfg.divergenceCorrection) * (ops.DIV*Vn);

A = [Avel,      dt*ops.GRAD; ...
     ops.DIV*L, dt*ops.LAP];

b = [rhsV; ...
     ops.DIV*rhsV - targetDiv];

end


function C = cross_right_matrix(z, N)
% Sparse matrix C such that C*v = v x z.

zx = z(1:N);
zy = z(N+1:2*N);
zz = z(2*N+1:3*N);

Z = sparse(N,N);
Dx = spdiags(zx,0,N,N);
Dy = spdiags(zy,0,N,N);
Dz = spdiags(zz,0,N,N);

C = [ Z,   Dz, -Dy; ...
     -Dz,    Z,  Dx; ...
      Dy,  -Dx,   Z];

end


function [H, ha] = invariants(V, CURL, fOverRo, volume)

H = 0.5 * volume * sum(V.^2);

ha = volume * sum( ...
    fOverRo.*V + 0.5*V.*(CURL*V));

end


function Vex = projected_exact( ...
        exactHandle, t, ops, projectionTol)

Vex0 = exactHandle(t);

[Vex, ~] = project_divergence_free( ...
    Vex0(:), ops.GRAD, ops.DIV, ...
    ops.LAP, projectionTol);

end


function Bha = wall_helicity_flux(Q, Vmid, P, cfg, ops)
% Discrete boundary contribution in Eq. (157), specialized to the
% inviscid, unforced reference problem:
%
%   B_ha = -(1/Ro) sum_e int_e P* n.f dGamma
%          - 1/2 b_boundary(Q,Vmid),
%
% where
%
%   b_boundary(Q,V) = sum_e int_e (n x V).Q dGamma.

faces = ops.boundaryFaces;
N = ops.N;

if isempty(faces.cell)
    Bha = 0;
    return;
end

Qx = Q(1:N);
Qy = Q(N+1:2*N);
Qz = Q(2*N+1:3*N);

Vx = Vmid(1:N);
Vy = Vmid(N+1:2*N);
Vz = Vmid(2*N+1:3*N);

idx = faces.cell;
nrm = faces.normal;
area = faces.area;

q = [Qx(idx), Qy(idx), Qz(idx)];
v = [Vx(idx), Vy(idx), Vz(idx)];

nxv = cross(nrm, v, 2);

bBoundary = sum( ...
    area .* sum(nxv .* q, 2));

pressureBackground = 0;

if ~isinf(cfg.Ro)
    ndotf = nrm * cfg.f(:);

    pressureBackground = ...
        (1/cfg.Ro) * sum( ...
        area .* P(idx) .* ndotf);
end

Bha = -pressureBackground - 0.5*bBoundary;

end


function count = gmres_iteration_count(iter, restart)

if isempty(iter)
    count = 0;
elseif numel(iter) == 1
    count = iter;
else
    count = (iter(1)-1)*restart + iter(2);
end

end


function [dt, tEnd, totalSteps, timeMode] = resolve_time_grid(cfg)
%RESOLVE_TIME_GRID Resolve the two supported time-integration interfaces.

hasPeriodMode = isfield(cfg,'period') || isfield(cfg,'nPeriods');
hasDirectMode = isfield(cfg,'tEnd') || isfield(cfg,'totalSteps');

if hasPeriodMode && hasDirectMode
    error(['Specify either period/nPeriods/nSteps OR ' ...
           'tEnd/totalSteps, not both.']);
end

if hasPeriodMode
    required = {'period','nPeriods','nSteps'};
    for i = 1:numel(required)
        if ~isfield(cfg, required{i})
            error('Period mode requires cfg.%s.', required{i});
        end
    end

    if cfg.period <= 0
        error('cfg.period must be positive.');
    end
    if cfg.nPeriods < 1 || cfg.nPeriods ~= round(cfg.nPeriods)
        error('cfg.nPeriods must be a positive integer.');
    end
    if cfg.nSteps < 1 || cfg.nSteps ~= round(cfg.nSteps)
        error('cfg.nSteps must be a positive integer (steps per period).');
    end

    dt = cfg.period / cfg.nSteps;
    totalSteps = cfg.nPeriods * cfg.nSteps;
    tEnd = cfg.nPeriods * cfg.period;
    timeMode = 'period';

elseif hasDirectMode
    required = {'tEnd','totalSteps'};
    for i = 1:numel(required)
        if ~isfield(cfg, required{i})
            error('Direct-time mode requires cfg.%s.', required{i});
        end
    end

    if cfg.tEnd <= 0
        error('cfg.tEnd must be positive.');
    end
    if cfg.totalSteps < 1 || cfg.totalSteps ~= round(cfg.totalSteps)
        error('cfg.totalSteps must be a positive integer.');
    end

    tEnd = cfg.tEnd;
    totalSteps = cfg.totalSteps;
    dt = tEnd / totalSteps;
    timeMode = 'direct';

else
    error(['No time integration mode specified. Use either ' ...
           'cfg.period + cfg.nPeriods + cfg.nSteps, or ' ...
           'cfg.tEnd + cfg.totalSteps.']);
end

end


function cfg = set_defaults(cfg)

required = {'Nel','Re','Ro','f','exact'};

for i = 1:numel(required)
    if ~isfield(cfg, required{i})
        error('Missing required cfg field: %s', required{i});
    end
end

defaults.solidWalls = [false false false];
defaults.divergenceCorrection = 1;
defaults.nonlinear = true;
defaults.picardTol = 1e-10;
defaults.maxPicard = 25;
defaults.linearTol = 1e-10;
defaults.gmresRestart = 50;
defaults.gmresMaxOuter = 20;
defaults.projectionTol = 1e-12;
defaults.verbose = true;

names = fieldnames(defaults);

for i = 1:numel(names)
    if ~isfield(cfg, names{i})
        cfg.(names{i}) = defaults.(names{i});
    end
end

if numel(cfg.solidWalls) ~= 3
    error('solidWalls must contain [Sx Sy Sz].');
end

cfg.solidWalls = logical(cfg.solidWalls(:).');

if cfg.divergenceCorrection < 0 || ...
        cfg.divergenceCorrection >= 2

    error(['divergenceCorrection must satisfy ' ...
           '0 <= a < 2.']);
end

end
