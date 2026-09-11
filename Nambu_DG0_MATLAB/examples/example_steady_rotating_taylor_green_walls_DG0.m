%% Steady rotating Taylor-Green flow with impermeable walls: DG(0)
%
% Reproduces the DG(0) version of the wall-bounded verification problem
% in Section 5.2.3 of the Nambu DGFEM paper.
%
% Domain:
%   [0,1]^3
%   impermeable walls at x = 0,1 and y = 0,1
%   periodic in z
%
% Open this file in MATLAB and press Run.

clear; clc; close all;

%% Add the MATLAB reference implementation to the path
thisFile = mfilename('fullpath');
exampleDir = fileparts(thisFile);
matlabDir = fileparts(exampleDir);
addpath(matlabDir);

%% Problem parameters
Ro = 1;

% Paper spatial-convergence endpoint.
T = 1/(2*pi);

cfg.Nel = [8 8 8];
cfg.tEnd = T;
cfg.totalSteps = 200;

cfg.Re = Inf;
cfg.Ro = Ro;
cfg.f = [0 0 1];

% Walls in x and y, periodic in z.
cfg.solidWalls = [true true false];

cfg.divergenceCorrection = 1;
cfg.nonlinear = true;

cfg.picardTol = 1e-10;
cfg.linearTol = 1e-10;
cfg.maxPicard = 25;
cfg.gmresRestart = 50;
cfg.gmresMaxOuter = 20;
cfg.projectionTol = 1e-12;
cfg.verbose = true;

cfg.exact = @(t) exact_steady_rotating_taylor_green_DG0(cfg.Nel, t);

%% Run DG(0)
result = nambu_DG0(cfg);

%% Report
fprintf('\nSteady rotating Taylor-Green walls DG(0) result\n');
fprintf('Final velocity error              : %.6e\n', result.velocityError(end));
fprintf('Maximum divergence                : %.6e\n', max(result.divergenceInf));
fprintf('Maximum energy defect             : %.6e\n', ...
    max(abs(result.energy-result.energy(1))));
fprintf('Maximum CURL*GRAD compatibility   : see solver startup diagnostic\n');
fprintf('Maximum helicity balance residual : %.6e\n', ...
    max(abs(result.helicityBalanceResidual)));
fprintf('Final cumulative helicity residual: %.6e\n', ...
    result.cumulativeHelicityResidual(end));
fprintf('Maximum Picard iterations         : %d\n', ...
    max(result.picardIterations));

%% Plots
figure;
semilogy(result.time, max(result.divergenceInf, eps), 'LineWidth', 1.5);
xlabel('t');
ylabel('||DIV V||_\infty');
title('Discrete divergence');
grid on;

figure;
semilogy(result.time, max(abs(result.energy-result.energy(1)), eps), ...
    'LineWidth', 1.5);
xlabel('t');
ylabel('|H(t)-H(0)|');
title('Kinetic-energy conservation');
grid on;

figure;
plot(result.time, result.helicity-result.helicity(1), 'LineWidth', 1.5);
xlabel('t');
ylabel('h_a(t)-h_a(0)');
title('Absolute-helicity change due to wall flux');
grid on;

figure;
semilogy(result.time(2:end), ...
    max(abs(result.helicityBalanceResidual), eps), ...
    'LineWidth', 1.5);
xlabel('t');
ylabel('|R_{h_a}|');
title('Pointwise absolute-helicity balance residual');
grid on;

figure;
semilogy(result.time, ...
    max(abs(result.cumulativeHelicityResidual), eps), ...
    'LineWidth', 1.5);
xlabel('t');
ylabel('|\hat{R}_{h_a}|');
title('Cumulative absolute-helicity balance residual');
grid on;

