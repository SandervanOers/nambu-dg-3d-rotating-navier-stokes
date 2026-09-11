%% Propagating rotating inviscid flow: DG(0)
%
% Reproduces the DG(0) version of the periodic verification problem in
% Section 5.2.1 of the Nambu DGFEM paper.
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
W = 2;

% One period of the exact solution.
T = 2*pi / abs(2*pi*W - sqrt(3)/(3*Ro));

cfg.Nel = [8 8 8];
cfg.period = T;
cfg.nPeriods = 1;
cfg.nSteps = 200;   % steps per period

cfg.Re = Inf;
cfg.Ro = Ro;
cfg.f = [0 0 1];

cfg.solidWalls = [false false false];

cfg.divergenceCorrection = 1;
cfg.nonlinear = true;

cfg.picardTol = 1e-10;
cfg.linearTol = 1e-10;
cfg.maxPicard = 25;
cfg.gmresRestart = 50;
cfg.gmresMaxOuter = 20;
cfg.projectionTol = 1e-12;
cfg.verbose = true;

cfg.exact = @(t) exact_rotating_inviscid_DG0(cfg.Nel, t, Ro, W);

%% Run DG(0)
result = nambu_DG0(cfg);

%% Report
fprintf('\nRotating inviscid DG(0) result\n');
fprintf('Final velocity error       : %.6e\n', result.velocityError(end));
fprintf('Maximum divergence         : %.6e\n', max(result.divergenceInf));
fprintf('Maximum energy defect      : %.6e\n', ...
    max(abs(result.energy - result.energy(1))));
fprintf('Maximum helicity defect    : %.6e\n', ...
    max(abs(result.helicity - result.helicity(1))));
fprintf('Maximum Picard iterations  : %d\n', max(result.picardIterations));

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
semilogy(result.time, max(abs(result.helicity-result.helicity(1)), eps), ...
    'LineWidth', 1.5);
xlabel('t');
ylabel('|h_a(t)-h_a(0)|');
title('Absolute-helicity conservation');
grid on;

