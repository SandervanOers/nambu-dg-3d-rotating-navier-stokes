%% Decaying viscous flow: DG(0)
%
% Reproduces the DG(0) version of the periodic verification problem in
% Section 5.2.2 of the Nambu DGFEM paper.
%
% Open this file in MATLAB and press Run.

clear; clc; close all;

%% Add the MATLAB reference implementation to the path
thisFile = mfilename('fullpath');
exampleDir = fileparts(thisFile);
matlabDir = fileparts(exampleDir);
addpath(matlabDir);

%% Problem parameters
Re = 1e3;
Ubar = [1/2, sqrt(2)/2, sqrt(3)/2];

% Paper definition: viscous e-folding time of the velocity amplitude.
tau_d = Re/(12*pi^2);

% Spatial-convergence verification time used in the paper.
T = 0.1*tau_d;

cfg.Nel = [8 8 8];
cfg.tEnd = T;
cfg.totalSteps = 200;

cfg.Re = Re;
cfg.Ro = Inf;
cfg.f = [0 0 0];

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

cfg.exact = @(t) exact_decaying_viscous_DG0(cfg.Nel, t, Re, Ubar);

%% Run DG(0)
result = nambu_DG0(cfg);

%% Report
fprintf('\nDecaying viscous DG(0) result\n');
fprintf('Final time / tau_d         : %.3f\n', result.time(end)/tau_d);
fprintf('Final velocity error       : %.6e\n', result.velocityError(end));
fprintf('Maximum divergence         : %.6e\n', max(result.divergenceInf));
fprintf('Maximum Picard iterations  : %d\n', max(result.picardIterations));

%% Plots
figure;
semilogy(result.time/tau_d, max(result.divergenceInf, eps), ...
    'LineWidth', 1.5);
xlabel('t / \tau_d');
ylabel('||DIV V||_\infty');
title('Discrete divergence');
grid on;

figure;
plot(result.time/tau_d, result.energy, 'LineWidth', 1.5);
xlabel('t / \tau_d');
ylabel('H');
title('Kinetic-energy decay');
grid on;

figure;
plot(result.time/tau_d, result.helicity, 'LineWidth', 1.5);
xlabel('t / \tau_d');
ylabel('h_a');
title('Absolute-helicity decay');
grid on;

