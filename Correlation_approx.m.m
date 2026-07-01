clear; clc; close all;

xi = [0.01, 0.1, 0.78, 1.57, 2.36, 3.0];

% Measured — 6R
PR_OLF_6R = [0.000714,0.003580357,0.00472449,0.000653061,0.00016,0.000157];
PR_BCE_6R = [0.005595238,0.029465909,0.0119520408,0.001163265,0.000199,0.000167];

% Measured — 4R
PR_OLF_4R = [0.00300603, 0.00889683, 0.0167273, 0.00172727, 0.000571429, 0.00038889];
PR_BCE_4R = [0.0136032, 0.0501028, 0.0239091, 0.00263636, 0.000734127, 0.00044444];

% Parameters
R6 = 6; R4 = 4;
l = 10; K = 1.58; f_D = 0.01;
delta = l;

% Single-resource outage
P1_6R = [0.120821,0.114648,0.0267831,0.00126701,0.000242,0.000211];
P1_4R = [0.170055556, 0.161847222, 0.027909091, 0.003181818, 0.000734127, 0.000481481];

sinc_u = @(x) (x==0) + (x~=0).*sin(x)./x;

%% Compute approximations
cterm = cos(2*pi*f_D*delta);
approx_sinc_6R = zeros(1,length(xi));
approx_j0_6R   = zeros(1,length(xi));
approx_sinc_4R = zeros(1,length(xi));
approx_j0_4R   = zeros(1,length(xi));

for kidx = 1:length(xi)
    varphi_s = sinc_u(xi(kidx) * delta);
    f_s = (K^2 + varphi_s^2 + 2*K*varphi_s*cterm) / (K+1)^2;

    varphi_j = besselj(0, xi(kidx) * delta);
    f_j = (K^2 + varphi_j^2 + 2*K*varphi_j*cterm) / (K+1)^2;

    approx_sinc_6R(kidx) = (P1_6R(kidx)^R6)*f_s + P1_6R(kidx)*(1 - f_s);
    approx_j0_6R(kidx)   = (P1_6R(kidx)^R6)*f_j + P1_6R(kidx)*(1 - f_j);

    approx_sinc_4R(kidx) = (P1_4R(kidx)^R4)*f_s + P1_4R(kidx)*(1 - f_s);
    approx_j0_4R(kidx)   = (P1_4R(kidx)^R4)*f_j + P1_4R(kidx)*(1 - f_j);
end

%% Colors
c6 = [0.00, 0.00, 1.00];   % blue — 6R
c4 = [0.85, 0.10, 0.10];   % red  — 4R

ms = 12;    % marker size
lw = 1.2;   % line width

%% Plot
figure; hold on; grid on;
set(gca, 'YScale', 'log', 'TickLabelInterpreter', 'latex', 'FontSize', 24);
xlim([0 3.0]);

% 6R measured
semilogy(xi, PR_OLF_6R, 's-', 'Color', c6, 'LineWidth', lw, ...
    'MarkerFaceColor', c6, 'MarkerSize', ms, 'HandleVisibility', 'off');
semilogy(xi, PR_BCE_6R, 'p-', 'Color', c6, 'LineWidth', lw, ...
    'MarkerFaceColor', c6, 'MarkerSize', ms, 'HandleVisibility', 'off');

% 4R measured
semilogy(xi, PR_OLF_4R, 's-', 'Color', c4, 'LineWidth', lw, ...
    'MarkerFaceColor', c4, 'MarkerSize', ms, 'HandleVisibility', 'off');
semilogy(xi, PR_BCE_4R, 'p-', 'Color', c4, 'LineWidth', lw, ...
    'MarkerFaceColor', c4, 'MarkerSize', ms, 'HandleVisibility', 'off');

% 6R approximations (no markers)
semilogy(xi, approx_j0_6R,   '-',  'Color', c6, 'LineWidth', lw, 'HandleVisibility', 'off');
semilogy(xi, approx_sinc_6R, '-.', 'Color', c6, 'LineWidth', lw, 'HandleVisibility', 'off');

% 4R approximations (no markers)
semilogy(xi, approx_j0_4R,   '-',  'Color', c4, 'LineWidth', lw, 'HandleVisibility', 'off');
semilogy(xi, approx_sinc_4R, '-.', 'Color', c4, 'LineWidth', lw, 'HandleVisibility', 'off');

xlabel('Phase shift $\xi$',  'Interpreter', 'latex', 'FontSize', 24);
ylabel('Outage probability', 'Interpreter', 'latex', 'FontSize', 24);

%% Legend dummies
% Measured (black dummies — colour conveyed by colour entries below)
hO  = semilogy(nan, nan, 'ks-', 'LineWidth', lw, 'MarkerFaceColor', 'k', 'MarkerSize', ms);
hB  = semilogy(nan, nan, 'kp-', 'LineWidth', lw, 'MarkerFaceColor', 'k', 'MarkerSize', ms);

% Approximations — one pair only for line style
hJ  = semilogy(nan, nan, 'k-',  'LineWidth', lw);
hS  = semilogy(nan, nan, 'k-.', 'LineWidth', lw);

% Colour identity dummies
h6  = semilogy(nan, nan, '-', 'Color', c6, 'LineWidth', 4.0);
h4  = semilogy(nan, nan, '-', 'Color', c4, 'LineWidth', 4.0);

legend([hB, hO, hJ, hS, h6, h4], ...
    { ...
    '$P_{|\mathsf{R}|}(\mathsf{\gamma_{th}}, \mathtt{q_{th}},\xi)$ : using BCE', ...
    '$P_{|\mathsf{R}|}(\mathsf{\gamma_{th}}, \mathtt{q_{th}},\xi)$ : using OLF', ...
    '$\tilde{P}_{|\mathsf{R}|}(\mathsf{\gamma_{th}}, \mathtt{q_{th}},\xi)$ using (14)', ...
    '$\tilde{P}_{|\mathsf{R}|}(\mathsf{\gamma_{th}}, \mathtt{q_{th}},\xi)$ using (15)', ...
    '$\mathsf{R} = 6$', ...
    '$\mathsf{R} = 4$' ...
    }, ...
    'Location', 'southwest', 'Interpreter', 'latex', ...
    'FontSize', 19.8, 'Box', 'on');
