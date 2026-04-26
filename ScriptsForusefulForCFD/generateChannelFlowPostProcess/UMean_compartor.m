% ======================================================================
% Utility: U+ vs y+ comparison (Planar SCFL vs Cyclic)
% ======================================================================
clear; clc; close all;

% ------------------------- USER SETTINGS -------------------------------%
matPlanar = "/shared_home/sritej/Downloads/395_withprofile_temporal_validation_CFL/postProcessing/sampleDict/0.39644/Dn1.mat";
matCyclic = "/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation/postProcessing/sampleDict/0.644/Dn1.mat";

namePlanar = "NCFL";
nameCyclic = "CFL";

uTauPlanar = 0.80;
uTauCyclic = 0.81;

nu    = 1.84e-5;
yWall = -0.00685;

yPlusMax = 10000;
lineW = 2.2;
% ----------------------------------------------------------------------%

outDir = fullfile(fileparts(matPlanar), "figs_Uplus_compare_10000DCFLNCFL");
if ~exist(outDir,"dir"), mkdir(outDir); end

% ===================== LOAD & PROCESS PLANAR ===========================
load(matPlanar,"M");

dyP    = M(:,1) - yWall;
yPlusP = (uTauPlanar .* dyP) ./ nu;
UplusP = M(:,7) ./ uTauPlanar;

keepP = yPlusP > 0 & yPlusP <= yPlusMax;
yPlusP = yPlusP(keepP);
UplusP = UplusP(keepP);

[yPlusP, idx] = sort(yPlusP);
UplusP = UplusP(idx);

% ===================== LOAD & PROCESS CYCLIC ===========================
load(matCyclic,"M");

dyC    = M(:,1) - yWall;
yPlusC = (uTauCyclic .* dyC) ./ nu;
UplusC = M(:,7) ./ uTauCyclic;

keepC = yPlusC > 0 & yPlusC <= yPlusMax;
yPlusC = yPlusC(keepC);
UplusC = UplusC(keepC);

[yPlusC, idx] = sort(yPlusC);
UplusC = UplusC(idx);

% ============================= PLOT ====================================
f = figure('Color','w'); hold on; box on;
set(gca,'XScale','log');

% Cyclic ? blue
semilogx(yPlusC, UplusC, 'b-', 'LineWidth', lineW);

% Planar SCFL ? red
semilogx(yPlusP, UplusP, 'r-', 'LineWidth', lineW);

xlabel('y^+','Interpreter','tex');
ylabel('U^+','Interpreter','tex');
title('Mean Velocity Profile Comparison (U^+ vs y^+)');

legend({nameCyclic, namePlanar}, 'Location','south');

grid on; grid minor;
xlim([1 yPlusMax]);
ylim([0 max([UplusP; UplusC])*1.05]);

set(gcf,'PaperPositionMode','auto');
print(f, fullfile(outDir,'Uplus_vs_yPlus_Cyclic_vs_PlanarSCFL_200.png'), ...
      '-dpng','-r300');
close(f);

fprintf("? Saved: %s\n", ...
    fullfile(outDir,'Uplus_vs_yPlus_Cyclic_vs_PlanarSCFL_200.png'));
