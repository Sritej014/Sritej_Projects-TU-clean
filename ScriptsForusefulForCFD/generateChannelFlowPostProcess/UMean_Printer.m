% ========= Mean Velocity Profile ? wall-layer demarcation (no arrows) ====
clear; clc; close all;

% ---------------- Load data ----------------
matIn = "/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation/postProcessing/sampleDict/0.644/Dn1.mat";
outDir = fullfile(fileparts(matIn), "figs_yPlus1000_UMean_Dn");
if ~exist(outDir,"dir"), mkdir(outDir); end

load(matIn,"M");

% ---------------- Compute y+ and U+ ----------------
yWall = -0.00685;
nu    = 1.84e-5;
uTau  = 0.802;

dy    = M(:,1) - yWall;
yPlus = (uTau .* dy) ./ nu;

keep = yPlus > 0 & yPlus <= 10000;
yPlus = yPlus(keep);
Uplus = M(keep,7) ./ uTau;

[yPlus, idx] = sort(yPlus);
Uplus = Uplus(idx);
yPlus = max(yPlus,1e-8);

% ======================================================================
% Detect log region via slope plateau
% ======================================================================
dU   = gradient(Uplus);
dlny = gradient(log(yPlus));
slope = smoothdata(dU ./ dlny,'sgolay',11);

kappa_guess = 0.41;
logMask = abs(slope - 1/kappa_guess) < 0.35 & yPlus > 30;
logIdx  = contiguousBlock(logMask,12);

yl = yPlus(logIdx);
Ul = Uplus(logIdx);

% Log-law fit
p_log = polyfit(log(yl), Ul, 1);
kappa = 1/p_log(1);
B     = p_log(2);

% ======================================================================
% Viscous sublayer = intersection of U+ with u+=y+
% ======================================================================
f_int  = @(y) interp1(yPlus,Uplus,y,'linear','extrap') - y;
y_visc = fzero(f_int,10);
u_visc = y_visc;

% ======================================================================
% Plot
% ======================================================================
f = figure('Color','w'); hold on;
set(gca,'XScale','log');

% Mean profile
semilogx(yPlus, Uplus,'b','LineWidth',2);

% Viscous law u+=y+
yV = yPlus(yPlus <= yl(1));
semilogx(yV, yV,'r--','LineWidth',2);

% Log-law (dotted + extended to viscous limit)
% --- Log-law (dotted, straight across entire right side) ---
ylog_ext = logspace(log10(y_visc), log10(max(yPlus)), 500);
ulog_ext = (1/kappa)*log(ylog_ext) + B;

semilogx(ylog_ext, ulog_ext, 'k:', 'LineWidth', 2.8);


% Intersection marker
semilogx(y_visc, u_visc,'ro','MarkerFaceColor','r','MarkerSize',8);

% ======================================================================
% Shaded regions (textbook-consistent)
% ======================================================================
ymax = max(Uplus)*1.05;

% Viscous sublayer
% Log region (capped at y+ = 300)
yLogEnd = min(300, max(yPlus));

patch([yl(1) yLogEnd yLogEnd yl(1)], ...
      [0 0 ymax ymax],[0.85 0.9 1.0], ...
      'FaceAlpha',0.25,'EdgeColor','none');


% Buffer layer
patch([y_visc yl(1) yl(1) y_visc], ...
      [0 0 ymax ymax],[1 1 0.75], ...
      'FaceAlpha',0.25,'EdgeColor','none');

% Log region
patch([yl(1) yl(end) yl(end) yl(1)], ...
      [0 0 ymax ymax],[0.85 0.9 1.0], ...
      'FaceAlpha',0.25,'EdgeColor','none');

% ======================================================================
% Dotted demarcation lines ONLY (no arrows)
% ======================================================================
xline(y_visc,'r--','LineWidth',1.5);   % viscous limit
xline(yl(1),'k:','LineWidth',1.5);     % log-region start
xline(yLogEnd,'k:','LineWidth',1.5);   % log-region end at y+ = 300


% ======================================================================
% Text annotations
% ======================================================================


% ======================================================================
% Formatting & save
% ======================================================================
xlabel('y^+');
ylabel('u^+');
title('Mean Velocity Profile(Non-Dimensionalised)');
grid on; box on;
xlim([min(yPlus) max(yPlus)]);
ylim([0 ymax]);

legend({'U^+','u^+=y^+','Log-Law'}, ...
       'Location','south');

set(gcf,'PaperPositionMode','auto');
print(f, fullfile(outDir,'Uplus_wall_regions_no_arrows.png'), '-dpng','-r300');
close(f);

fprintf('? Saved: %s\n', fullfile(outDir,'Uplus_wall_regions_no_arrows.png'));

% ======================================================================
% Helper
% ======================================================================
function idx = contiguousBlock(mask,minLen)
    d = diff([0; mask(:); 0]);
    s = find(d==1); e = find(d==-1)-1;
    if isempty(s), idx=[]; return; end
    len = e-s+1;
    k = find(len>=minLen,1,'first');
    if isempty(k), idx=[]; else, idx=s(k):e(k); end
end
