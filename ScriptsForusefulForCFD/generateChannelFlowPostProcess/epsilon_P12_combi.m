% ===================== USER SETTINGS ======================
matInCyclic       = "/WORK/sritej/TASKFILES/Task1/FOR_REUSCH_WORK/395_withprofile_20ms/postProcessing/sampleDict/0.42966101/Mid1.mat";
matInMappedCFL   = "/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/sampleDict/0.39644/Up1.mat";
matInMappedNonCFL = "/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation/postProcessing/sampleDict/0.644/Up1.mat";

% Friction velocities for y+ computation (per dataset)
uTau_yplus_MappedCFL    = 0.805;   % Mapped CFL
uTau_yplus_Cyclic       = 0.810;   % Cyclic
uTau_yplus_MappedNonCFL = 0.800;   % Mapped Non-CFL

% Flow constants
nu    = 1.84e-5;
yWall = -0.00685;
yPlusMax = 80;

% Index mapping
idx.kResolved        = 3;
idx.kModel           = 6;
idx.epsilonResolved  = 2;
idx.epsilonModel     = 5;
idx.UPrime2Mean      = 10:15;
idx.gradU            = 22:30;

% Output directory
outDir = "./figs_threeCases";
if ~exist(outDir,"dir"), mkdir(outDir); end
% ===========================================================


%% =============== PROCESS THREE DATASETS ===================
mustExist(matInMappedCFL);
mustExist(matInCyclic);
mustExist(matInMappedNonCFL);

aux_mapCFL = process_case_for_eps_P12(matInMappedCFL,    idx, uTau_yplus_MappedCFL,    nu, yWall, yPlusMax);
aux_cyc    = process_case_for_eps_P12(matInCyclic,       idx, uTau_yplus_Cyclic,       nu, yWall, yPlusMax);
aux_mapNon = process_case_for_eps_P12(matInMappedNonCFL, idx, uTau_yplus_MappedNonCFL, nu, yWall, yPlusMax);

% Pick reference y+ grid (densest dataset)
[~, refIdx] = max([numel(aux_mapCFL.yp), numel(aux_cyc.yp), numel(aux_mapNon.yp)]);
cands = {aux_mapCFL.yp, aux_cyc.yp, aux_mapNon.yp};   % <-- fix: create cell first
yp_ref = cands{refIdx};                               % then index the cell

% Ensure column vector
yp_ref = yp_ref(:);

% Interpolate each dataset onto the reference grid
eps_mapCFL = interp1(aux_mapCFL.yp, aux_mapCFL.epsTot, yp_ref, 'linear','extrap');
eps_cyc    = interp1(aux_cyc.yp,    aux_cyc.epsTot,    yp_ref, 'linear','extrap');
eps_mapNon = interp1(aux_mapNon.yp, aux_mapNon.epsTot, yp_ref, 'linear','extrap');

P12_mapCFL = interp1(aux_mapCFL.yp, aux_mapCFL.P12, yp_ref, 'linear','extrap');
P12_cyc    = interp1(aux_cyc.yp,    aux_cyc.P12,    yp_ref, 'linear','extrap');
P12_mapNon = interp1(aux_mapNon.yp, aux_mapNon.P12, yp_ref, 'linear','extrap');

% Save plots (Cyclic = dashed blue; Mapped CFL = solid red; Mapped Non-CFL = solid orange)
save_three_lines(yp_ref, eps_cyc, eps_mapCFL, eps_mapNon, ...
    'epsilonTotal', 'epsilonTotal_threeCases', outDir);
save_three_lines(yp_ref, P12_cyc, P12_mapCFL, P12_mapNon, ...
    'P_{12}', 'P12_threeCases', outDir);

fprintf("? Combined plots written to %s\n", outDir);


%% ===================== HELPER FUNCTIONS ==================
function mustExist(p)
    if ~isfile(p)
        error('File not found: %s', p);
    end
end

function aux = process_case_for_eps_P12(matPath, idx, uTau_yplus, nu, yWall, yPlusMax)
    S = load(matPath,"M"); M = S.M;

    % epsilonTotal
    epsilonTotal = M(:,idx.epsilonResolved) + M(:,idx.epsilonModel);

    % grad(U) and UPrime2Mean
    g = M(:,idx.gradU);       % 9 columns
    p = M(:,idx.UPrime2Mean); % 6 columns

    % P12 term (consistent with your mapping: -g(:,4).*p(:,2))
    P12 = - g(:,4) .* p(:,2);

    % yPlus using dataset-specific uTau
    dy    = M(:,1) - yWall;
    yPlus = uTau_yplus .* dy ./ nu;

    keep       = (yPlus <= yPlusMax);
    aux.yp     = yPlus(keep);
    aux.epsTot = epsilonTotal(keep);
    aux.P12    = P12(keep);

    % Ensure column vectors for safety
    aux.yp     = aux.yp(:);
    aux.epsTot = aux.epsTot(:);
    aux.P12    = aux.P12(:);
end

function save_three_lines(yp, Y_cyc, Y_mapCFL, Y_mapNon, yLabel, tag, outDir)
    f = figure('Visible','off'); hold on; box on;

    % Colors (fixed palette)
    colCyc    = [0 0.4470 0.7410];   % blue
    colMapCFL = [0.85 0.10 0.10];    % red
    colMapNon = [0.95 0.60 0.10];    % orange

    % Enforce column vectors
    yp       = yp(:);
    Y_mapCFL = Y_mapCFL(:);
    Y_mapNon = Y_mapNon(:);
    Y_cyc    = Y_cyc(:);

    % Plot datasets: Mapped solid; Cyclic dashed
    plot(yp, Y_mapCFL,  '-',  'LineWidth',1.8, 'Color', colMapCFL);
    plot(yp, Y_mapNon,  '-',  'LineWidth',1.8, 'Color', colMapNon);
    plot(yp, Y_cyc,    '-',  'LineWidth',1.8, 'Color', colCyc);

    legend({'Mapped CFL','Mapped Non-CFL','Cyclic (dashed)'}, ...
           'Location','northeast','Interpreter','tex');

    xlabel('y^+','Interpreter','tex');
    ylabel(yLabel,'Interpreter','tex');
    grid on; grid minor;

    print(f, fullfile(outDir, sprintf('%s.png',tag)), '-dpng','-r300');
    close(f);
end