clear;clc;
clear idx;
idx = struct();
%------------------------- USER SETTINGS -------------------------------%
matInA   = "/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation/postProcessing/sampleDict/0.644/Dn1.mat";  % Case A (e.g., Mapped)
matOutA  = matInA;        % same as before
matInB   = "/WORK/sritej/TASKFILES/Task1/FOR_REUSCH_WORK/395_withprofile_20ms/postProcessing/sampleDict/0.41106316/Mid1.mat";            % OPTIONAL second dataset (e.g., Cyclic). Leave "" to skip dual plot
nameA    = "NCFL Mid";
nameB    = "Cyclic";

% Different friction velocities per dataset
uTauA_yplus = 0.80;   % used to compute yPlus for dataset A (Mapped)
uTauB_yplus = 0.81;    % used to compute yPlus for dataset B (Cyclic)

% Plus-scaling for U' U'^+ (if empty, defaults to the yPlus uTau above)
uTauA_plus  = [];       % if [], uses uTauA_yplus
uTauB_plus  = [];       % if [], uses uTauB_yplus

nu      = 1.84e-5;      % kinematic viscosity (keep your value)
yWall   = -0.00685;     % wall location (keep your value)

askUser  = true;        % keep your interactive index override
lineW    = 1.6;         % line width for dual plot
yPlusMax = 80;          % x-range limit for UPrime2MeanPlusCombined
%----------------------------------------------------------------------%

%% default indices taken from the screenshot

idx.kResolved        =  3;
idx.kModel           =  6;
idx.epsilonResolved  =  2;
idx.epsilonModel     =  5;
idx.UPrime2Mean      = 10:15;      % _0 ? _5
idx.gradU            = 22:30;      % _0 ? _8   (must be 9 long)

%% optionally override from the keyboard
if askUser
    fprintf("Hit <Enter> to accept the default you see in [ ]\n");
    fn = fieldnames(idx);
    for k = 1:numel(fn)
        def  = idx.(fn{k});
        user = input(sprintf("%-18s [%s] : ", fn{k}, mat2str(def)), "s");
        if ~isempty(user), idx.(fn{k}) = str2num(user); end %#ok<ST2NM>
    end
    fprintf("\nUsing these indices:\n"); disp(idx)
end

%% ------------------------- LOAD A & AUGMENT --------------------------
load(matInA,"M");  % Case A (Mapped)
[M, auxA] = augment_and_vectors(M, idx, yPlusMax, uTauA_yplus, nu, yWall);  % yPlus uses uTauA_yplus
save(matOutA,"M");
fprintf("Augmented matrix written to  %s  (%d×%d)\n", matOutA, size(M,1), size(M,2))

outDir = fullfile(fileparts(matOutA),"figs_CombiyPlus200D" + ...
    "");
if ~exist(outDir,"dir"), mkdir(outDir); end

% Columns for UPrime2Mean (can still be changed interactively)
colsStr = input('6 col numbers for UPrime2Mean [10:15] : ','s');
uCols   = 10:15;
if ~isempty(colsStr), uCols = str2num(colsStr); end %#ok<ST2NM>
assert(numel(uCols)==6,'Need exactly six columns!');

% Plus-scaling uTau for Case A (defaults to uTauA_yplus if empty)
if isempty(uTauA_plus), uTauA_plus = uTauA_yplus; end

% ----- build Upp (Case A) -----
UppA      = auxA.Mkeep(:,uCols) ./ ( uTauA_plus^2 );
UppNames  = {'uu^+','uv^+','uw^+','vv^+','vw^+','ww^+'};
ypA       = auxA.yp;
comboTitle = 'UPrime2MeanPlusCombined';

%% ================== NEW: DUAL-DATASET COMBINED PLOT ==================
% If a second .mat is provided, build a single *linear-x* combined plot with:
%   - same colors per component
%   - solid (A=Mapped) vs dashed (B=Cyclic)
if ~isempty(matInB) && isfile(matInB)
    load(matInB,"M");  % Case B (Cyclic)
    [~, auxB] = augment_and_vectors(M, idx, yPlusMax, uTauB_yplus, nu, yWall); % yPlus uses uTauB_yplus

    if isempty(uTauB_plus), uTauB_plus = uTauB_yplus; end
    UppB = auxB.Mkeep(:,uCols) ./ ( uTauB_plus^2 );

    % Interpolate B onto A's y+ grid so the curves overlay perfectly
    yp    = ypA;
    UppB_ = interp1(auxB.yp, UppB, yp, 'linear', 'extrap');
    UppA_ = UppA;

    % Save the dual combined plot (linear x only)
    tagDual = sprintf('%s_%s_vs_%s_linx', comboTitle, char(nameA), char(nameB));
    savecombo_dual(yp, UppA_, UppB_, UppNames, nameA, nameB, outDir, lineW);
    fprintf('? Dual dataset plot written: %s\n', fullfile(outDir,[tagDual '.png']));
else
    fprintf('(No second .mat provided ? skipping dual combined plot)\n');
end

fprintf('? PNGs written to  %s\n', outDir);

%% ===================== LOCAL HELPER FUNCTIONS ========================
function [M, aux] = augment_and_vectors(M, idx, yPlusMax, uTau_yplus, nu, yWall)
    % Totals
    kTotal       =  M(:,idx.kResolved) + M(:,idx.kModel);
    epsilonTotal =  M(:,idx.epsilonResolved) + M(:,idx.epsilonModel);

    g = M(:,idx.gradU);          % 9 columns
    p = M(:,idx.UPrime2Mean);    % 6 columns

    Production = - p(:,1) .* g(:,1)  - ...
                  g(:,2) .* p(:,2)    - ...
                  g(:,3) .* p(:,3)    - ...
                  g(:,4) .* p(:,2)    - ...
                  p(:,4) .* g(:,5)    - ...
                  g(:,6) .* p(:,5)    - ...
                  g(:,7) .* p(:,3)    - ...
                  p(:,5) .* g(:,8)    - ...
                  g(:,9) .* p(:,6);

    P2E = Production .* epsilonTotal;
    M(:,31:34) = [kTotal, epsilonTotal, Production, P2E];

    % yPlus using per-dataset uTau_yplus
    dy     = M(:,1) - yWall;          % true wall distance
    yPlus  = uTau_yplus .* dy ./ nu;  % <-- per-dataset u_tau here
    M(:,35)= yPlus;

    % Keep mask and vectors
    keep       = yPlus <= yPlusMax;
    aux.yp     = yPlus(keep);
    aux.kTot   = kTotal(keep);
    aux.epsTot = epsilonTotal(keep);
    aux.Prod   = Production(keep);
    aux.P2E    = P2E(keep);
    aux.Mkeep  = M(keep,:);

    % Production terms for per-component plots
    aux.ProdTermsMat = [ ...
        -p(:,1).*g(:,1), ...   % P11
        -g(:,2).*p(:,2), ...   % P22
        -g(:,9).*p(:,6), ...   % P33
        -g(:,4).*p(:,2)  ];    % P12
    aux.ProdTermsMat = aux.ProdTermsMat(keep,:);
end



function savecombo_dual(x,YA,YB,legNames,nameA,nameB,outDir,lineW)
    % YA,YB: [N x 6] each  (uu+,uv+,uw+,vv+,vw+,ww+)
    f = figure('Visible','off'); hold on; box on;
    C = lines(6);  % consistent colors per component

    % Case A (Mapped) ? solid
    for i = 1:6
        plot(x, YA(:,i), 'LineWidth', lineW, 'Color', C(i,:), 'LineStyle', '-');
    end
    % Case B (Cyclic) ? dashed
    for i = 1:6
        plot(x, YB(:,i), 'LineWidth', lineW, 'Color', C(i,:), 'LineStyle', '--');
    end

    % Legend
    compLegA = strcat(legNames, " (" + nameA + " )");
    compLegB = strcat(legNames, " (" + nameB + " )");
    legend([compLegA, compLegB], 'Location','bestoutside','Interpreter','tex');

    xlabel('y^+','Interpreter','tex');
    ylabel('U''U''^+ components','Interpreter','tex');
    grid on; grid minor;

    tag = sprintf('UPrime2MeanPlusCombined_%s_vs_%s_linx', char(nameA), char(nameB));
    print(f, fullfile(outDir,[tag '.png']), '-dpng','-r300'); 
    close(f);
end