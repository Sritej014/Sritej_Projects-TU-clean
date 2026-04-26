clear idx;
%?????????????????????????? USER SETTINGS ??????????????????????????
matIn   = "/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_nearest/postProcessing/sampleDict/0.7106/Mid1.mat";        % ? your .mat with matrix  M
matOut  = "/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_nearest/postProcessing/sampleDict/0.7106/Mid1.mat";    % ? augmented file to save
askUser = true;                    % true = let me change indices
%???????????????????????????????????????????????????????????????????

%% default indices taken from the screenshot

idx = struct();
idx.kResolved        =  3;
idx.kModel           =  6;         % turbulenceProperties:k
idx.epsilonResolved  =  2;
idx.epsilonModel     =  5;         % turbulenceProperties:epsilon
idx.UPrime2Mean      = 10:15;      % _0 ? _5
idx.gradU            = 22:30;      % _0 ? _8   (must be 9 long)

%% optionally override from the keyboard
if askUser
    fprintf("Hit <Enter> to accept the default you see in [ ]\n");
    fn = fieldnames(idx);
    for k = 1:numel(fn)
        def  = idx.(fn{k});
        user = input(sprintf("%-18s [%s] : ", fn{k}, mat2str(def)), "s");
        if ~isempty(user)
            idx.(fn{k}) = str2num(user); %#ok<ST2NM> 
        end
    end
    fprintf("\nUsing these indices:\n"); disp(idx)
end

%% -----------------------------------------------------------------
load(matIn,"M");                 % brings raw matrix into workspace

% new columns -------------------------------------------------------
kTotal       =  M(:,idx.kResolved) + M(:,idx.kModel);
epsilonTotal =  M(:,idx.epsilonResolved) + M(:,idx.epsilonModel);

g = M(:,idx.gradU);              % 9 grad(U) columns  (g0 ? g8)
p = M(:,idx.UPrime2Mean);        % 6 UPrime2Mean cols (p0 ? p5)

% Production = ?( gij * rij )  with the mapping you specified
Production = - p(:,1) .* g(:,1)  - ...
              g(:,2) .* p(:,2)    - ...
              g(:,3) .* p(:,3)    - ...
              g(:,4) .* p(:,2)    - ...
              p(:,4) .* g(:,5)  - ...
              g(:,6) .* p(:,5)    - ...
              g(:,7) .* p(:,3)    - ...
              p(:,5) .* g(:,8)  - ...
              g(:,9) .* p(:,6);

Production1 = - p(:,1) .* g(:,1) ;
Production2 = - p(:,4) .* g(:,5);
Production3 =  - g(:,9) .* p(:,6);
Production4 = - g(:,2) .* p(:,2);

% ========== NEW CODE (production?to?epsilonT
% 
% otal) ==========  
Prod2Eps = Production .* epsilonTotal;   % column 34

% append the three old and one new columns
M(:,31:34) = [kTotal, epsilonTotal, Production, Prod2Eps];
% ==========================================================

yWall = -0.00685;                    % adjust for top wall if needed
dy     = M(:,1) - yWall;             % TRUE distance from that wall
M(:,35)= 0.8 .* dy ./ 1.84e-5;     % yPlus now uses ?y, not y?coord

% =========================================================
save(matOut,"M");
fprintf("Augmented matrix written to  %s  (%d×%d)\n", ...
        matOut, size(M,1), size(M,2))


%% =========  BUILD VECTORS limited to yPlus ? 80  ====================
outDir = fullfile(fileparts(matOut),"figs_yPlus200D");
if ~exist(outDir,"dir"), mkdir(outDir); end

yPlus = M(:,35);
keep  = yPlus <= 80;                    % mask

yp      = yPlus(keep);                 % x?axis for every plot
kTot    = M(keep,31);
epsTot  = M(keep,32);
Prod    = M(keep,33);
P2E     = M(keep,34);

% ---------- ask once for uTau and UPrime2Mean columns -----------------
uTau = input('uTau for "+" scaling  [0.8] : ');
if isempty(uTau), uTau = 0.8; end

colsStr = input('6 col?numbers for UPrime2Mean [10:15] : ','s');
uCols   = 10:15;
if ~isempty(colsStr)
    uCols = str2num(colsStr); %#ok<ST2NM>
end
assert(numel(uCols)==6,'Need exactly six columns!');

Upp      = M(keep,uCols) ./ ( uTau*uTau );
UppNames = {'uu^+','uv^+','uw^+','vv^+','vw^+','ww^+'};

TI_u = sqrt(Upp(:,1));                 % ?(uu') / uTau
TI_v = sqrt(Upp(:,4));                 % ?(vv') / uTau
TI_w = sqrt(Upp(:,6));                 % ?(ww') / uTau

%% ==========  SINGLE?CURVE PLOTS  ====================================

ProdEpsMat = [Prod , epsTot];                     % 4700×2 numeric matrix
ProdEpsLeg = {'ProductionTotal','epsilonTotal'};  % legend entries

savecombo(yp, ProdEpsMat, ProdEpsLeg, true , 'ProdEpsCombined', outDir); % log?x
savecombo(yp, ProdEpsMat, ProdEpsLeg, false, 'ProdEpsCombined', outDir); % lin?x

Productionterms      = {'P11','P22','P33','P12'};
ProdTermsMat         = [ ...
    -p(:,1).*g(:,1), ...   % P11
    -g(:,2).*p(:,2), ...   % P22
    -g(:,9).*p(:,6), ...   % P33
    -g(:,4).*p(:,2)];      % P12
ProdTermsMat = ProdTermsMat(keep,:);   % apply same mask as above

for n = 1:4
    saveplot(yp, ProdTermsMat(:,n), Productionterms{n}, true,  outDir);  % log?x
    saveplot(yp, ProdTermsMat(:,n), Productionterms{n}, false, outDir);  % lin?x
end

singleNames = {'kTotal','epsilonTotal','Production','ProdOverEps'};
singleData  = {kTot,    epsTot,        Prod,        P2E};

for n = 1:numel(singleNames)
    saveplot(yp, singleData{n}, singleNames{n}, true,  outDir);          % log?x
    saveplot(yp, singleData{n}, singleNames{n}, false, outDir);          % lin?x
end

TI_names = {'TI_u','TI_v','TI_w'};
TI_data  = {TI_u,   TI_v,   TI_w};

for n = 1:3
    saveplot(yp, TI_data{n}, TI_names{n}, true,  outDir);                % log?x
    saveplot(yp, TI_data{n}, TI_names{n}, false, outDir);                % lin?x
end

%% ==========  COMBINED CURVE PLOTS  ==================================
% 1) six U''U''+ together
comboTitle = 'UPrime2MeanPlusCombined';
savecombo(yp, Upp, UppNames, true,  comboTitle, outDir);  % log?x
savecombo(yp, Upp, UppNames, false, comboTitle, outDir);  % lin?x

% 2) TI combined
TIcombo = [TI_u, TI_v, TI_w];
TIleg   = {'TI_u','TI_v','TI_w'};
savecombo(yp, TIcombo, TIleg, true,  'TIcombined', outDir);   % log?x
savecombo(yp, TIcombo, TIleg, false, 'TIcombined', outDir);   % lin?x

% 3) Production terms combined
savecombo(yp, ProdTermsMat, Productionterms, true,  'ProductionIndividualcombined', outDir); % log?x
savecombo(yp, ProdTermsMat, Productionterms, false, 'ProductionIndividualcombined', outDir); % lin?x

fprintf('?  PNGs written to  %s\n', outDir);
%% ==========  LOCAL HELPER FUNCTIONS  ================================
function saveplot(x,y,tag,isLog,outDir)
    f = figure('Visible','off');
    if isLog
        semilogx(x,y,'LineWidth',1.2);
        suffix = 'logx';
    else
        plot(x,y,'LineWidth',1.2);
        suffix = 'linx';
    end
    xlabel('y^+'); ylabel(tag); grid on; grid minor;
    print(f, fullfile(outDir,sprintf('%s_%s.png',tag,suffix)), ...
          '-dpng','-r300'); close(f);
end

function savecombo(x,Y,legNames,isLog,tag,outDir)
    f = figure('Visible','off'); hold on; box on;
    set(gca,'ColorOrder',lines(size(Y,2)),'NextPlot','add');
    if isLog
        semilogx(x,Y,'LineWidth',1.2);
        suffix = 'logx';
    else
        plot(x,Y,'LineWidth',1.2);
        suffix = 'linx';
    end
    xlabel('y^+'); ylabel('U''U''^+ components');
    legend(legNames,'Location','bestoutside');
    grid on; grid minor;
    print(f, fullfile(outDir,sprintf('%s_%s.png',tag,suffix)), ...
          '-dpng','-r300'); close(f);
end
