% EnergySpectrum.m
%
% Turbulence 1D energy spectrum E(k) from OpenFOAM/CSV line data:
%   lineX_subtract(U,UMean).csv   and   lineZ_subtract(U,UMean).csv
%
% Your CSV format (example):
%   z,subtract(U,UMean)_0,subtract(U,UMean)_1,subtract(U,UMean)_2
%   -0.031,0,0,0
%   ...
%
% This script:
%  1) Reads multiple timesteps for X and Z lines
%  2) Computes one-sided spectra using Hann window + window-power correction
%  3) Overlays spectra for all times
%  4) Detects inertial-range band by sliding log-log slope near -5/3
%  5) Anchors a -5/3 line through the detected inertial-range midpoint
%  6) Marks large-scale / inertial / dissipative ranges
%  7) Plots compensated spectra E(k)*k^(5/3)
%
% NOTE (fix for your error):
% - inferTimeFromFilename() now returns a STRING SCALAR (not a cell)
% - comparison uses strcmp (not ~=)


clear; clc; close all;

         
         

% --- EDIT THESE LISTS ---
xFiles = { ...
    
    '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.63954/lineX_subtract(U,UMean).csv', ...
    '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.77418/lineX_subtract(U,UMean).csv', ...
     '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.77044/lineX_subtract(U,UMean).csv', ...
      '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.78166/lineX_subtract(U,UMean).csv', ...
       '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.786148/lineX_subtract(U,UMean).csv', ...
        '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.77792/lineX_subtract(U,UMean).csv', ...
         '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.7854/lineX_subtract(U,UMean).csv', ...
        '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.78727/lineX_subtract(U,UMean).csv', ...
        '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.788018/lineX_subtract(U,UMean).csv', ...
        '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.786896/lineX_subtract(U,UMean).csv', ...
        '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.787644/lineX_subtract(U,UMean).csv', ...
         };

zFiles = { ...
   
'/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.63954/lineZ_subtract(U,UMean).csv', ...
    '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.77418/lineZ_subtract(U,UMean).csv', ...
     '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.77044/lineZ_subtract(U,UMean).csv', ...
      '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.78166/lineZ_subtract(U,UMean).csv', ...
       '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.786148/lineZ_subtract(U,UMean).csv', ...
        '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.77792/lineZ_subtract(U,UMean).csv', ...
         '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.7854/lineZ_subtract(U,UMean).csv', ...
        '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.78727/lineZ_subtract(U,UMean).csv', ...
        '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.788018/lineZ_subtract(U,UMean).csv', ...
        '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.786896/lineZ_subtract(U,UMean).csv', ...
        '/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation_CFL/postProcessing/turbspectrum/0.787644/lineZ_subtract(U,UMean).csv', ...
     ...
};

component = 0;        % 0,1,2 -> subtract(U,UMean)_0/1/2
tolSlope  = 0.35;     % slope tolerance around -5/3 for inertial detection
winFrac   = 0.08;     % sliding slope window fraction (of points)
nGrid     = 300;      % common log-grid size for averaging
outPrefix = 'Ek_all'; % output prefix for PNGs

assert(numel(xFiles)==numel(zFiles), 'xFiles and zFiles must have same length');

% -------------------------
% Load & compute spectra
% -------------------------
results = struct();
times   = strings(numel(xFiles),1);

for i = 1:numel(xFiles)
    tx = inferTimeFromFilename(xFiles{i});
    tz = inferTimeFromFilename(zFiles{i});

    % FIX: use strcmp/isequal for strings, not ~=
    if ~strcmp(tx, tz)
        warning('Time mismatch x:%s z:%s, using x time as key.', tx, tz);
    end
    times(i) = tx;

    [coordNameX, X, uX] = readLineCsvAuto(xFiles{i}, component);
    [coordNameZ, Z, uZ] = readLineCsvAuto(zFiles{i}, component);

    dx = mean(diff(X));
    dz = mean(diff(Z));

    [kx, Ex] = energySpectrumOneSided(uX, dx);
    [kz, Ez] = energySpectrumOneSided(uZ, dz);

    key = matlab.lang.makeValidName(char(tx));
    results.(key).t  = tx;
    results.(key).coordNameX = coordNameX;
    results.(key).coordNameZ = coordNameZ;
    results.(key).kx = kx; results.(key).Ex = Ex;
    results.(key).kz = kz; results.(key).Ez = Ez;
end

% -------------------------
% Mean spectra on common grid (for inertial detection + anchoring)
% -------------------------
[kxMean, ExMean] = commonLogGridMean(results, "kx", "Ex", nGrid);
[kzMean, EzMean] = commonLogGridMean(results, "kz", "Ez", nGrid);

bandX = estimateInertialBand(kxMean, ExMean, -5/3, winFrac, tolSlope);
bandZ = estimateInertialBand(kzMean, EzMean, -5/3, winFrac, tolSlope);

% -------------------------
% Plots (with ranges + anchored -5/3)
% -------------------------
plotOverlayWithRanges(results, "x", kxMean, ExMean, bandX, outPrefix + "_x_with_ranges.png");
plotOverlayWithRanges(results, "z", kzMean, EzMean, bandZ, outPrefix + "_z_with_ranges.png");

% Compensated plots
plotCompensated(results, "x", outPrefix + "_x_comp.png");
plotCompensated(results, "z", outPrefix + "_z_comp.png");

disp("Saved plots:");
disp("  " + outPrefix + "_x_with_ranges.png");
disp("  " + outPrefix + "_z_with_ranges.png");
disp("  " + outPrefix + "_x_comp.png");
disp("  " + outPrefix + "_z_comp.png");

if ~isempty(bandX)
    fprintf("x inertial band: kL=%.4g, kEta=%.4g, fitted slope=%.3f\n", bandX(1), bandX(2), bandX(3));
else
    disp("x inertial band: not detected reliably");
end
if ~isempty(bandZ)
    fprintf("z inertial band: kL=%.4g, kEta=%.4g, fitted slope=%.3f\n", bandZ(1), bandZ(2), bandZ(3));
else
    disp("z inertial band: not detected reliably");
end


% ========================================================================
% Local functions
% ========================================================================

function t = inferTimeFromFilename(fname)
    % FIXED: returns STRING SCALAR, not cell.
    % Prefer parent folder if it is numeric (your case: .../0.17715453/lineX_...csv).
    fname = string(fname);
    partsPath = split(fname, filesep);

    if numel(partsPath) >= 2
        parent = partsPath(end-1);
        if ~isnan(str2double(parent))
            t = parent;
            return;
        end
    end

    % Fallback: token after last underscore in file stem
    [~, stem, ~] = fileparts(char(fname));
    tokens = split(string(stem), "_");
    t = tokens(end);
end

function [coordName, coord, ufluc] = readLineCsvAuto(fname, component)
    % Reads CSV with header, returns coord column and subtract(U,UMean)_component
    T = readtable(fname, 'Delimiter', ',', 'PreserveVariableNames', true);

    % coordinate column
    coordName = "";
    candidates = ["x","z","y","coord"];
    for c = candidates
        if any(strcmp(T.Properties.VariableNames, c))
            coordName = c; break;
        end
    end
    if coordName == ""
        coordName = string(T.Properties.VariableNames{1});
    end

    target = "subtract(U,UMean)_" + string(component);
    if any(strcmp(T.Properties.VariableNames, target))
        ucol = target;
    else
        % fallback: assume columns are [coord, comp0, comp1, comp2]
        ucol = string(T.Properties.VariableNames{2+component});
    end

    coord = T.(coordName);
    ufluc = T.(ucol);

    coord = double(coord(:));
    ufluc = double(ufluc(:));
end

function [k, E] = energySpectrumOneSided(u, d)
    % One-sided spectrum with Hann window + window power correction.
    % E(k) = (d/(N*W2)) |FFT(u*w)|^2, interior bins doubled, k=2*pi*f
    u = double(u(:));
    N = numel(u);
    if N < 8
        error("Signal too short for spectrum.");
    end

    w  = hann(N, 'periodic');
    W2 = mean(w.^2);

    uw = (u - mean(u)) .* w;
    F  = fft(uw);

    % Keep rfft-like half spectrum:
    if mod(N,2)==0
        % even: indices 1..N/2+1 are [0..Nyquist]
        Fp = F(1:(N/2+1));
    else
        % odd: indices 1..(N+1)/2 are [0..max positive]
        Fp = F(1:((N+1)/2));
    end

    E = (d/(N*W2)) * (abs(Fp).^2);

    % double interior bins (exclude DC and Nyquist if present)
    if numel(E) > 2
        E(2:end-1) = 2*E(2:end-1);
    end

    % frequencies and wavenumbers
    f = (0:numel(E)-1).' / (N*d);
    k = 2*pi*f;

    % drop k=0
    k = k(2:end);
    E = E(2:end);
end

function [kCommon, EMean] = commonLogGridMean(results, kField, EField, nGrid)
    keys = fieldnames(results);

    % k-range intersection
    kmins = []; kmaxs = [];
    for i = 1:numel(keys)
        k = results.(keys{i}).(kField);
        kmins(end+1) = k(1); %#ok<AGROW>
        kmaxs(end+1) = k(end); %#ok<AGROW>
    end
    kmin = max(kmins);
    kmax = min(kmaxs);
    if ~(kmin>0 && kmax>kmin)
        error("No valid overlapping k-range across datasets.");
    end

    kCommon = logspace(log10(kmin), log10(kmax), nGrid).';

    Eall = zeros(nGrid, numel(keys));
    for i = 1:numel(keys)
        k = results.(keys{i}).(kField);
        E = results.(keys{i}).(EField);
        Eall(:,i) = interp1(k, E, kCommon, 'linear', 'extrap');
    end
    EMean = mean(Eall, 2);
end

function band = estimateInertialBand(k, E, slopeTarget, winFrac, tol)
    % Sliding log-log slope; choose longest region near slopeTarget
    k = k(:); E = E(:);
    ok = isfinite(k) & isfinite(E) & (k>0) & (E>0);
    k = k(ok); E = E(ok);

    N = numel(k);
    if N < 40
        band = [];
        return;
    end

    w = max(7, round(N*winFrac));
    if mod(w,2)==0, w = w+1; end
    half = floor(w/2);

    slopes = nan(N,1);
    for i = 1+half : N-half
        ki = k(i-half:i+half);
        Ei = E(i-half:i+half);
        ok2 = (ki>0) & (Ei>0) & isfinite(ki) & isfinite(Ei);
        if nnz(ok2) < 5, continue; end
        X = log10(ki(ok2));
        Y = log10(Ei(ok2));
        p = polyfit(X, Y, 1);
        slopes(i) = p(1);
    end

    good = isfinite(slopes) & abs(slopes - slopeTarget) <= tol;
    if ~any(good)
        band = [];
        return;
    end

    idx = find(good);
    breaks = find(diff(idx) > 1);

    blocks = {};
    s = 1;
    for b = 1:numel(breaks)
        blocks{end+1} = idx(s:breaks(b)); %#ok<AGROW>
        s = breaks(b)+1;
    end
    blocks{end+1} = idx(s:end);

    % longest block
    lens = cellfun(@numel, blocks);
    [~, imax] = max(lens);
    block = blocks{imax};

    kL   = k(block(1));
    kEta = k(block(end));

    X = log10(k(block));
    Y = log10(E(block));
    p = polyfit(X, Y, 1);

    band = [kL, kEta, p(1)];
end

function plotOverlayWithRanges(results, direction, kMean, EMean, band, outFile)
    figure('Color','w'); hold on;

    keys = fieldnames(results);

    % sort by time (numeric if possible)
    tvals = zeros(numel(keys),1);
    for i=1:numel(keys)
        tstr = string(results.(keys{i}).t);
        tv = str2double(tstr);
        if isnan(tv), tv = i; end
        tvals(i)=tv;
    end
    [~, order] = sort(tvals);
    keys = keys(order);

    for i = 1:numel(keys)
        r = results.(keys{i});
        if direction=="x"
            tval = str2double(r.t);
            loglog(r.kx, r.Ex, ...
            'DisplayName', sprintf('t=%.2f', tval), ...
            'LineWidth', 1.0);

        else
            tval = str2double(r.t);
            loglog(r.kx, r.Ex, ...
            'DisplayName', sprintf('t=%.2f', tval), ...
            'LineWidth', 1.0);

        end
    end

    subtitle = "inertial band not reliably detected";
    if ~isempty(band)
        kL   = band(1);
        kEta = band(2);

        kmid = sqrt(kL*kEta);
        Emid = interp1(kMean, EMean, kmid, 'linear', 'extrap');

        Eref = Emid * (kMean/kmid).^(-5/3);
        loglog(kMean, Eref, 'k--', 'DisplayName', 'k^{-5/3} ', 'LineWidth', 1.5);

        xline(kL,  ':k', 'HandleVisibility','off');
        xline(kEta,':k', 'HandleVisibility','off');


        
    end

    xlabel("k_" + direction + "[radian/meter]");
    ylabel("E(k_" + direction + ")");
    title("Streamwise Spectrum(x-line) 1D CFL " + direction, 'Interpreter','none');
    grid on;
    legend off;
    set(gca, 'XScale','log', 'YScale','log');
    saveas(gcf, outFile);
    close(gcf);
end

function plotCompensated(results, direction, outFile)
    figure('Color','w'); hold on;

    keys = fieldnames(results);

    % sort by time (numeric if possible)
    tvals = zeros(numel(keys),1);
    for i=1:numel(keys)
        tstr = string(results.(keys{i}).t);
        tv = str2double(tstr);
        if isnan(tv), tv = i; end
        tvals(i)=tv;
    end
    [~, order] = sort(tvals);
    keys = keys(order);

    for i = 1:numel(keys)
        r = results.(keys{i});
        if direction=="x"
            k = r.kx; E = r.Ex;
        else
            k = r.kz; E = r.Ez;
        end
        semilogx(k, E .* (k.^(5/3)), 'DisplayName', "t=" + string(r.t), 'LineWidth', 1.0);
    end

    xlabel("k_" + direction);
    ylabel("E(k) k^{5/3}");
    title("Compensated spectra along " + direction, 'Interpreter','none');
    grid on;
    legend('Location','eastoutside');         % RECOMMENDED
    saveas(gcf, outFile);
    close(gcf);
end
