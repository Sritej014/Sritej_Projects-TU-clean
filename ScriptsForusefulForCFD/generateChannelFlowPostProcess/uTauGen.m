%% Friction velocity vs Time from surfaceFieldValue_*.dat

fileName = '/WORK/sritej/TASKFILES/Task1/FOR_REUSCH_WORK/395_withprofile_20ms/postProcessing/patchAverage(name=bottom,wallShearStressMean)/0.54662408/surfaceFieldValue.dat';

rho = 1.225;   % air density [kg/m^3]

%% 1) Read data robustly
fid = fopen(fileName, 'r');
if fid == -1
    error('Could not open file: %s', fileName);
end

% Columns: time  (tau_x tau_y tau_z)
C = textscan(fid, '%f (%f %f %f)', ...
    'CommentStyle', '#');
fclose(fid);

time  = C{1};
tau_x = C{2};
tau_y = C{3};
tau_z = C{4};

fprintf('Read %d time samples\n', numel(time));

%% 2) Compute |tau_w| and u_tau
tau_mag = sqrt(tau_x.^2 + tau_y.^2 + tau_z.^2);   % shear stress magnitude
uTau    = sqrt(tau_mag ./ rho);                   % friction velocity

%% 3) Plot (DO NOT round data)
figure;
plot(time, uTau, '-s', ...
    'LineWidth', 1.5, ...
    'MarkerFaceColor', 'b');
hold on;

%% 4) Axes formatting
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

xlabel('Time (s)', 'FontName', 'Times New Roman');
ylabel('Friction velocity (m s^{-1})', 'FontName', 'Times New Roman');
title('Friction velocity (m s^{-1}) vs Time (s)', 'FontName', 'Times New Roman');

ytickformat('%.4f')   % show precision without corrupting data

grid on;
box on;
