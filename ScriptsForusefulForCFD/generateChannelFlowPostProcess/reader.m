% ??? EDIT THIS ???
xyFile   = "/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation/postProcessing/sampleDict/0.0552/lineUp_epsilonResolved_kResolved_nut_turbulenceProperties:epsilon_turbulenceProperties:k_UMean_UPrime2Mean_turbulenceProperties:R_grad(U).xy";
outMat   = "/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/395_withprofile_temporal_validation/postProcessing/sampleDict/0.0552/Up1.mat";
% ?????????????????

% Load entire .xy into a matrix
M = load(xyFile);

% Save it as a .mat (variable name ?M?)
save(outMat, "M");