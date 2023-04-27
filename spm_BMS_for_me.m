function [T, alpha, exp_r, xp,g] = spm_BMS_for_me (exp)
% [T, alpha, exp_r, xp]=spm_BMS_for_me("MainExperiment")
    
    % load file
    T = readtable(strcat('data/bms/inputs/', exp , '.csv'))
    
    % add spm script
    addpath('spm12/');
    
    [alpha, exp_r, xp,pxp,bor,g] =  spm_BMS(T{:,2:end});
    
    writematrix(g, strcat('data/bms/outputs/', exp , '.csv'));