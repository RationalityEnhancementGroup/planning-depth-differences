function [T, alpha, exp_r, xp,g] = spm_BMS_for_me (exp)
    % load file
    T = readtable(strcat('data/bms/inputs/', exp , '.csv'))
    
    % add spm script
    addpath('spm12/');
    
    [alpha, exp_r, xp,pxp,bor,g] =  spm_BMS(T{:,2:end});
    
    model_names = T.Properties.VariableNames;
    expected_number_of_participants = [alpha-1,1];
    bms_struct = struct("Model",model_names,"Participants",expected_number_of_participants);%, "Probability",exp_r,"Exceedance",xp);

    writematrix([T{:,1}, g], strcat('data/bms/outputs/', exp , '.csv'));
