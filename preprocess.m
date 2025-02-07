% Author: Lucas Jeay-Bizot
% Code for the dataset (DOI): 
% Preprint for the dataset (DOI): https://doi.org/10.31219/osf.io/ghs95
% Purpose: Preprocess EEG data for analysis 

% NOTES:
% - Participant P04 and P16 data were excluded due to excessive trial rejection (>50%).

% REQUIREMENTS:
% 1. To run, edit ft_rejectvisual s.t. the top line reads
% % function [data, chansel, trlsel] = ft_rejectvisual(cfg, data)
% % This allows the saving of chansel and trlsel
% 2. src/ folder added to path
% 3. BIOSEMI_labels in path
% 4. a 'data_cube' and a 'data_steps' folder both need to be in this
% environment

% Load BIOSEMI channel labels for future use and adds src folder to path
addpath('src/')
load(string("src/" + "BIOSEMI_labels.mat"));

% Define excluded participants and loop over remaining participants
excluded_participants = [4 16];

for subj = setdiff(1:17, excluded_participants)
    
    % Check if participant has already been preprocessed
    if ismember({sprintf('P%02d_data_cube.mat', subj)}, {dir('data_cubes').name})
        continue % Skips current 'subj' if 3D data cube already exists
    end
        
%% Step 01: Load EEG data

    if ~ismember({sprintf('P%02d_data_step02.mat', subj)}, {dir('data_steps').name}) % If step02 has already been processed skip
    
        if ismember(subj, [1:5 7:10 14])                                   % for participants with 2 sessions
            filenames           = {sprintf('data/P%02dS01.bdf', subj),  sprintf('data/P%02dS02.bdf', subj)};
            cfg                 = [];
            cfg.dataset         =  filenames;
            cfg.channel         = {'A*','B*'};
            data_step01         = ft_preprocessing(cfg);

        elseif subj == 6                                                   % One session of participant 6 was ran in the wrong workspace. To concatenate, each session needs to be load separately.
            filenames1          = {sprintf('data/P%02dS01.bdf', subj)};
            cfg                 = [];
            cfg.dataset         =  filenames1;
            cfg.channel         = {'A*','B*'};
            data_step01a        = ft_preprocessing(cfg);
    
            filenames2          = {sprintf('data/P%02dS02.bdf', subj)};
            cfg                 = [];
            cfg.dataset         =  filenames2;
            cfg.channel         = {'A*','B*'};
            data_step01b        = ft_preprocessing(cfg);
    
        else                                                               % for participants with 1 session
            filenames           = {sprintf('data/P%02dS01.bdf', subj)};
            cfg                 = [];
            cfg.dataset         =  filenames;
            cfg.channel         = {'A*','B*'};
            data_step01         = ft_preprocessing(cfg);

        end
    
%% Step 02: Epoch to -4000 ms to +500 ms
        
        % Extract the event markers array with FieldTrip
        if ismember(subj, [1:5 7:10 14])                                   % participants with 2 sessions
            ev                  = ft_read_event(data_step01.cfg.dataset);
        elseif subj == 6                                                   % same problem as for loading data
            ev1                 = ft_read_event(data_step01a.cfg.dataset);
            ev2                 = ft_read_event(data_step01b.cfg.dataset);
            ev1(1:3)            = [];                                      % remove weird initial trigger at t0
            ev1                 = ev1(strcmp({ev1.type},'STATUS'));
            ev2                 = ev2(strcmp({ev2.type},'STATUS'));

            for st_i = 1:length(ev2)
                ev2(st_i).sample = ev2(st_i).sample + ev1(end).sample;
            end

            ev                  = [ev1 ev2];
            data_step01         = data_step01a;
            data_step01.trial{1,1} = cat(2,data_step01a.trial{1,1},data_step01b.trial{1,1});
            data_step01.sampleinfo(2) = data_step01a.sampleinfo(2) + data_step01b.sampleinfo(2);
            data_step01.time{1,1} = linspace(0, data_step01.sampleinfo(2)/data_step01.fsample, data_step01.sampleinfo(2));

            % Clears working space
            clear data_step01a data_step01b

        else                                                               % participants with 1 session
            ev                  = ft_read_event(data_step01.cfg.dataset{:});

        end

        % Subselects events of interest (slide transitions): 8 (active) and
        % 16 (passive)
        ev                      = ev(strcmp({ev.type},'STATUS'));
        trl_pos                 = floor([ev([ev.value]==8 | [ev.value]==16).sample]);
        trl_info                = [ev([ev.value]==8 | [ev.value]==16).value] == 8;  % set manual to 1 and passive to 0 for trial labels  
    
        % Define epochs from 4 seconds before to 0.5 seconds after
        event_trl               = [ trl_pos-4*data_step01.fsample'; trl_pos+0.5*data_step01.fsample'; -4*ones(1,length(trl_pos))*data_step01.fsample]';
        event_trialinfo         = trl_info';
        
        % Epoch data
        cfg                     = [];
        cfg.trl                 = event_trl;
        data_step02             = ft_redefinetrial(cfg, data_step01);
        data_step02.trialinfo   = event_trialinfo;
        
        % Save new data
        save(sprintf('data_steps/P%02d_data_step02.mat', subj), 'data_step02', '-v7.3')

        % Clears working space
        clear data_step01

    else
        % Loads already processed data_step
        load(sprintf('data_steps/P%02d_data_step02.mat', subj))
    end

    %% Step 03: Downsample to 500Hz
    
    if ~ismember({sprintf('P%02d_data_step03.mat', subj)}, {dir('data_steps').name})

        cfg                     = [];
        cfg.resamplefs          = 500;
        data_step03             = ft_resampledata(cfg, data_step02);

        % Save new data
        save(sprintf('data_steps/P%02d_data_step03.mat', subj), 'data_step03')

        % Clears working space
        clear data_step02

    else
        % Loads already processed data_step
        load(sprintf('data_steps/P%02d_data_step03.mat', subj))
    end
    
    %% Step 04: Prep data for ICA (remove bad channels and bad trials)
    
    if ~ismember({sprintf('P%02d_data_step04.mat', subj)}, {dir('data_steps').name})
        
        % Prepare the data to identify ocular artifact
        cfg                     = [];
        cfg.method              = 'trial';
        cfg.preproc.demean      = 'yes';
        [~, chansel_ica, trlsel_ica] = ft_rejectvisual(cfg, data_step03);  % needs modified ft_rejectvisual to output chansel and trlsel
        
        cfg                     = [];
        cfg.channel             = data_step03.label(chansel_ica);
        data_step04             = ft_selectdata(cfg, data_step03);

        % Save new data
        save(sprintf('data_steps/P%02d_data_step04.mat', subj), 'data_step04', 'chansel_ica')

        % Clears working space
        clear data_step03

    else
        % Loads already processed data_step
        load(sprintf('data_steps/P%02d_data_step04.mat', subj))
    end
    
    %% Step 05: Perform ICA

    if ~ismember({sprintf('P%02d_data_step07.mat', subj)}, {dir('data_steps').name})
    
        cfg                     = [];
        cfg.method              = 'runica';
        cfg.runica.stop         = 0.01;
        cfg.numcomponent        = 50; 
        comp                    = ft_componentanalysis(cfg, data_step04);
        
        cfg                     = [];
        cfg.layout              = 'biosemi64.lay';
        lay                     = ft_prepare_layout(cfg);
        lay.pos(65:end,:)       = [];
        lay.width(65:end)       = [];
        lay.height(65:end)      = [];
        lay.label(65:end)       = [];
        lay.label               = {'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32'};
        correct_layout          = lay;
        
        cfg                     = [];
        cfg.layout              = correct_layout;
        cfg.viewmode            = 'component';
        ft_databrowser(cfg, comp)                                          % Identify components that reflect eye blinks or movements
        
        cfg                     = [];
        cfg.component           = input('Which components? ');             % Enter component index as e.g. [1 3] for components 1 and 3.
        cfg.demean              = 'no';
        data_step05             = ft_rejectcomponent(cfg, comp, data_step04);
        comp_sel                = cfg.component;
        
        % Interpolate bad channels rejected pre-ICA
        data_step06         = interpolate_fieldtrip(correct_layout.pos, data_step05); % custom function in src/ - puts back noisy channels that were rejected back by interpolating them
        
        % Re-reference to common average
        cfg                 = [];
        cfg.reref           = 'yes';
        cfg.refchannel      = 'all';                                       % Common average
        data_step07         = ft_preprocessing(cfg, data_step06);
        
        % Save new data
        save(sprintf('data_steps/P%02d_data_step07.mat', subj), 'data_step07', 'comp', 'comp_sel')
        
        % Clears working space
        clear data_step04 data_step05 data_step06
        clear comp

    else
        % Loads already processed data_step
        load(sprintf('data_steps/P%02d_data_step07.mat', subj))
    end
    
    %% Step 06: Select crown channels
    
   if ~ismember({sprintf('P%02d_data_step08.mat', subj)}, {dir('data_steps').name}) % Checks if data step has already been processed

        % Loads into label an array that contains the correctly ordered
        % labels converted from A*, B* to FCz, P3, etc. notation
        data_step07.label   = BIOSEMI_labels;
        
        % Subselect crown of EEG (top 23 channels)
        cfg                 = [];
        cfg.channel         = {'F3', 'F1', 'Fz', 'F2', 'F4', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2'};
        data_step08         = ft_selectdata(cfg, data_step07);
        
        % Save new data
        save(sprintf('data_steps/P%02d_data_step08.mat', subj), 'data_step08')

        % Clears working space
        clear data_step07

    else
        % Loads already processed data_step
        load(sprintf('data_steps/P%02d_data_step08.mat', subj))
    end

    %% Step 07: Reject samples visually
    
    if ~ismember({sprintf('P%02d_data_step10.mat', subj)}, {dir('data_steps').name}) % Checks if data step has already been processed

        % Inspect & Reject trials with visual artifacts
        cfg                 = [];
        cfg.method          = 'trial';
        cfg.preproc.demean  = 'yes';
        [~, chansel, trlsel] = ft_rejectvisual(cfg, data_step08);

        % Removes identified trials
        cfg                 = [];
        cfg.trials          = trlsel;
        data_step10         = ft_selectdata(cfg, data_step08);

        % Save new data and trlsel containing excluded trials info
        save(sprintf('data_steps/P%02d_data_step10.mat', subj), 'data_step10', 'trlsel')

        % Clears working space
        clear data_step08

    else
        % Loads already processed data_step
        load(sprintf('data_steps/P%02d_data_step10.mat', subj))
    end
    
    

    %% Step 08: Save 3D array of data with labels
    
    % Extract labels: 1 (active) and 0 (passive) from trialinfo
    labels              = data_step10.trialinfo;
    
    % Initializes 3D array: channels-by-samples-by-trials
    data_cube           = zeros(length(data_step10.label), 2251, length(labels));
    
    % Loop over trials
    for trl = 1:length(labels)
        data_cube(:,1:2251,trl) = data_step10.trial{trl};
    end
    
    % Save 3D array with labels
    save(sprintf('data_cubes/P%02d_data_cube.mat', subj), 'data_cube', 'labels')
    
    % Clears working space
    clear data_step10
    clear data_cube
end
