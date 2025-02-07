function Get_ERP()
% GET_ERP Computes and saves the ERP data cube.
% 
% This function loads individual participant EEG data, extracts event-related
% potentials (ERP) for two conditions (Passive and Active), and computes the
% average across trials. The result is saved in 'data_figure/erp_cube.mat'.
%
% Data Structure:
% - erp_cube: 4D array (condition x channel x time x participant)
%   - Condition 1: Passive
%   - Condition 2: Active
% 
% Excluded Participants: 4, 16
%
% Dependencies:
% - Data files must be stored as 'data_cubes/PXX_data_cube.mat'
%
% Output:
% - Saves 'erp_cube.mat' in the 'data_figure' folder

% Exclude specific participants
part_sel = setdiff(1:17, [4 16]); % List of participants, excluding 4 and 16

% Preallocate array for individual ERP
erp_cube = nan(2, 23, 2251, length(part_sel)); % Individual ERP (condition-channel-time-participant)

% Loop through selected participants
for subj = 1:length(part_sel)
    % Load participant-specific data cube
    load(sprintf('data_cubes/P%02d_data_cube.mat', part_sel(subj)), 'data_cube', 'labels')

    % Compute condition-wise average across trials
    erp_cube(1, :, :, subj) = mean(data_cube(:, :, labels == 0), 3); % Passive condition
    erp_cube(2, :, :, subj) = mean(data_cube(:, :, labels == 1), 3); % Active condition
end

% Save the newly computed grand average cube
save('data_figure/erp_cube.mat', 'erp_cube')
end

