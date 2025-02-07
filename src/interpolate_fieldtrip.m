function [data_out] = interpolate_fieldtrip(electrodes_position, data_in)
% INTERPOLATE_FIELDTRIP Reconstructs missing channel data by interpolation.
%
%   Syntax:
%       [data_out] = interpolate_fieldtrip(electrodes_position, data_in)
%
%   Description:
%       This function identifies missing EEG channels in the input data 
%       and replaces their data with a weighted average of the nearest 
%       neighboring channels. The weights are calculated based on the 
%       inverse distance between electrodes. The function is designed 
%       to work with FieldTrip structures containing EEG trial data.
%
%   Inputs:
%       electrodes_position - A matrix of size Nx3 containing the 3D coordinates 
%                             of the electrode positions. Each row corresponds 
%                             to a channel, and the order of channels matches 
%                             the labels in 'data_in.label'.
%       data_in             - A FieldTrip structure with the following fields:
%                               - label: Cell array of channel labels (1xM).
%                               - trial: Cell array where each element is a matrix 
%                                        (CxT) containing EEG data, with C channels 
%                                        and T time points.
%
%   Outputs:
%       data_out - A FieldTrip structure with the same format as 'data_in', 
%                  but with interpolated data for missing channels. Missing 
%                  channels are identified based on a standard set of labels 
%                  ('A1' to 'A32', 'B1' to 'B32').
%
%   Notes:
%       - The function assumes that missing channels are not listed in 'data_in.label'.
%       - Missing channels are interpolated using the 5 nearest neighbors, based on 
%         the Euclidean distances calculated from 'electrodes_position'.
%       - Channels with no valid neighbors are left as NaN.
%
%   Example:
%       % Example usage
%       electrodes_position = rand(64, 3); % Example 3D coordinates for 64 electrodes
%       data_in.label = {'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12'}; % Example labels
%       data_in.trial = {rand(24, 1000)};    % Example data (12 channels, 1000 time points)
%
%       data_out = interpolate_fieldtrip(electrodes_position, data_in);


% Create a copy of data_in to store output
data_out = data_in;

% Get the order of the input channels
channel_order = data_in.label;

% Generate standard labels A1:A32 and B1:B32
standard_labels = [strcat('A', arrayfun(@(x) num2str(x), 1:32, 'UniformOutput', false)) strcat('B', arrayfun(@(x) num2str(x), 1:32, 'UniformOutput', false))];

% Identify channels that are missing
missing_chans        = ~ismember(standard_labels, channel_order);


% Calculate distance matrix between channels
dist_matrix = pdist2(electrodes_position, electrodes_position);

% Create temp variable to retain valid channels data
temp = data_out.trial;

% Reset all channels to NaN
data_out.trial = cellfun(@(x) nan(64, size(temp{1,1}, 2)), temp, 'UniformOutput', false);

% Iterate over trials and put valid data back in for valid channels
for i = 1:numel(data_out.trial)
    data_out.trial{1,i}(~missing_chans,:) = temp{1,i};
end

% Iterate over missing channels
for i = find(missing_chans)
    % Find 5 nearest neighbors
    [~, sorted_indices] = sort(dist_matrix(i, :));
    neighbors = sorted_indices(2:6); % Considering the 5 nearest neighbors
    
    % Exclude NaN values from neighbors
    valid_neighbors = neighbors(~isnan(data_out.trial{1,1}(neighbors, 1)));

    % Calculate weights based on inverse distance
    weights = 1 ./ dist_matrix(i, valid_neighbors);

    % Normalize weights
    weights = weights / sum(weights);

    % Iterate over trials to replace missing channel data with weighted
    % average of valid neighbors
    for j = 1:numel(data_out.trial)
        data_out.trial{1,j}(i, :) = sum(data_out.trial{1,j}(valid_neighbors, :) .* weights');
    end 
end

data_out.label = standard_labels;

end
