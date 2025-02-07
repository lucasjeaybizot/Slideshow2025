function Get_ERD()
%GET_ERD Computes time-frequency power using complex Morlet wavelets.
%
%   This function applies a time-frequency decomposition to EEG data using 
%   complex Morlet wavelets. It calculates event-related desynchronization (ERD) 
%   and synchrony for a selected EEG channel across trials in a given condition (manual). 
%
%   The function performs the following steps:
%   1. Loads EEG data from multiple participants, excluding specified ones.
%   2. Defines wavelet parameters (frequency range, wavelet standard deviations).
%   3. Computes time-frequency representations using convolution in the frequency domain.
%   4. Applies baseline normalization.
%   5. Saves the computed time-frequency power in '../data_figure/ga_power_cmw.mat'.
%
%   Input:
%       - This function does not require explicit input arguments; it directly 
%         loads EEG data files from the '../data_cubes/' directory.
%
%   Output:
%       - The processed power values (78 frequencies x 1901 time points x 15 participants) 
%         are saved as 'ga_power.mat'.
%
%   Data Sources:
%       - EEG data is loaded from individual participant files: 'PXX_data_cube.mat'.
%       - The processed results are saved in '../data_figure/'.
%
%   Notes:
%       - The analysis focuses on a single EEG channel (default: 'C3').
%       - Uses logarithmically spaced frequencies between 2 Hz and 80 Hz.
%       - Baseline correction is applied from -3.5s to -3s before stimulus onset.
%
% Adapted from Analyzing Neural Time Series Data (Cohen, 2014)

%% Define parameters

% EEG channel labels
chan_labels = {'F1';'F3';'FC3';'FC1';'C1';'C3';'CP3';'CP1';'P1';'Pz';'CPz';'Fz';'F2';'F4';'FC4';'FC2';'FCz';'Cz';'C2';'C4';'CP4';'CP2';'P2'};

% Initialize power matrix (frequency x time x participant)
ga_power = zeros(78,1901,15);

% Frequency parameters: 2 Hz to 80 Hz in 78 logarithmic steps
min_freq            = 2;
max_freq            = 80;
num_frex            = 78;

% Time and wavelet kernel setup
time_kern           = -1:1/500:1; % Creates a time vector from -1 to 1 at 500 Hz
frex                = logspace(log10(min_freq),log10(max_freq),num_frex); % logarithmically spaces frequency bands
s                   = logspace(log10(3),log10(10),num_frex)./(2*pi*frex); % logarithmically decreases cycle count
n_wavelet           = length(time_kern);

% Exclude specific participants (4 and 16)
part_sel            = setdiff(1:17, [4 16]);

%% Compute time-frequency representation for each participant

sub_id = 0; % Initialize participant index
for subj = 1:length(part_sel)
    sub_id                  = sub_id+1;

    % Load participant data
    load(sprintf('data_cubes/P%02d_data_cube.mat', part_sel(subj)), 'data_cube', 'labels')
    
    % Determine data size and FFT parameters
    n_data                  = prod(size(data_cube(:,:,labels==1), [2 3]));
    n_convolution           = n_wavelet+n_data-1;
    n_conv_pow2             = pow2(nextpow2(n_convolution));
    half_of_wavelet_size    = (n_wavelet-1)/2;

    % Compute wavelets in frequency domain
    wavelets                = zeros(num_frex,n_conv_pow2);
    for fi = 1:num_frex
        wavelets(fi,:)      = fft( sqrt(1/(s(fi)*sqrt(pi))) * exp(2*1i*pi*frex(fi).*time_kern) .* exp(-time_kern.^2./(2*(s(fi)^2))) , n_conv_pow2 );
    end
        
    % Compute FFT of EEG data at C3 for manual trials only
    eegfft                  = fft(reshape(data_cube(strcmpi('C3',chan_labels),:,labels==1),1,n_data),n_conv_pow2);

    % Initialize matrix for power
    eegpower                = zeros(num_frex,size(data_cube,2),size(data_cube(:,:,labels==1),3)); % frequencies X time X trials

    % Loop through frequencies and compute time-frequency decomposition
    for fi=1:num_frex
        
        % convolution
        eegconv             = ifft(wavelets(fi,:).*eegfft);
        eegconv             = eegconv(1:n_convolution);
        eegconv             = eegconv(half_of_wavelet_size+1:end-half_of_wavelet_size);
        
        % Reshape and extract power
        eegpower(fi,:,:)    = abs(reshape(eegconv,size(data_cube,2),size(data_cube(:,:,labels==1),3))).^2;
    end
        
    % Remove edge artifacts (discard first 500 ms and last 200 ms)
    time_s                  = dsearchn(linspace(-4,0.5,2251)',-3.5);
    time_e                  = dsearchn(linspace(-4,0.5,2251)',.3);

    time_eeg                = linspace(-4,0.5,2251);
        
    eegpower                = eegpower(:,time_s:time_e,:);
    tftimes                 = time_eeg(time_s:time_e);

    % Baseline normalization using -3.5 s to -3 s period and average over
    % trials

    baseidx(1)              = dsearchn(tftimes',-3.5);
    baseidx(2)              = dsearchn(tftimes',-3);

    realbaselines           = squeeze(mean(eegpower(:,baseidx(1):baseidx(2),:),2));
    ga_power(:,:,sub_id)    = 10*log10(bsxfun(@rdivide, mean(eegpower,3), mean(realbaselines,2)));

end

%% Save computed power data
save('data_figure/ga_power.mat', 'ga_power','frex','tftimes')

end

