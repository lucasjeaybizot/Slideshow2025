% EEG Data Processing and Analysis Script for Figure 2
%
% This script processes EEG data to analyze movement-related cortical potentials (MRCP),
% lateralized readiness potentials (LRP), and event-related desynchronization (ERD).
% It loads precomputed data files or computes them if missing computes them. 
% The script also determines MRCP onset using three methods:
% t-test, visual inspection, and a 90% area method.
%
% The script generates a figure displaying:
% - MRCPs over the C3 electrode
% - LRP as the difference between C3 and C4
% - Event-related desynchronization (ERD) of power changes at C3
% - Waiting time distributions for reaction times

% Clear workspace
close all
clear

% Ensure all dependencies are loaded
addpath('src\')

%% Load data

% Define file paths
files = {'data_figure/erp_cube.mat', 'data_figure/ga_power.mat', 'data_figure/RT.mat'};
functions = {@Get_ERP, @Get_ERD, @Get_WT};

% Check and load or compute missing files
for i = 1:length(files)
    if isfile(files{i})
        load(files{i})
    else
        functions{i}(); % Run corresponding function
    end
end

%% Set-up parameters & data

% Ordered crown labels
chan_labels     = {'F1';'F3';'FC3';'FC1';'C1';'C3';'CP3';'CP1';'P1';'Pz';'CPz';'Fz';'F2';'F4';'FC4';'FC2';'FCz';'Cz';'C2';'C4';'CP4';'CP2';'P2'};

% Create a time array for the x-axis
time            = linspace(-4, .5, 2251);

% Select channels for the MRCP (chan_1) and LRP (chan_2 - chan_1) 
chan_1          = strcmp(chan_labels, 'C3');
chan_2          = strcmp(chan_labels, 'C4');

% Set font size and line thickness
font_size       = 15;
line_thickness  = 3.5;

% Mean removal of the EEG data
demean_cube         = erp_cube - mean(erp_cube(:,:,:,:),3);

% Set a lowpass filter to smooth out the data for plotting
lp_filtfreq     = 30;
[b, a]          = butter(4, lp_filtfreq/250, 'low'); % order 4; sampling rate of 500Hz

% Preallocate filtered 4D array
flt_cube        = nan(size(demean_cube));

for i = 1:2 % active vs. passive
    for j = 1:23 % over crown channels
        flt_cube(i, j, :, :)        = filtfilt(b,a,squeeze(demean_cube(i,j,:,:))); % apply lowpass filter
    end
end

% Compute SEM:
erp_sem = nan(size(erp_cube, [1 2 3]));
for j = 1:23
    erp_sem(1,j,:) = std(squeeze(flt_cube(1,j,:,:)), [], 2) / sqrt(size(flt_cube(1,:,:,:),4));
    erp_sem(2,j,:) = std(squeeze(flt_cube(2,j,:,:)), [], 2) / sqrt(size(flt_cube(2,:,:,:),4));
end

%% Compute MRCP onsets

% MRCP Onset: t-test method
for i = fliplr(1:1999) % run from movement backwards
    % Check if signal is different from 0 for three consecutive samples
    if ~(ttest(squeeze(flt_cube(2,chan_1,i,:))) & ttest(squeeze(flt_cube(2,chan_1,i+1,:))) & ttest(squeeze(flt_cube(2,chan_1,i+2,:))))
        MRCPttest_method = time(i);
        break % Stop at the first instance that not all 3 consecutive samples are different from 0 running backwards from movement onset
    end
end

% MRCP Onset: by eye method
for i = fliplr(1:1999) % run from movement backwards
    % Check if signal is smaller 0 for three consecutive samples
    if ~(mean(squeeze(flt_cube(2,chan_1,i,:)))<0 & mean(squeeze(flt_cube(2,chan_1,i+1,:)))<0 & mean(squeeze(flt_cube(2,chan_1,i+2,:)))<0)
        MRCPvisual_method = time(i) ;
        break % Stop at the first instance that not all 3 consecutive samples are smaller than 0 running backwards from movement onset
    end
end

% MRCP Onset: 90% method
for j = fliplr(i:1999) % run from movement backwards
    % Checks if signal is 90% of the area under the curve from movement onset up to the by-eye onset
    if sum(squeeze(mean(flt_cube(2,chan_1,j:1999,:),4))) < sum(squeeze(mean(flt_cube(2,chan_1,i:1999,:),4)))*0.9
        MRCP90_method = time(j);
        break % Stops at the first instance the 90% area is reached
    end
end

% Save MRCP onsets for figure 3
save('data_figure\MRCP_onset.mat', 'MRCP90_method','MRCPvisual_method','MRCPttest_method')

%% %% Figure %% %%
% Generate figure - position values will determine the shape of the output
% (will depend on screen resolution)
fig = figure('Position',[-1919          41         973         955], 'Color','white');

%% ERP - top left (2 A)
subplot(2,2,1)

% Plot averages at chan_1 across participants (4th dimension)
plot(time, squeeze(mean(flt_cube(1,chan_1,:,:),4)), 'r', 'LineWidth', line_thickness), hold on
plot(time, squeeze(mean(flt_cube(2,chan_1,:,:),4)), 'b', 'LineWidth', line_thickness)

% Set plot label on figure
text(0, 1.07, 'A', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', font_size, 'FontWeight','bold');

% Get lower and upper SEM for manual and active
lower_sem_m     = squeeze(mean(flt_cube(2,chan_1,:,:),4)) - squeeze(mean(erp_sem(2,:,:),2));
lower_sem_a     = squeeze(mean(flt_cube(1,chan_1,:,:),4)) - squeeze(mean(erp_sem(2,:,:),2));
upper_sem_m     = squeeze(mean(flt_cube(2,chan_1,:,:),4)) + squeeze(mean(erp_sem(2,:,:),2));
upper_sem_a     = squeeze(mean(flt_cube(1,chan_1,:,:),4)) + squeeze(mean(erp_sem(2,:,:),2));

% Add shaded areas for the SEM around manual and active
fill([time fliplr(time)], [lower_sem_m' fliplr(upper_sem_m')],'b','EdgeColor', 'none', 'FaceAlpha', .2)
fill([time fliplr(time)], [lower_sem_a' fliplr(upper_sem_a')],'r','EdgeColor', 'none', 'FaceAlpha', .2)

xlim([-2.5 .3]) % window to focus closer to movement 
ylim([-2.25 2.25]) % range to focus on MRCP shape
xticks([-3 -2 -1 0]) % set the selected tickmarks on the x-axis
yticks([-3 -2 -1 0 1 2]) % set the selected tickmarks on the y-axis
yticklabels({'-3', '-2', '-1', '0', '1', '2'}) % set y-label values in microvolts
ylabel('Amplitude (µV)')
xlabel('Time (s)')
grid on
axis square
box on
legend('Passive', 'Active', 'Location', 'NorthWest','AutoUpdate','off')
xline(0,'--','LineWidth',2,'Alpha',0.5); % Line at time of movement
set(gca, 'TickDir', 'out', 'TickLength', [0.01 0.01], 'XAxisLocation', 'bottom', 'YAxisLocation','left')
set(gca, 'FontWeight', 'bold', 'FontSize',font_size, 'LineWidth',2)
title('Movement-Related Cortical Potential (C3)', 'FontSize', font_size)

% Add onsets to MRCP plot
xline(MRCP90_method, 'k--')
text(MRCPvisual_method-0.1, 0.5,'MRCP onset: by eye','Rotation',90)
xline(MRCPvisual_method, 'k--')
text(MRCP90_method-0.1, 0.5,'MRCP onset: 90%','Rotation',90)
xline(MRCPttest_method, 'k--')
text(MRCPttest_method-0.1, 0.5,'MRCP onset: t-test','Rotation',90)

%% LRP - top right (2 B)
subplot(2,2,2)

% Compute LRP as difference between C3 and C4
lrp_cube        = squeeze(flt_cube(:,chan_1,:,:) - flt_cube(:,chan_2,:,:));

% Compute SEM per condition (manual vs. automatic):
sem_lrp(1,:,:) = std(squeeze(lrp_cube(1,:,:)), [], 2) / sqrt(size(lrp_cube(1,:,:),3));
sem_lrp(2,:,:) = std(squeeze(lrp_cube(2,:,:)), [], 2) / sqrt(size(lrp_cube(2,:,:),3));

% Plot mean LRP per condition:
plot(time, squeeze(mean(lrp_cube(1,:,:),3)), 'r', 'LineWidth', line_thickness), hold on
plot(time, squeeze(mean(lrp_cube(2,:,:),3)), 'b', 'LineWidth', line_thickness)

% Set plot label on figure
text(0, 1.07, 'B', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', font_size, 'FontWeight','bold');

% Get lower and upper SEM for manual and active
lower_sem_lrp_m     = squeeze(mean(lrp_cube(2,:,:),3)) - squeeze(mean(sem_lrp(2,:),2));
lower_sem_lrp_a     = squeeze(mean(lrp_cube(1,:,:),3)) - squeeze(mean(sem_lrp(2,:),2));
upper_sem_lrp_m     = squeeze(mean(lrp_cube(2,:,:),3)) + squeeze(mean(sem_lrp(2,:),2));
upper_sem_lrp_a     = squeeze(mean(lrp_cube(1,:,:),3)) + squeeze(mean(sem_lrp(2,:),2));

% Add shaded areas for the SEM around manual and active LRP
fill([time fliplr(time)], [lower_sem_lrp_m fliplr(upper_sem_lrp_m)],'b','EdgeColor', 'none', 'FaceAlpha', .2)
fill([time fliplr(time)], [lower_sem_lrp_a fliplr(upper_sem_lrp_a)],'r','EdgeColor', 'none', 'FaceAlpha', .2)

xlim([-2.5 .3]) % window to focus closer to movement
ylim([-2.25 2.25]) % range to focus on LRP shape
xticks([-3 -2 -1 0])
yticks([-3 -2 -1 0 1 2])
yticklabels({'-3', '-2', '-1', '0', '1', '2'})
ylabel('Amplitude (µV)')
xlabel('Time (s)')
grid on
axis square
box on
legend('Passive', 'Active', 'Location', 'NorthWest', 'AutoUpdate','off')
xline(0,'--','LineWidth',2,'Alpha',0.5);
set(gca, 'TickDir', 'out', 'TickLength', [0.01 0.01], 'XAxisLocation', 'bottom', 'YAxisLocation','left')
set(gca, 'FontWeight', 'bold', 'FontSize',font_size, 'LineWidth',2)
title('Lateralized Readiness Potential (C3 - C4)', 'FontSize', font_size)


%% ERD - bottom left (2 C)
subplot(2,2,3)

load('data_figure/ga_power.mat');
contourf(tftimes,frex,squeeze(mean(ga_power(:,:,:),3)),40,'linecolor','none')
axis square
set(gca,'clim',[-3 3],'xlim',[-2.5 .3]) % exclude edge artifacts
title('Event-Related Desynchronization (C3)')
xlabel('Time (s)'), ylabel('Frequency (Hz)')
box on
line_tzero = xline(0,'--','LineWidth',2,'Alpha',0.5);
set(gca, 'TickDir', 'out', 'TickLength', [0.01 0.01], 'XAxisLocation', 'bottom', 'YAxisLocation','left')
set(gca, 'FontWeight', 'bold', 'FontSize',font_size, 'LineWidth',2)
% Set plot label on figure
text(0, 1.07, 'C', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', font_size, 'FontWeight','bold');
colorbar('west')

%% WT - bottom right (2 D)
subplot(2,2,4)
load('data_figure/RT.mat')

grayValues  = linspace(0,0.8,15);
grayRGB     = repmat((grayValues)', 1, 3);

sub_id = 0;
for subj = setdiff(1:17, [4 16])
    sub_id = sub_id + 1;
    hold on
    tp_hist = histcounts(RTs(subj).rt, 'BinWidth',1, 'Normalization','pdf');
    xx = linspace(0,max(RTs(subj).rt),1000);
    yy = interp1(linspace(0,max(RTs(subj).rt),length(tp_hist)), tp_hist, xx,'spline');
    plot(xx,yy,'Color',grayRGB(sub_id,:), 'LineWidth',2)
    xlim([0 20])
end

title('Waiting Time Distributions')
xlabel('Waiting Time (s)'), ylabel('Probability')
ylim([0 0.55])
axis square
box on
set(gca, 'TickDir', 'out', 'TickLength', [0.01 0.01], 'XAxisLocation', 'bottom', 'YAxisLocation','left')
set(gca, 'FontWeight', 'bold', 'FontSize',font_size, 'LineWidth',2)
text(0, 1.07, 'D', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', font_size, 'FontWeight','bold');