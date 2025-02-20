% ML Data Script for Figure 3

% This script generates figure 3.
%
% Key steps include:
% 1. Checking for required data files.
% 2. Extracting and restructuring EEG and MEG data ML results from csv files.
% 3. Determining earliest decoding times (EDT) based on based on both single trial and AUC method.
% 4. Generating a line plot to visualize AUC time courses (3 A).
% 5. Generating a box plot to visualize EDT comparisons (3 B).

%% Initialize workspace
% Clear workspace
close all
clear

% Ensure all dependencies are loaded
addpath('src\')

%% Load data

% Define required files and folder
folder              = "data_figure\";
required_files      = ["result_auc_timebased.csv", "result_auc_taskbased.csv", "result_auc_meg.xlsx", "timebased_edt_single.mat", "taskbased_edt_single.mat", "MRCP_onset.mat"];

% Check if data_figure folder exists in path
if ~isfolder(folder)
    error('Folder "%s" does not exist. Please create it before proceeding.', folder);
end

% Get list of missing files
current_files       = string({dir(folder).name});
missing_files       = setdiff(required_files, current_files);

% Check for missing files
if ~isempty(missing_files)
    fprintf('The following required files are missing:\n');
    fprintf('- %s\n', missing_files);
    error('Please generate or add the missing files before proceeding.');
else
    fprintf('All required files are present.\n');
end

% Load EEG and MEG data as both tables and matrices
timebased_mat       = readmatrix(folder + "result_auc_timebased.csv");
taskbased_mat       = readmatrix(folder + "result_auc_taskbased.csv");
timebased_tab       = readtable(folder + "result_auc_timebased.csv");
taskbased_tab       = readtable(folder + "result_auc_taskbased.csv");
meg_tab             = readtable(folder + "result_auc_meg.xlsx");

% Get time arrays
time_meg            = meg_tab.time; % MEG time array
time                = timebased_mat(2:152,3)/1000; % EEG time array

% Rearrange MEG data according to the 3 participants
meg_mat             = [meg_tab.auc_1'; meg_tab.auc_2'; meg_tab.auc_3'];
meg_sem             = [meg_tab.sem_1'; meg_tab.sem_2'; meg_tab.sem_3'];

% Initiate matrices for the EEG data
timebased_dat       = nan(151, 15);
taskbased_dat       = nan(151, 15);

% Rearrange EEG data according to the 15 participants

sub_id              = 0;
for subj=2:151:size(timebased_mat,1) % Each participant had 151 sliding windows
    sub_id                  = sub_id + 1;
    timebased_dat(:,sub_id) = timebased_mat(subj:(subj+150),4);
    taskbased_dat(:,sub_id) = taskbased_mat(subj:(subj+150),4);
end

% Compute SEM for EEG data:
taskbased_sem       = std(taskbased_dat, [], 2) / sqrt(size(taskbased_dat,2));
timebased_sem       = std(timebased_dat, [], 2) / sqrt(size(timebased_dat,2));

% Load EDT for single method and compute EDT for auc method

edt                 = nan(4,15);
load("data_figure\timebased_edt_single.mat")
edt(4,:)            = group_avg;
load("data_figure\taskbased_edt_single.mat")
edt(3,:)            = group_avg;

sub_id              = 0;
timebased_dat_lowerbound  = nan(151,15);
taskbased_dat_lowerbound  = nan(151,15);

% Get lower bound of the SEM
for subj=2:151:size(timebased_mat,1)
    sub_id                              = sub_id + 1;
    timebased_dat_lowerbound(:,sub_id)  = timebased_mat(subj:(subj+150),4) - timebased_mat(subj:(subj+150),5);
    taskbased_dat_lowerbound(:,sub_id)  = taskbased_mat(subj:(subj+150),4) - taskbased_mat(subj:(subj+150),5);
end

% Get 3 consecutive samples not above 0.5 (AUC method for participant level onset in 3 B)
for j = 1:15
    % For the task-based approach
    for i = fliplr(1:147)
        if (taskbased_dat_lowerbound(i,j) < 0.5) && (taskbased_dat_lowerbound(i+1,j) > 0.5) && (taskbased_dat_lowerbound(i+2,j) > 0.5) && (taskbased_dat_lowerbound(i+3,j) > 0.5)
            break
        end
        edt(1,j) = time(i+1);
    end
    % For the time-based approach
    for i = fliplr(1:147)
        if (timebased_dat_lowerbound(i,j) < 0.5) && (timebased_dat_lowerbound(i+1,j) > 0.5) && (timebased_dat_lowerbound(i+2,j) > 0.5) && (timebased_dat_lowerbound(i+3,j) > 0.5)
            break
        end
        edt(2,j) = time(i+1);
    end
end

%% Compute AUC onsets (3 A)
% according to the three methods compared to a baseline [-2.5 -2] s
bl_onset            = 1:25; % first 25 samples correspond to 500 ms which corresponds to the [-2.5 -2] s period

% AUC Onset: t-test method
for i = fliplr(1:149) % run from movement backwards
    % Check if signal is different from baseline for three consecutive samples
    if ~(ttest(squeeze(taskbased_dat(i,:)),mean(squeeze(taskbased_dat(bl_onset,:)))) && ttest(squeeze(taskbased_dat(i+1,:)),mean(squeeze(taskbased_dat(bl_onset,:)))) && ttest(squeeze(taskbased_dat(i+2,:)),mean(squeeze(taskbased_dat(bl_onset,:)))))
        AUCttest_method = time(i);
        break % Stop at the first instance that not all 3 consecutive samples are different from baseline running backwards from movement onset
    end
end

% AUC Onset: by eye method
for i = fliplr(1:149) % run from movement backwards
    % Check if signal is bigger than baseline for three consecutive samples
    if ~(mean(squeeze(taskbased_dat(i,:)),2)> mean(squeeze(taskbased_dat(bl_onset,:)), 'all') && mean(squeeze(taskbased_dat(i+1,:)),2)> mean(squeeze(taskbased_dat(bl_onset,:)), 'all') && mean(squeeze(taskbased_dat(i+2,:)),2)> mean(squeeze(taskbased_dat(bl_onset,:)), 'all'))
        AUCvisual_method = time(i);
        break % Stop at the first instance that not all 3 consecutive samples are bigger than baseline running backwards from movement onset
    end
end

% AUC Onset: 90% method
for j = fliplr(i:149) % run from movement backwards
    % Checks if signal is 90% of the area under the curve from movement onset up to the by-eye onset
    if sum(squeeze(mean(taskbased_dat(j:149,:),2))) > sum(squeeze(mean(taskbased_dat(i:149,:),2)))*0.9
        AUC90_method = time(j);
        break % Stops at the first instance the 90% area is reached
    end
end

% Load MRCP onsets [generated by make_figure_2]
load("data_figure\MRCP_onset.mat")

% Set font size and line thickness
font_size               = 15;
line_thickness          = 3.5;

%% %% Figure %% %%
% Generate figure - position values will determine the shape of the output
% (will depend on screen resolution)
fig                     = figure('Position',[-1919          41         973         955/2], 'Color','white'); % 

%%  AUC time course - Left plot (3 A)
subplot(1,2,1)
plot(time, mean(timebased_dat,2), 'Color', [0 .75 .5], 'LineWidth', line_thickness), hold on
plot(squeeze(time_meg/1000)', squeeze(nanmean(meg_mat,1)),'Color', [0.6 0 0.7], 'LineWidth', line_thickness)
plot(time, mean(taskbased_dat,2),'Color', [0 .5 .75], 'LineWidth', line_thickness)
text(0, 1.07, 'A', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', font_size, 'FontWeight','bold');

lower_sem_b             = squeeze(mean(timebased_dat,2) - timebased_sem);
lower_sem_c             = squeeze(mean(taskbased_dat,2) - taskbased_sem);
upper_sem_b             = squeeze(mean(timebased_dat,2) + timebased_sem);
upper_sem_c             = squeeze(mean(taskbased_dat,2) + taskbased_sem);

% Standard error of the mean
fill([time' fliplr(time')], [lower_sem_b' fliplr(upper_sem_b')], [0 .75 .5],'EdgeColor', 'none', 'FaceAlpha', .2)
fill([(time_meg/1000)' fliplr((time_meg/1000)')], [(squeeze(nanmean(meg_mat,1))-squeeze(nanstd(meg_mat,[],1))) fliplr(squeeze(nanmean(meg_mat,1))+squeeze(nanstd(meg_mat,[],1)))],[0.6, 0, 0.7],'EdgeColor', 'none', 'FaceAlpha', .2);
fill([time' fliplr(time')], [lower_sem_c' fliplr(upper_sem_c')],[0 .5 .75],'EdgeColor', 'none', 'FaceAlpha', .2)

xlim([-2.5 .5]) % window to focus closer to movement
ylim([0.45 1.05])
xticks([-2 -1 0])
yticks([0.5 0.6 0.7 0.8 0.9 1.0])
yticklabels({'0.5', '0.6', '0.7', '0.8', '0.9', '1.0'})
ylabel('Validation AUC')
xlabel('Time (s)')
grid on
axis square
box on
legend('Time-based_{EEG; OC}', 'Task-based_{MEG; PF}', 'Task-based_{EEG; OC}' , 'Location', 'NorthWest','AutoUpdate','off')
set(gca, 'TickDir', 'out', 'TickLength', [0.01 0.01], 'XAxisLocation', 'bottom', 'YAxisLocation','left')
set(gca, 'FontWeight', 'bold', 'FontSize',font_size, 'LineWidth',2)
title('Timecourse of validation AUC', 'FontSize', font_size)
xline(0,'--','LineWidth',2,'Alpha',0.5);

% Add onsets
xline(AUC90_method, 'k--')
text(AUCvisual_method-0.07, 0.63,'AUC onset: by eye','Rotation',90,'FontSize',10)
xline(AUCvisual_method, 'k--')
text(AUC90_method-0.07, 0.63,'AUC onset: 90%','Rotation',90,'FontSize',10)
xline(AUCttest_method, 'k--')
text(AUCttest_method-0.07, 0.63,'AUC onset: t-test','Rotation',90,'FontSize',10)

%% AUC EDT Boxplot - Right plot (3 B)
subplot(1,2,2)

boxplot(fliplr(edt'), 'Labels',{'Time-based', 'Task-based','Time-based', 'Task-based'})
ylabel('Earliest Decoding Times (s)')

axis square
text(0, 1.07, 'B', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', font_size, 'FontWeight','bold');
text(0.25, 1.03, 'Single trial', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', font_size, 'FontWeight','bold');
text(0.75, 1.03, 'AUC', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', font_size, 'FontWeight','bold');

set(gca, 'FontSize', font_size, 'FontWeight','bold', 'LineWidth',2)
axis square

h = findobj(gca, 'Tag', 'Box');
set(h, 'LineWidth', line_thickness)
set(h(1), 'Color',[0 .5 .75])
set(h(2), 'Color', [0 .75 .5])
set(h(3), 'Color',[0 .5 .75])
set(h(4), 'Color', [0 .75 .5])
grid on
yline(0)
yline(MRCPttest_method, 'r--','MRCP onset: t-test','LabelVerticalAlignment','bottom')
yline(MRCP90_method, 'r--','MRCP onset: 90%','LabelVerticalAlignment','bottom')
yline(MRCPvisual_method, 'r--','MRCP onset: by eye','LabelVerticalAlignment','bottom')
ylim([-2.5 0.5])
xline(2.48,'LineWidth',line_thickness/2,'Color','k')
xline(2.52,'LineWidth',line_thickness/2,'Color','k')
yline(0,'--','LineWidth',2,'Alpha',0.5);

% T-test for EDT (paired samples t-test, bidirectional - use default ttest(x,y) function)
[~,p_val_singleTrial,~,t_stat_singleTrial]  = ttest(edt(1,:),edt(2,:)); % For single trial method
[~,p_val_AUC,~,t_stat_AUC]                  = ttest(edt(3,:),edt(4,:)); % For AUC method