function Get_WT()
% GET_WT Extracts the wait times (RTs) for each subject from event files and saves them.
% 
%   This function processes EEG event data to calculate the wait times (reaction times, RT)
%   for multiple subjects in manual trials. It loops through the subjects, reading their corresponding event 
%   data from `.bdf` files, filtering for 'STATUS' events, and calculating RTs by extracting 
%   the difference between events with specific values. The wait times are stored in a structure 
%   and then saved to a MATLAB `.mat` file for future use.
%
%   RTs are calculated by the difference between the 'STATUS' events with specific values: 
%   `value == 1` (start) and `value == 8` (end). The resulting time is scaled by the sampling 
%   rate (2048 Hz), converting sample numbers to time in seconds.
%
%   The RTs are stored in a structure `RTs` where each subject's RT data is saved as an entry 
%   with the subject number as the field. After the loop, the RTs are saved to a `.mat` file in 
%   the directory `../data_figure/RT.mat` for further analysis or plotting.
%
% Inputs:
%   None (the function uses hardcoded file paths based on subject numbers)
%
% Outputs:
%   A `.mat` file named `RT.mat` containing a structure `RTs`, which has the RTs for all subjects.
%
% Notes:
%   - The function excludes subjects 4 and 16 from processing.
%   - For subject 6, special processing due a workspace mistake during recording.


% Initialize a structure to store RTs for each subject
RTs = struct();

% Loop through all subjects, excluding subjects 4 and 16
for subj = setdiff(1:17, [4 16])
    
    % For participants with 2 sessions
    if ismember(subj, [1:5 7:10 14])
        filenames = {sprintf('data/P%02dS01.bdf', subj),  sprintf('data/P%02dS02.bdf', subj)};
        ev = ft_read_event(filenames);
    % One session of participant 6 was ran in the wrong workspace. To concatenate, each session needs to be load separately.
    elseif subj == 6
        filenames1 = {sprintf('data/P%02dS01.bdf', subj)};
        ev1                 = ft_read_event(filenames1{:});
        filenames2 = {sprintf('data/P%02dS02.bdf', subj)};
        ev2                 = ft_read_event(filenames2{:});
        ev1(1:3)            = [];                                          % remove weird initial trigger at t0
        ev1                  = ev1(strcmp({ev1.type},'STATUS'));
        ev2                  = ev2(strcmp({ev2.type},'STATUS'));
        for st_i = 1:length(ev2)
            ev2(st_i).sample = ev2(st_i).sample + ev1(end).sample;
        end
        ev = [ev1 ev2];
    % For participants with 1 session
    else
        filenames = {sprintf('data/P%02dS01.bdf', subj)};
        ev = ft_read_event(filenames{:});
    end

    % Filter the events to keep only those of type 'STATUS'
    ev                  = ev(strcmp({ev.type},'STATUS'));
    

    % Calculate the wait times (RT) by subtracting the sample of event 1 (value==1; trial start) from event 8 (value==8; manual trial)
    % Divide by 2048 to convert samples to seconds.
    RTs(subj).rt=([ev([ev.value]==8).sample]- [ev([ev.value]==1).sample])/2048;
end

% Save the computed RTs structure to a .mat file
save('data_figure/RT.mat', 'RTs')

end

