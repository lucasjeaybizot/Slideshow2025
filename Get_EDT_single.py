"""
This script generates the Earliest Decoding Time (EDT) for each participant at the single trial level based on the AdaBoost model predictions.

Steps:
1. Predict labels for task and time using the AdaBoost model.
2. Compare the predicted labels to the true labels to determine correct and incorrect predictions at each single trial.
3. Initialize arrays to store EDT values for task and time.
4. For each trial, reverse the prediction data up to a certain point (movement onset).
5. Identify the first incorrect prediction in the reversed data.
6. Record the EDT value for each trial based on the time of the first incorrect prediction.

Variables:
- predict_label_task: Predicted labels for tasks.
- predict_label_time: Predicted labels for time.
- true_predictions_task: Array indicating correct (1) or incorrect (0) task predictions.
- true_predictions_time: Array indicating correct (1) or incorrect (0) time predictions.
- part_edt_task: Array to store average EDT values for task-based per participant.
- part_edt_time: Array to store average EDT values for time-based per participant.
- reversed_data: Reversed prediction data up to movement onset.
- time_flip: Array containing flipped time values.

Output:
- group_avg_task: Array containing average EDT values for task-based per participant.
- group_avg_time: Array containing average EDT values for time-based per participant.
"""

# Load dependencies
import numpy as np
from scipy.io import savemat

group_avg_task  = np.empty((15,1))
group_avg_time  = np.empty((15,1))
time            = np.linspace(-2.5,0.5,151)
time_flip       = np.flip(time[:125]) # :125

# Loop through each participant (N=15)
for subj in range(15):
    j               = (subj)*151 + 0

    # Load task-based window
    filename_task   = 'out_taskbased/out_{}/dump.npz'.format(j+1)
    dump_task       = np.load(filename_task, allow_pickle=True)

    # Load time-based window
    filename_time   = 'out_timebased/out_{}/dump.npz'.format(j+1)
    dump_time       = np.load(filename_time, allow_pickle=True)

    # Create empty arrays to store the predictions
    true_predictions_task = np.empty((len(dump_task['validation_predictions']), 151))
    true_predictions_time = np.empty((len(dump_time['validation_predictions']), 151))

    # Loop through each window (151 per participant; each window is one out folder)
    for wind in range(151):
        j               = (subj)*151 + wind
        filename_time   = 'out_timebased/out_{}/dump.npz'.format(j+1)
        filename_task   = 'out_taskbased/out_{}/dump.npz'.format(j+1)
        dump_task       = np.load(filename_task, allow_pickle=True)
        dump_time       = np.load(filename_time, allow_pickle=True)

        # Get labels -1 for automatic and 1 for manual
        true_label_task = np.sign(dump_task['train_label']-0.5)
        true_label_time = np.sign(dump_time['train_label']-0.5)

        # Predict the label based on AdaBoost model
        predict_label_task = np.sign(np.sum(dump_task['validation_predictions'],axis=1))
        predict_label_time = np.sign(np.sum(dump_time['validation_predictions'],axis=1))

        # Compare true label to predicted label (1 if correct, 0 if incorrect)
        true_predictions_task[:,wind] = true_label_task==predict_label_task
        true_predictions_time[:,wind] = true_label_time==predict_label_time

    ## Compute the EDT for each trial ##

    # Create empty arrays to store the EDT values
    part_edt_task    = np.empty((1,true_predictions_task.shape[0]))
    part_edt_time    = np.empty((1,true_predictions_time.shape[0]))

    # Loop through each trial - task-based
    for trl in range(true_predictions_task.shape[0]):
        # Cutting data at movement onset and flipping it
        reversed_data = np.flip(true_predictions_task[trl,:125])

        # Find the first samples that is not correctly classified
        for idx in range(len(reversed_data)):
            if reversed_data[idx] == 0:
                break

        # Save the EDT value for the trial in time (s)
        part_edt_task[:,trl] = time_flip[idx]

    # Loop through each trial - time-based
    for trl in range(true_predictions_time.shape[0]):
        # Cutting data at movement onset and flipping it
        reversed_data = np.flip(true_predictions_time[trl,:125])

        # Find the first samples that is not correctly classified
        for idx in range(len(reversed_data)):
            if reversed_data[idx] == 0:
                break

        # Save the EDT value for the trial in time (s)
        part_edt_time[:,trl] = time_flip[idx]

    # Add the mean EDT for the participant across trials to the group average
    group_avg_task[subj,:] = np.mean(part_edt_task)
    group_avg_time[subj,:] = np.mean(part_edt_time)

# Save the group average EDT values to a .mat file
savemat("taskbased_edt_single.mat", {"group_avg": group_avg_task})
savemat("timebased_edt_single.mat", {"group_avg": group_avg_time})
