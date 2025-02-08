# The Time Course of Neural Activity Predictive of Impending Movement
 This repository contains the code used in the study "The Time Course of Neural Activity Predictive of Impending Movement".
## Overview & Inventory
 This repository contains the code necessary to generate the figures of the preprint https://doi.org/10.31219/osf.io/ghs95. Data for that goes along with this code is available at https://osf.io/wm95q/ (DOI 10.17605/OSF.IO/WM95Q).<br><br>
 
 For all components to run you should have MATLAB (preferred 2023b) with fieldtrip installed (preferred 20240129), Python 3 (preferred 3.8.5) and the software singularity (version 3.4.1).<br>
 For all components to run, first you need to download, unzip and merge into the repository and into their respective folders all the data from OSF (folders: data, data_ml, data_cubes and code_ml).<br>
 For all components to run, first you need to create the following folder: data_steps.<br><br>
 
 **In the OSF repository you will find four folders:** <br>
 * **data** This contains the raw data (.bdf) zipped (.zip) per session for all 17 participants (including 2 excluded from analysis)
 * **data_cubes** This contains the preprocessed data (.mat) from the 15 included participants, one file each.<br>
   Each data file contains two MATLAB variables:
   * data_cubes: a 3D array of EEG activity for channels-by-time-by-trials.
     * There are 23 channels in the following order: F1, F3, FC3, FC1, C1, C3, CP3, CP1, P1, Pz, CPz, Fz, F2, F4, FC4, FC2, FCz, Cz, C2, C4, CP4, CP2 and P2.
     * There are 2251 time points at 500 Hz. Slide transition occurs at timepoint 2000.
   * labels: a 1D logical array of 1 for active and 0 for passive.
 * **data_ml** This contains the output of the AdaBoost model. There are two subfolders, one for the timebased and one for the taskbased approach (.zip). Each contains 2265 out subfolders.
   * Each out subfolder correspond to one sliding window for one participant with the following files:
     * _dump.npz_: a numpy file that contains the mean AUC for that fold 'val_auc_mean' and its associated standard deviation 'val_auc_std'.
     * _final_classifier.py_: a python file that can be used to predict new data using the fully trained model using all folds (unsused in this analysis).
     * _haar_wavelet_timebased.pyc_: The Haar wavelet base learner function.
     * _hypotheses.npy_: a numpy file that contains all 200 rounds with with the associated weights for class 1 'c1' and class 0 'c0' as well as the selected learner (e.g. haar_wavelet_XX.py or moving_average_XX.py with their inputs in 'fn_def').
     * _moving_average_timebased.pyc_: the moving average base learner function.
     * _roc.png_: The receiver operating curve of the final model (plotted).
     * _training_auc_: The training AUC of the model as a function of number of rounds of boosting (plotted).
     * _training_err_: The training error of the model as a function of number of rounds of boosting (plotted).
     * _validation_auc_: The validation AUC (in k-fold) of the model as a function of number of rounds of boosting (plotted).
     * _validation_err_: The validation error (in k-fold) of the model as a function of number of rounds of boosting (plotted).
 * **code_ml** This folder contains the container required to run pboost: ubuPy2_05_.img.<br><br>

**In the GitHub repository you will find three folders and four code files:** <br>
* **code_ml** This folder contains two subfolders (DATA and EXPERIMENTS), two python scripts and one bash script.<br>
  * To use this folder, put the container image (ubuPy2_05_.img) in the repository. Put the participants data_cubes (.mat) into their respective subject folders (DATA/SUBJXX/EEG/). Push the repository (code_ml) to your HPC cluster into a folder named 'pboost'. Run 'run_pboost.sh'.
    * _run.py_: parallelizes adaboost and run adaboost with the selected base learners.
    * _run_pboost.sh_: This will perform three main steps.
      * first preprocess the data cubes for pboost (baseline, cumulative sum, random 10-fold split using utils.transform()).
      * Then generate a config file for pboost using utils.cfg_generator().
      * Then run pboost using run.py. Note: to process faster, send configs to different nodes in parallel if possible. 
    * _utils.py_: contains 4 functions.
      * _transform()_ takes data cubes and generates .dat files for both timebased and taskbased approach. Uses cumulative sum for faster Haar wavelet dot products during boosting. This function will also perform k-fold split, note that each run will obtain slightly different models based on the random k-fold split.
      * _cfg_generator_taskbased()_: creates a config file indicate all parameters for AdaBoost for each sliding window for the taskbased approach.
      * _cfg_generator_timebased()_: creates a config file indicate all parameters for AdaBoost for each sliding window for the timebased approach.
      * _combine()_: This will create the result.csv file by extracting the mean AUC according to the .cfg file and the data in the /EXPERIMENTS/XXXXbased/eeg/out_XX folders.
    * **EXPERIMENTS** contains the outputs of the model. Needs to be structured with a timebased and a taskbased subfolder each containing an eeg subfolder. Each eeg subfolders need to contain the _haar_wavelet_XXXXbased.py_ and _moving_average_XXXXbased.py_ files that define the base learners.
    * **DATA** contains subfolders for each participants' data cube such that each participant's subfolder is named SUBXX where XX is the ID and each of these contains an EEG folder with the .mat data cube and a time.mat array with the time as 1D array for 2251 samples at 500 Hz.
* **data_figure** This folder contains 6 .mat files, 2 .csv and 1 .xlsx. These are all the files sufficient to generate figures 2 & 3.
  * _MRCP_onset.mat_: contains the 3 MRCP onset values.
  * _RT.mat_: contains a structure with each participant's waiting times in the active trials.
  * _erp_cube.mat_: contains a 4D array of condition-by-channel-by-time-by-participant EEG averages at each channels.
  * _ga_power.mat_: contains a 3D array of frequency-by-time-by-participant average power at C3. frex and tftimes contains the frequency and time indexes values.
  * _result_auc_meg.xlsx_: contains the AUC timecourses for each of the 3 PF MEG participants for the taskbased approach.
  * _result_auc_taskbased.csv_: contains the AUC timecourses (and feature info) for each of the 15 OC EEG participants for the taskbased approach.
  * _result_auc_timebased.csv_: contains the AUC timecourses (and feature info) for each of the 15 OC EEG participants for the timebased approach.
  * _taskbased_edt_single.mat_: contains the earliest decodable time (using the single trial method) for each of the 15 participants for the taskbased approach.
  * _timebased_edt_single.mat_: contains the earliest decodable time (using the single trial method) for each of the 15 participants for the timebased approach.
* **src** This folder contains 1 .mat file and 4 .m functions used in the analysis.
  * _BIOSEMI_labels.mat_: This file contains the correct channel labels in the order of A1-32 then B1-32 for the raw EEG data.
  * _Get_ERD.m_: This file extracts power at C3 in the active condition using complex morlet wavelets. Generates _ga_power.mat_. Needs data_cube folder to be operational.
  * _Get_ERP.m_: This file extracts average EEG activity for each participants. Generates _erp_cube.mat_. Needs data_cube folder to be operational.
  * _Get_WT.m_: This file extracts the wait time in the active condition for each participant. Generates _RT.mat_. Needs data folder to be operational.
  * _interpolate_fieldtrip.m_: For the channels missing, this function will interpolate them using the distance weighted average of the nearest 5 channels.
* _Get_EDT_single.py_: This code extracts from taskbased and timebased folders the earliest decodable time from the out subfolders. This code generates _taskbased_edt_single.mat_ and _timebased_edt_single.mat_. This code needs data_ml to be operational and needs to be ran from data_ml.
* _make_figure_2.m_: This code will generate figure 2. It will run without any processing needed. This code needs data_figure and src to be operational.
* _make_figure_3.m_: This code will generate figure 3. It will run without any processing needed. This code needs data_figure and src to be operational.
* _preprocess.m_: This code will turn raw .bdf files from **data** into data cubes in **data_cubes**. This code will prompt the user for identifying noisy channels, ICA components and noisy trials. For a description of the steps see the preprint and the in-line comments in the script.This code needs src, data, data_cubes and data_steps (data_cubes and data_steps can be empty). It also requires fieldtrip toolbox.
  

## Generate figures

* Step 1: clone this repository
* Step 2: run make_figure_2.m to generate Fig. 2 OR make_figure_3.m to generate Fig. 3
* Troubleshoot: edit paths if on mac/linux. 

## Preprocess EEG data

* Step 1: clone this repository
* Step 2: edit your ft_rejectvisual function by setting the top line to "function [data, chansel, trlsel] = ft_rejectvisual(cfg, data)"
* Step 3: create empty 'data_steps' and 'data_cubes' folders in the repository
* Step 4: Import and unzip the raw data from OSF into a 'data' folder
* Step 5: Run preprocess.m
* Step 6 (each participant): remove noisy channels
* Step 7 (each participant): identify ocular components and report them in the command window using e.g "[2 4]" for components 2 and 4
* Step 8 (each participant): remove noisy trials visually
* Troubleshoot: edit paths if on mac/linux. 
 
## Run Haar-AdaBoost

* Step 1: clone this repository
* Step 2: push code_ml to your HPC cluster (or environment of your choice that has the software singularity)
* Step 3: From OSF download data cubes and upload them to DATA/SUBJXX/EEG/ where XX stands for participant number (total 15).
* Step 4: From OSF download ubuPy2_05_.img and add it to code_ml
* Step 5: rename code_ml to pboost
* Step 6: Run 'bash run_pboost.sh'
* Tips: Parallelize line 11 and 12 of run_pboost.sh by spreading '1-2265' across nodes
* Results: resulting files will be in EXPERIMENTS/taskbased/eeg

Place data_cubes in the correct subject folders
Get_EDT_single.py
