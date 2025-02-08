# The Time Course of Neural Activity Predictive of Impending Movement
 This repository contains the code used in the study "The Time Course of Neural Activity Predictive of Impending Movement".
## Overview
 This repository contains the code necessary to generate the figures of the preprint https://doi.org/10.31219/osf.io/ghs95. Data for that goes along with this code is available at https://osf.io/wm95q/ (DOI 10.17605/OSF.IO/WM95Q).<br><br>
 
 For all components to run you should have MATLAB installed, Python 3 and the software singularity (version 3.4.1).<br>
 For all components to run, first you need to download, unzip and merge into the repository and into their respective folders all the data from OSF (folders: data, data_ml, data_cubes and code_ml).<br>
 For all components to run, first you need to create the following folder: data_steps.<br><br>
 
 In the OSF repository you will find four folders:<br>
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
In the GitHub repository you will find three folders and four code files


## Generate figures

## Preprocess EEG data

## Run Haar-AdaBoost
Place data_cubes in the correct subject folders
Get_EDT_single.py
