# This is python 2.7 ran from a singularity container only (ubuPy_05_.img).

import configparser
import os
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

def transform(indir, subj, outdir, dtype="EEG", n_xval=10, trend_bgn=-4, trend_end=-3):
  """
  Processes EEG data for a given subject, applies preprocessing steps, and formats the data 
  for use in AdaBoost classification.
  
  Parameters:
  -----------
  indir : str
    Input directory containing subject data.
  subj : str
    Subject identifier.
  outdir : str
    Output directory to save processed data.
  dtype : str, optional (default="EEG")
    Type of data to process.
  n_xval : int, optional (default=10)
    Number of cross-validation folds.
  trend_bgn : int, optional (default=-4)
    Start time for baseline correction.
  trend_end : int, optional (default=-3)
    End time for baseline correction.
  
  Processing Steps:
  -----------------
  1. **Data Loading:**
    - Loads `.mat` files containing EEG data and labels.
    - Data is stored in a 3D array (channels x time x trials).
    - Extracts and reorders data to (trials x time x channels).
    - Loads the time vector for baseline correction.
  
  2. **Channel Pairwise Differences:**
    - Computes differences between all pairs of EEG channels to extract comparative information.
  
  3. **Baseline Correction & Cumulative Sum:**
    - Subtracts the mean of a pre-defined baseline period.
    - Computes the cumulative sum of the data to enhance feature extraction.
  
  4. **Cross-Validation Splitting:**
    - Ensures balanced class representation across `n_xval` folds.
    - Randomly shuffles and assigns fold indices for AdaBoost training.
    - Checks for at least one sample of each class per fold before proceeding.
  
  5. **Data Saving:**
    - Saves processed data, labels, and fold indices in an AdaBoost-friendly format for both 
     **task-based** and **time-based** approaches.
    - For the **time-based approach**, positive class trials are duplicated and relabeled to 
     create balanced training samples.
  
  Output:
  -------
  Two `.dat` files are saved in the specified output directory:
  - `taskbased/{subject}_{dtype}_OCC.dat` (Task-based cross-validation).
  - `timebased/{subject}_{dtype}_BOCC.dat` (Time-based cross-validation).
  """
  # Create lists to store data and labels
  x_li              = []
  y_li              = []
  # Get file names (e.g. PXX_data_cube.mat and time.mat)
  fnames            = sorted(os.listdir(os.path.join(indir, subj, dtype)))
  # Load each file in participant's data type folder
  for fname in fnames:
    fmat            = sio.loadmat(os.path.join(indir, subj, dtype, fname))
    if 'data_cube' in fmat.keys():
      # Load the data cube and labels
      x             = np.transpose(fmat['data_cube'][...], [2, 1, 0]) # re-order dimensions to trial x time x channel
      y             = fmat['labels'][:, 0]
      # Append data and labels to lists
      x_li.append(x)
      y_li.append(y)
    if 'time' in fmat.keys():
      # Load the time vector
      t             = fmat['time'][0]
  # Concatenate data and labels (useful for multisessions participants)
  data              = np.concatenate(x_li, axis=0)
  label             = np.concatenate(y_li, axis=0)

  ## Make channel by channel comparisons
  # Get number of channels
  n                 = data.shape[2]
  # Create a temporary data array to store pairwise channel differences
  data_tp           = np.zeros((data.shape[0], data.shape[1], int(n*(n+1)/2)), dtype=data.dtype)
  # Loop through each pairs of channels
  idx               = 0
  # Store single channels
  for i in range(0, n):
    data_tp[:,:,idx] = data[:,:,i]
    idx += 1
  # Store pairwise channel differences
  for i in range(1,n):
    for j in range(i):
      data_tp[:,:,idx] = data[:,:,i] - data[:,:,j] # pairwise channel differences
      idx += 1  
  # Replace with data containing pairwise channel differences
  data              = data_tp
  # Baseline correct and do cumulative sum 
  trend_indices     = np.where(np.logical_and(t >= trend_bgn, t <= trend_end))[0]
  # Get baseline start and end indices
  ind1              = trend_indices[0]
  ind2              = trend_indices[-1]
  # Get the mean of the data during the baseline period
  data_mean         = np.mean(data[:, ind1:ind2, :], axis=1)[:, np.newaxis, :]
  # Subtract the mean from the data (baseline correction)
  data              = data- data_mean
  # Take the cumulative sum of the data to save computing power (difference in amplitude between two points of a cumulative sum are equivalent to the dot product of a Haar wavelet with of the same width)
  data              = np.cumsum(data, axis=1)
  # Assign new arrays for positive samples
  pos               = data[label == 1]
  
  ## Save data and labels into AdaBoost-friendly format
  # Split data into training and testing sets
  # For task-based approach ensure there is at least one sample of each class in each fold

  # Task-based approach #
  while True:
    # Create temporary indexes to shuffle data
    tp_idxs = np.arange(len(label))
    np.random.shuffle(tp_idxs) # shuffle data
    # Create temporary data and labels according to shuffled indexes
    tp_lab = label[tp_idxs]
    tp_dat = data[tp_idxs,:,:]
    # Get number of trials
    n_samples = len(tp_lab)
    # Get indexes of positive and negative samples
    pos_label = np.where(tp_lab==1)
    neg_label = np.where(tp_lab==0)
    # Create indexes for cross-validation
    pos_idx_xval = np.tile(np.arange(1, n_xval+1)[:np.shape(pos_label)[1]], np.shape(pos_label)[1]//n_xval+1)[:np.shape(pos_label)[1]]
    neg_idx_xval = np.tile(np.arange(1, n_xval+1)[:np.shape(neg_label)[1]], np.shape(neg_label)[1]//n_xval+1)[:np.shape(neg_label)[1]]
    # Shuffle cross-validation indexes
    np.random.shuffle(pos_idx_xval)
    np.random.shuffle(neg_idx_xval)
    # Create 'indices' array to store cross-validation indexes for both classes
    indices = np.arange(n_samples)
    indices[pos_label] = pos_idx_xval
    indices[neg_label] = neg_idx_xval
    # Check if there is at least one sample of each class in each fold
    if (np.amax(indices[tp_lab == 1]) == n_xval) and (np.amax(indices[tp_lab==0]) == n_xval):
      # Save data, labels and fold indices into AdaBoost-friendly format
      with h5py.File(os.path.join(outdir, 'taskbased', dtype.lower(),
                                    '%s_%s_OCC.dat' % (subj,dtype)), 'w') as h5f:
        h5f.create_dataset('data', data=tp_dat)
        h5f.create_dataset('label', data=tp_lab)
        h5f.create_dataset('indices', data=indices)
      break
    else:
      print('... looking for new k-fold split ..')
  # End of task-based approach #

  # Time-based approach #
  with h5py.File(os.path.join(outdir, 'timebased', dtype.lower(),
                              '%s_%s_BOCC.dat' % (subj, dtype)), 'w') as h5f:
    # Create new data array with manual trials (positive) repeated
    newdata = np.concatenate((pos, pos), axis=0)
    # Get indices for cross-validation
    N_ex = pos.shape[0]
    halfindices = np.repeat(np.arange(1, n_xval + 1), N_ex // n_xval + 1)[:N_ex]
    np.random.shuffle(halfindices)
    indices = np.r_[halfindices, halfindices]
    # Re-assign class (half is baseline class, half is manual class)
    new_label = np.ones(N_ex * 2)
    new_label[N_ex:] = 0
    # Save data, labels and fold indices into AdaBoost-friendly format
    h5f.create_dataset('data', data=newdata)
    h5f.create_dataset('label', data=new_label)
    h5f.create_dataset('indices', data=indices)
  # End of time-based approach #

## Configuration file generator for AdaBoost classification with pboost for task-based approach
def cfg_generator_taskbased(cfg_fname,
                  working_dir,
                  subjs,
                  dtype="EEG",
                  first_lead=-2500,
                  last_lead=501,
                  timestep=20,
                  window=500,
                  omp_threads=16,
                  n_xval=10, rounds=200,
                  factory_files=['moving_average_taskbased.pyc', 'haar_wavelet_taskbased.pyc'],
                  algorithm='adaboost-fast',
                  max_memory=9,
                  tree_depth=1,
                  wl_step='FINE',
                  wl_window='FINE',
                  wl_pulse='FINE'):
  """
  Generates a configuration file for AdaBoost classification based on EEG/physiological data.

  This function creates a configuration file (.cfg) used for machine learning classification, 
  specifically AdaBoost. It defines multiple configurations based on different time windows 
  for each participant's data.

  Parameters:
  - cfg_fname (str): Name of the output configuration file.
  - working_dir (str): Directory where results and temporary files will be stored.
  - subjs (list of str): List of subject identifiers.
  - dtype (str, optional): Data type (default: "EEG").
  - first_lead (int, optional): End time of the first analysis window (default: -2500 ms).
  - last_lead (int, optional): End time of the last analysis window (default: 501 ms).
  - timestep (int, optional): Step size for shifting the analysis window (default: 20 ms).
  - window (int, optional): Duration of each analysis window (default: 500 ms).
  - omp_threads (int, optional): Number of OpenMP threads to use (default: 16).
  - n_xval (int, optional): Number of cross-validation folds (default: 10).
  - rounds (int, optional): Number of boosting rounds for AdaBoost (default: 200).
  - factory_files (list of str, optional): List of precompiled feature extraction files 
  (default: ['moving_average_taskbased.pyc', 'haar_wavelet_taskbased.pyc']).
  - algorithm (str, optional): Machine learning algorithm to use (default: 'adaboost-fast').
  - max_memory (int, optional): Maximum memory usage in GB (default: 9).
  - tree_depth (int, optional): Depth of decision trees used in AdaBoost (default: 1; i.e. stump).
  - wl_step (str, optional): Weak learner step size setting (default: 'FINE').
  - wl_window (str, optional): Weak learner window setting (default: 'FINE').
  - wl_pulse (str, optional): Weak learner pulse setting (default: 'FINE').

  Functionality:
  1. Iterates over each subject and generates multiple time windows.
  2. Creates a new configuration section for each time window.
  3. Saves configurations to a file in the specified working directory.

  Output:
  - A configuration file containing multiple time-windowed configurations for AdaBoost training.
  """
  # Create configuration file for AdaBoost classification
  cfg = configparser.RawConfigParser()
  cfg.optionxform = str
  # Initialize configuration number
  confnum = 1
  # Loop through each participant
  for subj in subjs:
    for ending_time in np.arange(first_lead, last_lead, timestep):
      starting_time = ending_time - window
      strconfnum = 'Configuration %i' % confnum
      cfg.add_section(strconfnum)
      cfg.set(strconfnum, 'train_file', '%s_%s_OCC.dat' % (subj, dtype))
      cfg.set(strconfnum, 'test_file', '')
      cfg.set(strconfnum, 'factory_files', ','.join(factory_files))
      cfg.set(strconfnum, 'algorithm', algorithm)
      cfg.set(strconfnum, 'rounds', rounds)
      cfg.set(strconfnum, 'xval_no', n_xval)
      cfg.set(strconfnum, 'working_dir', working_dir)
      cfg.set(strconfnum, 'max_memory', max_memory)
      cfg.set(strconfnum, 'show_plots', 'n')
      cfg.set(strconfnum, 'omp_threads', omp_threads)
      cfg.set(strconfnum, 'deduplication', 'n')
      cfg.set(strconfnum, 'tree_depth', tree_depth)
      cfg.set(strconfnum, 'wl_step', wl_step)
      cfg.set(strconfnum, 'wl_window', wl_window)
      cfg.set(strconfnum, 'wl_pulse', wl_pulse)
      cfg.set(strconfnum, 'starting_time', starting_time)
      cfg.set(strconfnum, 'ending_time', ending_time)
      confnum += 1
  with open(cfg_fname, 'w') as configfile:
    cfg.write(configfile)

## Configuration file generator for AdaBoost classification with pboost for time-based approach
def cfg_generator_timebased(cfg_fname,
                  working_dir,
                  subjs,
                  dtype="EEG",
                  first_lead=-2500,
                  last_lead=501,
                  timestep=20,
                  window=500,
                  omp_threads=16,
                  n_xval=10, rounds=200,
                  factory_files=['moving_average_timebased.pyc', 'haar_wavelet_timebased.pyc'],
                  algorithm='adaboost-fast',
                  max_memory=9,
                  tree_depth=1,
                  wl_step='FINE',
                  wl_window='FINE',
                  wl_pulse='FINE',
                  subset='ALL',
                  data_type='ONLY_EEG'):
  """
  Generates a configuration file for AdaBoost classification based on EEG/physiological data.

  This function creates a configuration file (.cfg) used for machine learning classification, 
  specifically AdaBoost. It defines multiple configurations based on different time windows 
  for each participant's data.

  Parameters:
  - cfg_fname (str): Name of the output configuration file.
  - working_dir (str): Directory where results and temporary files will be stored.
  - subjs (list of str): List of subject identifiers.
  - dtype (str, optional): Data type (default: "EEG").
  - first_lead (int, optional): End time of the first analysis window (default: -2500 ms).
  - last_lead (int, optional): End time of the last analysis window (default: 501 ms).
  - timestep (int, optional): Step size for shifting the analysis window (default: 20 ms).
  - window (int, optional): Duration of each analysis window (default: 500 ms).
  - omp_threads (int, optional): Number of OpenMP threads to use (default: 16).
  - n_xval (int, optional): Number of cross-validation folds (default: 10).
  - rounds (int, optional): Number of boosting rounds for AdaBoost (default: 200).
  - factory_files (list of str, optional): List of precompiled feature extraction files 
  (default: ['moving_average_taskbased.pyc', 'haar_wavelet_taskbased.pyc']).
  - algorithm (str, optional): Machine learning algorithm to use (default: 'adaboost-fast').
  - max_memory (int, optional): Maximum memory usage in GB (default: 9).
  - tree_depth (int, optional): Depth of decision trees used in AdaBoost (default: 1; i.e. stump).
  - wl_step (str, optional): Weak learner step size setting (default: 'FINE').
  - wl_window (str, optional): Weak learner window setting (default: 'FINE').
  - wl_pulse (str, optional): Weak learner pulse setting (default: 'FINE').

  Functionality:
  1. Iterates over each subject and generates multiple time windows.
  2. Creates a new configuration section for each time window.
  3. Saves configurations to a file in the specified working directory.

  Output:
  - A configuration file containing multiple time-windowed configurations for AdaBoost training.
  """
  cfg = configparser.RawConfigParser()
  cfg.optionxform = str
  confnum = 1
  for subj in subjs:
    for ending_time in np.arange(first_lead, last_lead, timestep):
      starting_time = ending_time - window
      strconfnum = 'Configuration %i' % confnum
      cfg.add_section(strconfnum)
      cfg.set(strconfnum, 'train_file', '%s_%s_BOCC.dat' % (subj, dtype))
      cfg.set(strconfnum, 'test_file', '')
      cfg.set(strconfnum, 'factory_files', ','.join(factory_files))
      cfg.set(strconfnum, 'algorithm', algorithm)
      cfg.set(strconfnum, 'rounds', rounds)
      cfg.set(strconfnum, 'xval_no', n_xval)
      cfg.set(strconfnum, 'working_dir', working_dir)
      cfg.set(strconfnum, 'max_memory', max_memory)
      cfg.set(strconfnum, 'show_plots', 'n')
      cfg.set(strconfnum, 'omp_threads', omp_threads)
      cfg.set(strconfnum, 'deduplication', 'n')
      cfg.set(strconfnum, 'tree_depth', tree_depth)
      cfg.set(strconfnum, 'wl_step', wl_step)
      cfg.set(strconfnum, 'wl_window', wl_window)
      cfg.set(strconfnum, 'wl_pulse', wl_pulse)
      cfg.set(strconfnum, 'subset', subset)
      cfg.set(strconfnum, 'data_type', data_type)
      cfg.set(strconfnum, 'starting_time', starting_time)
      cfg.set(strconfnum, 'ending_time', ending_time)
      confnum += 1
  with open(cfg_fname, 'w') as configfile:
    cfg.write(configfile)

## Generate results csv from the output of AdaBoost classification (out_* folders)

def combine(working_dir, cfg_fname):
  """
  Combines classification results from multiple configurations into a CSV file.

  This function reads a configuration file (`.cfg`) and extracts relevant classification 
  results from multiple output files (`dump.npz` and `hypotheses.npy`) stored in 
  subdirectories. It compiles the extracted data into a structured pandas DataFrame and 
  saves it as `result.csv` in the specified working directory.

  Parameters:
  - working_dir (str): The directory where output files (`dump.npz` and `hypotheses.npy`) are stored.
  - cfg_fname (str): The configuration file (`.cfg`) specifying the experiment details.

  Functionality:
  1. Reads the `.cfg` file to extract relevant configurations.
  2. Iterates through each configuration section:
    - Retrieves the subject ID and lead time.
    - Loads the corresponding output data from `dump.npz` and `hypotheses.npy`.
    - Extracts validation AUC mean and standard deviation.
    - Extracts feature values and their corresponding weights.
  3. Constructs a pandas DataFrame with the following columns:
    - `subject`: Subject ID.
    - `lead`: Ending time of the time window.
    - `auc_mean`: Mean validation AUC.
    - `auc_std`: Standard deviation of validation AUC.
    - `feature_1, feature_2, ...`: Extracted feature values from hypotheses.
    - `weight_1, weight_2, ...`: Corresponding weights for each feature.
  4. Saves the DataFrame as `result.csv` in `working_dir`.

  Output:
  - A CSV file (`result.csv`) containing the aggregated classification results.

  Dependencies:
  - `configparser` for parsing the configuration file.
  - `numpy` (`np.load()`) for loading `.npz` and `.npy` files.
  - `pandas` for DataFrame manipulation and saving results as CSV.

  """
  cfg = configparser.RawConfigParser()
  cfg.optionxform = str
  with open(cfg_fname, 'r') as configfile:
    cfg.read_file(configfile)
  out = []
  for section in cfg.sections():
    subj = cfg[section]['train_file'].split('_')[0]
    lead = cfg[section]['ending_time']
    confnum = int(section.split(' ')[-1])
    fpath = os.path.join(working_dir, 'out_%i' % confnum, 'dump.npz')
    ff = np.load(fpath)
    f2path = os.path.join(working_dir, 'out_%i' % confnum, 'hypotheses.npy')
    ff2 = np.load(f2path, allow_pickle=True)
    val_auc_mean = ff['val_auc_mean'][...]
    val_auc_std = ff['val_auc_std'][...]
    values_vector = [item['hypothesis']['fn_def'][2] for item in ff2]
    weights_vector = [item['c1'] for item in ff2]
    out.append([subj, lead, val_auc_mean, val_auc_std] + values_vector + weights_vector)
  # Create a DataFrame from the output list
  columns = ['subject', 'lead', 'auc_mean', 'auc_std'] + ['feature_{}'.format(i+1) for i in range(len(values_vector))] + ['weight_{}'.format(i+1) for i in range(len(weights_vector))]
  df = pd.DataFrame(out, columns=columns)
  fpath = os.path.join(working_dir, 'result.csv')
  df.to_csv(fpath)
