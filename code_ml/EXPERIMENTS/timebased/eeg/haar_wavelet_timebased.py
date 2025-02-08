import itertools
import numpy as np
import ujson
import inspect
import os
import sqlite3
from pboost.feature.factory import BaseFeatureFactory, BaseFeatureFactoryManager

# Get initial parameters
SAMPLING_RATE = 500 # 500 Hz
INIT_TIME = -4000 # start of epoch in ms relative to stimulus onset
WINDOW = 500 # window size in ms

class BipolarFeatureFactoryManager(BaseFeatureFactoryManager):

  def produce(self, **kwargs):
    # Get parameters
    wl_step = kwargs.get('wl_step', 'FINE')
    wl_window = kwargs.get('wl_window', 'FINE')
    wl_pulse = kwargs.get('wl_pulse', 'FINE')
    starting_time = float(kwargs.get('starting_time', -3000))
    ending_time = float(kwargs.get('ending_time', -3000))
    # Get time indices for start and end of window
    t0 = int((starting_time - INIT_TIME) * SAMPLING_RATE / 1000)
    tf = int((ending_time - INIT_TIME) * SAMPLING_RATE / 1000)
    # Define step sizes, window sizes, and pulse sizes
    step_dict = {'COARSE': 50,
                 'MEDIUM': 20,
                 'FINE': 10}
    window_dict = {'FINE': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 23, 27, 31, 38, 49, 67, 106, 239]} 
    pulse_dict = {'COARSE': [0.125, ],
                  'MEDIUM': [0.125, 0.25],
                  'FINE': [0.125, 0.25, 0.375]}
    step = step_dict[wl_step]
    pulse = pulse_dict[wl_pulse]
    window = np.array(window_dict[wl_window])
    # Channel array
    channels = np.arange(0, self.data_handle.shape[2])
    # Make a feature for each wavelet
    for w in window:
      no_step = np.int(np.ceil((tf - t0 - w) / step))
      for n in np.arange(no_step):
        te = tf - n * step - 1
        ts = te - w
        for pls in pulse:
          pw = int(pls * w)
          # Skip if pulse width is 0
          if pw > 0:
            for chnl in channels:
              self.make(int(chnl),
                        int(ts),
                        int(te),
                        int(pw)
                        )


class BipolarFeatureFactory(BaseFeatureFactory):

  def load_data(self, data_handle, **kwargs):
    # Get channel array
    self.channels = np.arange(0, data_handle.shape[2])
    # Get time indices for start and end of window
    starting_time = float(kwargs.get('starting_time', -3000))
    ending_time = float(kwargs.get('ending_time', -3000))
    # Get baseline indices
    baseline_ending_time = -2500
    baseline_starting_time = -3000
    self.t0 = int((starting_time - INIT_TIME) * SAMPLING_RATE / 1000)
    self.tf = int((ending_time - INIT_TIME) * SAMPLING_RATE / 1000)
    self.bt0 = int((baseline_starting_time - INIT_TIME) * SAMPLING_RATE / 1000)
    self.btf = int((baseline_ending_time - INIT_TIME) * SAMPLING_RATE / 1000)
    # Get data for window
    N_ex = data_handle.shape[0] / 2
    pos = data_handle[:N_ex, self.t0:self.tf, self.channels]
    neg = data_handle[N_ex:, self.bt0:self.btf, self.channels]
    self.data = np.concatenate((pos, neg), axis=0)

  def blueprint(self, chnl, ts, te, pw=10):
    # Get position for wavelet
    te = te - self.t0
    ts = ts - self.t0
    chnl_ind = int(np.where(self.channels == chnl)[0])
    # Get value for positive peak of wavelet (double difference in cumulative sum ~ dot product with Haar wavelet)
    valp = self.data[:, te, chnl_ind] - self.data[:, te - pw, chnl_ind]
    # Get value for negative peak of wavelet (double difference in cumulative sum ~ dot product with Haar wavelet)
    valn = self.data[:, ts + pw, chnl_ind] - self.data[:, ts, chnl_ind]
    return valp - valn
