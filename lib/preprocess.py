#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:33:12 2017

@author: jannes
"""

import numpy as np
from scipy.signal import butter, lfilter

import lib.io as io

def strip_data(data, rinfo_path, tinfo_path, onset, start=-100, length=1400):
    """Cuts everything before 100 ms before stim or match onset and after a 
    defined stop point (trial length, passed in ms).
    
    Args:
        align_on: A str ['stim'|'match']. Start and stop either relative to 
            stimulus onset or to match onset.
        start: An int. Drop everything before, relative to stim/match onset.
        stop: An int. Drop everything after, relative to stim/match onset.
    """
    
    # Get srate and sample onset times from recording/trial info files
    srate = io.get_sfreq(rinfo_path)
    
    # Drop leading and trailing samples
    data = drop_lead(data, srate, onset, start)
    data = drop_trail(data, srate, length)
    
    return data


def drop_lead(raw, srate, onset, start=-100):
    """drop_lead takes a chunk of raw data and cuts everything until 100 ms 
    before sample onset.
    
    Args:
        raw: An ndarray. Raw data ([channels x timepoints]).
        srate: A float. Sampling rate of the raw data.
        onset: A float. Stimulus/match onset in ms.
        start: An int. Start relative to stimulus onset (negative: include n ms
            before stimulus onset; positive: start from onset + n ms)
    
    Returns: 
        An ndarray of the sliced data.
    """
    samples_per_ms = srate/1000
    samples_to_drop = (onset + start) * samples_per_ms
    new_start = int(np.floor(samples_to_drop))
    return raw[:,new_start:]


def drop_trail(raw, srate, length=1400):
    """Cuts off every trial at a certain treshold (by default: 100 ms pre + 
    500 ms sample + 800 ms delay = 1400 ms).
    
    Args:
        raw: An ndarray. Raw data ([channels x timepoints]).
        srate: A float. Sampling rate of the raw data.
        trial_length: An int. Intended length of trial (in ms).
        
    Returns: 
        An ndarray of the sliced data.
    """
    samples_per_ms = srate/1000
    no_of_samples = int(np.floor(length * samples_per_ms))
    return raw[:,:no_of_samples]


def downsample(data, init_srate, target_srate):
    """
    Takes the data set and initial and new sampling rate as inputs to downsample
    the data by a given factor.
    
    :param data: dataset
    :param init_srate: original sampling rate
    :param target_srate: new sampling rate
    """
    steps = int(np.floor(init_srate / target_srate))  # can only take integer indices
    return data[:,::steps]


def butter_bandpass_filter(data, lowcut, highcut, srate, order=1):
    """
    Applies butterworth filter to the data. Takes lowcut and highcut params as input
    as well as the srate of the dataset.
    
    :param data: dataset
    :param lowcut: high-pass frequency
    :param highcut: low-pass frequency
    :param srate: sampling rate
    :return: filtered data (array)
    """
    nyq = 0.5 * srate  # nyqvist frequency
    low = lowcut / nyq  # high pass filter
    high = highcut / nyq  # low pass filter
    b, a = butter(order, [low, high], btype='bandpass')
    return lfilter(b, a, data)  # apply filter to the raw data and return filtered