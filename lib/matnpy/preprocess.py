#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:33:12 2017

@author: jannes
"""
import numpy as np
from scipy.signal import butter, lfilter

import matnpyio as io

def strip_data(data, rinfo_path, onset, start, length=500):
    """Strips data to relevant interval defined by start (relative to onset) and
    length.
    
    Args:
        data: An ndarray. Raw data to strip.
        rinfo_path: A str. Path to the rinfo file containing the sfreq/onsets.
        onset: An int. Time point to align on (usually stimulus/match onset).
        start: An int. Drop everything before (relative to onset).
        length: An int. Length of interval.

    Returns:
        An ndarray of the stripped data.
    """
    srate = io.get_sfreq(rinfo_path)
    data = drop_lead(data, srate, onset, start)
    data = drop_trail(data, srate, length)
    return data

def drop_lead(raw, srate, onset, start):
    """Drops all samples before start parameter (in ms, relative to onset).
    
    Args:
        raw: An ndarray. Raw data ([channels x timepoints]).
        srate: A float. Sampling rate of the raw data.
        onset: An int. Stimulus/match onset in ms.
        start: An int. Start relative to stimulus onset (negative: include n ms
            before stimulus onset; positive: start from onset + n ms). Default
            value will drop everything before 500 ms before stimulus/match
            onset.
    
    Returns: 
        An ndarray of the stripped data.
    """
    samples_per_ms = srate/1000
    samples_to_drop = (onset + start) * samples_per_ms
    new_start = int(np.floor(samples_to_drop))
    return raw[:,new_start:]

def drop_trail(raw, srate, length=500):
    """Drops all samples falling out of the length of the interval.
    
    Args:
        raw: An ndarray. Raw data ([channels x timepoints]).
        srate: A float. Sampling rate of the raw data.
        trial_length: An int. Intended length of trial (in ms).
        
    Returns: 
        An ndarray of the stripped data.
    """
    samples_per_ms = srate/1000
    no_of_samples = int(np.floor(length * samples_per_ms))
    return raw[:,:no_of_samples]

def downsample(data, init_srate, target_srate):
    """Downsamples a dataset from init_srate to target_srate.

    Args:
        data: An ndarray. The data set.
        init_srate: A float. The original sampling rate.
        target_srate: A float. The new sampling rate.

    Returns:
        An ndarray of the downsampled data.
    """
    steps = int(np.floor(init_srate / target_srate))  # only integer indices
    return data[:,::steps]

def butter_bandpass_filter(data, lowcut, highcut, srate, order=1):
    """Applies butterworth filter to the data.
    
    Args:
        data: An ndarray. The data set.
        lowcut: An int. The high-pass frequency.
        highcut: An int. The low-pass frequency.
        srate: A float. The sampling rate.

    Returns:
        An ndarray of the filtered data.
    """
    nyq = 0.5 * srate  # nyqvist frequency
    low = lowcut / nyq  # high-pass
    high = highcut / nyq  # low-pass
    b, a = butter(order, [low, high], btype='bandpass')
    return lfilter(b, a, data)