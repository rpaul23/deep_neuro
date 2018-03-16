#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:07:39 2017

@author: jannes
"""
from os import listdir

import h5py
import numpy as np

    
def get_data(file_path):
    """get_data gets raw data for a given trial from the Grey data set (*.mat).
    returns an ndarray."""
    with h5py.File(file_path, 'r') as f:
        #return np.transpose(np.array([sample for sample in f['raw_data']]))
        return np.transpose(f['lfp_data'][:])
    
    
def get_channel_names(rinfo_path):
    """get_ch_names gets the channel numbers for a given recording.
    """
    with h5py.File(rinfo_path, 'r') as f:
        # 'recording_info.mat' holds 2 structures: 'recording_info' and #refs.
        rinfo = f['recording_info']
        
        # Extract and return ch_numbers
        return [str(int(el)) for el in rinfo['channel_numbers']]
    

def get_number_of_channels(rinfo_path):
    with h5py.File(rinfo_path, 'r') as f:
        # 'recording_info.mat' holds 2 structures: 'recording_info' and #refs.
        rinfo = f['recording_info']
        
        # Extract and return number of working electrodes/channels used
        return int(np.array(rinfo['channel_count']).item())
    
    
def get_sfreq(rinfo_path):
    """ get_sfreq gets the sampling rate for a given recording.
    """
    with h5py.File(rinfo_path, 'r') as f:
        # 'recording_info.mat' holds 2 structures: 'recording_info' and #refs.
        rinfo = f['recording_info']
        
        # Extract and return sampling rate
        return np.array(rinfo['lfp_sampling_rate']).item()
    

def get_responses(tinfo_path):
    """get_responses gets the responses for a given session."""
    with h5py.File(tinfo_path, 'r') as f:
        # 'trial_info.mat' holds only 1 structure
        tinfo = f['trial_info']
        
        # Extract and return responses
        return np.array(tinfo['behavioral_response'])
    

def get_sample_on(tinfo_path):
    """Gets sample onset times for all trials in a session. """
    with h5py.File(tinfo_path, 'r') as f:
        # 'trial_info.mat' holds only 1 structure
        tinfo = f['trial_info']
        
        # Extract and return sample onset
        return np.array(tinfo['sample_on'])
    
    
def get_match_on(tinfo_path):
    """Gets match onset times for all trials in a session. """
    with h5py.File(tinfo_path, 'r') as f:
        # 'trial_info.mat' holds only 1 structure
        tinfo = f['trial_info']
        
        # Extract and return sample onset
        return np.array(tinfo['match_on'])


def get_trial_ids(session_path):
    """get_trial_ids scans (raw) files in a session directory and thereby
    checks what trials are actually available"""
    # Directory will contain trials of format 'session.trial.mat' and rinfo,
    # tinfo. In order to get only trial fnames, split at '.' and select only
    # files that were split into 3 (vs. 2) parts.
    trial_ids = [f.split('.')[1].replace('_raw', '') for f in listdir(session_path) if len(f.split('.')) == 3]
    trial_ids.sort()
    return np.array(trial_ids)


def get_number_of_trials(tinfo_path):
    with h5py.File(tinfo_path, 'r') as f:
        # 'trial_info.mat' holds only 1 structure
        tinfo = f['trial_info']
        
        return int(tinfo['num_trials'][0].item())