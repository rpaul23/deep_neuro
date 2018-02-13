#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:07:39 2017

@author: jannes
"""
import h5py
import numpy as np


def get_subset(file_path, target_area, raw_path, elec_type, 
               return_nchans=False, only_correct_trials=False):
    """Gets prepocessed data for a given session and filters/subsets for the 
    target area.
    
    Args:
        file_path: A str. Path to pre-processed data (*.npy).
        target_area: A list. String(s) indicating the target area.
        raw_path: A str. Path to the folder containing trial/recording info.
        elec_type: A str. One of 'single' (use all electrodes within area as
            single trials), 'grid' (use whole electrode grid), 'average' (mean
            over all electrodes in area).
        return_nchans: A boolean. Whether number of channels within area is
            returned alongside the data.
        only_correct_trials: A boolean. Indicating whether to subset for 
            trials with correct behavioral response only.
            
    Returns:
        data: The subsetted data for a given target area.
        n_chans: Number of channels in the target area (optional; only returned
            if return_nchans is set to True).
    """
    # Read pre-processed
    data = get_preprocessed(file_path)
    rinfo_path = raw_path + 'recording_info.mat'
    tinfo_path = raw_path + 'trial_info.mat'
    
    # Call get_roi to subset loaded data
    data, n_chans = get_roi(data, target_area, rinfo_path, return_nchans)
    
    # Get responses and keep only trials with non-NA responses
    responses = get_responses(tinfo_path)
    ind_to_keep = (responses == responses).flatten()
    
    # If only_correct_trials set to True, drop all incorrect trials
    if only_correct_trials == True:
        ind_to_keep = (responses == 1).flatten()
        
    # Subset data
    data = data[ind_to_keep]
    
    # If elec_type == 'single' we want to make every electrode in an area their
    # own trials. Therefore, we must reshape from trials x channels x samples 
    # to trials*channels x samples; 
    # f. e. 681 trials, 17 channels, 500 samples turns into 11577 trials by 500
    if elec_type == 'single':
        data = data.reshape(data.shape[0]*data.shape[1], data.shape[2])
        data = np.expand_dims(data, axis=1)
    elif elec_type == 'average':
        data = np.mean(data, axis=1, keepdims=True)
    elif elec_type == 'grid':
        data = data
    else:
        raise ValueError('Type \'' + elec_type + '\' not supported. Please ' + 
                         'choose one of \'single\'|\'grid\'|\'average\'.')
    
    # Also return n_chans if return_nchans is set to true else return data only
    if not return_nchans == True:
        return data
    else:
        return data, n_chans

def get_targets(decode_for, raw_path, elec_type, n_chans, 
                only_correct_trials=True):
    """Gets behavioral responses or stimulus classes from the trial_info file
    and encodes them as one-hot.
    
    Args:
        decode_for: A str. One of 'stim' (stimulus classes) or 'resp' (behav. 
            response), depending on which info to classify the data on.
            (Defines number of classes: stim -> 5 classes, resp -> 2 classes)
        raw_path: A str. Path to the trial_info file.
        elec_type: A str. One of 'single' (use all electrodes within area as
            single trials), 'grid' (use whole electrode grid), 'average' (mean
            over all electrodes in area).
        n_chans: An int. Number of channels in the target area.
        only_correct_trials: A boolean. Indicating whether to subset for 
            trials with correct behavioral response only.
        
    Returns:
        Ndarray of one-hot targets.
    """
    # Trial info holds behavioral responses and stimulus classes
    tinfo_path = raw_path + 'trial_info.mat'
    
    # Get behavioral responses or stimulus classes, depending on user input
    if decode_for == 'stim':
        classes = 5
        targets = get_samples(tinfo_path)
    elif decode_for == 'resp':
        classes = 2
        targets = get_responses(tinfo_path)
    else:
        print('Can decode for behavioral response ("resp") or stimulus ' +
              'identity ("stim"), you entered \"' + decode_for + '\". Please ' +
              'adapt your input.')
    
    # Only keep non-NA targets
    ind_to_keep = (targets == targets).flatten()
    
    # If only_correct_trials set to True, drop targets for all incorrect trials
    if only_correct_trials == True:
        responses = get_responses(tinfo_path)
        ind_to_keep = (responses == 1).flatten()
    
    targets = targets[ind_to_keep].astype(int)
    
    # If every electrode (in an area) shall be regarded as their own trials,
    # we need to reshape targets accordingly. Every target es repeated as many
    # times as there are electrodes in the area.
    if elec_type == 'single':
        targets = np.repeat(targets, n_chans, axis=0)
    
    # Convert to one-hot, return
    return np.eye(classes)[targets].reshape(targets.shape[0], classes)
    
    
def get_preprocessed(file_path):
    """Gets prepocessed data for a given session from the Grey data set.
    returns an ndarray."""
    # Load file, return data
    return np.load(file_path)


def get_roi(data, list_of_areas, rinfo_path, return_nchans=False):
    """Subsets the data to area of interest.
    
    Args:
        data: An ndarray. Pre-processed data for a given session.
        list_of_areas: A list. Str of target area(s).
        rinfo_path: A str. Path to the recording_info file.
        return_nchans: A boolean. Whether number of channels in target area is
            to be returned.
            
    Returns:
        data: An ndarray. Subsetted data.
        n_chans: An int. Number of channels in the area of interest.
    """    
    # Open file for given recording
    with h5py.File(rinfo_path, 'r') as f:
        rinfo = f.get('recording_info')
        
        # Get area names
        area = rinfo['area']
        area_names = []
        for i in range(area.shape[0]):
            for j in range(area.shape[1]):
                curr_idx = area[i][j]
                curr_area = rinfo[curr_idx]
                curr_str = ''.join(chr(k) for k in curr_area[:])
                area_names.append(curr_str)
        
        # Get channel numbers
        c_nums = [int(c.item()) for c in rinfo['channel_numbers']]
        
        # Convert to list if input is str/int
        if not isinstance(list_of_areas, list):
            list_of_areas = [list_of_areas]
        
        # For number of areas in list_of_areas
        target_indices = [count for count, name in enumerate(area_names) 
                          if name in list_of_areas]
        target_channels = [c_nums[i] for i in target_indices]
        
        # Get indices of target electrodes
        idx = []
        for count, ch in enumerate(c_nums):
            if ch in target_channels:
                idx.append(count)
                
        # Subset data
        data = data[:, idx, :]

        #data = np.array(curr_area)
        if not return_nchans == True:
            return data 
        else:
            return data, len(idx)


def get_responses(tinfo_path):
    """Gets the responses for all trials in a given session."""
    with h5py.File(tinfo_path, 'r') as f:
        # 'trial_info.mat' holds only 1 structure
        tinfo = f['trial_info']
        
        # Extract and return responses
        return np.array(tinfo['behavioral_response'])


def get_samples(tinfo_path):
    """Gets sample image classes for all trials in a given session. """
    with h5py.File(tinfo_path, 'r') as f:
        # 'trial_info.mat' holds only 1 structure
        tinfo = f['trial_info']
        
        # Extract and return sample classes. substract 1 to have classes range
        # from 0 to 4 instead of 1 to 5
        return np.array([k-1 for k in tinfo['sample_image']])
    

def get_area_names(rinfo_path):
    """Returns array of unique area names for recordings of a given session. """
    with h5py.File(rinfo_path, 'r') as f:
        info = f.get('recording_info')
        area = info['area']

        area_names = []
        for i in range(area.shape[0]):
            for j in range(area.shape[1]):
                curr_idx = area[i][j]
                curr_area = info[curr_idx]
                curr_str = ''.join(chr(k) for k in curr_area[:])
                area_names.append(curr_str)
        area_names = np.array(area_names)
        area_names.shape = (area.shape[0], area.shape[1])
        return np.unique(area_names)