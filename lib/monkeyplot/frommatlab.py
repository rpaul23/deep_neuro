#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:10:49 2018

@author: jannes
"""

from scipy.io import loadmat
import h5py
import numpy as np
import pandas as pd

def load_coords(coords_path):
    """Gets coords from MATLAB file and returns them as ndarray. """
    return loadmat(coords_path)['xy']

def channel_numbers(rinfo_path):
    """Loads all channel numbers of electrodes used in a given session. """
    # Get channel numbers from recording info
    with h5py.File(rinfo_path, 'r') as f:
        rinfo = f.get('recording_info')
        channel_numbers = [int(c.item()) for c in rinfo['channel_numbers']]
    return channel_numbers

def area_names(rinfo_path, unique=True, as_list=False):
    """Loads all area names recorded in a given session. 
    
    Args:
        rinfo_path: A str. Path to the recording_info file.
        unique: A boolean. Returns unique values by default, if set to False,
            instead returns list of area names (one per electrode).
        as_list: A boolean. Return list if set to True.
            
    Returns:
        An ndarray of area_names, either unique values or one per electrode;
        optionally returns a list if as_list is set to True.
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
    
    # Convert to ndarray
    area_names = np.array(area_names)
    
    # If unique is True, return only unique values, optionally as list
    if unique == True:
        area_names = np.unique(area_names)
        if as_list == True:
            area_names = area_names.tolist()
    return area_names
    
def target_channels(list_of_areas, rinfo_path, return_nchans=False):
    """Returns indices of channels in a given set of target areas.
    
    Args:
        list_of_areas: A list. Str of target area(s).
        rinfo_path: A str. Path to the recording_info file.
        return_nchans: A boolean. Whether number of channels in target area is
            to be returned.
            
    Returns:
        idx: An ndarray. Indices of channels in target area (ROI).
        n_chans: An int. Number of channels in the area of interest (optional).
    """    
    # Get area names and channel numbers
    areas = area_names(rinfo_path, unique=False)
    
    # Convert to list if input is str/int
    if not isinstance(list_of_areas, list):
        list_of_areas = [list_of_areas]
    
    # For number of areas in list_of_areas
    target_indices = [count for count, name in enumerate(areas) 
                      if name in list_of_areas]
    
    # If return_nchans set to True, also return number of channels in ROI.
    if not return_nchans == True:
        return target_indices 
    else:
        return target_indices, len(target_indices)

def flatmap_coords(path, area):
    """Gets flatmap area coordinates from MATLAB file and returns ndarray."""
    cols = ['x', 'y']
    if area in [el for el in loadmat(path)]:
        coords = pd.DataFrame([el for el in loadmat(path)[area]], 
                              columns=cols)
    elif area.lower() in [el for el in loadmat(path)]:
        coords = pd.DataFrame([el for el in loadmat(path)[area.lower()]], 
                              columns=cols)
    elif area.upper() in [el for el in loadmat(path)]:
        coords = pd.DataFrame([el for el in loadmat(path)[area.upper()]], 
                              columns=cols)
    elif area.replace('/', '_') in [el for el in loadmat(path)]:
        coords = pd.DataFrame([el for el in loadmat(path)[area.replace('/', '_')]],
                              columns=cols)
    else:
        print("Area not available ({}).".format(area))
    coords['area'] = area
    return coords

if __name__ == '__main__':
    coords_path = '/media/jannes/disk2/raw/brainmap/all_flatmap_areas.mat'
    rinfo_path = '/media/jannes/disk2/raw/141023/session01/recording_info.mat'
    flatmap_coords(coords_path, 'V1')