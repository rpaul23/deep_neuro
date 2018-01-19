#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:10:49 2018

@author: jannes
"""

from scipy.io import loadmat
import h5py
import numpy as np


def loadcoords(coords_path):
    """Gets coords from MATLAB file and returns them as ndarray. """
    return loadmat(coords_path)['xy']


def channelnumbers(rinfo_path):
    """Loads all channel numbers of electrodes used in a given session. """
    # Get channel numbers from recording info
    with h5py.File(rinfo_path, 'r') as f:
        rinfo = f.get('recording_info')
        channelnumbers = [int(c.item()) for c in rinfo['channel_numbers']]
    return channelnumbers
    

def areanames(rinfo_path, unique=True, as_list=False):
    """Loads all area names recorded in a given session. 
    
    Args:
        rinfo_path: A str. Path to the recording_info file.
        unique: A boolean. Returns unique values by default, if set to False,
            instead returns list of area names (one per electrode).
        as_list: A boolean. Return list if set to True.
            
    Returns:
        An ndarray of areanames, either unique values or one per electrode;
        optionally returns a list if as_list is set to True.
    """
    # Open file for given recording
    with h5py.File(rinfo_path, 'r') as f:
        rinfo = f.get('recording_info')
        
        # Get area names
        area = rinfo['area']
        areanames = []
        for i in range(area.shape[0]):
            for j in range(area.shape[1]):
                curr_idx = area[i][j]
                curr_area = rinfo[curr_idx]
                curr_str = ''.join(chr(k) for k in curr_area[:])
                areanames.append(curr_str)
    
    # Convert to ndarray
    areanames = np.array(areanames)
    
    # If unique is True, return only unique values, optionally as list
    if unique == True:
        areanames = np.unique(areanames)
        if as_list == True:
            areanames = areanames.tolist()
    return areanames

    
def targetchannels(list_of_areas, rinfo_path, return_nchans=False):
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
    area_names = areanames(rinfo_path, unique=False)
    
    # Convert to list if input is str/int
    if not isinstance(list_of_areas, list):
        list_of_areas = [list_of_areas]
    
    # For number of areas in list_of_areas
    target_indices = [count for count, name in enumerate(area_names) 
                      if name in list_of_areas]
    
    # If return_nchans set to True, also return number of channels in ROI.
    if not return_nchans == True:
        return target_indices 
    else:
        return target_indices, len(target_indices)