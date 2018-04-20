#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:15:35 2018

@author: jannes
"""
import sys

import numpy as np
import h5py

def get_area_names(rinfo_path):
    """Returns array of unique area names for recordings of a given session."""
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

# Paths
session = sys.argv[1]
accountname = sys.argv[2]
base_path = '/home/' + accountname + '/'
rinfo_path = (base_path + 'data/raw/' + session
              + '/session01/recording_info.mat')
path_out = base_path + 'scripts/_params/training.txt'

# Params
decoders = ['resp', 'stim']
intervals = ['matchfrom-800to-300', 'matchfrom-700to-200',
             'matchfrom-600to-100', 'matchfrom-500to0', 'stimfrom0to500',
             'stimfrom100to600', 'stimfrom200to700', 'stimfrom300to800',
             'stimfrom400to900', 'stimfrom500to1000', 'stimfrom600to1100',
             'stimfrom700to1200', 'stimfrom800to1300']
areas = get_area_names(rinfo_path)
areas = [[el] for el in areas]
# groups = [['V1', 'V2', 'V4t', 'V6A', 'DP', 'MT'],
#           ['a5', 'AIP', 'LIP', 'a7A', 'a7B', 'TPt'],
#           ['a1', 'a2', 'a3'],
#           ['a24c', 'a44', 'a45B', 'a8B', 'a8L', 'a8M', 'a8r', 'a46D'],
#           ['F1', 'F2', 'F6', 'F7']]
groups = [['a24c', 'a44', 'a45B', 'a8B', 'a8L', 'a8M', 'a8r', 'a46D']]

runs_per_combination = 10
total_runs = len(decoders) * len(intervals) * len(groups) * runs_per_combination

# Append to file
with open(path_out, 'w') as f:
    f.write('')
count = total_runs + 1
for decode_for in decoders:
    for interval in intervals:
        for area in groups:
            for i in range(runs_per_combination):
                count -= 1
                print_str = '{}, {}, {}: {}, {}/10. Total: {}/{}'.format(
                    session, decode_for, interval, area,
                    runs_per_combination-i, str(count), str(total_runs))
                params = [session, decode_for, area, interval, print_str]
                with open(path_out, 'a') as f:
                    f.write('\n' + str(params))

print(total_runs)  # print to pass length of array to shell