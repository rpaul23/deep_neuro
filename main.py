#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:08:48 2018

@author: jannes
"""

# Imports
import matplotlib.pyplot as plt

import lib.frommatlab as fmat
import lib.monkeyplot as mplt
import lib.helpers as hlp


# Params
decode_for = 'stim'
phase = 'sample_500'


# Paths
elecs_path = '/media/jannes/disk2/raw/brainmap/lucy_xy_coordinates_wholebrain.mat'
areas_path = '/media/jannes/disk2/raw/brainmap/all_flatmap_areas.mat'
jpg_path = '/media/jannes/disk2/raw/brainmap/Flatmap_outlines.jpg'
rinfo_path = '/media/jannes/disk2/raw/141023/session01/recording_info.mat'
pval_path = ('/home/jannes/dat/results/accuracy/summary/pval/' 
             + '141023_acc_' + decode_for + '_' + phase + '_pval.csv')


# Data
areas = hlp.pvals_from_csv(pval_path, 
                           only_significant=True, 
                           decode_for=decode_for, 
                           alevel=.1)
areas = areas[areas['area'].str.len() < 5]  # drop data for aggregated areas
areas = areas.sort_values(by=['mean'])
colors = mplt.color_map(areas)


# Plot
plt.figure(figsize=(10,10))
for idx, area in enumerate(areas['area']):
    coords = fmat.flatmap_coords(areas_path, area)
    mplt.plot_area(coords, colors[idx])
mplt.plot_jpg(jpg_path)
plt.legend(['test'])
plt.title('Decode: ' + decode_for + ', phase: ' + phase)