#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:08:48 2018

@author: jannes
"""

# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
pvals = hlp.pvals_from_csv(pval_path,
                           only_significant=True,
                           decode_for=decode_for,
                           alevel=.05)
pvals = pvals[[len(el) < 5 for el in pvals['area']]]
unique_areas = np.unique(pvals['area'])
coords = pd.DataFrame(columns=['x', 'y', 'area'])
for area in unique_areas:
    coords = coords.append(fmat.flatmap_coords(areas_path, area))
df = coords.merge(pvals)


# Plot
cmap = plt.get_cmap('afmhot')
new_cmap = mplt.truncate_colormap(cmap, 0, 1)
ax = mplt.colored_map(df, jpg_path, new_cmap)
plt.title(decode_for + ', ' + phase)
plt.show()

