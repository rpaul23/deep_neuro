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


decoders = ['stim', 'resp']
intervals = ['sample_500', 'delay_500', 'pre-match_500', 'match_500']
alevels = [.05, .001]


for decoder in decoders:
    for count, phase in enumerate(intervals):
        for alevel in alevels:
            # Params
            sess = '141023'
            decode_for = decoder
            phase = phase
            alevel = alevel
            
            
            # Paths
            elecs_path = '/media/jannes/disk2/raw/brainmap/lucy_xy_coordinates_wholebrain.mat'
            areas_path = '/media/jannes/disk2/raw/brainmap/all_flatmap_areas.mat'
            labels_path = '/media/jannes/disk2/raw/brainmap/area_labels.csv'
            jpg_path = '/media/jannes/disk2/raw/brainmap/Flatmap_no_areas.jpg'
            rinfo_path = '/media/jannes/disk2/raw/' + sess + '/session01/recording_info.mat'
            pval_path = ('/home/jannes/dat/results/accuracy/summary/pval/' 
                         + sess + '_acc_' + decode_for + '_' + phase + '_pval.csv')
            path_out = '/home/jannes/dat/results/accuracy/plots/brainmap/a' + str(alevel)[2:] + '/'
            
            
            # Data
            pvals = hlp.pvals_from_csv(pval_path,
                                       only_significant=True,
                                       decode_for=decode_for,
                                       alevel=alevel)
            pvals = pvals[[len(el) < 5 for el in pvals['area']]]
            unique_areas = np.unique(pvals['area'])
            coords = pd.DataFrame(columns=['x', 'y', 'area'])
            for area in unique_areas:
                coords = coords.append(fmat.flatmap_coords(areas_path, area))
            df = coords.merge(pvals)
            chance_level = .5 if decode_for == 'resp' else .2
            
            
            # Plot
            cmap = plt.get_cmap('afmhot')
            new_cmap = mplt.truncate_colormap(cmap, 0, 1)
            ax = mplt.colored_map(df, jpg_path, labels_path, new_cmap, chance_level)
            plt.text(x=1800, y=1950, s='p < ' + str(alevel), color='black', fontsize=25)
            if decode_for == 'resp':
                title_string = ('Decoding responses in the ' 
                                + phase.replace('_500','') + ' period')
            else:
                title_string = ('Decoding stimulus class in the ' 
                                + phase.replace('_500','') + ' period')
            plt.title(title_string, fontsize=60, family='Helvetica Neue LT')
            plt.savefig(path_out + sess + '_' + decode_for + '_0' 
                        + str(count+1) + phase + '_' + str(alevel) + '.png')
                
