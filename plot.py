#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:08:48 2018

@author: jannes
"""

# Imports
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lib.monkeyplot.frommatlab as fmat
import lib.monkeyplot.monkeyplot as mplt
import lib.monkeyplot.helpers as hlp

def plot_areas(accountname, decoders, intervals, sessions, alevel):
    for decoder in decoders:
        for count, phase in enumerate(intervals):
            for session in sessions:
                # Params
                decode_for = decoder
                phase = phase
                alevel = alevel

                # Paths
                system_path = '/home/' + accountname + '/'
                raw_path = system_path + 'data/raw/'
                results_path = system_path + 'results/'
                areas_path = raw_path + 'brainmap/all_flatmap_areas.mat'
                labels_path = raw_path + 'brainmap/area_labels.csv'
                jpg_path = raw_path + 'brainmap/Flatmap_no_areas.jpg'
                pval_path = (results_path + 'training/pvals/' + session
                             + '_' + decode_for + '_pvals.csv')
                path_out = results_path + 'plots/'

                # Data
                pvals = hlp.pvals_from_csv(pval_path,
                                           only_significant=False,
                                           decode_for=decode_for,
                                           interval=phase,
                                           alevel=alevel)
                coords = pd.DataFrame(columns=['x', 'y', 'area'])
                chance_level = .5 if decode_for == 'resp' else .2

                # Some phases for a given session might not have any significant
                # areas. If this is the case, we just pass an empty array/df.
                if pvals.shape[0] > 0:
                    unique_areas = np.unique(pvals['area'])
                else:
                    unique_areas = np.array([])

                for area in unique_areas:
                    coords = coords.append(fmat.flatmap_coords(areas_path, area))

                if coords.shape[0] > 0:
                    df = coords.merge(pvals)
                else:
                    df = coords.join(pd.DataFrame(
                        columns=['mean', 'pval', 'sign', 'mean_reg',
                                 'pval_reg', 'sign_reg'])
                    )

                # Plot
                cmap = plt.get_cmap('afmhot')
                new_cmap = mplt.truncate_colormap(cmap, 0, 1)
                ax = mplt.colored_map(df, jpg_path, labels_path, new_cmap,
                                      chance_level)
                plt.text(x=1800, y=1950, s='p < ' + str(alevel),
                         color='black', fontsize=12.5)
                if decode_for == 'resp':
                    title_string = ('Decoding responses in the '
                                    + phase.replace('_500','') + ' period')
                else:
                    title_string = ('Decoding stimulus class in the '
                                    + phase.replace('_500','') + ' period')
                plt.title(title_string, fontsize=30,
                          family='Helvetica Neue LT')
                curr_dir = path_out + session + '/'
                if not os.path.exists(curr_dir):
                    os.makedirs(curr_dir)
                plt.savefig(curr_dir + session + '_'+ decode_for + '_' + phase
                            + '_' + str(alevel) + '.png')

def plot_lobes(accountname, session):
    # Paths
    system_path = '/home/' + accountname + '/'
    raw_path = system_path + 'data/raw/'
    areas_path = raw_path + 'brainmap/all_flatmap_areas.mat'
    jpg_path = raw_path + 'brainmap/Flatmap_no_areas.jpg'
    rinfo_path = raw_path + session + '/session01/recording_info.mat'
    coords = pd.DataFrame(columns=['x', 'y', 'area'])

    # Some phases for a given session might not have any significant
    # areas. If this is the case, we just pass an empty array/df.
    unique_areas = fmat.area_names(rinfo_path)
    for area in unique_areas:
        coords = coords.append(fmat.flatmap_coords(areas_path, area))
    # Plot
    ax = mplt.colored_lobes(coords, jpg_path)

def main():
    # Params
    sess_no = sys.argv[1]
    accountname = sys.argv[2]
    decoders = ['resp','stim']
    intervals = ['pre-sample_500', 'sample_500', 'delay_500', 'pre_match_500',
                 'match_500']
    alevel = .05
    plot_areas(accountname, decoders, intervals, sess_no, alevel)

if __name__ == '__main__':
    main()