#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:43:55 2018

@author: jannes
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
import numpy as np
import imageio

if __name__ == '__main__':
    import frommatlab as fmat
    import helpers as hlp
else:
    from . import frommatlab as fmat
    from . import helpers as hlp


def elecs_by_area(list_of_areas, coords_path, rinfo_path, rotate=True):
    """Plots electrodes for a given set of target areas onto a map. """
     # Load and subset coords
    idx = fmat.target_channels(list_of_areas, rinfo_path)
    coords = fmat.load_coords(coords_path)[idx]
    
    # Clock-wise rotate by 90 degrees (x, y) --> (y, -x)
    if rotate == True:
        coords[:,0], coords[:,1] = coords[:,1], -coords[:,0].copy()
        
    # Plot
    plt.scatter(coords[:,0], coords[:,1])


def elecs_wholebrain(coords_path, rinfo_path, rotate=True, by_significance=False,
               pvals=None, alevel=.05, decode_for=None):
    """Plots all electrodes onto map, colors by region."""
    # Load coords
    area_names = fmat.area_names(rinfo_path, unique=False)
    area_names = area_names.reshape([area_names.shape[0], 1])
    unique_areas = fmat.area_names(rinfo_path, unique=True, as_list=True)
    idx = fmat.target_channels(unique_areas, rinfo_path)
    coords = fmat.load_coords(coords_path)[idx]
    df = pd.DataFrame(np.concatenate((coords, area_names), axis=1), columns=[
            'x', 'y', 'area'])
    
    threshold_mean = .2 if decode_for == 'stim' else .5
    ms_factor = 30 if decode_for == 'stim' else 200
    ms = 4  # default markersize
    
    # Clock-wise rotate by 90 degrees (x, y) --> (y, -x)
    if rotate == True:
        #coords[:,0], coords[:,1] = coords[:,1], -coords[:,0].copy()
        df[['x', 'y']] = df[['x', 'y']].apply(pd.to_numeric, errors='coerce')
        df['x'], df['y'] = df['y'], -df['x']
    
    # Plot only significant regions if by_significance is equal True
    if by_significance == True:
        if not isinstance(pvals, pd.DataFrame):
            print("If you want to plot significant areas only, you have to "
                  + "pass a pd.DataFrame of pvalues.")
        else:
            df = df.merge(pvals)
            df = df[(df['pval'] < alevel) & (df['mean'] > threshold_mean)]
            df['ms'] = ((df['mean']-min(df['mean'])+.01) 
                        / max(df['mean']) * ms_factor)
            
    # Use pandas for grouping
    df_grouped = df.groupby('area')
    
    # Plot
    fig, ax = plt.subplots()
    #plt.style.use('fivethirtyeight')
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in df_grouped:
        if by_significance == True:
            ms = np.mean(group.ms)
        ax.plot(group.x, group.y, marker='o', ms=ms, linestyle='', label=name)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.xlim((0, 1500))
    plt.ylim((-1000, 0))
    ax.legend()
    plt.show()
    
    
def plot_jpg(path):
    img = imageio.imread(path)
    plt.imshow(img, zorder=0)
    
    
def plot_area(coords, color):
    """Gets coords from all_flatmap_areas.mat and plots them."""
    # Plot
    plt.plot(coords['x'], coords['y'], color='black', linewidth=2, zorder=2)
    plt.fill(coords['x'], coords['y'], color=color, zorder=1)
    plt.xticks([])
    plt.yticks([])
    
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """"""
    import matplotlib.colors as colors

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
    

def colored_map(df, jpg_path, cmap):
    """Test2"""
    from matplotlib.collections import PatchCollection
    
    fig, ax = plt.subplots(figsize=(10,10))
    patches = []
    areas = df['area'].unique()
    means = df['mean'].unique()
    
    for area in areas:
        subset = df[df['area'] == area]
        ax.plot(subset['x'], subset['y'], color='black', linewidth=1.5)
        curr_area = Polygon(subset[['x', 'y']].as_matrix(), True)
        patches.append(curr_area)

    p = PatchCollection(patches, cmap=cmap, alpha=.9)
    colors = np.hstack((100*means, 0, 100))  # Scale from 0 to 100 by appending
    p.set_array(colors)
    ax.add_collection(p)
    ax.set_xlim([-50, 2050])
    ax.set_ylim([2000, -150])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plot_jpg(jpg_path)
    plt.colorbar(p)
    plt.plot()
    return ax


if __name__ == '__main__':
    print('is_main')
        