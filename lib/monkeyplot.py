#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:43:55 2018

@author: jannes
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
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
    
    
def plot_jpg(path, return_ax=False):
    img = imageio.imread(path)
    plt.imshow(img, zorder=0)
    
    if return_ax == True:
        fig, ax = plt.subplots()
        ax.imshow(img, zorder=0)
        return ax
    
    
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

def label_positions(area, labels_path):
    """"""
    df = pd.read_csv(labels_path)
    x = df[df['area'] == area]['x']
    y = df[df['area'] == area]['y']
    arr = df[df['area'] == area]['arr']
    x_arr = df[df['area'] == area]['x_arr']
    y_arr = df[df['area'] == area]['y_arr']
    label = area
    return x, y, label, arr, x_arr, y_arr
    

def colored_map(df, jpg_path, labels_path, cmap, threshold=.2):
    """Test2"""    
    # Data
    patches = []
    areas = df['area'].unique()

    # Styling
    fontsize = 20
    fontfamily = 'Helvetica Neue LT'

    # Plot colored map
    fig, ax = plt.subplots(figsize=(15, 15))

    # If significant areas for current session and phase
    if areas.shape[0] > 0:
        means = df.groupby('area')['mean'].mean()

        for area in areas:
            subset = df[df['area'] == area]
            ax.plot(subset['x'], subset['y'], color='black', linewidth=.75)
            x, y, label, arr, x_arr, y_arr = label_positions(area, labels_path)
            if arr.item() == 0:
                t = ax.text(x, y, label, fontsize=fontsize, color='white',
                        family=fontfamily, va='center')
                t.set_bbox(dict(facecolor='black', alpha=.7, edgecolor='white'))
            else:
                t = ax.annotate(label,
                            xy=(x, y),
                            xytext=(x_arr, y_arr),
                            fontsize=fontsize, family=fontfamily,
                            color='white',
                            arrowprops=dict(facecolor='blac',
                                            width=1, headwidth=5, headlength=5,
                                            shrink=0, color='black'))
                t.set_bbox(dict(facecolor='black', alpha=.7, edgecolor='white'))

            curr_area = Polygon(subset[['x', 'y']].as_matrix(), True)
            patches.append(curr_area)
        p = PatchCollection(patches, cmap=cmap, alpha=1)
        colors = np.hstack((100*means, 0, 100))  # Scale from 0 to 100 by appending
        p.set_array(colors)
        ax.add_collection(p)
    # Have to create PatchCollection to draw colormap onto empty plot
    else:
        p = PatchCollection([], cmap=cmap, alpha=1)
        p.set_array(np.hstack((0, 100)))
    ax.set_xlim([-50, 2050])
    ax.set_ylim([2000, -150])
    ax.set_xticks([])
    ax.set_yticks([])
    plot_jpg(jpg_path)

    # Plot colorbar
    areas = df[['area', 'mean']].drop_duplicates().sort_values(['mean'],
              ascending=False)['area'][0:3]
    cbar = plt.colorbar(p, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.set_ylabel('Prediction accuracy', rotation=270, fontsize=20,
                       verticalalignment='center', family='Helvetica Neue LT')
    cbar.ax.plot([0,1], [threshold, threshold], 'white', linewidth=4)
    cbar.ax.text(x=-1, y=threshold, s='chance\nlevel', color='white',
                 backgroundcolor='black', va='center', ha='center',
                 fontsize=15, zorder=99)

    for count, area in enumerate(areas):
        subset = df[df['area'] == area]
        x, ha = (-.25, 'right') if count % 2 == 0 else (1.25, 'left')
        y = np.mean(subset['mean'])
        cbar.ax.plot([0,1], [y, y], 'white', linewidth=2, zorder=1)
        t = cbar.ax.text(x=x, y=y, s=area, color='white', fontsize=15,
                         ha='center', va='center', zorder=3-count+1)
        t.set_bbox(dict(facecolor='black', alpha=.7, edgecolor='red'))
    return ax

if __name__ == '__main__':
    print('is_main')