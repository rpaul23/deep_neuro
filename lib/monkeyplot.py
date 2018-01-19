#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:43:55 2018

@author: jannes
"""

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    import frommatlab as fmat
else:
    from . import frommatlab as fmat


def plotareas(list_of_areas, coords_path, rinfo_path, rotate=True):
    """Plots electrodes for a given set of target areas onto a map. """
     # Load and subset coords
    idx = fmat.targetchannels(list_of_areas, rinfo_path)
    coords = fmat.loadcoords(coords_path)[idx]
    
    # Clock-wise rotate by 90 degrees (x, y) --> (y, -x)
    if rotate == True:
        coords[:,0], coords[:,1] = coords[:,1], -coords[:,0].copy()
        
    # Plot
    plt.scatter(coords[:,0], coords[:,1])


def plotwholebrain(coords_path, rinfo_path, rotate=True):
    """Plots all  electrodes onto map, colors by region. """
    # Load coords
    area_names = fmat.areanames(rinfo_path, unique=False)
    unique_areas = fmat.areanames(rinfo_path, unique=True, as_list=True)
    idx = fmat.targetchannels(unique_areas, rinfo_path)
    coords = fmat.loadcoords(coords_path)[idx]
    
    # Clock-wise rotate by 90 degrees (x, y) --> (y, -x)
    if rotate == True:
        coords[:,0], coords[:,1] = coords[:,1], -coords[:,0].copy()
    
    # Use pandas for grouping
    df = pd.DataFrame(dict(x=coords[:,0], y=coords[:,1], label=area_names))
    df_grouped = df.groupby('label')
    
    # Plot
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in df_grouped:
        ax.plot(group.x, group.y, marker='o', linestyle='', label=name)
    ax.legend()
    plt.show()
    
plotwholebrain(coords_path, rinfo_path, rotate=True)
        