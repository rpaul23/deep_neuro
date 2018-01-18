#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:43:55 2018

@author: jannes
"""

import matplotlib.pyplot as plt

from . import frommatlab as fmat

def plotareas(list_of_areas, coords_path, rinfo_path, rotate=True):
    """Plots electrodes for a given set of target areas onto a map. """
     # Load and subset coords
    idx = fmat.targetchannels(list_of_areas, rinfo_path)
    coords = fmat.loadcoords(coords_path)[idx]
    
    # Clock-wise rotate by 90 degrees (x, y) --> (y, -x)
    if rotate == True:
        coords[:,0], coords[:,1] = coords[:,1], -coords[:,0].copy()
    
    # Plot, return coords if indicated
    plt.scatter(coords[:,0], coords[:,1])