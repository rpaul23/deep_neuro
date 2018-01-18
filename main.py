#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:08:48 2018

@author: jannes
"""

import lib.monkeyplot as mplt

coords_path = '/media/jannes/disk2/raw/brainmap/lucy_xy_coordinates_wholebrain.mat'
rinfo_path = '/media/jannes/disk2/raw/141023/session01/recording_info.mat'

list_of_areas = ['V1', 'V2', 'F1', 'LIP']
for area in list_of_areas:
    mplt.plotareas(area, coords_path, rinfo_path, rotate=True)
