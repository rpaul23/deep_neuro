#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:44:03 2018

@author: jannes
"""

import pandas as pd


def pvals_from_csv(path, only_significant=False, decode_for=None, alevel=.05):
    """Gets (all|only significant) pvalues from CSV and returns a pandas df."""
    pval = pd.read_csv(path) 
    if only_significant == True:
        threshold_mean = .5 if decode_for == 'resp' else .2
        pval = pval[(pval['pval'] < alevel) & (pval['mean'] > threshold_mean)]
    return pval