#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:08:40 2018

@author: jannes
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp

sessions = ['141023', '141017', '141016', '141015', '141014']
decoders = ['stim', 'resp']
phases = ['sample_500', 'delay_500', 'pre-match_500', 'match_500']

for curr_sess in sessions:
    for curr_dec in decoders:
        for curr_phase in phases:
    
            sess = curr_sess
            decode_for = curr_dec
            phase = curr_phase
            
            dir_in = '/home/jannes/dat/results/accuracy/' + sess + '/'
            dir_out = '/home/jannes/dat/results/accuracy/summary/pval/'
            file_in = sess + '_acc_' + decode_for + '_' + phase + '.csv'
            file_out = sess + '_acc_' + decode_for + '_' + phase + '_pval.csv'
            
            path_in = dir_in + file_in
            path_out = dir_out + file_out
            pop_mean = .5 if decode_for == 'resp' else .2
            
            df = pd.read_csv(path_in)
            df = df[df['time'] > '2018-01-16 13:30:00']
            
            areas = np.unique(df['area'])
            
            acc = []
            for area in areas:
                subset = df[df['area'] == area]
                # CNN
                mean = np.mean(subset['acc'])
                var = np.var(subset['acc'])
                n = subset.shape[0]
                t, p = ttest_1samp(subset['acc'], pop_mean)
                sign = '**' if p < .001 else '*' if p < .05 else 0
                # Regression
                mean_reg = np.mean(subset['acc_reg'])
                _, p_reg = ttest_1samp(subset['acc_reg'], pop_mean)
                sign_reg = '**' if p_reg < .001 else '*' if p_reg < .05 else 0
                acc.append([area, mean, p, sign, mean_reg, p_reg, sign_reg])
                
            df = pd.DataFrame(acc, columns=['area', 'mean', 'pval', 'sign', 'mean_reg', 
                                            'pval_reg', 'sign_reg'])
            df = df.sort_values(by=['mean'], ascending=False)
            df.to_csv(path_out, header=True, index=False)