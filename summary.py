#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:24:46 2017

@author: jannes
"""

# imports
import sys

import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp

# params
sess_no = sys.argv[1]
accountname = sys.argv[2]
decoders = ['stim', 'resp']
intervals = ['pre-sample_500', 'sample_500', 'delay_500', 'pre-match_500',
             'match_500']

columns = ['acc_reg', 'acc', 'iterations', 'batch_size', 'l2_penalty',
           'learning_rate', 'patch_dim', 'pool_dim', 'output_channels',
           'fc_units', 'dist', 'time_of_bn', 'nonlinearity', 'area', 'std',
           'only_correct_trials', 'empty', 'train_accuracy', 'keep_prob',
           'n_layers', 'time', 'probs', 'test_labels', 'test_indices',
           'session', 'interval', 'decode_for']
cols_to_groupby = ['iterations', 'batch_size', 'l2_penalty', 'learning_rate',
                   'patch_dim', 'pool_dim', 'output_channels', 'fc_units',
                   'dist', 'time_of_bn', 'nonlinearity', 'area', 'std',
                   'only_correct_trials', 'keep_prob', 'n_layers',
                   'decode_for', 'interval', 'session']
# paths
file_path = '/home/' + accountname + '/results/training/'
file_name = sess_no + '_training_test.csv'

# read file
df = pd.read_csv(file_path + file_name, header=None, names=columns)
acc = df.copy()

# manipulations
acc['N'] = 1
acc['acc'] = pd.to_numeric(acc['acc'])
acc['acc_reg'] = pd.to_numeric(acc['acc_reg'])
acc['var'] = pd.to_numeric(acc['acc'])
acc['train_accuracy'] = pd.to_numeric(acc['train_accuracy'])

# aggregate on columns
acc = acc.groupby(cols_to_groupby).agg({
    'var': np.var,
    'train_accuracy': np.mean,
    'acc': np.mean,
    'acc_reg': np.mean,
    'N': np.count_nonzero
})

# save file
acc = acc.sort_values(by=['acc'], ascending=False)
acc.to_csv(file_path + 'summary/' + sess_no + '_training_summary_test.csv',
           float_format='%.6f')

# pvals
for decode_for in decoders:
    df_subset = df[df['decode_for'] == decode_for]
    areas = np.unique(df_subset['area'])
    pvals = []
    for area in areas:
        df_area = df_subset[df_subset['area'] == area]
        pop_mean = .5 if decode_for == 'resp' else .2
        for curr_int in intervals:
            # CNN
            df_int = df_area[df_area['interval'] == curr_int]
            mean = np.mean(df_int['acc'])
            var = np.var(df_int['acc'])
            n = df_int.shape[0]
            t, p = ttest_1samp(df_int['acc'], pop_mean)
            sign = '**' if p < .001 else '*' if p < .05 else 0
            # Regression
            mean_reg = np.mean(df_int['acc_reg'])
            _, p_reg = ttest_1samp(df_int['acc_reg'], pop_mean)
            sign_reg = '**' if p_reg < .001 else '*' if p_reg < .05 else 0
            pvals.append([area, mean, curr_int, sign, p, mean_reg, p_reg,
                          sign_reg])
    # merge, sort, save
    df_pvals = pd.DataFrame(pvals, columns=['area', 'mean', 'curr_int', 'sign',
                                            'pval', 'mean_reg', 'pval_reg',
                                            'sign_reg'])
    df_pvals = df_pvals.sort_values(by=['mean'], ascending=False)
    df_pvals.to_csv(file_path + 'pvals/' + sess_no + '_' + decode_for
                    + '_pvals_test.csv', header=True, index=False)
