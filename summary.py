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

def metrics(values, pop_mean):
    """Returns mean, pvals (1-sample t-test) and sign for array of values."""
    mean = np.mean(values)
    _, p = ttest_1samp(values, pop_mean)
    sign = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 0
    return mean, p, sign

# params
sess_no = sys.argv[1]
accountname = sys.argv[2]
decoders = ['stim', 'resp']
intervals = ['pre-sample_500', 'sample_500', 'delay_500', 'pre-match_500',
             'match_500']

columns = ['acc_reg', 'acc_svm_lin', 'acc_svm_rbf', 'acc_rdf', 'acc_cnn',
           'iterations', 'batch_size', 'l2_penalty', 'learning_rate',
           'patch_dim', 'pool_dim', 'output_channels', 'fc_units', 'dist',
           'time_of_bn', 'nonlinearity', 'area', 'std', 'only_correct_trials',
           'empty', 'train_accuracy', 'keep_prob', 'n_layers', 'time', 'probs',
           'test_labels', 'test_indices', 'session', 'interval', 'decode_for']
cols_to_groupby = ['iterations', 'batch_size', 'l2_penalty', 'learning_rate',
                   'patch_dim', 'pool_dim', 'output_channels', 'fc_units',
                   'dist', 'time_of_bn', 'nonlinearity', 'area', 'std',
                   'only_correct_trials', 'keep_prob', 'n_layers',
                   'decode_for', 'interval', 'session']
cols_to_numeric = ['acc_reg', 'acc_svm_lin', 'acc_svm_rbf', 'acc_rdf',
                   'acc_cnn']
# paths
file_path = '/home/' + accountname + '/results/training/'
file_name = sess_no + '_training_allmodels.csv'

# read file
df = pd.read_csv(file_path + file_name, header=None, names=columns)
acc = df.copy()

# manipulations
acc['N'] = 1
acc[cols_to_numeric] = acc[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
acc['var'] = pd.to_numeric(acc['acc_cnn'])
acc['train_accuracy'] = pd.to_numeric(acc['train_accuracy'])

# aggregate on columns
acc = acc.groupby(cols_to_groupby).agg({
    'var': np.var,
    'train_accuracy': np.mean,
    'acc_cnn': np.mean,
    'acc_svm_lin': np.mean,
    'acc_svm_rbf': np.mean,
    'acc_rdf': np.mean,
    'acc_reg': np.mean,
    'N': np.count_nonzero
})

# save file
acc = acc.sort_values(by=['acc_cnn'], ascending=False)
acc.to_csv(file_path + 'summary/' + sess_no + '_training_summary_allmodels.csv',
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
            df_int = df_area[df_area['interval'] == curr_int]
            # CNN
            mean_cnn, p_cnn, sign_cnn = metrics(values=df_int['acc_cnn'],
                                                pop_mean=pop_mean)
            # Regression
            mean_reg, p_reg, sign_reg = metrics(values=df_int['acc_reg'],
                                                pop_mean=pop_mean)
            # SVM (linear + rbf kernel)
            mean_svm_lin, p_svm_lin, sign_svm_lin = metrics(
                values=df_int['acc_svm_lin'],
                pop_mean=pop_mean)
            mean_svm_rbf, p_svm_rbf, sign_svm_rbf = metrics(
                values=df_int['acc_svm_rbf'],
                pop_mean=pop_mean)
            # Random Forest
            mean_rdf, p_rdf, sign_rdf = metrics(values=df_int['acc_rdf'],
                                                pop_mean=pop_mean)
            # Append all
            pvals.append([area, mean_cnn, curr_int, sign_cnn, p_cnn, mean_reg,
                          p_reg, sign_reg, mean_svm_lin, p_svm_lin,
                          sign_svm_lin, mean_svm_rbf, p_svm_rbf, sign_svm_rbf,
                          mean_rdf, p_rdf, sign_rdf])
    # merge, sort, save
    df_pvals = pd.DataFrame(pvals, columns=['area', 'mean_cnn', 'curr_int',
                                            'sign_cnn', 'pval_cnn',
                                            'mean_reg', 'pval_reg', 'sign_reg',
                                            'mean_svm_lin', 'pval_svm_lin',
                                            'sign_svm_lin', 'mean_svm_rbf',
                                            'pval_svm_rbf', 'sign_svm_rbf',
                                            'mean_rdf', 'pval_rdf', 'sign_rdf'])
    df_pvals = df_pvals.sort_values(by=['mean_cnn'], ascending=False)
    df_pvals.to_csv(file_path + 'pvals/' + sess_no + '_' + decode_for
                    + '_pvals_allmodels.csv', header=True, index=False)
