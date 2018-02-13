#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:56:33 2018

@author: jannes
"""

import numpy as np
import pandas as pd
import warnings


def custom_warning(msg, *a):
    return(str(msg)) + '\n'


class Log():

    def __init__(self, session, decode_for, interval, path):
        self.session = session
        self.decode_for = decode_for
        self.interval = interval
        self.path = path
        self.df = self.from_csv(path)
        self.column_names = self.column_names()
        
        # only temp
        self.df = self.df.dropna(axis=0, how='any')
        self.df['run_id'] = self.df.index.values % 10
        
    def from_csv(self, path):
        return pd.read_csv(path)

    def column_names(self):
        return list(self.df)

    def subset(self, columns=[], area_column_label='area', area=None):
        if not isinstance(area, list):
            area = [area]
        if columns == []:
            columns = self.column_names
            warnings.formatwarning = custom_warning
            warnings.warn('Warning: Column parameter left empty, will return '
                          'complete data instead of a subset.')
        if area != None:
            return self.df[self.df[area_column_label].isin(area)][columns]
        else:
            return self.df[columns]
        
    def mutual_information(self, var1, var2):
        
        
        


def main():
    session = '141023'
    decode_for = 'stim'
    period = 'sample_500'

    base_path = '/home/jannes/dat/results/accuracy/'
    session_path = base_path + session + '/'
    file_name = session + '_acc_' + decode_for + '_' + period + '.csv'

    path = session_path + file_name
    log = Log(session, decode_for, period, path)

    # Create one subset for each area, then merge by run_id
    subset1 = log.subset(
        columns = ['acc', 'area', 'probs', 'test_labels', 'test_indices', 
                   'run_id'],
        area_column_label='area',
        area='V1'
    )
    subset2 = log.subset(
        columns = ['acc', 'area', 'probs', 'test_labels', 'test_indices', 
                   'run_id'],
        area_column_label='area',
        area='V2'
    )
    df = subset1.merge(subset2, on='run_id')
    
    # Probs are of type string, manipulate, then convert to ndarray
    df['probs_x'] = df['probs_x'].apply(lambda val: np.array(eval(
            val.replace('   ', ' ').replace('  ', ' ').replace(' [ ', '[')
            .replace('[ ', '[').replace('] ', ']').replace(' ]', ']')
            .replace(']\n','],\n').replace(' ', ', ').replace(', , ', ', '))))
    df['probs_y'] = df['probs_y'].apply(lambda val: np.array(eval(
            val.replace('   ', ' ').replace('  ', ' ').replace(' [ ', '[')
            .replace('[ ', '[').replace('] ', ']').replace(' ]', ']')
            .replace(']\n','],\n').replace(' ', ', ').replace(', , ', ', '))))
    df['test_labels'] = df['test_labels_x'].apply(lambda val: np.array(eval(
            val.replace('  ', ', ').replace(']', '],')))[0])
    
    # Manipulations
    df['multiplied_probs'] = df['probs_x'] * df['probs_y']  # new probs
    df['K'] = df['multiplied_probs'].apply(lambda row: np.sum(row, axis=1))
    df['K'] = df['K'].apply(lambda val: np.repeat(val, 5).reshape(-1, 5))
    df['classifier'] = df['multiplied_probs'] / df['K']  # normalize
    max_indices = df['classifier'].apply(lambda row: np.argmax(row, axis=1))
    n_classes = 5
    df['classif_one_hot'] = max_indices.apply(lambda row: np.eye(n_classes)[row])
    argmax_test_labels = df['test_labels'].apply(lambda row: np.argmax(row, axis=1))
    argmax_classif = df['classif_one_hot'].apply(lambda row: np.argmax(row, axis=1))
    correct_predictions = []
    n = []
    for i in range(argmax_test_labels.shape[0]):
        correct_predictions.append(sum(argmax_test_labels[i] == argmax_classif[i]))
        n.append(argmax_test_labels[i].shape[0])
    df['correct_predictions'] = correct_predictions
    df['acc_new_classif'] = df['correct_predictions'] / n
        
    return df
    
if __name__ == '__main__':
    main()


def main2():
    # TO DO: Everything
    new_softmax = []
    trials = 'All trials in one session'
    areas_to_combine = ['V1', 'V2']
    n_areas = len(areas_to_combine)
    n_classes = 2

    subset = 'must be dataframe'
    runs = [row for row in subset.iterrows()]
    for run in runs:
        observations = 1
        for observation in observations:
            softmax = np.zeros(n_areas, dtype=np.ndarray)
        for count, area in enumerate(areas_to_combine):
            softmax[count] = 'Softmax output for current trial and current area'

        # Sum of products
        K = np.sum(np.prod(softmax, axis=0))

        # New classifier
        new_classifier = []
        for curr_class in range(n_classes):
            new_classifier.append(
                np.prod(softmax, axis=0)[curr_class] / K
            )
        new_softmax.append(new_classifier)

    # One-hot encoding of new_softmax
    # Compare one-hot to labels
    # Compute MI
    # Compute simple difference