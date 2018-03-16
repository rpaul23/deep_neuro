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
    return (str(msg)) + '\n'


class Log():

    def __init__(self, session, decode_for, path):
        self.session = session
        self.decode_for = decode_for
        self.path = path
        self.df = self.from_csv(path, decode_for)
        self.column_names = self.column_names()

    def from_csv(self, path, decode_for):
        columns = ['acc_reg', 'acc', 'iterations', 'batch_size', 'l2_penalty',
                   'learning_rate', 'patch_dim', 'pool_dim', 'output_channels',
                   'fc_units', 'dist', 'time_of_BN', 'nonlinearity', 'area',
                   'std', 'only_correct_trials', 'empty', 'train_accuracy',
                   'keep_prob', 'n_layers', 'time', 'probs', 'test_labels',
                   'test_indices', 'session', 'interval', 'decode_for',
                   'run_id', 'target_areas']
        df = pd.read_csv(path, header=None, names=columns)
        df = df[df['decode_for'] == decode_for]
        df = df[df['interval'] == 'delay_500']
        df = df.sort_values(by=['run_id', 'time'], ascending=True)
        #df['area'] = df['area'].apply(lambda row: eval(row))
        #df['target_areas'] = df['target_areas'].apply(lambda row: eval(row))
        return df

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
        pass


def main():
    sessions = ['141023']
    decoders = ['stim', 'resp']

    for decode_for in decoders:
        cols = ['area', 'acc_single', 'acc_combination', 'information_gain']
        df = pd.DataFrame([], columns=cols)

        for session in sessions:
            print(session)
            file_in = session + '_training_combi2.csv'
            file_out = session + '_' + decode_for +'_combinations2.csv'

            base_path = '/Users/jannes/Projects/Marseille/cluster/results/training/'
            path_in = base_path + file_in
            path_out = base_path + 'combinations/' + file_out

            log = Log(session, decode_for, path_in)
            unique_area_combinations = [el for el in np.unique(log.df['target_areas'])]
            for curr_target_areas_str in unique_area_combinations:
                df_temp = log.df[log.df.target_areas == curr_target_areas_str]
                df_temp = df_temp[['acc', 'area', 'probs', 'test_labels', 'test_indices',
                                   'run_id']]

                # Create one subset for each area, then merge by run_id
                curr_target_areas_lst = eval(curr_target_areas_str)
                subsets = {}
                for count, area in enumerate(curr_target_areas_lst):
                    subsets[count] = df_temp[df_temp.area == str(area)]
                df_temp = subsets[0].merge(subsets[1], on='run_id').merge(subsets[2], on='run_id')
                df_temp.columns = ['acc_x', 'area_x', 'probs_x', 'test_labels_x',
                                   'test_indices_x', 'run_id', 'acc_y', 'area_y', 'probs_y',
                                   'test_labels_y', 'test_indices_y', 'acc_z', 'area_z',
                                   'probs_z', 'test_labels_z', 'test_indices_z']
                # Probs are of type string, manipulate, then convert to ndarray
                df_temp['probs_x'] = df_temp['probs_x'].apply(lambda val: np.array(eval(
                    val.replace(' ', ',').replace(',,,', ',').replace(',,',',').replace(',,',',').replace('[,','[')
                )))
                df_temp['probs_y'] = df_temp['probs_y'].apply(lambda val: np.array(eval(
                    val.replace(' ', ',').replace(',,,', ',').replace(',,', ',').replace(',,', ',').replace('[,','[')
                )))
                df_temp['probs_z'] = df_temp['probs_z'].apply(lambda val: np.array(eval(
                    val.replace(' ', ',').replace(',,,', ',').replace(',,', ',').replace(',,', ',').replace('[,','[')
                )))
                df_temp['test_labels'] = df_temp['test_labels_x'].apply(lambda val: np.array(eval(
                    val.replace(' ', ',').replace(',,,', ',').replace(',,', ',').replace(',,', ',').replace('[,','[')
                )))
                # # Manipulations
                n_classes = len(df_temp['probs_x'][0][0])
                df_temp['multiplied_probs'] = df_temp['probs_x'] * df_temp['probs_y']  # new probs
                df_temp['K'] = df_temp['multiplied_probs'].apply(lambda row: np.sum(row, axis=1))
                df_temp['K'] = df_temp['K'].apply(lambda val: np.repeat(val, n_classes).reshape(-1, n_classes))
                df_temp['classifier'] = df_temp['multiplied_probs'] / df_temp['K']  # normalize
                max_indices = df_temp['classifier'].apply(lambda row: np.argmax(row, axis=1))
                df_temp['classif_one_hot'] = max_indices.apply(lambda row: np.eye(n_classes)[row])
                argmax_test_labels = df_temp['test_labels'].apply(lambda row: np.argmax(row, axis=1))
                argmax_classif = df_temp['classif_one_hot'].apply(lambda row: np.argmax(row, axis=1))
                correct_predictions = []
                n = []
                for i in range(argmax_test_labels.shape[0]):
                    correct_predictions.append(sum(argmax_test_labels[i] == argmax_classif[i]))
                    n.append(argmax_test_labels[i].shape[0])
                df_temp['correct_predictions'] = correct_predictions
                df_temp['acc_new_classif'] = df_temp['correct_predictions'] / n
                df_temp['information_gain'] = df_temp['acc_z'] - df_temp['acc_new_classif']
                df_temp = df_temp[['area_z', 'acc_new_classif', 'acc_z', 'information_gain']]
                df_temp.columns = cols
                df = df.append(df_temp)

            df = df.dropna()
            df = df.groupby(['area']).agg({
                'acc_single': np.mean,
                'acc_combination': np.mean,
                'information_gain': np.mean
            })
            df = df.sort_values(by='information_gain', ascending=False)

            df.to_csv(path_out)


if __name__ == '__main__':
    main()
