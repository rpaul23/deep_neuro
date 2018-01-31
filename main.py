#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:01:26 2017

@author: jannes
"""

###########
# IMPORTS #
###########

import datetime

from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np

import lib.io as io
import lib.cnn as cnn
import lib.helpers as hlp


# Add a few lines to have the program run over multiple areas and datasets.
# Load file of areas, get the last, then save new list to file to call on
# next iteration
with open('/home/jannes/dat/scripts/areas_temp.txt', 'rb') as f:
    areas = np.loadtxt(f, dtype='object', delimiter='\n')
areas = areas.tolist()
curr_area = areas.pop()
areas = np.array(areas)
with open('/home/jannes/dat/scripts/areas_temp.txt', 'wb') as f:
    np.savetxt(f, areas, fmt='%s')
    
# Define the intervals to loop over    
intervals = ['sample_500', 'delay_500', 'pre-match_500', 'match_500']
sessions = ['141023']
for curr_sess in sessions:
    for curr_int in intervals:
        for i in range(10):
            print('Interval {}, iteration {}, area {}:'.format(curr_int, i+1, curr_area))
            
            
            ##########
            # PARAMS #
            ##########
                    
            # global params
            sess_no = curr_sess
            interval = curr_int  # 'sample_500'
            target_area = curr_area
            decode_for = 'stim'  # 'resp' for behav resp, 'stim' for stimulus class
            elec_type = 'grid' # any one of single|grid|average
            only_correct_trials = False  # if pre-sample, only look at correct trials
            
            # hyper params
            n_iterations = 100
            size_of_batches = 50
            dist = 'random_normal'
            batch_norm = 'after'
            nonlin = 'elu'
            normalized_weights = True
            learning_rate = 1e-5
            l2_regularization_penalty = 5
            keep_prob_train = .5
            
            # layer dimensions
            n_layers = 7
            patch_dim = [1, 5]  # 1xN patch
            pool_dim = [1, 2]
            in1, out1 = 1, 3
            in2, out2 = 3, 6
            in3, out3 = 6, 12
            in4, out4 = 12, 36
            in5, out5 = 36, 72
            in6, out6 = 72, 256
            in7, out7 = 256, 500
            in8, out8 = 500, 1000
            fc_units = 200
            channels_in = [in1, in2, in3, in4, in5, in6, in7, in8][:n_layers]
            channels_out = [out1, out2, out3, out4, out5, out6, out7, out8][:n_layers]
            
            
            #########
            # PATHS #
            #########
            
            data_path = '/media/jannes/disk2/pre-processed/' + interval + '/'
            raw_path = '/media/jannes/disk2/raw/' + sess_no + '/session01/'
            file_name = sess_no + '_freq1000low5hi450order3.npy'
            file_path = data_path + file_name
            rinfo_path = raw_path + 'recording_info.mat'
            tinfo_path = raw_path + 'trial_info.mat'
            
            
            ########
            # DATA #
            ########
            
            # Auto-define number of classes
            classes = 2 if decode_for == 'resp' else 5
            
            # Load data and targets
            data, n_chans = io.get_subset(file_path, target_area, raw_path, 
                                          elec_type, return_nchans=True, 
                                          only_correct_trials=only_correct_trials)
            targets = io.get_targets(decode_for, raw_path, elec_type, n_chans,
                                     only_correct_trials=only_correct_trials)
            
            # train/test params
            samples_per_trial = data.shape[2]
            n_chans = data.shape[1]
            train_size = .8
            test_size = .2
            
            # split into train and test
            indices = np.arange(data.shape[0])
            train, test, train_labels, test_labels, idx_train, idx_test = train_test_split(
                    data, 
                    targets, 
                    indices,
                    test_size=test_size, 
                    random_state=42)
            
            
            ##########
            # LAYERS #
            ##########
            
            # placeholders
            x = tf.placeholder(tf.float32, shape=[None, n_chans, samples_per_trial])
            y_ = tf.placeholder(tf.float32, shape=[None, classes])
            training = tf.placeholder_with_default(True, shape=())
            keep_prob = tf.placeholder(tf.float32)
            
            # Network
            out, weights = cnn.create_network(
                    n_layers=n_layers, 
                    x_in=x, 
                    n_in=channels_in, 
                    n_out=channels_out, 
                    patch_dim=patch_dim,
                    pool_dim=pool_dim,
                    training=training, 
                    n_chans=n_chans,
                    n_samples=samples_per_trial,
                    weights_dist=dist, 
                    normalized_weights=normalized_weights,
                    nonlin=nonlin,
                    bn=True)
            
            
            ###################
            # FULLY-CONNECTED #
            ###################
            
            # Fully-connected layer (BN)
            fc1, weights[n_layers] = cnn.fully_connected(out, 
                        bn=True, 
                        units=fc_units, 
                        nonlin=nonlin,
                        weights_dist=dist,
                        normalized_weights=normalized_weights)
            
            
            ###################
            # DROPOUT/READOUT #
            ###################
            
            # Dropout (BN)
            fc1_drop = tf.nn.dropout(fc1, keep_prob)
            
            # Readout
            weights[n_layers+1] = cnn.init_weights([fc_units, classes])
            y_conv = tf.matmul(fc1_drop, weights[n_layers+1])
            softmax_probs = tf.contrib.layers.softmax(y_conv)
            weights_shape = [tf.shape(el) for el in weights.values()]
            
            #############
            # OPTIMIZER #
            #############
            
            # Loss
            loss = cnn.l2_loss(weights, l2_regularization_penalty, y_, y_conv, 'loss')
            
            # Optimizer
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            
            ######################
            # SOFTMAX REGRESSION #
            ######################
            
            # Implement softmax regression for comparison
            x_reg = tf.placeholder(tf.float32, shape=[None, n_chans * samples_per_trial])
            y_reg = tf.placeholder(tf.float32, shape=[None, classes])
            y_hat_reg = cnn.softmax_regression(x_reg, classes)
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_reg * tf.log(y_hat_reg), reduction_indices=[1]))
            train_step_reg = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
            correct_prediction_reg = tf.equal(tf.argmax(y_hat_reg, 1), tf.argmax(y_reg, 1))
            accuracy_reg = tf.reduce_mean(tf.cast(correct_prediction_reg, tf.float32))
            
            
            ############
            # TRAINING #
            ############
            
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                
                # Number of batches to train on
                for i in range(n_iterations):
                   # Subsetting to get equal class proportions in the training data
                    indices = hlp.subset_train(train_labels, classes, size_of_batches)
                    
                    # Every n iterations, print training accuracy
                    if i % 10 == 0:
                        train_accuracy = accuracy.eval(feed_dict={
                                x: train[indices,:,:],
                                y_: train_labels[indices,:],
                                keep_prob: 1.0
                                })
                        print('step %d, training accuracy: %g' % (
                                i, train_accuracy))
              
                    # Training
                    train_step.run(feed_dict={
                            x: train[indices,:,:], 
                            y_: train_labels[indices,:],
                            keep_prob: keep_prob_train
                            })
                    train_step_reg.run(feed_dict={
                            x_reg: train[indices,:,:].reshape(len(indices), -1), 
                            y_reg: train_labels[indices,:].reshape(len(indices), -1),
                            })
                
                # Subsetting to get equal class proportions in the test data
                ind = hlp.subset_test(test_labels, classes)
                
                # Print test accuracy
                acc = accuracy.eval(feed_dict={
                        x: test[ind,:,:],
                        y_: test_labels[ind,:],
                        keep_prob: 1.0
                        })
                acc_reg = accuracy_reg.eval(feed_dict={
                        x_reg: test[ind,:,:].reshape(len(ind), -1),
                        y_reg: test_labels[ind,:].reshape(len(ind), -1)
                        })
                probs = softmax_probs.eval(feed_dict={
                        x: test[ind,:,:],
                        y_: test_labels[ind,:],
                        keep_prob: 1.0
                        })
                print('test accuracy: CNN %g, Regression %g' % (acc, acc_reg))
                
                # Get size of weights
                size_weights = sess.run(weights_shape)
            
            
            #################
            # DOCUMENTATION #
            #################
            
            time = str(datetime.datetime.now())
            
            # Store data in list, convert to df
            data = [acc_reg, acc, n_iterations, size_of_batches, l2_regularization_penalty,
                    learning_rate, str(patch_dim), str(pool_dim), str(channels_out), 
                    fc_units, dist, batch_norm, nonlin, str(target_area), 
                    normalized_weights, only_correct_trials, 0, 
                    train_accuracy, keep_prob_train, n_layers, time, probs,
                    test_labels[ind,:], idx_test[ind]]
            df = pd.DataFrame([data], 
                              columns=['acc_reg', 'acc', 'iterations', 'batch size', 'l2 penalty', 
                                       'learning rate', 'patch dim', 'pool_dim', 
                                       'output channels', 'fc_units', 'dist', 
                                       'time of BN', 'nonlinearity', 'area','std', 
                                       'only_correct_trials', '', 
                                       'train_accuracy', 'keep_prob', 'n_layers', 
                                       'time', 'probs', 'test_labels', 
                                       'test_indices'],
                              index=[0])
            
            # Save to file
            with open('/home/jannes/dat/results/accuracy/' + sess_no + '/' +
                      sess_no + '_acc_' + decode_for + '_' + interval + '.csv', 'a') as f:
               df.to_csv(f, index=False, header=False)
