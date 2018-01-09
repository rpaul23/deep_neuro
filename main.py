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

import lib.io as io
import lib.cnn as cnn
import lib.helpers as hlp


##########
# PARAMS #
##########

# global params
sess_no = '141023'
interval = 'sample'
target_area = 'V1'
decode_for = 'stim'  # 'resp' for behav resp, 'stim' for stimulus class
elec_type = 'grid' # any one of single|grid|average

# hyper params
size_of_batches = 50
n_iterations = 100
dist = 'random_normal'
batch_norm = 'after'
nonlin = 'elu'
normalized_weights = True
learning_rate = 1e-6
l2_regularization_penalty = 5
keep_prob_train = .5

# layer dimensions
n_layers = 6
patch_dim = [1, 2]  # 1xN patch
in1, out1 = 1, 2
in2, out2 = 2, 4
in3, out3 = 4, 8
in4, out4 = 8, 16
in5, out5 = 16, 64
in6, out6 = 64, 256
fc_units = 100
channels_in = [in1, in2, in3, in4, in5, in6][:n_layers]
channels_out = [out1, out2, out3, out4, out5, out6][:n_layers]


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
data, n_chans = io.get_subset(file_path, target_area, raw_path, elec_type, 
                              return_nchans=True)    
targets = io.get_targets(decode_for, raw_path, elec_type, n_chans)

# train/test params
samples_per_trial = data.shape[2]
n_chans = data.shape[1]
train_size = .8
test_size = .2

# split into train and test
train, test, train_labels, test_labels = train_test_split(
        data, 
        targets, 
        test_size=test_size, 
        random_state=42)


##########
# LAYERS #
##########

# placeholders
x = tf.placeholder(tf.float32, shape=[None, n_chans, samples_per_trial])
y_ = tf.placeholder(tf.float32, shape=[None, classes])
training = tf.placeholder_with_default(True, shape=())
keep_prob_bn = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

# Network
out_bn, weights_bn = cnn.create_network(
        n_layers=n_layers, 
        x_in=x, 
        n_in=channels_in, 
        n_out=channels_out, 
        patch_dim=patch_dim, 
        training=training, 
        n_chans=n_chans,
        n_samples=samples_per_trial,
        weights_dist=dist, 
        normalized_weights=normalized_weights,
        nonlin=nonlin,
        bn=True)

out, weights = cnn.create_network(
        n_layers=n_layers, 
        x_in=x, 
        n_in=channels_in, 
        n_out=channels_out, 
        patch_dim=patch_dim, 
        training=training, 
        n_chans=n_chans,
        n_samples=samples_per_trial,
        weights_dist=dist, 
        normalized_weights=normalized_weights,
        nonlin=nonlin,
        bn=False)


###################
# FULLY-CONNECTED #
###################

# Fully-connected layer (BN)
fc1_bn, weights_bn[n_layers] = cnn.fully_connected(out_bn, bn=True, units=fc_units, nonlin=nonlin)
fc1, weights[n_layers] = cnn.fully_connected(out, bn=False, units=fc_units, nonlin=nonlin)


###################
# DROPOUT/READOUT #
###################

# Dropout (BN)
fc1_drop_bn = tf.nn.dropout(fc1_bn, keep_prob_bn)
fc1_drop = tf.nn.dropout(fc1, keep_prob)

# Readout
weights_bn[n_layers+1] = cnn.init_weights([fc_units, classes])
weights[n_layers+1] = cnn.init_weights([fc_units, classes])
y_conv_bn = tf.matmul(fc1_drop_bn, weights_bn[n_layers+1])
y_conv = tf.matmul(fc1_drop, weights[n_layers+1])


#############
# OPTIMIZER #
#############

# Loss
loss_bn = cnn.l2_loss(weights_bn, l2_regularization_penalty, y_, y_conv_bn, 'loss_bn')
loss = cnn.l2_loss(weights, l2_regularization_penalty, y_, y_conv, 'loss')

# Optimizer
train_step_bn = tf.train.AdamOptimizer(learning_rate).minimize(loss_bn)
correct_prediction_bn = tf.equal(tf.argmax(y_conv_bn, 1), tf.argmax(y_, 1))
accuracy_bn = tf.reduce_mean(tf.cast(correct_prediction_bn, tf.float32))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


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
            train_accuracy_bn = accuracy_bn.eval(feed_dict={
                    x: train[indices,:,:], 
                    y_: train_labels[indices,:],
                    keep_prob_bn: 1.0
                    })
            print('step %d, training accuracy: no BN %g, BN %g' % (
                    i, train_accuracy, train_accuracy_bn))
        
        # Training
        # without BN
        train_step.run(feed_dict={
                x: train[indices,:,:], 
                y_: train_labels[indices,:],
                keep_prob: keep_prob_train
                })
        # with BN
        train_step_bn.run(feed_dict={
                x: train[indices,:,:], 
                y_: train_labels[indices,:],
                keep_prob_bn: keep_prob_train
                })
    
    # Subsetting to get equal class proportions in the test data
    ind = hlp.subset_test(test_labels, classes)
    
    # Print test accuracy
    acc = accuracy.eval(feed_dict={
            x: test[ind,:,:],
            y_: test_labels[ind,:],
            keep_prob: 1.0
            })
    acc_bn = accuracy_bn.eval(feed_dict={
            x: test[ind,:,:],
            y_: test_labels[ind,:],
            keep_prob_bn: 1.0
            })
    print('test accuracy: no BN %g, BN %g' % (acc, acc_bn))


#################
# DOCUMENTATION #
#################

# Store params and accuracy in file
time = str(datetime.datetime.now())
data = [acc, acc_bn, n_iterations, size_of_batches, l2_regularization_penalty,
        learning_rate, str(patch_dim), str(channels_out), fc_units, dist, 
        batch_norm, nonlin, str(target_area), normalized_weights, 
        train_accuracy, train_accuracy_bn, keep_prob_train, n_layers, time]
df = pd.DataFrame([data], 
                  columns=['acc (no BN)', 'acc (BN)', 'iterations',
                           'batch size', 'l2 penalty', 'learning rate',
                           'patch dim', 'output channels', 'fc_units', 'dist',
                           'time of BN', 'nonlinearity', 'area', 'std',
                           'train_accuracy', 'train_accuracy_bn', 'keep_prob',
                           'n_layers', 'time'],
                  index=[0])
with open('/home/jannes/dat/results/accuracy/' + 
          sess_no + '_acc_' + decode_for + '_' + interval + '.csv', 'a') as f:
    df.to_csv(f, index=False, header=False)