#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:25:16 2018

@author: jannes
"""

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

# =============================================================================
# areas = ['AIP', 'DP', 'F1', 'F2', 'F6', 'F7', 'LIP', 'MT', 'TPt', 'V1', 'V2',
#          'V4t', 'V6A', 'a1', 'a2', 'a24c', 'a3', 'a44', 'a45B', 'a46D', 'a5',
#          'a7A', 'a7B', 'a8B', 'a8L', 'a8M', 'a8r']
# =============================================================================
        
# global params
sess_no = '141023'
interval = 'pre-match'
target_area = ['a1']
decode_for = 'resp'  # 'resp' for behav resp, 'stim' for stimulus class
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
l2_regularization_penalty = 20
keep_prob_train = .1

# layer dimensions
n_layers = 6
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
data, n_chans = io.get_subset(file_path, target_area, raw_path, elec_type, 
                              return_nchans=True, 
                              only_correct_trials=only_correct_trials)
targets = io.get_targets(decode_for, raw_path, elec_type, n_chans,
                         only_correct_trials=only_correct_trials)

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
fc1, weights[n_layers] = cnn.fully_connected(out, bn=True, units=fc_units, nonlin=nonlin)


###################
# DROPOUT/READOUT #
###################

# Dropout (BN)
fc1_drop = tf.nn.dropout(fc1, keep_prob)

# Readout
weights[n_layers+1] = cnn.init_weights([fc_units, classes])
y_conv = tf.matmul(fc1_drop, weights[n_layers+1])


#############
# OPTIMIZER #
#############

# Loss
loss = cnn.l2_loss(weights, l2_regularization_penalty, y_, y_conv, 'loss')

# Optimizer
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
            print('step %d, training accuracy: %g' % (
                    i, train_accuracy))
  
        # Training
        train_step.run(feed_dict={
                x: train[indices,:,:], 
                y_: train_labels[indices,:],
                keep_prob: keep_prob_train
                })
    
    # Subsetting to get equal class proportions in the test data
    ind = hlp.subset_test(test_labels, classes)
    
    # Print test accuracy
    acc = accuracy.eval(feed_dict={
            x: test[ind,:,:],
            y_: test_labels[ind,:],
            keep_prob: 1.0
            })
    print('test accuracy: %g' % (acc))


#################
# DOCUMENTATION #
#################

time = str(datetime.datetime.now())

# Store data in list, convert to df
data = [0, acc, n_iterations, size_of_batches, l2_regularization_penalty,
        learning_rate, str(patch_dim), str(pool_dim), str(channels_out), 
        fc_units, dist, batch_norm, nonlin, str(target_area), 
        normalized_weights, only_correct_trials, 0, train_accuracy, 
        keep_prob_train, n_layers, time]
df = pd.DataFrame([data], 
                  columns=['', 'acc', 'iterations', 'batch size', 'l2 penalty', 
                           'learning rate', 'patch dim', 'pool_dim', 
                           'output channels', 'fc_units', 'dist', 
                           'only_correct_trials', 'time of BN', 'nonlinearity',
                           'area', 'std', '', 'train_accuracy', 'keep_prob', 
                           'n_layers', 'time'],
                  index=[0])

# Save to file
with open('/home/jannes/dat/results/accuracy/' + 
          sess_no + '_acc_' + decode_for + '_' + interval + '.csv', 'a') as f:
    df.to_csv(f, index=False, header=False)