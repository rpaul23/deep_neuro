import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import io as io
import helpers as hlp

counter = int(sys.argv[1])
user_name = sys.argv[2]
session = sys.argv[3]
base_path = 'home/' + user_name + '/'
params = {
    'intervals': ['sample_500', 'delay_500'],
    'n_trees': [10, 20, 50, 100, 250, 500, 1000, 5000, 10000]
    }

ind_interval = 0 if counter < len(params['n_trees']) else 1
ind_n = int(np.floor(counter/len(params['n_trees'])))

interval = params['intervals'][ind_interval]
n_trees = params['n_trees'][ind_n]

data_path = base_path + 'data/pre-processed/' + interval + '/'
raw_path = base_path + 'data/raw/' + session + '/session01/'
file_name = session + '_freq1000low5hi450order3.npy'
file_path = data_path + file_name
file_out = base_path + 'results/tuning/rf_tuning.csv'

decode_for = 'stim' if interval == 'sample_500' else 'resp'
target_area = ['V1'] if interval == 'sample_500' else ['F1']
elec_type = 'grid'
only_correct_trials = False
# Auto-define number of classes
classes = 2 if decode_for == 'resp' else 5

# Load data and targets
data, n_chans = io.get_subset(file_path, target_area, raw_path,
                              elec_type, return_nchans=True,
                              only_correct_trials=only_correct_trials)
targets = io.get_targets(decode_for, raw_path, elec_type, n_chans,
                         only_correct_trials=only_correct_trials,
                         onehot=False)
samples_per_trial = data.shape[2]
n_chans = data.shape[1]
train_size = .8
test_size = .2
seed = np.random.randint(1, 10000)
indices = np.arange(data.shape[0])

for _ in range(10):
    # train/test params
    train, test, train_labels, test_labels, idx_train, idx_test = (
            train_test_split(
                data,
                targets,
                indices,
                test_size=test_size,
                random_state=seed))
    ind_test = hlp.subset_test(test_labels, classes)  # equal class proportions

    clf = RandomForestClassifier(n_estimators=n_trees)
    clf.fit(train.reshape(train.shape[0], -1),
            train_labels.reshape(-1))
    acc = np.mean(clf.predict(test[ind_test,:,:].reshape(len(ind_test), -1)) == test_labels[ind_test])
    df = pd.DataFrame({'interval': interval,
                       'area': target_area,
                       'trees': n_trees,
                       'acc': acc})
    df = df[['interval', 'area', 'trees', 'acc']]
    print(df)
    # Save to file
    with open(file_out, 'a') as f:
       df.to_csv(f, index=False, header=False)