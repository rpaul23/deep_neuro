import sys

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import myio as io
import helpers as hlp

def train_classif(x_train, y_train, layer_sizes, alpha, batch_size):
    """Trains a multi-layer perceptron."""
    clf = MLPClassifier(hidden_layer_sizes=layer_sizes,
                        activation='relu',
                        solver='adam',
                        alpha=alpha,
                        batch_size=batch_size,
                        learning_rate_init=1e-5,
                        max_iter=200)
    clf.fit(x_train, y_train)
    return clf

def evaluate_classif(clf, x_test, y_test):
    """Evaluates MLP classifier against test data."""
    return np.mean(clf.predict(x_test) == y_test)

counter = int(sys.argv[1])
base_path = '/home/jannesschaefer/'
data_path = base_path + 'data/pre-processed/sample_500/'
raw_path = base_path + 'data/raw/141023/session01/'
file_in = '141023_freq1000low5hi450order3.npy'
file_path = data_path + file_in
file_out = base_path + 'results/tuning/nn_tuning.csv'

decode_for = 'stim'
classes = 5
only_correct_trials = False
target_area = ['V1']
elec_type = 'grid'

# Load data and targets
data, n_chans = io.get_subset(file_path, target_area, raw_path,
                              elec_type, return_nchans=True,
                              only_correct_trials=only_correct_trials)
targets = io.get_targets(decode_for, raw_path, elec_type, n_chans,
                         only_correct_trials=only_correct_trials,
                         onehot=False)

params = {
    'batch_size': [50, 100],
    'alpha': [.01, .1, 1, 10, 50],
    'layer_sizes': [(10,), (20,), (50,), (20, 10), (50, 20)]
    }

ind_batch_size = 0 if counter < len(params['alpha']) * len(params['layer_sizes']) else 1
temp = counter if ind_batch_size == 0 else counter - 25
ind_alpha = int(np.floor(temp/len(params['layer_sizes'])))
ind_layer_sizes = counter % len(params['layer_sizes'])

batch_size = params['batch_size'][ind_batch_size]
alpha = params['alpha'][ind_alpha]
layer_sizes = params['layer_sizes'][ind_layer_sizes]

# train/test params
samples_per_trial = data.shape[2]
n_chans = data.shape[1]
train_size = .8
test_size = .2
seed = np.random.randint(1,10000)
indices = np.arange(data.shape[0])

for i in range(5):
    train, test, train_labels, test_labels, idx_train, idx_test = (
            train_test_split(
                data,
                targets,
                indices,
                test_size=test_size,
                random_state=seed))
    ind_test = hlp.subset_test(test_labels, classes)  # equal class proportions

    x_train = train.reshape(train.shape[0], -1)
    y_train = train_labels.reshape(-1)
    x_test = test.reshape(test.shape[0], -1)
    y_test = test_labels.reshape(-1)

    clf = train_classif(x_train, y_train,
                        layer_sizes=layer_sizes,
                        alpha=alpha,
                        batch_size=batch_size)
    acc = evaluate_classif(clf, x_test, y_test)
    df = pd.DataFrame({'batch_size': [batch_size],
                       'alpha': [alpha],
                       'layers': str(layer_sizes),
                       'acc': [acc]})
    df = df[['batch_size', 'alpha', 'layers', 'acc']]
    # Save to file
    with open(file_out, 'a') as f:
       df.to_csv(f, index=False, header=False)
