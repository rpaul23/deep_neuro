# Imports
import sys
import os

import numpy as np

import matnpyio as io
import preprocess as pp

# Passed params
sess_no = sys.argv[1]
user_name = sys.argv[2]
align_on = sys.argv[3]
from_time = int(sys.argv[4])
to_time = int(sys.argv[5])
trial_length = abs(from_time - to_time)

# Filter params
lowcut = 5
highcut = 450
order = 3

# Paths
base_path = '/home/' + user_name + '/'
raw_path = base_path + 'data/raw/' + sess_no + '/session01/'
prep_path = base_path + 'data/pre-processed/'
rinfo_path = raw_path + 'recording_info.mat'
tinfo_path = raw_path + 'trial_info.mat'

# Define and loop over intervals
srate = io.get_sfreq(rinfo_path)
n_trials = io.get_number_of_trials(tinfo_path)
last_trial = int(max(io.get_trial_ids(raw_path)))
n_chans = io.get_number_of_channels(rinfo_path)
channels = [ch for ch in range(n_chans)]

# Pre-process data
filtered = np.empty([n_trials,
                     len(channels),
                     int(trial_length * srate/1000)])
trial_counter = 0; counter = 0
while trial_counter < last_trial:
    n_zeros = 4-len(str(trial_counter+1))
    trial_str = '0' * n_zeros + str(trial_counter+1)  # fills leading 0s
    file_in = sess_no + '01.' + trial_str + '.mat'
    if align_on == 'stim':
        onset = io.get_sample_on(tinfo_path)[trial_counter].item()
    else:
        onset = io.get_match_on(tinfo_path)[trial_counter].item()
    if np.isnan(onset):  # drop trials for which there is no onset info
        print('No onset for ' + file_in)
        trial_counter += 1
        if trial_counter == last_trial:
            break
        else:
            counter += 1
            continue
    print(file_in)
    try:
        raw = io.get_data(raw_path + file_in)
        temp = pp.strip_data(raw,
                             rinfo_path,
                             onset,
                             start=from_time,
                             length=trial_length)
        temp = pp.butter_bandpass_filter(temp,
                                         lowcut,
                                         highcut,
                                         srate,
                                         order)
        if temp.shape[1] == trial_length:  # drop trials shorter than length
            filtered[counter] = temp
        counter += 1
    except IOError:
        print('No file ' + file_in)
    trial_counter += 1

# Store data
filtered = np.array(filtered)
dir_out = prep_path + 'intervals' + '/'
file_out = (sess_no
            + '_' + align_on
            + 'from' + str(from_time)
            + 'to' + str(to_time)
            + '.npy')
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
np.save(dir_out + file_out, filtered)
print('Successfully saved ' + file_out)