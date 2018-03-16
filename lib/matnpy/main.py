# Imports
import numpy as np

import lib.io as io
import lib.preprocess as pp


# Hyper params
sessions = ['141014', '141015', '141016', '141017', '141023']
for sess in sessions:
    sess_no = sess
    align_on = 'match'  # choose to align on either 'stim' or 'match' onset
    save_in_folder = 'match_100'
    start = 0
    stop = 100
    trial_length = 100
    
    # Filter params
    lowcut = 5
    highcut = 450
    order = 3
    
    # Paths
    session_path = '/media/jannes/disk2/raw/' + sess_no + '/session01/'
    rinfo_path = session_path + 'recording_info.mat'
    tinfo_path = session_path + 'trial_info.mat'
    path_out = '/media/jannes/disk2/pre-processed/' + save_in_folder + '/'
    
    # Slicing params
    srate = io.get_sfreq(rinfo_path)
    no_of_trials = io.get_number_of_trials(tinfo_path)
    highest_filename = int(max(io.get_trial_ids(session_path)))
    n_chans = io.get_number_of_channels(rinfo_path)
    channels = [ch for ch in range(n_chans)]
    
    # preallocate space, declare counters
    filtered = np.empty([no_of_trials, 
                         len(channels), 
                         int(trial_length * srate/1000)])
    trial_counter = 0; counter = 0
    
    while trial_counter < highest_filename:
        no_of_zeros = 4-len(str(trial_counter+1))
        trial_str = '0' * no_of_zeros + str(trial_counter+1)  # creates 4-digit string containing str(i+1) at the end and fills up with 0s
        filename = sess_no + '01.' + trial_str + '.mat'
        if align_on == 'stim':
            onset = io.get_sample_on(tinfo_path)[trial_counter].item()  # sample onset in ms
        else:
            onset = io.get_match_on(tinfo_path)[trial_counter].item()  # match onset in ms
        if np.isnan(onset):
            print('No onset for ' + filename)
            trial_counter += 1
            counter += 1
            continue
        
        print(filename)
        try:
            raw = io.get_data(session_path + filename)
            temp = pp.strip_data(raw, 
                                 rinfo_path, 
                                 tinfo_path, 
                                 onset, 
                                 start=start, 
                                 length=trial_length)
            temp = pp.butter_bandpass_filter(temp, 
                                             lowcut, 
                                             highcut, 
                                             srate, 
                                             order)
            
            # Some trials might be shorter than defined trial_length, drop them
            if temp.shape[1] == trial_length:
                filtered[counter] = temp
                
            counter += 1
            
        except IOError as e:
            print('No file ' + filename)
        
        trial_counter += 1
     
    # Convert list to ndarray
    filtered = np.array(filtered)
        
    # save
    np.save(path_out + sess_no + '_freq1000low' + str(lowcut) + 'hi' + str(highcut) + 'order' + str(order) + '.npy', filtered)
