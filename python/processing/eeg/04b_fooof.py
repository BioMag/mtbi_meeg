#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fits a FOOOF (fitting oscillations & one over f) model for PSDs.
Creates alternative/additional features to channel bandpowers 

Created on Fri Feb 24 12:26:23 2023

@author: heikkiv
"""

import numpy as np
import h5py 
import mne
import argparse
from pathlib import Path
from fooof import FOOOF
import datetime
import time
from config_eeg import fname, f_bands

import sys
sys.path.append('../analysis/')
#from 01_read_processed_data import define_subtasks #THIS DOES NOT WORK due to number in the beginning


def define_subtasks(task):
    """
    Define the subtasks to be used for the analysis
    
    
    Input parameters
    ---------
    - task: chosen task (eyes open, eyes closed, Paced Auditory Serial Addition Test 1 or PASAT 2)
    
    Returns
    -------
    - chosen_tasks: The list of chosen subtasks
    """
    tasks = [['ec_1', 'ec_2', 'ec_3'], 
             ['eo_1', 'eo_2', 'eo_3'], 
             ['PASAT_run1_1', 'PASAT_run1_2'], 
             ['PASAT_run2_1', 'PASAT_run2_2']]
       
    # Define which files to read for each subject
    if task == 'ec':
        chosen_tasks = tasks[0]
    elif task == 'eo':
        chosen_tasks = tasks[1]
    elif task == 'PASAT_1':
        chosen_tasks = tasks[2]
    elif task == 'PASAT_2': 
        chosen_tasks = tasks[3]
    else:
        raise("Incorrect task")
    
    
    return chosen_tasks


# Save time of beginning of the execution to measure running time
start_time = time.time()

# Deal with command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', help='The subject to process')
parser.add_argument('task', help='Which measurement condition to use')
args = parser.parse_args()



subject_psds = fname.psds(subject=args.subject, ses='01')
  
f = h5py.File(subject_psds, 'r')
psds_keys = list(f.keys())
psds_data = f[psds_keys[0]]
data_keys = list(psds_data)
data = dict()

# Add the data for each PSD to the dictionary 'data'
for i in data_keys:
    if 'eo' in i or 'ec' in i or 'PASAT' in i:
        dict_key = i.removeprefix('key_')
        data[dict_key]=np.array(psds_data[i])
freqs = np.array(psds_data['key_freqs'])
info_keys = list(psds_data['key_info'])

f.close()

  # Calculate the average bandpower for each PSD
for data_obj in list(data.keys()):
    data_bandpower =[] 
    for band in f_bands:
        fmin, fmax = band[0], band[1]
        
        min_index = np.argmax(freqs > fmin) - 1
        max_index = np.argmax(freqs > fmax) -1
        
        bandpower = np.trapz(data[data_obj][:, min_index: max_index], freqs[min_index: max_index], axis = 1)
        
        data_bandpower.append(bandpower)
    
    avg_bandpower=np.array([np.mean(power) for power in data_bandpower])
    freqs=np.arange(1,90, step=1)
    
    # Initialize a FOOOF object (Here on a rougher scale)
    fm = FOOOF()
    # Set the frequency range to fit the model
    freq_range = [2, 80]
    # Report: fit the model, print the resulting parameters, and plot the reconstruction
    fm.report(freqs, avg_bandpower, freq_range)


chosen_task = define_subtasks(task=args.task) #TODO: different name perhaps?
spectra = data[chosen_task[0]]
global_avg = np.mean(spectra, axis=0)  #Global characteristics OR analysis on some/all chs?

# I do not yet know how I want the script to be...
# Loop through everyhing or do one task only? Also where to save results?

# Initialize a FOOOF object
fm = FOOOF()

# Set the frequency range to fit the model
freq_range = [2, 60]

# Report: fit the model, print the resulting parameters, and plot the reconstruction
fm.report(freqs, global_avg, freq_range)
    
# Combine peak representations
fm.plot(plot_aperiodic=True, plot_peaks='line-shade-outline', plt_log=False)



# Calculate time that the script takes to run
execution_time = (time.time() - start_time)
print('\n###################################################\n')
print(f'Execution time of 04_bandpower.py is: {round(execution_time,2)} seconds\n')
print('###################################################\n')

    