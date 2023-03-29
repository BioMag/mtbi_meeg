#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:29:27 2022

@author: aino

Calculates log band power (absolute) for each subject

Running:
import subprocess
subprocess.run('/net/tera2/home/aino/work/mtbi-eeg/python/processing/eeg/runall.sh', shell=True)
"""


import argparse
import h5py
import numpy as np
from pathlib import Path
import time
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config_eeg import fname, thin_bands, wide_bands, processed_data_dir

# Save time of beginning of the execution to measure running time
start_time = time.time()

# Deal with command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', help='The subject to process')
parser.add_argument('--freq_band_type', type=str, help="Define the frequency bands. 'thin' are 1hz bands from 1 to 40hz. 'wide' are conventional delta, theta, etc. Default is 'thin'.", default="thin")

args = parser.parse_args()

if args.freq_band_type == 'wide':
    f_bands =  wide_bands  
elif args.freq_band_type == 'thin':
    f_bands = thin_bands


normalize_ch_power = False

# A list for corruprted or missing psds files
corrupted_psds_files = []

try:
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
    
    
    #Create a directory to save the .csv files
    directory = f'{processed_data_dir}sub-{args.subject}/ses-01/eeg/bandpowers/'
    Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Calculate the average bandpower for each PSD
    for data_obj in list(data.keys()):
        data_bandpower =[] 
        data_arr = data[data_obj]
        
        if normalize_ch_power:
            ch_tot_powers = np.sum(data_arr, axis = 1)
            data_arr = data_arr / ch_tot_powers[:,None]
        
        for fmin, fmax  in f_bands:           
            min_index = np.argmax(freqs > fmin) - 1
            max_index = np.argmax(freqs > fmax) - 1
            
            #NOTE: trapezoidal rule gives weird results. Changed to mean.
            #bandpower = np.trapz(data_arr[:, min_index: max_index], freqs[min_index: max_index], axis = 1)
            bandpower = np.mean(data_arr[:, min_index: max_index],axis=1)
            
            data_bandpower.append(bandpower)
        
        # Save the calculated bandpowers
        np.savetxt(directory + '/' + data_obj + '.csv', data_bandpower, delimiter=',' )
except:
    print("Psds file corrupted or missing")
    corrupted_psds_files.append(args.subject)

# TODO: Once the folder structure is defined, re-code the path depending on where is this expected
# with open('psds_corrupted_or_missing.txt', 'a') as  
try: 
    with open('/net/tera2/home/heikkiv/work_s2022/mtbi-eeg/python/processing/eeg/psds_corrupted_or_missing.txt', 'a') as file:
        for bad_file in corrupted_psds_files:
            file.write(bad_file+'\n')
        file.close()
except PermissionError:
    print('No permission to access this file')
    
    
# Calculate time that the script takes to run
execution_time = (time.time() - start_time)
print('\n###################################################\n')
print(f'Execution time of 04_bandpower.py is: {round(execution_time,2)} seconds\n')
print('###################################################\n')