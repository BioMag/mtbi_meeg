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

# Save time of beginning of the execution to measure running time
start_time = time.time()

from config_eeg import fname, f_bands

# Deal with command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', help='The subject to process')
args = parser.parse_args()


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
    directory = "/net/theta/fishpool/projects/tbi_meg/k22_processed/sub-" + args.subject + "/ses-01/eeg/bandpowers/"
    Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Calculate the average bandpower for each PSD
    for data_obj in list(data.keys()):
        data_bandpower =[] 
        for band in f_bands:
            fmin, fmax = band[0], band[1]
            
            min_index = np.argmax(freqs > fmin) - 1
            max_index = np.argmax(freqs > fmax) -1
            
            bandpower = np.trapz(data[data_obj][:, min_index: max_index], freqs[min_index: max_index], axis = 1)
            
            data_bandpower.append(bandpower)
        
        # Save the calculated bandpowers
        np.savetxt(directory + '/' + data_obj + '.csv', data_bandpower, delimiter=',' )
except:
    print("Psds file corrupted or missing")
    corrupted_psds_files.append(args.subject)
    
with open('/net/tera2/home/portae1/biomag/mtbi-eeg/python/processing/eeg/psds_corrupted_or_missing.txt', 'a') as file:
    for bad_file in corrupted_psds_files:
        file.write(bad_file+'\n')
    file.close()

# Calculate time that the script takes to run
execution_time = (time.time() - start_time)
print('\n###################################################\n')
print(f'Execution time in seconds of 04_bandpower is: {round(execution_time,2)} seconds\n')
print('###################################################\n')

