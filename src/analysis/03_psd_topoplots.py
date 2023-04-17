#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots PSD info in an interactive manner.

Plot PSDS as a topoplot:
    - channelwise average PSDS
    - global averages


Created on Fri Apr 14 10:15:13 2023

# TODO: Refactor into modular structure

@author: heikkiv
"""

import os
import sys
import mne
import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
from mne.viz import iter_topography
#from mne.time_frequency import read_spectrum

parent_dir = os.path.abspath(os.path.join(os.path.dirname('src'), '..'))
sys.path.append(parent_dir)
from config_eeg import fname

# COMMENT/Question: is it ok if this cannot be run from console?

#read in the subjects
with open('subjects.txt', 'r') as subjects_file:
    subjects = [line.rstrip() for line in subjects_file.readlines()]

#define task to be plotted
task = 'PASAT_run2'

# Idea: store the data in a ?nested? dictionary
# subj_data = {
#  'task' : task name (maybe unnecessary)
#  'group': patient / control
#  'data' : ndarray of psds?
#  'age' : int, could be added but omitted for now. 
# }

PSD_allsubj = {}

for subject in subjects:
    subject_psds = fname.psds(subject=subject, ses='01')

    try:
        f = h5py.File(subject_psds, 'r')
    except:
        print("Psds file corrupted or missing")
        
    psds_keys = list(f.keys())
    psds_data = f[psds_keys[0]]
    data_keys = list(psds_data)
    data = dict()
    
    freqs = np.array(psds_data['key_freqs']) #extract freq info
    
    if 'P' in subject:
        group='Patient' 
    elif 'C' in subject:
        group='Control'
    
    for i in data_keys:
        if 'eo' in i or 'ec' in i or 'PASAT' in i:
            dict_key = i.removeprefix('key_')
            # Take only the first segment run of the task, for now.
            if task in dict_key and dict_key.endswith('_1'): 
                psds = np.array(psds_data[i])
                # scale to dB
                psds = 20 * np.log10(psds)
                #define the task name 
                task = dict_key.removesuffix('_2') 
                
                PSD_allsubj[subject] = {'task': task, 
                                        'group': group,
                                        'data': psds}            
    f.close()

#%%Create a df from dict
PSD_df = pd.DataFrame.from_dict(PSD_allsubj, orient='index')

# Calculate ch-wise mean psds per group
clinical_groups = PSD_df.groupby('group')

group_ch_means = []
names = [] 
for name, group_df in clinical_groups:
    group_data = group_df['data']
    Arr = np.array([i for i in group_data]) #nsubj x n_chs x n_freq
    mean_data = np.mean(Arr, axis=0) #get mean data over all subjects within group
    group_ch_means.append(mean_data)
    names.append(name)

group_n_mean1 = zip(names, group_ch_means)
group_n_mean = [(name, psd_data) for name, psd_data in group_n_mean1]

#read in one raw data file to get sensor location info
raw = mne.io.read_raw_fif(fname.clean(subject=subject, task='ec', run=1,ses='01'),
                   preload=True)

#%% Plotting functions and utils
def my_callback1(ax, ch_idx):
    """
    This block of code is executed once you click on one of the channel axes
    in the plot. To work with the viz internals, this function should only take
    two parameters, the axis and the channel or data index.
    """
    for name, data in group_n_mean:
        ax.plot(freqs, data[ch_idx], label=name) #for all the group averages
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (dB)')
        ax.legend(loc="upper right")
        
def my_callback2(ax, ch_idx):
    """
    Once clicked in the topoplot, plots cohort mean PSD with individual traces.
    It does so separately for each cohort, otherwise plots become too crowded.
    """
    name, group_mean = group_n_mean[i] #which group. TODO: name!
    cohort_data = clinical_groups.get_group(name)

    ax.plot(freqs, group_mean[ch_idx], color=colors[i], label=f'{name} mean', lw=2) #for all the group averages
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB)')
    ax.legend(loc="upper right")
    
    for index, row in cohort_data.iterrows(): #generally it's ill-advised to loop over df rows
        data = row['data']
        ax.plot(freqs, data[ch_idx], color=colors[i], alpha=0.2) #for all the group averages
        ax.text(90, data[ch_idx,-1], index, size='small')

# Define colours
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

i=1 #zero for controls, 1 for patients

#loop through all channels and create an axis for them
for ax, idx in iter_topography(raw.info,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white',
                               on_pick=my_callback1):
    ax.plot(psds[idx], color='grey') #just to show some general output for the big figure
    
plt.gcf().suptitle(f'Power spectral densities, {task}')
plt.show()