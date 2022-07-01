#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:21:38 2022

@author: aino

Reads bandpower data from csv files and creates a matrix whose rows represent each subject. 
Plots control vs patient grand average and ROI averages. Plots spectra for different tasks and a single subject and channel.
"""

import numpy as np
import os
import csv
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import log

# Get the list of subjects
with open('/net/tera2/home/aino/work/mtbi-eeg/python/processing/eeg/subjects.txt', 'r') as subjects_file:
    subjects = subjects_file.readlines()
    subjects_file.close()
    # subjects = ['14P'] # Choose subjects manually

# Define which files to read for each subject
tasks = ['ec_1', 'ec_2', 'ec_3', 'eo_1', 'eo_2', 'eo_3', 'PASAT_run1_1', 
         'PASAT_run1_2', 'PASAT_run2_1', 'PASAT_run2_2']
chosen_tasks = ['PASAT_run2_1'] # Choose tasks
subjects_and_tasks = [(x,y) for x in subjects for y in chosen_tasks]

# Create a two dimensional list (and a 3D list) to which the data will be saved
data_vectors = []
data_matrices = []

# Choose one channel and subject to be plotted
channel = 11
chosen_subject = '24P'
plot_array = []

# Lists for grand average and ROI
averages_controls = [[],[],[] ] # all, frontal, occipital
averages_patients= [[],[],[] ] # all, frontal, occipital
averages_problem = []
tp_channels = []
tp_freqs = []

plot_tasks = False
plot_averages = False

# Go through all the subjects
for pair in subjects_and_tasks:
    subject = pair[0].rstrip()
    task = pair[1]
    bandpower_file = "/net/theta/fishpool/projects/tbi_meg/k22_processed/sub-" + subject + "/ses-01/eeg/bandpowers/" + task + '.csv'
    # Create a 2D list to which the read data will be added
    f_bands_list = []
    # Read csv file and save the data to the two dimensional list 'f_bands'
    with open(bandpower_file, 'r') as file:
        reader = csv.reader(file)
        for f_band in reader:
            f_bands_list.append([float(f) for f in f_band])
        file.close()
        
    
    # Vectorize 'f_bands'
    f_bands_array = np.array(f_bands_list)
    f_bands_vector = np.concatenate(f_bands_array)
    
    # Add the vector to 'data_vectors' (this is not in dB)
    data_vectors.append(f_bands_vector)
    data_matrices.append(f_bands_array)
    
    # Total power 
    tp_channel = np.sum(f_bands_array, axis=0)
    tp_freq = np.sum(f_bands_array, axis=1)
    tp_channels.append(tp_channel)
    tp_freqs.append(tp_freq)
    
    # Convert the array to dB
    log_array = 10* np.log10(f_bands_array) 
    
    # Plot different tasks for one subject and channel
    if chosen_subject in subject:
        plot_array.append(log_array[:, channel])
    

    # Grand average and ROI
    sum_all = np.sum(log_array, axis = 1)
    sum_frontal = np.sum(log_array[:, 0:22], axis = 1)
    sum_occipital = np.sum(log_array[:, 54:63], axis = 1)
    
    if 'P' in subject:
        averages_patients[0].append(np.divide(sum_all, 64))
        averages_patients[1].append(np.divide(sum_frontal, 22))
        averages_patients[2].append(np.divide(sum_occipital, 10))
    elif 'C' in subject:
        averages_controls[0].append(np.divide(sum_all, 64))
        averages_controls[1].append(np.divide(sum_frontal, 22))
        averages_controls[2].append(np.divide(sum_occipital, 10))
    else:
        averages_problem.append(subject)
    
"""
Creating a data frame
"""

# Convert 'data_vectors' (2D list) to 2D numpy array
data_matrix = np.array(data_vectors)

# Create indices for dataframe
indices = []
for i in subjects_and_tasks:
    i = i[0].rstrip()+'_'+i[1]
    indices.append(i)

# Convert numpy array to dataframe
data_frame = pd.DataFrame(data_matrix, indices)   

tp_c_dataframe = pd.DataFrame(tp_channels, indices)
tp_f_dataframe = pd.DataFrame(tp_freqs, indices)

# Add column 'Group'
groups = []
for subject in indices:
    if 'P' in subject[2]:
        groups.append(1)
    elif 'C' in subject[2]:
        groups.append(0)
    else:
        groups.append(2) # In case there is a problem
data_frame.insert(0, 'Group', groups)
tp_c_dataframe.insert(0, 'Group', groups)
tp_f_dataframe.insert(0,'Group', groups)

"""
Plotting
"""

# Plot the chosen tasks for some subject and channel
if plot_tasks:
    fig3, ax3 = plt.subplots()
    for index in range(len(chosen_tasks)):
        ax3.plot([x for x in range(1,40)], plot_array[index], label=chosen_tasks[index])
    plt.title('Sub-'+chosen_subject+' Channel '+str(channel + 1))
    ax3.legend()


# Calculate and plot grand average patients vs controls 
if plot_averages:
    controls_total_power = np.sum(averages_controls[0], axis = 0)
    controls_average = np.divide(controls_total_power, 41)
    patients_total_power = np.sum(averages_patients[0], axis = 0)    
    patients_average = np.divide(patients_total_power, 31)
    
    fig, ax = plt.subplots()
    ax.plot([x for x in range(1,40)], controls_average, label='Controls')
    ax.plot([x for x in range(1,40)], patients_average, label='Patients')
    ax.legend()
    
    # fig11, ax11 = plt.subplots()
    # ax11.plot([x for x in range(1,40)], controls_total_power, label= 'Controls')
    # ax11.plot([x for x in range(1,40)], patients_total_power, label= 'Patients')
    
    # Plot region of interest
    # Occipital lobe (channels 55-64)
    controls_sum_o = np.sum(averages_controls[1], axis = 0)
    controls_average_o = np.divide(controls_sum_o, 41)
    patients_sum_o = np.sum(averages_patients[1], axis = 0)    
    patients_average_o = np.divide(patients_sum_o, 31)
    
    fig1, ax1 = plt.subplots()
    ax1.plot([x for x in range(1,40)], controls_average_o, label='Controls')
    ax1.plot([x for x in range(1,40)], patients_average_o, label='Patients')
    plt.title('Occipital lobe')
    ax1.legend()
    
    # Frontal lobe (channels 1-22 (?))
    controls_sum_f = np.sum(averages_controls[2], axis = 0)
    controls_average_f = np.divide(controls_sum_f, 41)
    patients_sum_f = np.sum(averages_patients[2], axis = 0)    
    patients_average_f = np.divide(patients_sum_f, 31)
    
    fig2, ax2 = plt.subplots()
    ax2.plot([x for x in range(1,40)], controls_average_f, label='Controls')
    ax2.plot([x for x in range(1,40)], patients_average_f, label='Patients')
    plt.title('Frontal lobe')
    ax2.legend()






