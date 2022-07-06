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
from sklearn.preprocessing import scale

# Get the list of subjects
with open('/net/tera2/home/aino/work/mtbi-eeg/python/processing/eeg/subjects.txt', 'r') as subjects_file:
    subjects = subjects_file.readlines()
    subjects_file.close()
    
tasks = [['ec_1', 'ec_2', 'ec_3'], ['eo_1', 'eo_2', 'eo_3'], ['PASAT_run1_1', 'PASAT_run1_2'], ['PASAT_run2_1', 'PASAT_run2_2']]
five_bands = [(0,3), (3,7), (7,11), (11,34), (34,40)] # List of tuples (Note: if the bands are changed in 04_bandpower.py, these need to be modified too.)

"""
Choose parameters

"""
# Choose frequency bands
change_bands = False 
new_bands = five_bands

# Define which files to read for each subject
chosen_tasks = tasks[0] # Choose tasks (ec: 0, eo: 1, pasat1: 2, pasat2: 3)
subjects_and_tasks = [(x,y) for x in subjects for y in chosen_tasks] # length = subjects x chosen_tasks

# Choose region of interest (not implemented yet)
region_of_interest = False
channels = []

# Choose normalization methods
channel_scaling = False

# Choose what to plot
plot_tasks = False
plot_averages = False

# Choose one channel and subject to be plotted
channel = 59
chosen_subject = '39C'
plot_array = [] # Contains len(chosen_tasks) vectors (length = 39) (39 frequency bands)



# Create a two dimensional list to which the data will be saved
all_bands_vectors = [] # Contains n (n = subjects x chosen_tasks) vectors (length = 2496 = 64 x 39) (64 channels, 39 frequency bands)


# Lists for grand average and ROI
averages_controls = [[], [], []] # Contains 3 lists (all, frontal, occipital)(length = len(chosen_tasks) x controls) of vectors (lenght = 39)(39 frequency bands)
averages_patients= [[], [], []] # all, frontal, occipital
averages_problem = []


"""
Reading data
"""
# Go through all the subjects
for pair in subjects_and_tasks:
    subject = pair[0].rstrip() # Get subject from subjects_and_tasks
    task = pair[1] # Get task from subjects_and_tasks
    bandpower_file = "/net/theta/fishpool/projects/tbi_meg/k22_processed/sub-" + subject + "/ses-01/eeg/bandpowers/" + task + '.csv'
    
    # Create a 2D list to which the read data will be added
    sub_bands_list = [] # 39 x 64 matrix (64 channels, 39 frequency bands)
    
    # Read csv file and save the data to f_bands_list
    with open(bandpower_file, 'r') as file:
        reader = csv.reader(file)
        for f_band in reader:
            sub_bands_list.append([float(f) for f in f_band])
        file.close()
        
    # Convert list to array    
    sub_bands_array = np.array(sub_bands_list) # n x m matrix (n = channels, m = frequency bands)
    
    # Vectorize 'f_bands'
    f_bands_array = np.array(f_bands_list)
    f_bands_vector = np.concatenate(f_bands_array)
    
    # Add the vector to 'data_vectors' (this is not in dB)
    data_vectors.append(f_bands_vector)
    data_matrices.append(f_bands_array)
    
    # Total power 
    tp_channel = np.sum(f_bands_array, axis=0)
    tp_freq = np.sum(f_bands_array) 
    tp_channels.append(tp_channel)
    tp_freqs.append(tp_freq)
    
    # Vectorize array
    sub_bands_vector = np.concatenate(sub_bands_array)

    # Normalize vectors
    if channel_scaling:
        # Calculate channel total power
        channel_total_power = np.sum(sub_bands_array, axis = 0) # Vector (length = 64) (64 channels)
        # Vectorize matrix and divide channel bandpowers by channel total powers
        sub_bands_vector = np.concatenate(np.divide(sub_bands_array, channel_total_power))
        
    # Add vector to matrix
    all_bands_vectors.append(sub_bands_vector)
        
    """
    For plotting
    """
    # Convert the array to dB
    log_array = 10* np.log10(sub_bands_array)  # 64 x 39 matrix
    
    
    # Plot different tasks for one subject and channel
    if chosen_subject in subject:
        plot_array.append(log_array[:, channel])
    

    # Grand average and ROI 
    sum_all = np.sum(log_array, axis = 1) # Vector (length = 39)
    sum_frontal = np.sum(log_array[:, 0:22], axis = 1) # Vector (length = 39)
    sum_occipital = np.sum(log_array[:, 54:63], axis = 1) # Vector (length = 39)
    
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

# Create indices for dataframe
indices = []
for i in subjects_and_tasks:
    i = i[0].rstrip()+'_'+i[1]
    indices.append(i)

# Convert numpy array to dataframe
dataframe = pd.DataFrame(np.array(all_bands_vectors), indices) #n x m matrix where n = subjects x tasks, m = channels x frequency bands

# Add column 'Group'
groups = []
for subject in indices:
    if 'P' in subject[2]:
        groups.append(1)
    elif 'C' in subject[2]:
        groups.append(0)
    else:
        groups.append(2) # In case there is a problem
dataframe.insert(0, 'Group', groups)




"""
Plotting
"""
patients = sum(groups)/len(chosen_tasks)
controls = len(groups)/len(chosen_tasks)-patients


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
    controls_average = np.divide(controls_total_power, controls)
    patients_total_power = np.sum(averages_patients[0], axis = 0)    
    patients_average = np.divide(patients_total_power, patients)
    
    fig, axes = plt.subplots(1,3)
    axes[0].plot([x for x in range(1,40)], controls_average, label='Controls')
    axes[0].plot([x for x in range(1,40)], patients_average, label='Patients')
    axes[0].legend()
    
    # Plot region of interest
    # Occipital lobe (channels 55-64)
    controls_sum_o = np.sum(averages_controls[1], axis = 0)
    controls_average_o = np.divide(controls_sum_o, controls)
    patients_sum_o = np.sum(averages_patients[1], axis = 0)    
    patients_average_o = np.divide(patients_sum_o, patients)
    
    axes[1].plot([x for x in range(1,40)], controls_average_o, label='Controls')
    axes[1].plot([x for x in range(1,40)], patients_average_o, label='Patients')
    axes[1].title.set_text('Frontal lobe')
    axes[1].legend()
    
    # Frontal lobe (channels 1-22 (?))
    controls_sum_f = np.sum(averages_controls[2], axis = 0)
    controls_average_f = np.divide(controls_sum_f, controls)
    patients_sum_f = np.sum(averages_patients[2], axis = 0)    
    patients_average_f = np.divide(patients_sum_f, patients)
    
    axes[2].plot([x for x in range(1,40)], controls_average_f, label='Controls')
    axes[2].plot([x for x in range(1,40)], patients_average_f, label='Patients')
    axes[2].title.set_text('Occipital lobe')
    axes[2].legend()



    




