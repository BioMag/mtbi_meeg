#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:21:38 2022

@author: aino

Reads bandpower data from csv files and creates a matrix whose rows represent each subject. 
"""

import numpy as np
import os
#import scikit 
import csv
import pandas as pd

# Get the list of subjects
with open('/net/tera2/home/aino/work/mtbi-eeg/python/processing/eeg/subjects.txt', 'r') as subjects_file:
    subjects = subjects_file.readlines()
    subjects_file.close()

# Define which files to read for each subject
tasks = ['ec_1', 'ec_2', 'ec_3', 'eo_1', 'eo_2', 'eo_3']
subjects_and_tasks = [(x,y) for x in subjects for y in [tasks[0], tasks[1]]]

# Create a two dimensional list to which the data will be saved
data_vectors = []


# Go through all the subjects
for pair in subjects_and_tasks:
    subject = pair[0].rstrip()
    task = pair[1]
    bandpower_file = "/net/theta/fishpool/projects/tbi_meg/k22_processed/sub-" + subject + "/ses-01/eeg/bandpowers/" + task + '.csv'
    # Create a 2D list which will be vectorized
    f_bands_list = []
    # Read csv file and save the data to the two dimensional list 'f_bands'
    with open(bandpower_file, 'r') as file:
        reader = csv.reader(file)
        for f_band in reader:
            f_bands_list.append(f_band)
        file.close()
        
    # Vectorize 'f_bands'
    f_bands_array = np.array(f_bands_list)
    f_bands_vector = np.concatenate(f_bands_array)
    
    # Add the vector to 'data_vectors'
    data_vectors.append(f_bands_vector)
    
# Convert 'data_vectors' (2D list) to 2D numpy array
data_matrix = np.array(data_vectors)

# Create indices for dataframe
indices = []
for i in subjects_and_tasks:
    i = i[0].rstrip()+'_'+i[1]
    indices.append(i)

# Convert numpy array to dataframe
data_frame = pd.DataFrame(data_matrix, indices)   

# Add column 'Group'
groups = []
for subject in indices:
    if 'P' in subject:
        groups.append(1)
    elif 'C' in subject:
        groups.append(0)
    else:
        groups.append(2) # In case there is a problem
data_frame.insert(0, 'Group', groups)

